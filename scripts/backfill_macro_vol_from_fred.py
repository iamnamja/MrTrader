"""Backfill missing vix / vix3m in macro_history.parquet from FRED (2026-09-01).

yfinance returns a frame for ^VIX3M whose Close is almost entirely NaN, so the column thinned
out without ever tripping the "No close data returned" warning. Coverage of rows carrying BOTH
series decayed 100% (through Apr 2026) -> 73% (Jun) -> 39% (Jul) -> 5% (Aug), last complete row
2026-08-05.

That silently disabled the crash governor: it needs vix and vix3m on the SAME settled date to
compute the term-structure ratio, and being fail-safe it just returned 1.0 (no de-risk). The only
symptom was one WARNING per weekly rebalance.

`_fetch_closes` now backstops new rows from FRED, but rows already written keep their NaNs — this
repairs the existing file. Idempotent: only ever writes where a value is absent, so a real
yfinance value is never overwritten.

    python scripts/backfill_macro_vol_from_fred.py --dry-run
    python scripts/backfill_macro_vol_from_fred.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    from app.data.macro_history import (
        MACRO_PATH, FRED_SERIES, _fetch_fred_series, load_macro_history,
    )

    df = load_macro_history()
    if df is None or df.empty:
        print("macro_history is empty — nothing to backfill.")
        return 1

    dates = df["date"].astype(str).str[:10]
    start, end = dates.min(), dates.max()
    both_before = int((df["vix"].notna() & df["vix3m"].notna()).sum())
    print(f"rows={len(df)}  range={start}..{end}")
    print(f"BEFORE: vix missing={int(df['vix'].isna().sum())}  "
          f"vix3m missing={int(df['vix3m'].isna().sum())}  both-present={both_before}")

    total_filled = 0
    for col, series_id in FRED_SERIES.items():
        missing = df[col].isna()
        if not missing.any():
            print(f"  {col}: nothing missing")
            continue
        obs = _fetch_fred_series(series_id, start, end)
        if not obs:
            print(f"  {col}: FRED returned nothing for {series_id} — skipped")
            continue
        filled = dates[missing].map(obs)
        n = int(filled.notna().sum())
        total_filled += n
        print(f"  {col}: {int(missing.sum())} missing, {len(obs)} FRED obs -> fill {n}")
        # Apply in BOTH modes so the projected after-count below is truthful; --dry-run simply
        # never writes the file. A preview that reports a number the real run would not produce
        # is worse than no preview.
        if n:
            df.loc[missing, col] = filled

    both_after = int((df["vix"].notna() & df["vix3m"].notna()).sum())
    print(f"AFTER : both-present={both_after}  (+{both_after - both_before})")

    if not args.apply:
        print("\nDRY RUN — nothing written.")
        return 0

    backup = MACRO_PATH.with_suffix(".parquet.bak")
    load_macro_history().to_parquet(backup, index=False)
    df.to_parquet(MACRO_PATH, index=False)
    print(f"\nWROTE {MACRO_PATH}  (backup: {backup})  filled {total_filled} value(s)")

    # The point of the exercise: can the governor actually compute now?
    chk = pd.read_parquet(MACRO_PATH)
    tail = chk.tail(10)
    print("\nlast 10 rows (vix / vix3m):")
    print(tail[["date", "vix", "vix3m"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
