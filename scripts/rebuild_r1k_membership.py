"""
D1: Augment data/universe/russell1000_membership.parquet with all R1K tickers.

Tickers already present keep their historical add/remove dates.
New tickers get added=2019-01-01 (conservative date within WF training window).

Run from repo root:
    python scripts/rebuild_r1k_membership.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

PARQUET = Path(__file__).parent.parent / "data" / "universe" / "russell1000_membership.parquet"
SEED_DATE = date(2019, 1, 1)


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.utils.constants import RUSSELL_1000_TICKERS

    df = pd.read_parquet(PARQUET)
    existing = set(df["ticker"].str.upper())
    new_rows = [
        {"ticker": sym, "added": SEED_DATE, "removed": None}
        for sym in RUSSELL_1000_TICKERS
        if sym not in existing
    ]

    print(f"Existing: {len(existing)}  New: {len(new_rows)}")

    if new_rows:
        combined = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_parquet(PARQUET, index=False)
        print(f"Written: {len(combined)} total rows → {PARQUET}")
    else:
        print("Nothing to add — parquet already complete.")


if __name__ == "__main__":
    main()
