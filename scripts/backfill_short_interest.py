"""
Backfill short interest + short volume (FINRA-sourced, via Polygon) into
data/short_interest.parquet and data/short_volume.parquet.

Survivorship-safe: defaults to the point-in-time UNION of Russell-1000 membership
over the data window (captures index adds/removes incl. delisted names), so a
backtest as-of date sees exactly what was knowable then.

See docs/reference/SHORT_INTEREST_DATA.md for the PIT contract.

Usage
-----
    python scripts/backfill_short_interest.py --workers 6
    python scripts/backfill_short_interest.py --symbols AAPL GME --dry-run
    python scripts/backfill_short_interest.py --incremental
    python scripts/backfill_short_interest.py --no-sv          # short interest only
    python scripts/backfill_short_interest.py --sv-years 5     # deeper daily history
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import pandas as pd  # noqa: E402

from app.data import short_interest_provider as sip  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_short_interest")

_SI_START = date(2017, 1, 1)   # Polygon SI history begins ~2017-12


def _resolve_symbols(arg_value) -> list[str]:
    if arg_value and arg_value != ["all"]:
        return list(arg_value)
    # Survivorship-safe default: PIT union of R1K membership over the data window.
    today = date.today()
    try:
        from app.data.universe_history import pit_union, historical_trade_symbols
        extra = historical_trade_symbols(_SI_START, today)
        syms = pit_union("russell_1000", _SI_START, today, extra_symbols=extra)
        if syms:
            return syms
    except Exception as exc:  # pragma: no cover - infra fallback
        logger.warning("pit_union unavailable (%s); falling back to current R1K", exc)
    from app.utils.constants import RUSSELL_1000_TICKERS
    return list(RUSSELL_1000_TICKERS)


def _latest_per_ticker(df: pd.DataFrame, date_col: str) -> dict:
    if df is None or df.empty:
        return {}
    g = df.groupby("ticker")[date_col].max()
    return {t: pd.Timestamp(v).date() for t, v in g.items()}


def _fetch_one(symbol: str, do_sv: bool, sv_start: date,
               si_skip_before: date | None, sv_since: dict) -> tuple:
    """Returns (symbol, si_df, sv_df). Empty frames on no-new / error."""
    si_df = pd.DataFrame(columns=sip._SI_COLS)
    sv_df = pd.DataFrame(columns=sip._SV_COLS)
    try:
        # Short interest: cheap (bi-monthly) -> always fetch full, dedupe on merge.
        # In incremental mode skip tickers whose latest settlement is still fresh.
        if si_skip_before is None:
            rows = sip.fetch_short_interest(symbol)
            si_df = sip.short_interest_to_df(symbol, rows)
        if do_sv:
            start = sv_since.get(symbol, sv_start)
            # re-fetch the last stored day too (cheap) to fill any gaps
            start = max(sv_start, start - timedelta(days=3))
            rows = sip.fetch_short_volume(symbol, start=start)
            sv_df = sip.short_volume_to_df(symbol, rows)
    except Exception as exc:
        logger.warning("fetch failed for %s: %s", symbol, exc)
    return symbol, si_df, sv_df


def _merge(existing: pd.DataFrame, new_parts: list[pd.DataFrame],
           cols: list[str], key: list[str]) -> pd.DataFrame:
    frames = [existing] if existing is not None and not existing.empty else []
    frames += [p for p in new_parts if p is not None and not p.empty]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)
    return out[cols]


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill short interest + short volume (Polygon/FINRA)")
    p.add_argument("--symbols", nargs="+", default=None, help="Symbols (default: PIT R1K union)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--incremental", action="store_true",
                   help="Skip SI for tickers whose latest settlement is <12d old; fetch SV only since last stored day")
    p.add_argument("--sv-years", type=int, default=3, help="Daily short-volume history depth (default 3y)")
    p.add_argument("--no-sv", action="store_true", help="Short interest only (skip daily short volume)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    symbols = _resolve_symbols(args.symbols)
    do_sv = not args.no_sv
    sv_start = date.today() - timedelta(days=365 * args.sv_years)

    existing_si = sip.load_short_interest(refresh=True)
    existing_sv = sip.load_short_volume(refresh=True)
    si_latest = _latest_per_ticker(existing_si, "settlement_date")
    sv_latest = _latest_per_ticker(existing_sv, "date")

    fresh_cut = date.today() - timedelta(days=12)
    logger.info("symbols=%d workers=%d sv=%s sv_years=%d incremental=%s dry_run=%s",
                len(symbols), args.workers, do_sv, args.sv_years, args.incremental, args.dry_run)

    if args.dry_run:
        logger.info("[dry-run] would fetch %d symbols; existing SI rows=%d SV rows=%d",
                    len(symbols), len(existing_si), len(existing_sv))
        logger.info("[dry-run] sample: %s", symbols[:10])
        return 0

    si_parts, sv_parts = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for s in symbols:
            si_skip = None
            if args.incremental and si_latest.get(s) and si_latest[s] >= fresh_cut:
                si_skip = si_latest[s]  # fresh -> skip SI fetch
            since = {s: sv_latest[s]} if (args.incremental and sv_latest.get(s)) else {}
            futs[ex.submit(_fetch_one, s, do_sv, sv_start, si_skip, since)] = s
        for fut in as_completed(futs):
            sym, si_df, sv_df = fut.result()
            if not si_df.empty:
                si_parts.append(si_df)
            if not sv_df.empty:
                sv_parts.append(sv_df)
            done += 1
            if done % 50 == 0:
                logger.info("  %d/%d fetched", done, len(symbols))

    merged_si = _merge(existing_si, si_parts, sip._SI_COLS, ["ticker", "settlement_date"])
    sip.SI_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged_si.to_parquet(sip.SI_PARQUET, index=False)
    logger.info("wrote %s  (%d rows, %d tickers)", sip.SI_PARQUET,
                len(merged_si), merged_si["ticker"].nunique() if not merged_si.empty else 0)

    if do_sv:
        merged_sv = _merge(existing_sv, sv_parts, sip._SV_COLS, ["ticker", "date"])
        merged_sv.to_parquet(sip.SV_PARQUET, index=False)
        logger.info("wrote %s  (%d rows, %d tickers)", sip.SV_PARQUET,
                    len(merged_sv), merged_sv["ticker"].nunique() if not merged_sv.empty else 0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
