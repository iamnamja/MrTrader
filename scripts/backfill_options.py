"""
Backfill historical option OHLCV + contract universe into data/options_bars.parquet and
data/options_contracts.parquet from Polygon S3 flat files (us_options_opra/day_aggs_v1).

Survivorship-safe by construction: the universe is built FROM the daily flat files (every
contract that actually traded that day, expired ones included), so a backtest as-of date
sees exactly what existed then. PIT-stamped: each bar's knowable_date = trade_date + 1 bday
(the EOD bar prints after the close).

See docs/reference/OPTIONS_DATA.md for the PIT + survivorship contract.

Usage
-----
    python scripts/backfill_options.py --underlyings SPY QQQ AAPL --years 4
    python scripts/backfill_options.py --start 2024-01-01 --end 2024-03-31 --workers 6
    python scripts/backfill_options.py --dry-run
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import pandas as pd  # noqa: E402

from app.data import options_provider as op  # noqa: E402
from app.data.options_provider import knowable_date  # noqa: E402  (holiday-aware)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_options")

# Liquid optionable default set (broad-index ETFs + large single names). The earnings
# IV-crush and VRP strategies (OPT-3/4) only need a focused universe, so we don't pull
# all of OPRA — just the names we'll actually trade/test.
DEFAULT_UNDERLYINGS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX",
    "JPM", "BAC", "XLE", "GLD", "TLT",
]


def _s3():
    from app.config import settings
    if not all([settings.polygon_s3_access_key, settings.polygon_s3_secret_key,
                settings.polygon_s3_endpoint, settings.polygon_s3_bucket]):
        return None
    from app.data.polygon_s3 import PolygonS3
    return PolygonS3(
        access_key=settings.polygon_s3_access_key,
        secret_key=settings.polygon_s3_secret_key,
        endpoint_url=settings.polygon_s3_endpoint,
        bucket=settings.polygon_s3_bucket,
    )


def _business_days(start: date, end: date) -> list:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def filter_day_panel(panel: pd.DataFrame, day: date, pattern, loose=None) -> pd.DataFrame:
    """Keep only standard-OCC rows for our target underlyings, PIT-stamped. Pure (no I/O)
    so it's unit-testable. `loose` (optional) is an O:ROOT-prefix regex used only to count
    adjusted/non-standard roots (e.g. post-split 'AAPL1...') that we deliberately drop —
    logging the count makes split-driven coverage gaps visible instead of silent."""
    if panel is None or panel.empty:
        return pd.DataFrame(columns=op.BARS_COLS)
    ext = panel["contract"].str.extract(pattern)  # 0=root,1=ymd,2=cp,3=strike
    mask = ext[0].notna()
    if loose is not None:
        loose_hits = int(panel["contract"].str.match(loose).sum())
        dropped = loose_hits - int(mask.sum())
        if dropped > 0:
            logger.debug("%s: dropped %d non-standard/adjusted-root contracts", day, dropped)
    if not mask.any():
        return pd.DataFrame(columns=op.BARS_COLS)
    sub = panel[mask].copy()
    sub["underlying"] = ext[0][mask].values
    # Force datetime64[ns] (pandas 2.x infers second-resolution from a date scalar, which
    # would clash with the ns store on concat). Keep one canonical resolution everywhere.
    sub["date"] = pd.Series([pd.Timestamp(day)] * len(sub),
                            index=sub.index).astype("datetime64[ns]")
    kd = pd.Timestamp(knowable_date(day, op.OPT_BAR_LAG_BDAYS))
    sub["knowable_date"] = pd.Series([kd] * len(sub),
                                     index=sub.index).astype("datetime64[ns]")
    return sub[op.BARS_COLS]


def _fetch_day(s3, day: date, pattern, loose=None) -> pd.DataFrame:
    """Download one OPRA day file and filter it (see filter_day_panel)."""
    panel = s3.get_options_day_file(day)
    return filter_day_panel(panel, day, pattern, loose=loose)


def _merge(existing: pd.DataFrame, parts: list, cols: list, key: list) -> pd.DataFrame:
    frames = [existing] if existing is not None and not existing.empty else []
    frames += [p for p in parts if p is not None and not p.empty]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)
    return out[cols]


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill option OHLCV + universe (Polygon S3)")
    p.add_argument("--underlyings", nargs="+", default=None,
                   help=f"Underlyings (default: {len(DEFAULT_UNDERLYINGS)} liquid names)")
    p.add_argument("--years", type=int, default=4, help="History depth if --start omitted")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: yesterday)")
    p.add_argument("--workers", type=int, default=4, help="Parallel day-file downloads")
    p.add_argument("--max-days", type=int, default=None, help="Cap (debug)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    underlyings = [u.upper() for u in (args.underlyings or DEFAULT_UNDERLYINGS)]
    end = (date.fromisoformat(args.end) if args.end
           else date.today() - timedelta(days=1))
    start = (date.fromisoformat(args.start) if args.start
             else end - timedelta(days=365 * args.years))
    days = _business_days(start, end)
    if args.max_days:
        days = days[-args.max_days:]
    pattern = op.occ_root_pattern(underlyings)
    # loose O:ROOT-prefix matcher (any root form) — used only to count dropped
    # adjusted/non-standard roots so split-driven coverage gaps are visible.
    import re
    loose = re.compile(r"^O:(" + "|".join(re.escape(u) for u in underlyings) + r")")

    logger.info("underlyings=%d window=%s→%s (%d business days) workers=%d",
                len(underlyings), start, end, len(days), args.workers)
    if args.dry_run:
        existing = op.load_options_bars(refresh=True)
        logger.info("[dry-run] %s; existing bars=%d contracts=%d; sample underlyings=%s",
                    f"{len(days)} day-files to scan", len(existing),
                    existing["contract"].nunique() if not existing.empty else 0,
                    underlyings[:6])
        return 0

    s3 = _s3()
    if s3 is None:
        logger.error("Polygon S3 credentials missing (polygon_s3_*). Cannot backfill.")
        return 2

    parts, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_fetch_day, s3, d, pattern, loose): d for d in days}
        for fut in as_completed(futs):
            df = fut.result()
            if not df.empty:
                parts.append(df)
            done += 1
            if done % 50 == 0:
                rows = sum(len(x) for x in parts)
                logger.info("  %d/%d day-files scanned (%d bar rows so far)",
                            done, len(days), rows)

    existing = op.load_options_bars(refresh=True)
    merged = _merge(existing, parts, op.BARS_COLS, ["contract", "date"])
    op.OPTIONS_BARS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(op.OPTIONS_BARS_PARQUET, index=False)
    logger.info("wrote %s  (%d bar rows, %d contracts, %d underlyings)",
                op.OPTIONS_BARS_PARQUET, len(merged),
                merged["contract"].nunique() if not merged.empty else 0,
                merged["underlying"].nunique() if not merged.empty else 0)

    contracts = op.contracts_from_bars(merged)
    contracts.to_parquet(op.OPTIONS_CONTRACTS_PARQUET, index=False)
    logger.info("wrote %s  (%d contracts; expired+active, survivorship-safe)",
                op.OPTIONS_CONTRACTS_PARQUET, len(contracts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
