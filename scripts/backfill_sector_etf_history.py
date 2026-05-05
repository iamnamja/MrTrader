"""
Phase 89b — Historical Sector ETF Backfill

Fetches 5yr daily bars for all 11 sector ETFs (XLK, XLF, XLE, etc.) from
Polygon (via the same S3/REST provider used for regular symbols) and writes
them into data/cache/daily/ — the same cache training reads from.

Training then looks up sector ETF bars PIT the same way it looks up stock bars.

Output:
  data/cache/daily/XLK.parquet  (and XLC, XLY, XLP, XLF, XLV, XLI, XLE, XLB, XLRE, XLU)
  Also writes data/sector_etf/sector_etf_history.parquet for standalone use.

Usage:
  python scripts/backfill_sector_etf_history.py [--days 1260] [--dry-run]

Runtime: <1 min (11 ETFs, Polygon S3 bulk download).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sector_etf_backfill")

OUTPUT_PATH = Path("data/sector_etf/sector_etf_history.parquet")

SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLB", "XLRE", "XLU"]


def run(days: int = 1260, dry_run: bool = False) -> None:
    import pandas as pd
    from app.data.polygon_provider import PolygonProvider
    from app.data.cache import get_cache

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=days)

    logger.info("Sector ETF backfill: %d ETFs, %d days back (%s to %s), dry_run=%s",
                len(SECTOR_ETFS), days, start_dt, end_dt, dry_run)

    if dry_run:
        logger.info("[DRY-RUN] Would fetch %s from %s to %s via Polygon", SECTOR_ETFS, start_dt, end_dt)
        return

    provider = PolygonProvider()
    data = provider.get_daily_bars_bulk(SECTOR_ETFS, start_dt, end_dt)

    ok = len(data)
    errors = len(SECTOR_ETFS) - ok
    logger.info("Fetch complete: %d ETFs ok, %d errors", ok, errors)

    for etf, df in data.items():
        logger.info("%s: %d bars (%s to %s)", etf, len(df),
                    df.index.min().date(), df.index.max().date())

    if not data:
        logger.warning("No data fetched — exiting")
        return

    # Also write standalone parquet for direct use
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for etf, df in data.items():
        df2 = df[["open", "high", "low", "close", "volume"]].copy()
        df2["date"] = df2.index.strftime("%Y-%m-%d")
        df2["etf"] = etf
        all_frames.append(df2.reset_index(drop=True))

    df_out = pd.concat(all_frames, ignore_index=True)
    df_out = df_out[["etf", "date", "open", "high", "low", "close", "volume"]]
    df_out = df_out.sort_values(["etf", "date"]).reset_index(drop=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Written: %s (%d rows, %d ETFs)", OUTPUT_PATH, len(df_out), df_out["etf"].nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical sector ETF bars via Polygon")
    parser.add_argument("--days", type=int, default=1260, help="Calendar days to backfill (default 1260 = ~5yr)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(days=args.days, dry_run=args.dry_run)
