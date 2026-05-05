"""
Phase 89b — Historical Sector ETF Backfill

Fetches daily OHLCV bars for all 11 sector ETFs (XLK, XLF, XLE, etc.) from
Alpaca and writes a point-in-time parquet store. Training loads from this store
instead of making live Alpaca calls, allowing sector_momentum and
sector_momentum_5d to be un-pruned from PRUNED_FEATURES.

Output:
  data/sector_etf/sector_etf_history.parquet
  Columns: etf (str), date (str YYYY-MM-DD), open, high, low, close, volume

Usage:
  python scripts/backfill_sector_etf_history.py [--days 1260] [--dry-run]

Default --days 1260 covers ~5 years (enough for all training windows).
Runtime: <1 min (11 ETFs, 1 bulk Alpaca call each).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
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
    from app.integrations import get_alpaca_client

    client = get_alpaca_client()
    end_dt = datetime.utcnow().date()
    start_dt = end_dt - timedelta(days=days)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Sector ETF backfill: %d ETFs, %d days back (from %s to %s), dry_run=%s",
                len(SECTOR_ETFS), days, start_dt, end_dt, dry_run)

    import pandas as pd
    all_frames: list[pd.DataFrame] = []
    ok = errors = 0

    for etf in SECTOR_ETFS:
        try:
            df = client.get_bars(
                etf,
                timeframe="1Day",
                start=start_dt.isoformat(),
                end=end_dt.isoformat(),
            )
            if df is None or df.empty:
                logger.warning("%s: no data returned", etf)
                errors += 1
                continue

            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df["date"] = df.index.strftime("%Y-%m-%d")
            df["etf"] = etf
            df = df.reset_index(drop=True)
            all_frames.append(df)
            ok += 1
            logger.info("%s: %d bars", etf, len(df))
        except Exception as exc:
            errors += 1
            logger.warning("%s: failed — %s", etf, exc)

    logger.info("Fetch complete: %d ETFs ok, %d errors", ok, errors)

    if dry_run:
        total = sum(len(f) for f in all_frames)
        logger.info("[DRY-RUN] Would write %d rows to %s", total, OUTPUT_PATH)
        if all_frames:
            logger.info("Sample rows:\n%s", all_frames[0].head(3).to_string())
        return

    if not all_frames:
        logger.warning("No rows to write — exiting")
        return

    df_out = pd.concat(all_frames, ignore_index=True)
    df_out = df_out[["etf", "date", "open", "high", "low", "close", "volume"]]
    df_out = df_out.sort_values(["etf", "date"]).reset_index(drop=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Written: %s (%d rows, %d ETFs)", OUTPUT_PATH, len(df_out), df_out["etf"].nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical sector ETF bars from Alpaca")
    parser.add_argument("--days", type=int, default=1260, help="Calendar days to backfill (default 1260 = ~5yr)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(days=args.days, dry_run=args.dry_run)
