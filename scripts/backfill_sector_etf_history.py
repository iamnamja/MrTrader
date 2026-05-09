"""
Phase 89b — Historical Sector ETF Backfill

Fetches daily bars for all 11 sector ETFs (XLK, XLF, XLE, etc.) and writes
data/sector_etf/sector_etf_history.parquet plus the per-symbol cache.

Defaults to Polygon (S3 bulk) when available; falls back to yfinance otherwise.
Supports --incremental to append only new dates after the latest in the file.

Usage:
  python scripts/backfill_sector_etf_history.py                    # full 10yr backfill
  python scripts/backfill_sector_etf_history.py --incremental      # append latest dates
  python scripts/backfill_sector_etf_history.py --days 1260        # custom window
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sector_etf_backfill")

OUTPUT_PATH = Path("data/sector_etf/sector_etf_history.parquet")

SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLB", "XLRE", "XLU"]


def _fetch_polygon(start_dt: date, end_dt: date) -> dict:
    try:
        from app.data.polygon_provider import PolygonProvider
        provider = PolygonProvider()
        return provider.get_daily_bars_bulk(SECTOR_ETFS, start_dt, end_dt)
    except Exception as exc:
        logger.warning("Polygon unavailable (%s) — falling back to yfinance", exc)
        return {}


def _fetch_yfinance(start_dt: date, end_dt: date) -> dict:
    import pandas as pd
    import yfinance as yf

    logger.info("Fetching %d ETFs from yfinance %s → %s", len(SECTOR_ETFS), start_dt, end_dt)
    try:
        raw = yf.download(
            SECTOR_ETFS,
            start=start_dt.isoformat(),
            end=(end_dt + timedelta(days=1)).isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as exc:
        logger.error("yfinance bulk download failed: %s", exc)
        return {}

    result = {}
    for etf in SECTOR_ETFS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[etf].copy()
            else:
                df = raw.copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=["close"])
            if not df.empty and len(df) >= 5:
                result[etf] = df[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as exc:
            logger.warning("yfinance parse failed for %s: %s", etf, exc)
    return result


def run(days: int = 2520, dry_run: bool = False, incremental: bool = False) -> None:
    import pandas as pd

    end_dt = date.today()

    if incremental and OUTPUT_PATH.exists():
        try:
            existing = pd.read_parquet(OUTPUT_PATH)
            latest = existing["date"].max()
            from datetime import datetime as _dt
            start_dt = _dt.strptime(latest, "%Y-%m-%d").date() + timedelta(days=1)
            logger.info("Incremental mode: latest stored=%s, fetching from %s", latest, start_dt)
            if start_dt > end_dt:
                logger.info("Already up to date — nothing to do")
                return
        except Exception as exc:
            logger.warning("Could not read existing parquet for incremental — full fetch (%s)", exc)
            existing = None
            start_dt = end_dt - timedelta(days=days)
    else:
        existing = None
        start_dt = end_dt - timedelta(days=days)

    logger.info("Sector ETF backfill: %d ETFs, %s → %s, dry_run=%s, incremental=%s",
                len(SECTOR_ETFS), start_dt, end_dt, dry_run, incremental)

    if dry_run:
        logger.info("[DRY-RUN] Would fetch %s from %s to %s", SECTOR_ETFS, start_dt, end_dt)
        return

    # Try Polygon first; fall back to yfinance
    data = _fetch_polygon(start_dt, end_dt)
    if not data:
        data = _fetch_yfinance(start_dt, end_dt)

    ok = len(data)
    errors = len(SECTOR_ETFS) - ok
    logger.info("Fetch complete: %d ETFs ok, %d errors", ok, errors)

    if not data:
        logger.warning("No data fetched — exiting")
        return

    for etf, df in data.items():
        logger.info("%s: %d bars (%s to %s)", etf, len(df),
                    df.index.min().date(), df.index.max().date())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for etf, df in data.items():
        df2 = df[["open", "high", "low", "close", "volume"]].copy()
        df2["date"] = df2.index.strftime("%Y-%m-%d")
        df2["etf"] = etf
        all_frames.append(df2.reset_index(drop=True))

    new_rows = pd.concat(all_frames, ignore_index=True)
    new_rows = new_rows[["etf", "date", "open", "high", "low", "close", "volume"]]

    if existing is not None and incremental:
        # Append only rows past existing latest per-ETF
        existing_max_per_etf = existing.groupby("etf")["date"].max().to_dict()
        keep = []
        for etf, grp in new_rows.groupby("etf"):
            cutoff = existing_max_per_etf.get(etf)
            if cutoff is not None:
                grp = grp[grp["date"] > cutoff]
            if not grp.empty:
                keep.append(grp)
        if not keep:
            logger.info("No new rows to append — file already current")
            return
        appended = pd.concat(keep, ignore_index=True)
        df_out = pd.concat([existing, appended], ignore_index=True)
        df_out = df_out.drop_duplicates(subset=["etf", "date"], keep="last")
        logger.info("Appending %d new rows (%d → %d total)", len(appended), len(existing), len(df_out))
    else:
        df_out = new_rows

    df_out = df_out.sort_values(["etf", "date"]).reset_index(drop=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Written: %s (%d rows, %d ETFs, %s → %s)",
                OUTPUT_PATH, len(df_out), df_out["etf"].nunique(),
                df_out["date"].min(), df_out["date"].max())


def update_sector_etf_history_incremental() -> None:
    """Programmatic entry point for app startup hooks."""
    run(incremental=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical sector ETF bars")
    parser.add_argument("--days", type=int, default=2520,
                        help="Calendar days to backfill in full mode (default 2520 = ~10yr)")
    parser.add_argument("--incremental", action="store_true",
                        help="Append only dates after latest in existing parquet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(days=args.days, dry_run=args.dry_run, incremental=args.incremental)
