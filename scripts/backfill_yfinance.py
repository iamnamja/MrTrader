"""
Backfill yfinance daily bars to an earlier start date.

Downloads OHLCV data for the full Russell 1000 universe (+ SPY, QQQ, IWM,
VIX proxies) from --start through today, writing per-symbol Parquet cache
files that ModelTrainer._fetch_data() will pick up automatically on the
next retrain.

Usage:
  python scripts/backfill_yfinance.py --start 2005-01-01          # full backfill
  python scripts/backfill_yfinance.py --start 2005-01-01 --dry-run
  python scripts/backfill_yfinance.py --start 2005-01-01 --symbols AAPL MSFT
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
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
logger = logging.getLogger("backfill_yfinance")

CACHE_DIR = Path("data/price_cache")


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def _fetch_symbol(symbol: str, start: date, end: date):
    import yfinance as yf
    import pandas as pd

    try:
        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).normalize()
        return df
    except Exception as exc:
        logger.warning("%s: fetch error — %s", symbol, exc)
        return None


def _merge_with_existing(symbol: str, new_df) -> any:
    import pandas as pd
    path = _cache_path(symbol)
    if not path.exists():
        return new_df
    try:
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined
    except Exception:
        return new_df


def run(symbols: list[str], start: date, dry_run: bool = False, batch_size: int = 50):
    import pandas as pd
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    end = date.today()

    logger.info("Backfilling %d symbols: %s → %s (dry_run=%s)", len(symbols), start, end, dry_run)

    errors = []
    saved = 0

    # Batch download for efficiency (yfinance handles multi-ticker well)
    for batch_start in range(0, len(symbols), batch_size):
        batch = symbols[batch_start: batch_start + batch_size]
        logger.info("Batch %d–%d / %d", batch_start + 1, batch_start + len(batch), len(symbols))

        try:
            raw = yf.download(
                batch,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            logger.warning("Batch download failed (%s), falling back to per-symbol", exc)
            raw = None

        for symbol in batch:
            try:
                if raw is not None and len(batch) > 1:
                    try:
                        df = raw[symbol].dropna(how="all")
                    except KeyError:
                        df = pd.DataFrame()
                else:
                    df = _fetch_symbol(symbol, start, end) or pd.DataFrame()

                if df.empty:
                    errors.append(symbol)
                    continue

                df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
                df.index = pd.to_datetime(df.index).normalize()

                if dry_run:
                    logger.info("  DRY RUN %s: %d rows", symbol, len(df))
                    saved += 1
                    continue

                merged = _merge_with_existing(symbol, df)
                merged.to_parquet(_cache_path(symbol))
                saved += 1

            except Exception as exc:
                logger.warning("  %s: error — %s", symbol, exc)
                errors.append(symbol)

        time.sleep(0.5)  # be polite between batches

    logger.info(
        "Backfill complete: %d saved, %d errors%s",
        saved, len(errors), " (dry run)" if dry_run else "",
    )
    if errors:
        logger.warning("Symbols with errors: %s", errors[:20])
    return saved, errors


def main():
    from app.utils.constants import RUSSELL_1000_TICKERS

    parser = argparse.ArgumentParser(description="Backfill yfinance daily bars")
    parser.add_argument("--start", default="2005-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbol list (default: Russell 1000 + benchmarks)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Symbols per yfinance batch request (default 50)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)

    benchmarks = ["SPY", "QQQ", "IWM", "DIA", "VIX", "^VIX", "^VIX3M",
                  "HYG", "IEF", "TLT", "GLD", "RSP"]
    symbols = args.symbols or (RUSSELL_1000_TICKERS + [s for s in benchmarks
                                                        if s not in RUSSELL_1000_TICKERS])

    run(symbols, start=start, dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
