"""
Phase 93 — FMP quarterly fundamentals backfill.

Fetches income statement / balance sheet / cash flow (quarterly) for each
symbol and writes data/fundamentals/fmp_fundamentals_history.parquet.

Usage:
    python scripts/backfill_fmp_fundamentals.py --workers 4
    python scripts/backfill_fmp_fundamentals.py --symbols AAPL MSFT --dry-run
    python scripts/backfill_fmp_fundamentals.py --incremental

Cost: covered by existing FMP standard plan (~10 req/s).
Runtime: ~400 symbols × 3 endpoints × 0.15s / 4 workers ≈ 5 min.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fmp_fundamentals_backfill")


def _resolve_symbols(arg_value) -> list[str]:
    if not arg_value or arg_value == ["all"]:
        from app.utils.constants import RUSSELL_1000_TICKERS
        return list(RUSSELL_1000_TICKERS)
    return list(arg_value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill FMP quarterly fundamentals")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help='Symbols (default: SP500). Example: --symbols AAPL MSFT')
    parser.add_argument("--lookback-quarters", type=int, default=None,
                        help="How many quarters per symbol (default from retrain_config)")
    parser.add_argument("--incremental", action="store_true",
                        help="Only fetch symbols whose latest row is >45 days old")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be fetched without writing parquet")
    args = parser.parse_args()

    from app.config import settings
    from app.data.fmp_fundamentals import (
        backfill_fmp_fundamentals,
        update_fmp_fundamentals_incremental,
    )
    from app.ml.retrain_config import FMP_QUARTERLY_LOOKBACK_QUARTERS

    if not settings.fmp_api_key:
        logger.error("FMP_API_KEY not set in environment — aborting")
        return 1

    symbols = _resolve_symbols(args.symbols)
    lookback = args.lookback_quarters or FMP_QUARTERLY_LOOKBACK_QUARTERS

    logger.info("Resolved %d symbols, lookback=%dq, workers=%d, incremental=%s, dry_run=%s",
                len(symbols), lookback, args.workers, args.incremental, args.dry_run)

    if args.incremental and not args.dry_run:
        update_fmp_fundamentals_incremental(symbols, workers=args.workers)
    else:
        backfill_fmp_fundamentals(
            symbols,
            workers=args.workers,
            lookback_quarters=lookback,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
