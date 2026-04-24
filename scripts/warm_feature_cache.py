"""
Phase 31 — Nightly feature store pre-compute script.

Run after market close (e.g. 17:30 ET) to compute features for any date not
already in the feature store.  By the time the 17:00 ET retrain runs next
day, the store has today's entries — no cold-cache overhead during training.

After filling, trims the store to 5 years to keep it bounded at ~170k entries.

Usage:
    python scripts/warm_feature_cache.py [--years N] [--symbols AAPL MSFT ...]
"""
import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.feature_store import FeatureStore  # noqa: E402
from app.ml.features import FeatureEngineer  # noqa: E402
from app.utils.constants import SP_500_TICKERS, SECTOR_MAP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KEEP_YEARS = 5
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _fetch_bars(symbols: list, start: date, end: date):
    """Fetch daily OHLCV bars via Polygon or yfinance fallback."""
    try:
        from app.data.polygon_fetcher import PolygonDataProvider
        provider = PolygonDataProvider()
        data = provider.get_daily_bars_bulk(symbols, start, end)
        if data:
            return data
    except Exception as exc:
        logger.warning("Polygon fetch failed (%s), falling back to yfinance", exc)

    try:
        import yfinance as yf
        raw = yf.download(
            symbols, start=start.isoformat(), end=end.isoformat(),
            progress=False, auto_adjust=True, group_by="ticker",
        )
        data = {}
        for sym in symbols:
            try:
                df = raw[sym].dropna()
                df.columns = [c.lower() for c in df.columns]
                if not df.empty:
                    data[sym] = df
            except Exception:
                pass
        return data
    except Exception as exc:
        logger.error("yfinance fallback also failed: %s", exc)
        return {}


def warm(symbols: list, years: int) -> None:
    store = FeatureStore()
    fe = FeatureEngineer()

    today = date.today()
    start = today - timedelta(days=365 * years + 90)  # extra buffer for features

    logger.info("Warming feature store: %d symbols, %d years back to %s", len(symbols), years, start)
    t0 = time.time()

    logger.info("Fetching daily bars...")
    bars = _fetch_bars(symbols, start, today)
    logger.info("Got bars for %d / %d symbols", len(bars), len(symbols))

    if not bars:
        logger.error("No bar data — aborting")
        return

    # Build date spine from today back to start (business days only)
    # Use the union of all bar dates as the spine
    all_bar_dates = sorted(set(
        d for df in bars.values() for d in df.index.date
    ))

    # Identify which (symbol, date) pairs are already cached
    logger.info("Checking cache for %d dates...", len(all_bar_dates))
    cached_bulk = store.get_all_for_dates(all_bar_dates)

    new_entries = 0
    skip_entries = 0
    errors = 0

    for sym, df in bars.items():
        sector = SECTOR_MAP.get(sym, "Unknown")
        cached_for_sym = cached_bulk.get(sym, {})
        idx = df.index.date

        for i, as_of in enumerate(all_bar_dates):
            if as_of not in idx:
                continue
            if as_of in cached_for_sym:
                skip_entries += 1
                continue

            # Need at least WINDOW_DAYS bars ending at as_of
            window_df = df.loc[idx <= as_of]
            if len(window_df) < FeatureEngineer.MIN_BARS:
                continue

            try:
                entry_price = float(df.loc[idx == as_of, "close"].iloc[0])
                feats = fe.engineer_features(window_df, entry_price, sector=sector)
                if feats:
                    store.put(sym, as_of, feats)
                    new_entries += 1
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    logger.debug("Feature compute failed %s/%s: %s", sym, as_of, exc)

    elapsed = time.time() - t0
    logger.info(
        "Warm complete: %d new entries, %d skipped, %d errors in %.1fs",
        new_entries, skip_entries, errors, elapsed,
    )

    # Trim to keep store bounded
    trimmed = store.trim_to_years(KEEP_YEARS)
    count = store.count()
    logger.info("Feature store: %d total entries after trim (%d deleted)", count, trimmed)


def main():
    parser = argparse.ArgumentParser(description="Nightly feature store pre-compute (Phase 31)")
    parser.add_argument("--years", type=int, default=5, help="Years of history to warm (default: 5)")
    parser.add_argument("--symbols", nargs="+", default=None, metavar="TICKER",
                        help="Symbols to warm (default: full SP500 universe)")
    args = parser.parse_args()

    symbols = args.symbols or SP_500_TICKERS
    warm(symbols, args.years)


if __name__ == "__main__":
    main()
