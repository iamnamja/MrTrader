"""
fetch_intraday_history.py — pull 2-year 5-min bars from Polygon.io and write
to data/intraday/{SYMBOL}.parquet.

Usage:
    python scripts/fetch_intraday_history.py                # full Russell-1000
    python scripts/fetch_intraday_history.py --symbols AAPL MSFT TSLA
    python scripts/fetch_intraday_history.py --days 730 --workers 8
    python scripts/fetch_intraday_history.py --force-refresh

Environment requirements:
    POLYGON_API_KEY (or set in .env as polygon_api_key)

Exit codes:
    0 — success (>= 1 symbol written)
    1 — Polygon API key missing or no data returned
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/intraday")
FETCH_CHUNK = 50
FETCH_WORKERS = 6
RATE_LIMIT_SLEEP = 0.2  # seconds between Polygon REST calls


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m  {msg}")


def _warn(msg: str) -> None:
    print(f"  \033[33m⚠\033[0m  {msg}")


def _err(msg: str) -> None:
    print(f"  \033[31m✗\033[0m  {msg}")


def fetch_symbol(
    symbol: str,
    start: datetime,
    end: datetime,
    api_key: str,
    interval_minutes: int = 5,
) -> pd.DataFrame | None:
    """Fetch 5-min bars for one symbol via Polygon REST v2 aggregates."""
    import requests

    multiplier = interval_minutes
    timespan = "minute"
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range"
        f"/{multiplier}/{timespan}/{start_str}/{end_str}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )

    all_results = []
    while url:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("Polygon REST error %s: %s", symbol, exc)
            return None

        results = data.get("results", [])
        all_results.extend(results)

        # Polygon paginates via next_url
        url = data.get("next_url")
        if url:
            url = url + f"&apiKey={api_key}"
            time.sleep(RATE_LIMIT_SLEEP)

    if not all_results:
        return None

    df = pd.DataFrame(all_results)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    df.index.name = "timestamp"
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Filter to market hours only (09:30–16:00 ET = 13:30–20:00 UTC)
    df = df.between_time("13:30", "20:00")
    return df if not df.empty else None


def write_parquet(symbol: str, df: pd.DataFrame) -> None:
    path = CACHE_DIR / f"{symbol}.parquet"
    df.to_parquet(path)


def cache_is_fresh(symbol: str, start: datetime) -> bool:
    path = CACHE_DIR / f"{symbol}.parquet"
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 3600
    if age_hours > 24:
        return False
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return False
        earliest = pd.Timestamp(df.index.min())
        if earliest.tzinfo is None:
            earliest = earliest.tz_localize("UTC")
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        return earliest <= start + timedelta(days=5)
    except Exception:
        return False


def fetch_chunk(
    symbols: list[str],
    start: datetime,
    end: datetime,
    api_key: str,
    force: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sym in symbols:
        if not force and cache_is_fresh(sym, start):
            df = pd.read_parquet(CACHE_DIR / f"{sym}.parquet")
            counts[sym] = len(df)
            continue
        df = fetch_symbol(sym, start, end, api_key)
        time.sleep(RATE_LIMIT_SLEEP)
        if df is not None and not df.empty:
            write_parquet(sym, df)
            counts[sym] = len(df)
        else:
            logger.debug("No data for %s", sym)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch 2-year intraday Polygon bars")
    parser.add_argument("--symbols", nargs="*", help="Tickers (default: Russell 1000)")
    parser.add_argument("--days", type=int, default=730, help="Calendar days of history")
    parser.add_argument("--workers", type=int, default=FETCH_WORKERS)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    from app.config import settings
    api_key = settings.polygon_api_key or ""
    if not api_key:
        _err("polygon_api_key not configured. Set POLYGON_API_KEY in .env")
        return 1

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        from app.utils.constants import RUSSELL_1000_TICKERS
        symbols = list(RUSSELL_1000_TICKERS)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.days + 10)

    print(f"\n  Fetching {len(symbols)} symbols | {args.days}d history | "
          f"workers={args.workers} | force={args.force_refresh}\n")

    chunks = [symbols[i:i + FETCH_CHUNK] for i in range(0, len(symbols), FETCH_CHUNK)]
    total_ok = 0
    total_bars = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="poly") as pool:
        futures = {
            pool.submit(fetch_chunk, chunk, start, end, api_key, args.force_refresh): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                result = future.result()
                total_ok += len(result)
                total_bars += sum(result.values())
                logger.info(
                    "Chunk done: %d symbols, %d bars (running: %d/%d)",
                    len(result), sum(result.values()), total_ok, len(symbols),
                )
            except Exception as exc:
                _warn(f"Chunk {chunk[0]}… failed: {exc}")

    elapsed = time.time() - t0
    print()
    if total_ok == 0:
        _err("No symbols fetched. Check API key and network.")
        return 1

    _ok(f"{total_ok}/{len(symbols)} symbols fetched  |  "
        f"{total_bars:,} bars  |  {elapsed:.1f}s")
    _ok(f"Parquet files written to {CACHE_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
