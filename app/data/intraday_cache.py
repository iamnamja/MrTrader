"""
intraday_cache.py — read/write 5-min bar Parquet files from data/intraday/.

Used by:
  - scripts/fetch_intraday_history.py  (write)
  - IntradayModelTrainer               (read, optional fallback write)
  - IntradayBacktester                 (read)
  - IntradayAgentSimulator             (read)

File format: data/intraday/{SYMBOL}.parquet
  Index: DatetimeIndex named 'timestamp', UTC-aware
  Columns: open, high, low, close, volume (all float64)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

INTRADAY_CACHE_DIR = Path("data/intraday")
CACHE_TTL_HOURS = 24


def _cache_path(symbol: str) -> Path:
    return INTRADAY_CACHE_DIR / f"{symbol}.parquet"


def cache_exists(symbol: str) -> bool:
    return _cache_path(symbol).exists()


def cache_is_fresh(symbol: str, required_start: date | None = None) -> bool:
    """
    Return True if the Parquet for symbol exists, was written within CACHE_TTL_HOURS,
    and covers the requested start date (if given).
    """
    path = _cache_path(symbol)
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_h = (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 3600
    if age_h > CACHE_TTL_HOURS:
        return False
    if required_start is None:
        return True
    try:
        df = pd.read_parquet(path, columns=["close"])
        if df.empty:
            return False
        earliest = pd.Timestamp(df.index.min())
        if earliest.tzinfo is None:
            earliest = earliest.tz_localize("UTC")
        cutoff = pd.Timestamp(required_start, tz="UTC") + pd.Timedelta(days=5)
        return earliest <= cutoff
    except Exception:
        return False


def load(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
) -> Optional[pd.DataFrame]:
    """
    Load cached 5-min bars for symbol, optionally filtered to [start, end].
    Returns None if cache is missing or unreadable.
    """
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        if start is not None:
            df = df[df.index.date >= start]
        if end is not None:
            df = df[df.index.date <= end]
        return df if not df.empty else None
    except Exception as exc:
        logger.debug("Failed to read intraday cache for %s: %s", symbol, exc)
        return None


def load_many(
    symbols: List[str],
    start: date | None = None,
    end: date | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load cached bars for multiple symbols. Missing symbols are omitted."""
    result: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load(sym, start=start, end=end)
        if df is not None:
            result[sym] = df
    missing = len(symbols) - len(result)
    if missing:
        logger.info(
            "Intraday cache: %d/%d symbols loaded  |  %d missing",
            len(result), len(symbols), missing,
        )
    return result


def save(symbol: str, df: pd.DataFrame) -> None:
    """Write a DataFrame to the Parquet cache."""
    INTRADAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol)
    try:
        df.to_parquet(path)
    except Exception as exc:
        logger.warning("Failed to write intraday cache for %s: %s", symbol, exc)


def available_symbols() -> List[str]:
    """Return list of symbols that have a Parquet file in the cache dir."""
    if not INTRADAY_CACHE_DIR.exists():
        return []
    return [p.stem for p in INTRADAY_CACHE_DIR.glob("*.parquet")]


def cache_stats() -> dict:
    """Summary stats for the cache directory (for reporting)."""
    symbols = available_symbols()
    total_bars = 0
    oldest_date: date | None = None
    for sym in symbols:
        df = load(sym)
        if df is not None:
            total_bars += len(df)
            sym_start = df.index.min().date()
            if oldest_date is None or sym_start < oldest_date:
                oldest_date = sym_start
    return {
        "symbols": len(symbols),
        "total_bars": total_bars,
        "oldest_date": str(oldest_date) if oldest_date else "N/A",
        "cache_dir": str(INTRADAY_CACHE_DIR.resolve()),
    }
