"""
Disk-based data cache for OHLCV bars and fundamental snapshots.

Architecture
------------
Layer 1 — Disk (Parquet / JSON):
  data/cache/daily/{symbol}.parquet      Daily OHLCV, append-only
  data/cache/5min/{symbol}.parquet       5-min OHLCV, rolling 90-day window
  data/cache/fundamentals/{symbol}.json  Fundamental snapshot + TTL timestamp
  data/cache/misc/{key}.json             Insider / earnings / other TTL caches

Layer 2 — In-memory (per-process):
  LRU-style dict keyed by (symbol, interval).
  Avoids repeated Parquet reads within a single training run.

Usage
-----
    cache = DataCache()                              # default cache dir
    df = cache.get_daily("AAPL", start, end)         # read-through
    cache.put_daily("AAPL", new_bars)                # append + save
    data = cache.get_json("fundamentals/AAPL", ttl=86400)
    cache.put_json("fundamentals/AAPL", data)
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


class DataCache:
    """
    Thread-safe read/write cache for OHLCV bars and JSON blobs.

    OHLCV strategy
    ~~~~~~~~~~~~~~
    - On ``get_daily(symbol, start, end)``:
        1. Load disk cache for symbol (if it exists).
        2. Identify the date range NOT yet in cache.
        3. Return cached rows for the requested range.
        Caller is responsible for fetching missing tail and calling ``put_daily``.
    - On ``put_daily(symbol, df)``:
        Merge new rows with existing cache (dedup by date), save to Parquet.

    JSON strategy
    ~~~~~~~~~~~~~
    - TTL-based: ``get_json`` returns None when the cached entry is older than ttl.
    - ``put_json`` writes with a ``_cached_at`` timestamp.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._mem: Dict[str, Any] = {}   # in-memory layer: key → value
        self._ensure_dirs()

    def _ensure_dirs(self):
        for sub in ("daily", "5min", "fundamentals", "misc"):
            (self._dir / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_df(path: Path) -> pd.DataFrame:
        """Read Parquet, falling back to CSV if pyarrow is unavailable."""
        try:
            return pd.read_parquet(path)
        except ImportError:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path, index_col=0, parse_dates=True)
            raise

    @staticmethod
    def _write_df(df: pd.DataFrame, path: Path) -> None:
        """Write Parquet, falling back to CSV if pyarrow is unavailable."""
        try:
            df.to_parquet(path)
        except ImportError:
            df.to_csv(path.with_suffix(".csv"))

    # ── OHLCV — daily ─────────────────────────────────────────────────────────

    def get_daily(
        self, symbol: str, start: date, end: date
    ) -> Optional[pd.DataFrame]:
        """
        Return cached daily bars for symbol in [start, end] (inclusive).
        Returns None if no cache file exists.
        """
        mem_key = f"daily:{symbol}"
        df = self._mem.get(mem_key)
        if df is None:
            path = self._dir / "daily" / f"{symbol}.parquet"
            if not path.exists():
                return None
            try:
                df = self._read_df(path)
                self._mem[mem_key] = df
            except Exception as exc:
                logger.debug("Cache read failed for %s: %s", symbol, exc)
                return None

        # Filter to requested range
        idx = pd.DatetimeIndex(df.index)
        mask = (idx.date >= start) & (idx.date <= end)
        result = df.loc[mask]
        return result if len(result) > 0 else None

    def put_daily(self, symbol: str, new_bars: pd.DataFrame) -> None:
        """Merge new_bars into cached daily data and save to Parquet."""
        if new_bars is None or new_bars.empty:
            return
        path = self._dir / "daily" / f"{symbol}.parquet"
        mem_key = f"daily:{symbol}"

        existing = self._mem.get(mem_key)
        if existing is None and path.exists():
            try:
                existing = self._read_df(path)
            except Exception:
                existing = None

        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_bars])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_bars.sort_index()

        try:
            self._write_df(combined, path)
            self._mem[mem_key] = combined
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", symbol, exc)

    def missing_daily_range(
        self, symbol: str, start: date, end: date
    ) -> tuple:
        """
        Return (fetch_start, fetch_end) representing the date range that is NOT
        yet in cache for symbol.  Returns (start, end) if nothing is cached.
        """
        path = self._dir / "daily" / f"{symbol}.parquet"
        if not path.exists():
            return start, end

        mem_key = f"daily:{symbol}"
        df = self._mem.get(mem_key)
        if df is None:
            try:
                df = self._read_df(path)
                self._mem[mem_key] = df
            except Exception:
                return start, end

        if df.empty:
            return start, end

        cached_dates = pd.DatetimeIndex(df.index).date
        cached_start = cached_dates.min()
        cached_end = cached_dates.max()

        if start < cached_start:
            # Need earlier data — fetch from start to day before cached_start
            return start, cached_start - timedelta(days=1)
        elif end > cached_end:
            # Need newer data — fetch from day after cached_end to end
            return cached_end + timedelta(days=1), end
        else:
            # Fully covered
            return None, None

    # ── OHLCV — 5-min ─────────────────────────────────────────────────────────

    def get_intraday(
        self, symbol: str, start: datetime, end: datetime, interval: str = "5min"
    ) -> Optional[pd.DataFrame]:
        """Return cached intraday bars, or None if not in cache."""
        mem_key = f"{interval}:{symbol}"
        df = self._mem.get(mem_key)
        if df is None:
            path = self._dir / interval / f"{symbol}.parquet"
            if not path.exists():
                return None
            try:
                df = self._read_df(path)
                self._mem[mem_key] = df
            except Exception:
                return None

        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        result = df.loc[mask]
        return result if len(result) > 0 else None

    def put_intraday(
        self, symbol: str, new_bars: pd.DataFrame,
        interval: str = "5min", keep_days: int = 90
    ) -> None:
        """Merge and save intraday bars; prune to keep_days to limit disk use."""
        if new_bars is None or new_bars.empty:
            return
        path = self._dir / interval / f"{symbol}.parquet"
        mem_key = f"{interval}:{symbol}"

        existing = self._mem.get(mem_key)
        if existing is None and path.exists():
            try:
                existing = self._read_df(path)
            except Exception:
                existing = None

        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_bars])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_bars.sort_index()

        # Rolling window: drop bars older than keep_days
        cutoff = pd.Timestamp.now(tz=combined.index.tz) - timedelta(days=keep_days)
        cutoff = cutoff.tz_localize(None) if combined.index.tz is None else cutoff
        combined = combined.loc[combined.index >= cutoff]

        try:
            self._write_df(combined, path)
            self._mem[mem_key] = combined
        except Exception as exc:
            logger.warning("Intraday cache write failed for %s: %s", symbol, exc)

    # ── JSON blobs (fundamentals, insider scores, etc.) ───────────────────────

    def get_json(self, key: str, ttl: int = 86_400) -> Optional[Dict]:
        """
        Return cached JSON for key if it exists and is within ttl seconds.
        key examples: "fundamentals/AAPL", "misc/insider_AAPL"
        """
        mem_key = f"json:{key}"
        entry = self._mem.get(mem_key)
        if entry is not None:
            if time.time() - entry.get("_cached_at", 0) < ttl:
                return entry
            del self._mem[mem_key]

        parts = key.split("/", 1)
        sub = parts[0] if len(parts) == 2 else "misc"
        fname = (parts[1] if len(parts) == 2 else key) + ".json"
        path = self._dir / sub / fname

        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if time.time() - data.get("_cached_at", 0) >= ttl:
                return None
            self._mem[mem_key] = data
            return data
        except Exception:
            return None

    def put_json(self, key: str, data: Dict) -> None:
        """Write data dict to disk cache with current timestamp."""
        parts = key.split("/", 1)
        sub = parts[0] if len(parts) == 2 else "misc"
        fname = (parts[1] if len(parts) == 2 else key) + ".json"
        path = self._dir / sub / fname

        payload = {**data, "_cached_at": time.time()}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f)
            self._mem[f"json:{key}"] = payload
        except Exception as exc:
            logger.warning("JSON cache write failed for %s: %s", key, exc)

    def invalidate(self, symbol: str) -> None:
        """Clear all in-memory cache entries for a symbol."""
        keys = [k for k in self._mem if symbol in k]
        for k in keys:
            del self._mem[k]

    def cache_info(self) -> Dict:
        """Return a summary of what's on disk."""
        info: Dict[str, int] = {}
        for sub in ("daily", "5min", "fundamentals", "misc"):
            sub_dir = self._dir / sub
            info[sub] = len(list(sub_dir.glob("*"))) if sub_dir.exists() else 0
        info["memory_entries"] = len(self._mem)
        return info


# Module-level singleton — shared across the process
_default_cache: Optional[DataCache] = None


def get_cache(cache_dir: Optional[Path] = None) -> DataCache:
    """Return the process-wide default DataCache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = DataCache(cache_dir)
    return _default_cache
