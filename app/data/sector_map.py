"""
Sector map: symbol → GICS sector string.

Loaded from data/sector_map.parquet (pre-built cache). Falls back to
yfinance info for uncached symbols, caching results immediately. Falls
back to "UNKNOWN" gracefully on any error.

Build/refresh the cache:
    python scripts/build_sector_map.py
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_PATH = Path("data/sector_map.parquet")
_sector_cache: Dict[str, str] = {}
_cache_loaded = False


def _load_cache() -> None:
    global _sector_cache, _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    if _CACHE_PATH.exists():
        try:
            df = pd.read_parquet(_CACHE_PATH)
            if "symbol" in df.columns and "sector" in df.columns:
                _sector_cache = dict(zip(df["symbol"], df["sector"]))
                logger.debug("sector_map: loaded %d symbols from cache", len(_sector_cache))
        except Exception as exc:
            logger.warning("sector_map: failed to load cache — %s", exc)


def get_sector_map(symbols: List[str]) -> Dict[str, str]:
    """Return {symbol: sector} for each symbol. Falls back to 'UNKNOWN'."""
    _load_cache()
    result: Dict[str, str] = {}
    missing = []
    for sym in symbols:
        if sym in _sector_cache:
            result[sym] = _sector_cache[sym]
        else:
            missing.append(sym)

    if missing:
        # Try yfinance for uncached symbols (best-effort, no crash)
        try:
            import yfinance as yf
            new_entries: Dict[str, str] = {}
            for sym in missing:
                try:
                    info = yf.Ticker(sym).info
                    sector = info.get("sector") or "UNKNOWN"
                    new_entries[sym] = sector
                    result[sym] = sector
                except Exception:
                    result[sym] = "UNKNOWN"
            # Extend in-memory cache
            _sector_cache.update(new_entries)
            # Persist additions to parquet
            if new_entries:
                _save_cache()
        except ImportError:
            for sym in missing:
                result[sym] = "UNKNOWN"

    return result


def _save_cache() -> None:
    """Persist current in-memory cache to parquet."""
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [{"symbol": sym, "sector": sec} for sym, sec in _sector_cache.items()]
        )
        df.to_parquet(_CACHE_PATH, index=False)
    except Exception as exc:
        logger.warning("sector_map: failed to save cache — %s", exc)
