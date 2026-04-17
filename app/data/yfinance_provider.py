"""
yfinance data provider — with disk + in-memory caching.

Read path
---------
1. Check DataCache for bars covering the requested range.
2. Fetch only the missing tail from yfinance.
3. Merge into cache and return the full range.

Strengths : free, no API key, broad coverage, 20-year daily history
Limits    : 5-min bars only last ~60 days, rate-limited under heavy use
Use for   : model training (daily + intraday), backtesting
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

from app.data.base import DataProvider
from app.data.cache import get_cache

logger = logging.getLogger(__name__)

_POLITE_DELAY = 0.05   # seconds between yfinance calls


class YFinanceProvider(DataProvider):

    @property
    def name(self) -> str:
        return "yfinance"

    # ── Daily ─────────────────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> Optional[pd.DataFrame]:
        cache = get_cache()

        # 1. Identify what's missing
        fetch_start, fetch_end = cache.missing_daily_range(symbol, start, end)

        # 2. Fetch only the missing portion
        if fetch_start is not None and fetch_start <= fetch_end:
            fetched = self._fetch_daily(symbol, fetch_start, fetch_end)
            if fetched is not None:
                cache.put_daily(symbol, fetched)

        # 3. Return from (now-updated) cache
        result = cache.get_daily(symbol, start, end)
        return result

    def get_daily_bars_bulk(
        self,
        symbols: List[str],
        start: date,
        end: date,
    ) -> dict:
        """
        Download all symbols — served from cache where possible, fetching only
        the missing tail per symbol via a single batched yfinance call.
        """
        cache = get_cache()
        result = {}
        needs_fetch: List[str] = []
        fetch_ranges = {}

        for symbol in symbols:
            fs, fe = cache.missing_daily_range(symbol, start, end)
            if fs is None:
                # Fully cached — load directly
                df = cache.get_daily(symbol, start, end)
                if df is not None and len(df) >= 10:
                    result[symbol] = df
            else:
                needs_fetch.append(symbol)
                fetch_ranges[symbol] = (fs, fe)

        if needs_fetch:
            # Find the widest date range across symbols needing a fetch,
            # then do one bulk download and split per-symbol into cache.
            bulk_start = min(v[0] for v in fetch_ranges.values())
            bulk_end = max(v[1] for v in fetch_ranges.values())
            bulk = self._fetch_daily_bulk(needs_fetch, bulk_start, bulk_end)

            for symbol in needs_fetch:
                new_bars = bulk.get(symbol)
                if new_bars is not None:
                    cache.put_daily(symbol, new_bars)
                df = cache.get_daily(symbol, start, end)
                if df is not None and len(df) >= 10:
                    result[symbol] = df

        return result

    # ── Intraday ──────────────────────────────────────────────────────────────

    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> Optional[pd.DataFrame]:
        interval_key = f"{interval_minutes}min"
        cache = get_cache()

        cached = cache.get_intraday(symbol, start, end, interval=interval_key)
        if cached is not None and len(cached) >= 12:
            return cached

        # Not in cache — fetch from yfinance
        interval_map = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
        yf_interval = interval_map.get(interval_minutes, "5m")
        try:
            df = yf.download(
                symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                interval=yf_interval,
                progress=False,
                auto_adjust=True,
            )
            df = self._normalise(df)
            if df.empty or "close" not in df.columns:
                return None
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            time.sleep(_POLITE_DELAY)
            cache.put_intraday(symbol, df, interval=interval_key)
            # Return the slice matching the requested window
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            result = df.loc[mask]
            return result if len(result) > 0 else None
        except Exception as exc:
            logger.debug("yfinance intraday failed for %s: %s", symbol, exc)
            return None

    # ── Internal fetch helpers (no caching) ───────────────────────────────────

    def _fetch_daily(self, symbol: str, start: date, end: date) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(
                symbol,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # end is exclusive in yfinance
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            df = self._normalise(df)
            if df.empty or "close" not in df.columns:
                return None
            time.sleep(_POLITE_DELAY)
            return df[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as exc:
            logger.debug("yfinance daily fetch failed for %s: %s", symbol, exc)
            return None

    def _fetch_daily_bulk(
        self, symbols: List[str], start: date, end: date
    ) -> dict:
        try:
            raw = yf.download(
                symbols,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
        except Exception as exc:
            logger.warning("yfinance bulk download failed: %s — falling back", exc)
            return {}

        result = {}
        for symbol in symbols:
            try:
                df = raw[symbol].copy() if len(symbols) > 1 else raw.copy()
                df = self._normalise(df)
                df = df.dropna(subset=["close"])
                if not df.empty and len(df) >= 5:
                    result[symbol] = df[["open", "high", "low", "close", "volume"]].astype(float)
            except Exception:
                pass
        return result
