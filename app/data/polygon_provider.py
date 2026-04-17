"""
Polygon.io data provider.

Strategy
--------
Daily bars (training):
  1. Check DataCache for fully-covered range.
  2. If missing tail, try S3 bulk download (fast, covers all symbols at once).
  3. Fall back to REST API for very recent dates not yet on S3.

Intraday bars (training):
  1. Check DataCache.
  2. S3 1-minute bulk download, then resample to requested interval.
  3. Fall back to REST API.

Live signal generation:
  Always REST API for the latest bars.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from app.data.base import DataProvider
from app.data.cache import get_cache

logger = logging.getLogger(__name__)

_REST_BASE = "https://api.polygon.io"
# S3 lags REST by ~1 day; dates newer than this fall back to REST
_S3_LAG_DAYS = 2


class PolygonProvider(DataProvider):
    """
    DataProvider backed by Polygon.io REST API + S3 flat files.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_endpoint: Optional[str] = None,
        s3_bucket: Optional[str] = None,
    ):
        from app.config import settings
        self._api_key = api_key or settings.polygon_api_key or ""
        self._s3_access = s3_access_key or settings.polygon_s3_access_key
        self._s3_secret = s3_secret_key or settings.polygon_s3_secret_key
        self._s3_endpoint = s3_endpoint or settings.polygon_s3_endpoint
        self._s3_bucket = s3_bucket or settings.polygon_s3_bucket
        self._s3: Optional[Any] = None  # lazy init — PolygonS3 loaded on demand

    @property
    def name(self) -> str:
        return "polygon"

    # ── S3 client (lazy) ──────────────────────────────────────────────────────

    def _get_s3(self):
        if self._s3 is None:
            if not all([self._s3_access, self._s3_secret, self._s3_endpoint, self._s3_bucket]):
                return None
            from app.data.polygon_s3 import PolygonS3
            self._s3 = PolygonS3(
                access_key=self._s3_access,
                secret_key=self._s3_secret,
                endpoint_url=self._s3_endpoint,
                bucket=self._s3_bucket,
            )
        return self._s3

    # ── Daily bars ────────────────────────────────────────────────────────────

    def get_daily_bars(
        self, symbol: str, start: date, end: date
    ) -> Optional[pd.DataFrame]:
        cache = get_cache()

        # 1. Serve from cache where possible
        fetch_start, fetch_end = cache.missing_daily_range(symbol, start, end)
        if fetch_start is None:
            return cache.get_daily(symbol, start, end)

        # 2. Fetch missing range
        df = self._fetch_daily_rest(symbol, fetch_start, min(fetch_end, date.today()))
        if df is not None and not df.empty:
            cache.put_daily(symbol, df)

        return cache.get_daily(symbol, start, end)

    def get_daily_bars_bulk(
        self, symbols: List[str], start: date, end: date
    ) -> Dict[str, pd.DataFrame]:
        cache = get_cache()

        # Split symbols into cached vs needs-fetch
        needs_fetch = []
        result = {}
        for sym in symbols:
            fs, fe = cache.missing_daily_range(sym, start, end)
            if fs is None:
                df = cache.get_daily(sym, start, end)
                if df is not None:
                    result[sym] = df
            else:
                needs_fetch.append(sym)

        if not needs_fetch:
            return result

        # Try S3 bulk download for missing symbols
        s3_end = end - timedelta(days=_S3_LAG_DAYS)
        s3 = self._get_s3()
        if s3 and s3_end >= start:
            logger.info("S3 bulk download: %d symbols %s→%s", len(needs_fetch), start, s3_end)
            s3_data = s3.get_daily_bulk(needs_fetch, start, s3_end)
            for sym, df in s3_data.items():
                cache.put_daily(sym, df)

        # REST fallback for recent tail or symbols S3 missed
        rest_start = end - timedelta(days=_S3_LAG_DAYS + 1)
        still_missing = []
        for sym in needs_fetch:
            fs, fe = cache.missing_daily_range(sym, start, end)
            if fs is not None and fs >= rest_start:
                # Only need the recent tail
                df = self._fetch_daily_rest(sym, fs, fe)
                if df is not None:
                    cache.put_daily(sym, df)
            elif fs is not None:
                still_missing.append(sym)

        for sym in still_missing:
            df = self._fetch_daily_rest(sym, start, end)
            if df is not None:
                cache.put_daily(sym, df)

        # Assemble final result
        for sym in needs_fetch:
            df = cache.get_daily(sym, start, end)
            if df is not None:
                result[sym] = df

        logger.info("Polygon bulk: %d/%d symbols returned", len(result), len(symbols))
        return result

    # ── Intraday bars ─────────────────────────────────────────────────────────

    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> Optional[pd.DataFrame]:
        interval_str = f"{interval_minutes}min"
        cache = get_cache()

        cached = cache.get_intraday(symbol, start, end, interval=interval_str)
        if cached is not None and len(cached) > 0:
            return cached

        # Fetch 1-min from REST, then resample
        df_1m = self._fetch_intraday_rest(symbol, start, end, interval_minutes=1)
        if df_1m is None or df_1m.empty:
            return None

        df = _resample(df_1m, interval_minutes)
        if df is not None and not df.empty:
            cache.put_intraday(symbol, df, interval=interval_str)
        return cache.get_intraday(symbol, start, end, interval=interval_str)

    def get_intraday_bars_bulk(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """Use S3 1-min bulk download then resample — fast for training."""
        s3 = self._get_s3()
        interval_str = f"{interval_minutes}min"
        cache = get_cache()
        result = {}

        if s3:
            start_date = start.date() if hasattr(start, "date") else start
            end_date = end.date() if hasattr(end, "date") else end
            logger.info(
                "S3 minute bulk: %d symbols %s→%s", len(symbols), start_date, end_date
            )
            raw = s3.get_minute_bulk(symbols, start_date, end_date)
            for sym, df_1m in raw.items():
                df = _resample(df_1m, interval_minutes)
                if df is not None and not df.empty:
                    cache.put_intraday(sym, df, interval=interval_str)
                    result[sym] = df
        else:
            for sym in symbols:
                df = self.get_intraday_bars(sym, start, end, interval_minutes)
                if df is not None:
                    result[sym] = df

        logger.info(
            "Polygon intraday bulk: %d/%d symbols returned", len(result), len(symbols)
        )
        return result

    # ── REST helpers ──────────────────────────────────────────────────────────

    def _fetch_daily_rest(
        self, symbol: str, start: date, end: date
    ) -> Optional[pd.DataFrame]:
        """Fetch daily bars via Polygon REST aggregates endpoint."""
        try:
            rows = []
            url = f"{_REST_BASE}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
            params = {"apiKey": self._api_key, "limit": 50000, "adjusted": "true"}
            while url:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 403:
                    logger.warning("Polygon REST 403 for %s — check API key/plan", symbol)
                    return None
                if resp.status_code != 200:
                    logger.debug("Polygon REST %s for %s", resp.status_code, symbol)
                    return None
                data = resp.json()
                rows.extend(data.get("results", []))
                url = data.get("next_url")
                params = {"apiKey": self._api_key}  # next_url already has params

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.normalize()
            df.index = df.index.tz_localize(None)
            df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df
        except Exception as exc:
            logger.warning("Polygon REST daily failed for %s: %s", symbol, exc)
            return None

    def _fetch_intraday_rest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval_minutes: int = 1,
    ) -> Optional[pd.DataFrame]:
        """Fetch intraday bars via Polygon REST."""
        try:
            start_ms = int(pd.Timestamp(start).timestamp() * 1000)
            end_ms = int(pd.Timestamp(end).timestamp() * 1000)
            rows = []
            url = (
                f"{_REST_BASE}/v2/aggs/ticker/{symbol}/range/{interval_minutes}/minute"
                f"/{start_ms}/{end_ms}"
            )
            params = {"apiKey": self._api_key, "limit": 50000, "adjusted": "true"}
            while url:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                rows.extend(data.get("results", []))
                url = data.get("next_url")
                params = {"apiKey": self._api_key}

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
            df.index.name = None
            df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df
        except Exception as exc:
            logger.warning("Polygon REST intraday failed for %s: %s", symbol, exc)
            return None

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            resp = requests.get(
                f"{_REST_BASE}/v2/aggs/ticker/AAPL/range/1/day/2024-01-02/2024-01-02",
                params={"apiKey": self._api_key},
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resample(df_1m: pd.DataFrame, interval_minutes: int) -> Optional[pd.DataFrame]:
    """Resample 1-minute bars to *interval_minutes* OHLCV bars."""
    if interval_minutes == 1:
        return df_1m
    try:
        rule = f"{interval_minutes}min"
        rs = df_1m.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["close"])
        return rs
    except Exception as exc:
        logger.debug("Resample failed: %s", exc)
        return None
