"""
Alpaca data provider.

Strengths : real-time + historical, already authenticated, good intraday depth
Limits    : free tier has some history caps; requires API keys
Use for   : live signal generation, real-time intraday bars during market hours
"""

import logging
from datetime import date, datetime, timezone
from typing import List, Optional

import pandas as pd

from app.data.base import DataProvider

logger = logging.getLogger(__name__)


class AlpacaProvider(DataProvider):

    @property
    def name(self) -> str:
        return "alpaca"

    def _client(self):
        from app.integrations import get_alpaca_client
        return get_alpaca_client()

    # ── Daily ─────────────────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> Optional[pd.DataFrame]:
        try:
            client = self._client()
            bars = client.get_bars(
                symbol,
                timeframe="1Day",
                start=start.isoformat(),
                end=end.isoformat(),
            )
            df = pd.DataFrame(bars)
            if df.empty:
                return None
            df = self._normalise(df)
            # Alpaca uses 't' for timestamp
            if "t" in df.columns and df.index.dtype == object:
                df.index = pd.to_datetime(df["t"])
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[cols].astype(float)
        except Exception as exc:
            logger.debug("Alpaca daily bars failed for %s: %s", symbol, exc)
            return None

    # ── Intraday ──────────────────────────────────────────────────────────────

    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> Optional[pd.DataFrame]:
        timeframe_map = {1: "1Min", 5: "5Min", 15: "15Min", 30: "30Min", 60: "1Hour"}
        timeframe = timeframe_map.get(interval_minutes, "5Min")
        try:
            # Ensure UTC-aware datetimes
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)

            client = self._client()
            bars = client.get_bars(
                symbol,
                timeframe=timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
            )
            df = pd.DataFrame(bars)
            if df.empty:
                return None
            df = self._normalise(df)
            if "t" in df.columns and df.index.dtype == object:
                df.index = pd.to_datetime(df["t"])
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[cols].astype(float)
        except Exception as exc:
            logger.debug("Alpaca intraday bars failed for %s: %s", symbol, exc)
            return None

    def get_intraday_bars_bulk(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> dict:
        """Alpaca supports multi-symbol requests — more efficient than looping."""
        timeframe_map = {1: "1Min", 5: "5Min", 15: "15Min", 30: "30Min", 60: "1Hour"}
        timeframe = timeframe_map.get(interval_minutes, "5Min")
        try:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)

            client = self._client()
            multi = client.get_bars(
                symbols,
                timeframe=timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
            )
            result = {}
            for symbol in symbols:
                try:
                    bars = multi.get(symbol, [])
                    df = pd.DataFrame(bars)
                    if df.empty:
                        continue
                    df = self._normalise(df)
                    if "t" in df.columns:
                        df.index = pd.to_datetime(df["t"])
                    cols = [c for c in ["open", "high", "low", "close", "volume"]
                            if c in df.columns]
                    result[symbol] = df[cols].astype(float)
                except Exception:
                    pass
            return result
        except Exception as exc:
            logger.warning("Alpaca bulk intraday failed: %s — falling back", exc)
            return super().get_intraday_bars_bulk(symbols, start, end, interval_minutes)

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            self._client().get_account()
            return True
        except Exception:
            return False
