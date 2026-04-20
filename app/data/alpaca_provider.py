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
            df = client.get_bars(
                symbol,
                timeframe="1Day",
                start=start if isinstance(start, str) else start.isoformat(),
                end=end if isinstance(end, str) else end.isoformat(),
            )
            if df is None or df.empty:
                return None
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
            df = client.get_bars(
                symbol,
                timeframe=timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
            )
            if df is None or df.empty:
                return None
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
        """Use Alpaca multi-symbol request for efficiency."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from app.config import settings

        tf_map = {1: TimeFrame.Minute, 5: TimeFrame(5, TimeFrameUnit.Minute),
                  15: TimeFrame(15, TimeFrameUnit.Minute), 60: TimeFrame.Hour}
        tf = tf_map.get(interval_minutes, TimeFrame(5, TimeFrameUnit.Minute))

        try:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)

            data_client = StockHistoricalDataClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )
            req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf,
                                   start=start, end=end)
            bars_resp = data_client.get_stock_bars(req)

            result = {}
            for sym in symbols:
                try:
                    sym_bars = bars_resp[sym]
                    if not sym_bars:
                        continue
                    records = [{"open": b.open, "high": b.high, "low": b.low,
                                "close": b.close, "volume": b.volume}
                               for b in sym_bars]
                    timestamps = [b.timestamp for b in sym_bars]
                    df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps, name="timestamp"))
                    result[sym] = df.astype(float)
                except (KeyError, TypeError):
                    pass
            logger.info("Alpaca bulk intraday: got %d / %d symbols", len(result), len(symbols))
            return result
        except Exception as exc:
            logger.warning("Alpaca bulk intraday failed: %s — falling back to serial", exc)
            return super().get_intraday_bars_bulk(symbols, start, end, interval_minutes)

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            self._client().get_account()
            return True
        except Exception:
            return False
