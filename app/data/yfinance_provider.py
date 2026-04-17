"""
yfinance data provider.

Strengths : free, no API key, broad coverage, 3+ years daily history
Limits    : 5-min bars only last ~60 days, rate-limited under heavy use
Use for   : model training (daily + intraday), backtesting
"""

import logging
import time
from datetime import date, datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf

from app.data.base import DataProvider

logger = logging.getLogger(__name__)

# Seconds to sleep between yfinance calls to avoid rate limits
_POLITE_DELAY = 0.1


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
        try:
            df = yf.download(
                symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            df = self._normalise(df)
            if df.empty or "close" not in df.columns or len(df) < 10:
                return None
            time.sleep(_POLITE_DELAY)
            return df[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as exc:
            logger.debug("yfinance daily bars failed for %s: %s", symbol, exc)
            return None

    def get_daily_bars_bulk(
        self,
        symbols: List[str],
        start: date,
        end: date,
    ) -> dict:
        """Download all symbols in one yfinance call (much faster than looping)."""
        try:
            raw = yf.download(
                symbols,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
        except Exception as exc:
            logger.warning("yfinance bulk download failed: %s — falling back", exc)
            return super().get_daily_bars_bulk(symbols, start, end)

        result = {}
        for symbol in symbols:
            try:
                if len(symbols) == 1:
                    df = raw.copy()
                else:
                    df = raw[symbol].copy()
                df = self._normalise(df)
                df = df.dropna(subset=["close"])
                if not df.empty and len(df) >= 10:
                    result[symbol] = df[["open", "high", "low", "close", "volume"]].astype(float)
            except Exception:
                pass
        return result

    # ── Intraday ──────────────────────────────────────────────────────────────

    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> Optional[pd.DataFrame]:
        interval_map = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
        interval = interval_map.get(interval_minutes, "5m")
        try:
            df = yf.download(
                symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            df = self._normalise(df)
            if df.empty or "close" not in df.columns:
                return None
            time.sleep(_POLITE_DELAY)
            return df[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as exc:
            logger.debug("yfinance intraday bars failed for %s: %s", symbol, exc)
            return None
