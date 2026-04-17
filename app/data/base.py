"""
Abstract data provider interface.

All data sources (yfinance, Alpaca, Polygon, Databento, etc.) must
implement this contract. The rest of the system only depends on this
interface — swapping providers requires zero changes to agents or models.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import List, Optional
import pandas as pd


class DataProvider(ABC):
    """
    Contract for market data providers.

    All returned DataFrames use lowercase column names:
        open, high, low, close, volume
    Index is a DatetimeIndex (daily) or DatetimeTZIndex (intraday, UTC).
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, e.g. 'yfinance' or 'alpaca'."""

    # ── Daily bars ────────────────────────────────────────────────────────────

    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> Optional[pd.DataFrame]:
        """
        Return daily OHLCV bars for *symbol* between *start* and *end* (inclusive).

        Returns None if data is unavailable or too short.
        Columns: open, high, low, close, volume (all float).
        """

    def get_daily_bars_bulk(
        self,
        symbols: List[str],
        start: date,
        end: date,
    ) -> dict:
        """
        Return {symbol: DataFrame} for multiple symbols.

        Default implementation calls get_daily_bars() in a loop.
        Override for providers that support batch requests (e.g. Alpaca).
        """
        result = {}
        for symbol in symbols:
            df = self.get_daily_bars(symbol, start, end)
            if df is not None:
                result[symbol] = df
        return result

    # ── Intraday bars ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> Optional[pd.DataFrame]:
        """
        Return intraday OHLCV bars for *symbol*.

        Args:
            symbol:            Ticker.
            start / end:       Datetime range (timezone-aware recommended).
            interval_minutes:  Bar size — 1, 5, 15, 30, or 60.

        Returns None if data is unavailable.
        Columns: open, high, low, close, volume (all float).
        """

    def get_intraday_bars_bulk(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> dict:
        """
        Return {symbol: DataFrame} for multiple symbols.

        Default implementation calls get_intraday_bars() in a loop.
        Override for providers that support batch requests.
        """
        result = {}
        for symbol in symbols:
            df = self.get_intraday_bars(symbol, start, end, interval_minutes)
            if df is not None:
                result[symbol] = df
        return result

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if the provider is reachable. Override as needed."""
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case columns, rename 'adj close' -> 'close'."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns and "adj close" in df.columns:
            df = df.rename(columns={"adj close": "close"})
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
