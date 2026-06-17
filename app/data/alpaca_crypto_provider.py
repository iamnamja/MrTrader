"""Alpaca spot-crypto daily bars — P3-1.

Thin wrapper over Alpaca's `CryptoHistoricalDataClient` to fetch daily CLOSE panels for a
crypto basket (BTC/USD, ETH/USD, + liquid alts). Alpaca crypto history starts ~2021-01-01.
Mirrors the equity providers' "close-price DataFrame" contract so the existing TSMOM engine
and Sleeve Lab consume it unchanged.

Crypto trades 24/7/365 (one bar per CALENDAR day), so callers must run TSMOM with `ann=365`
(not the equity 252) — handled in the `crypto_trend` sleeve builder.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Liquid USD pairs available on Alpaca (provider drops any with no data for the window).
DEFAULT_CRYPTO_UNIVERSE: List[str] = [
    "BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "BCH/USD",
    "AVAX/USD", "LINK/USD", "UNI/USD", "AAVE/USD", "DOGE/USD",
]
CRYPTO_HISTORY_START = date(2021, 1, 1)   # Alpaca crypto data inception


class AlpacaCryptoProvider:
    """Daily crypto closes from Alpaca. Long-only spot data (no shorting on Alpaca crypto)."""

    def __init__(self):
        from alpaca.data.historical import CryptoHistoricalDataClient
        from app.config import settings
        # Crypto data is public on Alpaca; keys are still passed for rate-tier.
        self._client = CryptoHistoricalDataClient(
            api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key)

    def get_daily_closes(self, symbols: List[str], *, start: date = CRYPTO_HISTORY_START,
                         end: Optional[date] = None) -> pd.DataFrame:
        """Close-price panel: index = naive daily DatetimeIndex (UTC date), columns = symbols
        that returned data. Empty DataFrame if nothing came back. Never raises on a missing
        symbol — it's simply absent from the columns."""
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        syms = list(symbols)
        start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        end_dt = (datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
                  if end else None)
        req = CryptoBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Day,
                                start=start_dt, end=end_dt)
        bars = self._client.get_crypto_bars(req)
        df = bars.df
        if df is None or df.empty:
            logger.warning("crypto provider: no bars for %s", syms)
            return pd.DataFrame()
        # df is MultiIndex (symbol, timestamp); pivot close -> columns by symbol.
        closes = df["close"].unstack("symbol")
        # Normalize the timestamp index to a naive daily DatetimeIndex.
        idx = closes.index
        try:
            idx = idx.tz_convert("UTC").tz_localize(None)
        except (TypeError, AttributeError):
            try:
                idx = idx.tz_localize(None)
            except (TypeError, AttributeError):
                pass
        closes.index = pd.DatetimeIndex(idx).normalize()
        closes = closes.sort_index()
        # PIT: crypto has no daily settlement, so the bar for the CURRENT UTC day is a partial
        # (in-progress) bar — its close is just the live price at fetch time, which a backtest
        # could not have known at the prior decision. Drop it (unless caller pinned `end`).
        if end is None and len(closes):
            today_utc = datetime.now(timezone.utc).date()
            closes = closes[closes.index.normalize() < pd.Timestamp(today_utc)]
        # Keep only requested symbols that actually returned data.
        present = [s for s in syms if s in closes.columns and closes[s].notna().any()]
        return closes[present]
