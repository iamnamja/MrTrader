"""
Mean-reversion signal using Bollinger Band squeeze.

Entry conditions (BUY):
  1. Price is below lower Bollinger Band (oversold, potential bounce)
  2. Band width is narrow (squeeze: BB_width / price < SQUEEZE_THRESHOLD)
  3. Price is above its 50-period SMA (not in a downtrend)
  4. RSI is between 25-45 (oversold but not free-falling)

Exit: handled by the standard check_exit() in signals.py (ATR stop/target).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
SMA_PERIOD = 50
SQUEEZE_THRESHOLD = 0.04   # band_width / price < 4% → squeeze
RSI_LOW = 25.0
RSI_HIGH = 45.0


def _rsi(closes: pd.Series, period: int = RSI_PERIOD) -> float:
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("inf"))
    rsi_series = 100 - 100 / (1 + rs)
    return float(rsi_series.iloc[-1])


def check_mean_reversion_signal(symbol: str, bars: pd.DataFrame) -> Optional[str]:
    """
    Return "BUY" if a mean-reversion entry is warranted, None otherwise.

    bars must have at least BB_PERIOD + SMA_PERIOD rows with a 'close' column.
    """
    required = max(BB_PERIOD, SMA_PERIOD, RSI_PERIOD) + 5
    if len(bars) < required:
        return None

    close = bars["close"]
    last = float(close.iloc[-1])

    # Bollinger Bands
    rolling = close.rolling(BB_PERIOD)
    mid = rolling.mean()
    std = rolling.std()
    lower_bb = float(mid.iloc[-1]) - BB_STD * float(std.iloc[-1])
    upper_bb = float(mid.iloc[-1]) + BB_STD * float(std.iloc[-1])
    band_width = upper_bb - lower_bb

    if last >= lower_bb:
        return None  # not below lower band

    squeeze_ratio = band_width / last if last > 0 else float("inf")
    if squeeze_ratio >= SQUEEZE_THRESHOLD:
        return None  # bands too wide, not a squeeze

    sma50 = float(close.rolling(SMA_PERIOD).mean().iloc[-1])
    if last < sma50:
        return None  # below 50-SMA → downtrend, skip

    rsi = _rsi(close)
    if not (RSI_LOW <= rsi <= RSI_HIGH):
        return None  # not in oversold range

    logger.info(
        "%s: mean-reversion BUY — price %.2f below BB %.2f, squeeze %.3f, RSI %.1f",
        symbol, last, lower_bb, squeeze_ratio, rsi,
    )
    return "BUY"
