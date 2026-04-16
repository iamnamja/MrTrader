"""
Technical indicators for the Trader Agent.

All functions accept plain Python lists or numpy arrays and return scalar
floats (or tuples of floats). Pandas Series are accepted wherever lists are.

Edge-case handling:
- Returns None when there is insufficient data instead of raising.
- NaN values in input are forward-filled before computation.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_series(values: Sequence[float]) -> pd.Series:
    """Convert any sequence to a clean float Series (NaN forward-filled)."""
    s = pd.Series(values, dtype=float)
    return s.ffill().bfill()


# ─── RSI ─────────────────────────────────────────────────────────────────────

def calculate_rsi(prices: Sequence[float], period: int = 14) -> Optional[float]:
    """
    Relative Strength Index.

    Range: 0–100
    Signals: RSI > 70 overbought, RSI < 30 oversold.

    Returns None if insufficient data (need at least period + 1 prices).
    """
    s = _to_series(prices)
    if len(s) < period + 1:
        return None

    delta = s.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # When avg_loss == 0: pure uptrend → RSI = 100
    # When avg_gain == 0: pure downtrend → RSI = 0
    # When both == 0: flat → RSI = 50
    rsi = np.where(
        avg_loss == 0,
        np.where(avg_gain == 0, 50.0, 100.0),
        100 - (100 / (1 + avg_gain / avg_loss)),
    )
    val = float(pd.Series(rsi).iloc[-1])
    return val


# ─── MACD ────────────────────────────────────────────────────────────────────

def calculate_macd(
    prices: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[Tuple[float, float, float]]:
    """
    MACD (Moving Average Convergence Divergence).

    Returns (macd_line, signal_line, histogram) or None if insufficient data.
    Signal: macd_line > signal_line → bullish, < signal_line → bearish.
    """
    s = _to_series(prices)
    if len(s) < slow + signal:
        return None

    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return (
        float(macd_line.iloc[-1]),
        float(signal_line.iloc[-1]),
        float(histogram.iloc[-1]),
    )


# ─── Bollinger Bands ──────────────────────────────────────────────────────────

def calculate_bollinger_bands(
    prices: Sequence[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> Optional[Tuple[float, float, float]]:
    """
    Bollinger Bands.

    Returns (upper_band, middle_band, lower_band) or None if insufficient data.
    Signals: price near lower band → oversold, near upper band → overbought.
    """
    s = _to_series(prices)
    if len(s) < period:
        return None

    middle = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return (
        float(upper.iloc[-1]),
        float(middle.iloc[-1]),
        float(lower.iloc[-1]),
    )


# ─── ATR ─────────────────────────────────────────────────────────────────────

def calculate_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> Optional[float]:
    """
    Average True Range — volatility measure.

    Used for dynamic stop-loss sizing and position sizing.
    Returns None if insufficient data.
    """
    h = _to_series(highs)
    l = _to_series(lows)
    c = _to_series(closes)

    if len(c) < period + 1:
        return None

    prev_close = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return float(atr.iloc[-1])


# ─── EMA ─────────────────────────────────────────────────────────────────────

def calculate_ema(prices: Sequence[float], period: int = 20) -> Optional[float]:
    """
    Exponential Moving Average.

    Used to identify trend direction.
    Returns None if insufficient data.
    """
    s = _to_series(prices)
    if len(s) < period:
        return None

    ema = s.ewm(span=period, adjust=False).mean()
    return float(ema.iloc[-1])


# ─── SMA ─────────────────────────────────────────────────────────────────────

def calculate_sma(prices: Sequence[float], period: int = 20) -> Optional[float]:
    """
    Simple Moving Average.

    Returns None if insufficient data.
    """
    s = _to_series(prices)
    if len(s) < period:
        return None

    return float(s.rolling(period).mean().iloc[-1])


# ─── Signal helpers ───────────────────────────────────────────────────────────

def is_oversold(prices: Sequence[float], rsi_threshold: float = 30) -> bool:
    """True if the latest RSI is below rsi_threshold."""
    rsi = calculate_rsi(prices)
    return rsi is not None and rsi < rsi_threshold


def is_overbought(prices: Sequence[float], rsi_threshold: float = 70) -> bool:
    """True if the latest RSI is above rsi_threshold."""
    rsi = calculate_rsi(prices)
    return rsi is not None and rsi > rsi_threshold


def price_near_band(
    current_price: float,
    band_price: float,
    tolerance_pct: float = 0.005,
) -> bool:
    """True if current_price is within tolerance_pct of band_price."""
    if band_price == 0:
        return False
    return abs(current_price - band_price) / band_price <= tolerance_pct
