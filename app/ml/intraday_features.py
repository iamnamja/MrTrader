"""
Intraday feature engineering for 5-minute bar models.

Features (25 total):
  ── Price / structure ──────────────────────────────────────────────────────
  orb_position        Price position within opening 30-min range (0=low,1=high)
  orb_breakout        +1 above ORB high, -1 below ORB low, 0 inside
  vwap_distance       (close - VWAP) / VWAP — signed distance from fair value
  gap_pct             Overnight gap: (open - prior_close) / prior_close
  prev_day_high_dist  (close - prior_day_high) / close
  prev_day_low_dist   (close - prior_day_low) / close

  ── Trend / moving averages ────────────────────────────────────────────────
  ema9_dist           (close - EMA9) / close
  ema20_dist          (close - EMA20) / close
  ema_cross           (EMA9 - EMA20) / close — positive = bullish cross
  macd_hist           MACD histogram (12/26/9 EMA) normalised by close
  bb_position         Bollinger Band %B: 0=lower band, 1=upper band

  ── Momentum ──────────────────────────────────────────────────────────────
  rsi_14              RSI(14) on 5-min closes (0-100, normalised to 0-1)
  session_return      (close - first_open) / first_open
  ret_15m             Return over last 3 bars (~15 min)
  ret_30m             Return over last 6 bars (~30 min)
  stoch_k             Stochastic %K(14) — close position within 14-bar range

  ── Volume / order flow ───────────────────────────────────────────────────
  volume_surge        Last bar volume / 20-bar average volume
  cum_delta           Cumulative buying pressure: sum(close>open bars) / n_bars
  vol_trend           Volume EMA(10) slope (rising volume = momentum)

  ── Volatility ────────────────────────────────────────────────────────────
  atr_norm            ATR(14) / close — normalised intraday volatility
  range_compression   5-bar H-L range / close — tight = potential breakout

  ── Market context ────────────────────────────────────────────────────────
  spy_session_return  SPY return from session open (benchmark direction)
  spy_rsi_14          SPY RSI(14) — market momentum context
  rel_vol_spy         Stock 20-bar avg vol / SPY 20-bar avg vol

  ── Session timing ────────────────────────────────────────────────────────
  time_of_day         Fraction of 6.5-hr session elapsed (0=open, 1=close)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_BARS = 12  # 1 hour of 5-min bars


def compute_intraday_features(
    bars: pd.DataFrame,
    spy_bars: Optional[pd.DataFrame] = None,
    prior_close: Optional[float] = None,
    prior_day_high: Optional[float] = None,
    prior_day_low: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """
    Compute intraday features from a slice of 5-min OHLCV bars.

    Args:
        bars:           5-min OHLCV [open, high, low, close, volume].
        spy_bars:       Optional SPY 5-min bars for same session.
        prior_close:    Prior-day close for gap and S/R calculation.
        prior_day_high: Prior-day high for S/R proximity.
        prior_day_low:  Prior-day low for S/R proximity.

    Returns:
        Dict of feature_name -> float, or None if insufficient data.
    """
    if bars is None or len(bars) < MIN_BARS:
        return None

    bars = bars.copy()
    closes = bars["close"].values.astype(float)
    highs = bars["high"].values.astype(float)
    lows = bars["low"].values.astype(float)
    volumes = bars["volume"].values.astype(float)
    opens = bars["open"].values.astype(float)
    last_close = closes[-1]
    first_open = float(opens[0])

    feats: Dict[str, float] = {}

    # ── Opening range ─────────────────────────────────────────────────────
    orb_bars = min(6, len(bars))
    orb_high = float(highs[:orb_bars].max())
    orb_low = float(lows[:orb_bars].min())
    orb_range = orb_high - orb_low if orb_high > orb_low else 1e-6
    feats["orb_position"] = float((last_close - orb_low) / orb_range)
    feats["orb_breakout"] = float(
        1 if last_close > orb_high else (-1 if last_close < orb_low else 0)
    )

    # ── VWAP ──────────────────────────────────────────────────────────────
    typical = (highs + lows + closes) / 3.0
    cum_vol = np.cumsum(volumes)
    cum_tp_vol = np.cumsum(typical * volumes)
    vwap = cum_tp_vol[-1] / cum_vol[-1] if cum_vol[-1] > 0 else last_close
    feats["vwap_distance"] = float((last_close - vwap) / vwap) if vwap != 0 else 0.0

    # ── Gap ───────────────────────────────────────────────────────────────
    if prior_close and prior_close > 0:
        feats["gap_pct"] = float((first_open - prior_close) / prior_close)
    else:
        feats["gap_pct"] = 0.0

    # ── Prior-day S/R levels ──────────────────────────────────────────────
    if prior_day_high and prior_day_high > 0:
        feats["prev_day_high_dist"] = float((last_close - prior_day_high) / last_close)
    else:
        feats["prev_day_high_dist"] = 0.0

    if prior_day_low and prior_day_low > 0:
        feats["prev_day_low_dist"] = float((last_close - prior_day_low) / last_close)
    else:
        feats["prev_day_low_dist"] = 0.0

    # ── EMAs ──────────────────────────────────────────────────────────────
    ema9 = _ema(closes, 9)
    ema20 = _ema(closes, 20)
    feats["ema9_dist"] = float((last_close - ema9) / last_close) if last_close > 0 else 0.0
    feats["ema20_dist"] = float((last_close - ema20) / last_close) if last_close > 0 else 0.0
    feats["ema_cross"] = float((ema9 - ema20) / last_close) if last_close > 0 else 0.0

    # ── MACD histogram (12/26/9) ──────────────────────────────────────────
    macd_hist = _macd_histogram(closes, fast=12, slow=26, signal=9)
    feats["macd_hist"] = float(macd_hist / last_close) if last_close > 0 else 0.0

    # ── Bollinger Bands %B (20-period, 2 std) ─────────────────────────────
    feats["bb_position"] = _bollinger_pct_b(closes, period=20, num_std=2.0)

    # ── RSI(14) normalised to 0-1 ─────────────────────────────────────────
    feats["rsi_14"] = _rsi(closes, 14) / 100.0

    # ── Stochastic %K(14) normalised to 0-1 ──────────────────────────────
    feats["stoch_k"] = _stochastic_k(highs, lows, closes, period=14) / 100.0

    # ── Momentum ──────────────────────────────────────────────────────────
    feats["session_return"] = (
        float((last_close - first_open) / first_open) if first_open > 0 else 0.0
    )
    feats["ret_15m"] = (
        float((closes[-1] - closes[-4]) / closes[-4])
        if len(closes) >= 4 and closes[-4] > 0 else 0.0
    )
    feats["ret_30m"] = (
        float((closes[-1] - closes[-7]) / closes[-7])
        if len(closes) >= 7 and closes[-7] > 0 else 0.0
    )

    # ── ATR and range ─────────────────────────────────────────────────────
    atr = _atr(highs, lows, closes, 14)
    feats["atr_norm"] = float(atr / last_close) if last_close > 0 else 0.0
    range_5 = (
        float((highs[-5:].max() - lows[-5:].min()) / last_close)
        if len(bars) >= 5 else 0.0
    )
    feats["range_compression"] = range_5

    # ── Volume ────────────────────────────────────────────────────────────
    vol_window = min(20, len(volumes) - 1)
    avg_vol = (
        float(volumes[-vol_window - 1:-1].mean()) if vol_window > 0 else float(volumes[-1])
    )
    feats["volume_surge"] = float(volumes[-1] / avg_vol) if avg_vol > 0 else 1.0

    # Cumulative delta: fraction of bars where close > open (buying pressure)
    feats["cum_delta"] = float(np.sum(closes > opens) / len(closes))

    # Volume EMA slope: (vol_ema_last - vol_ema_prev) / vol_ema_prev
    feats["vol_trend"] = _ema_slope(volumes, period=10)

    # ── SPY context ───────────────────────────────────────────────────────
    if spy_bars is not None and len(spy_bars) >= 2:
        spy_closes = spy_bars["close"].values.astype(float)
        spy_opens = spy_bars["open"].values.astype(float)
        spy_open0 = float(spy_opens[0])
        feats["spy_session_return"] = (
            float((spy_closes[-1] - spy_open0) / spy_open0) if spy_open0 > 0 else 0.0
        )
        feats["spy_rsi_14"] = _rsi(spy_closes, 14) / 100.0
        spy_vol = (
            float(spy_bars["volume"].values[-20:].mean())
            if len(spy_bars) >= 20 else 1.0
        )
        stock_vol = (
            float(volumes[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())
        )
        feats["rel_vol_spy"] = float(stock_vol / spy_vol) if spy_vol > 0 else 1.0
    else:
        feats["spy_session_return"] = 0.0
        feats["spy_rsi_14"] = 0.5
        feats["rel_vol_spy"] = 1.0

    # ── Time of day ───────────────────────────────────────────────────────
    feats["time_of_day"] = float(min(len(bars) / 78, 1.0))  # 78 bars = 6.5 hr session

    return feats


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(values: np.ndarray, period: int) -> float:
    if len(values) == 0:
        return 0.0
    if len(values) < period:
        return float(values.mean())
    k = 2.0 / (period + 1)
    ema = float(values[:period].mean())
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def _ema_series(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    k = 2.0 / (period + 1)
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _macd_histogram(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    if len(closes) < slow + signal:
        return 0.0
    fast_ema = _ema_series(closes, fast)
    slow_ema = _ema_series(closes, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema_series(macd_line, signal)
    return float(macd_line[-1] - signal_line[-1])


def _bollinger_pct_b(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    mid = float(window.mean())
    std = float(window.std(ddof=1)) if len(window) > 1 else 1e-9
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_range = upper - lower
    if band_range < 1e-9:
        return 0.5
    return float(np.clip((closes[-1] - lower) / band_range, 0.0, 1.0))


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _stochastic_k(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> float:
    if len(closes) < period:
        return 50.0
    h = float(highs[-period:].max())
    lo = float(lows[-period:].min())
    if h == lo:
        return 50.0
    return float(np.clip((closes[-1] - lo) / (h - lo) * 100.0, 0.0, 100.0))


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < 2:
        return float(highs[-1] - lows[-1])
    n = min(period, len(closes) - 1)
    trs = []
    for i in range(-n, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs)) if trs else float(highs[-1] - lows[-1])


def _ema_slope(values: np.ndarray, period: int = 10) -> float:
    if len(values) < period + 1:
        return 0.0
    series = _ema_series(values, period)
    prev = series[-2]
    if prev == 0:
        return 0.0
    return float((series[-1] - prev) / prev)


FEATURE_NAMES = [
    # Price / structure
    "orb_position", "orb_breakout", "vwap_distance", "gap_pct",
    "prev_day_high_dist", "prev_day_low_dist",
    # Trend / moving averages
    "ema9_dist", "ema20_dist", "ema_cross", "macd_hist", "bb_position",
    # Momentum
    "rsi_14", "session_return", "ret_15m", "ret_30m", "stoch_k",
    # Volume / order flow
    "volume_surge", "cum_delta", "vol_trend",
    # Volatility
    "atr_norm", "range_compression",
    # Market context
    "spy_session_return", "spy_rsi_14", "rel_vol_spy",
    # Session timing
    "time_of_day",
]
