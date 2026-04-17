"""
Intraday feature engineering for 5-minute bar models.

Features:
  - Opening range breakout (ORB): price relative to first 30-min high/low
  - VWAP distance: how far price is from VWAP at feature time
  - Pre-market gap: overnight gap as % of prior close
  - Volume surge: bar volume vs 20-bar rolling average
  - RSI(14) on 5-min closes
  - SPY/QQQ direction: benchmark trend for the session
  - ATR-normalised range: intraday volatility context
  - Time of day: fraction of session elapsed (0=open, 1=close)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum bars needed to compute features
MIN_BARS = 12  # 1 hour of 5-min bars


def compute_intraday_features(
    bars: pd.DataFrame,
    spy_bars: Optional[pd.DataFrame] = None,
    prior_close: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """
    Compute intraday features from a slice of 5-min OHLCV bars.

    Args:
        bars:        5-min OHLCV with columns [open, high, low, close, volume].
                     Index is DatetimeTzAware or naive timestamps.
        spy_bars:    Optional SPY 5-min bars for same session (benchmark).
        prior_close: Prior-day close for gap calculation.

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

    feats: Dict[str, float] = {}

    # 1. Opening range (first 6 bars = 30 min)
    orb_bars = min(6, len(bars))
    orb_high = float(highs[:orb_bars].max())
    orb_low = float(lows[:orb_bars].min())
    orb_range = orb_high - orb_low if orb_high > orb_low else 1e-6
    last_close = closes[-1]
    feats["orb_position"] = float((last_close - orb_low) / orb_range)
    feats["orb_breakout"] = float(
        1 if last_close > orb_high else (-1 if last_close < orb_low else 0)
    )

    # 2. VWAP distance
    typical = (highs + lows + closes) / 3.0
    cum_vol = np.cumsum(volumes)
    cum_tp_vol = np.cumsum(typical * volumes)
    vwap = cum_tp_vol[-1] / cum_vol[-1] if cum_vol[-1] > 0 else last_close
    feats["vwap_distance"] = float((last_close - vwap) / vwap) if vwap != 0 else 0.0

    # 3. Pre-market gap
    first_open = float(opens[0])
    if prior_close and prior_close > 0:
        feats["gap_pct"] = float((first_open - prior_close) / prior_close)
    else:
        feats["gap_pct"] = 0.0

    # 4. Volume surge (last bar vs 20-bar average)
    vol_window = min(20, len(volumes) - 1)
    avg_vol = float(volumes[-vol_window - 1:-1].mean()) if vol_window > 0 else float(volumes[-1])
    feats["volume_surge"] = float(volumes[-1] / avg_vol) if avg_vol > 0 else 1.0

    # 5. RSI(14) on closes
    feats["rsi_14"] = _rsi(closes, 14)

    # 6. ATR-normalised range (last 14 bars)
    atr = _atr(highs, lows, closes, 14)
    feats["atr_norm"] = float(atr / last_close) if last_close > 0 else 0.0

    # 7. Momentum: return since open
    feats["session_return"] = (
        float((last_close - first_open) / first_open) if first_open > 0 else 0.0
    )

    # 8. Return over last 3 bars (~15 min)
    if len(closes) >= 4:
        feats["ret_15m"] = float((closes[-1] - closes[-4]) / closes[-4]) if closes[-4] > 0 else 0.0
    else:
        feats["ret_15m"] = 0.0

    # 9. Time of day fraction (assuming 6.5-hour session = 78 bars of 5-min)
    session_bars = 78
    feats["time_of_day"] = float(min(len(bars) / session_bars, 1.0))

    # 10. SPY/QQQ direction (return from session open)
    if spy_bars is not None and len(spy_bars) >= 2:
        spy_closes = spy_bars["close"].values.astype(float)
        spy_open = float(spy_bars["open"].values[0])
        feats["spy_session_return"] = float(
            (spy_closes[-1] - spy_open) / spy_open
        ) if spy_open > 0 else 0.0
    else:
        feats["spy_session_return"] = 0.0

    # 11. Relative volume vs SPY (rough proxy for institutional interest)
    if spy_bars is not None and len(spy_bars) > 0:
        spy_vol = float(spy_bars["volume"].values[-20:].mean()) if len(spy_bars) >= 20 else 1.0
        stock_vol = float(volumes[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())
        feats["rel_vol_spy"] = float(stock_vol / spy_vol) if spy_vol > 0 else 1.0
    else:
        feats["rel_vol_spy"] = 1.0

    # 12. Price range compression (tight range before breakout signal)
    range_5 = float((highs[-5:].max() - lows[-5:].min()) / last_close) if len(bars) >= 5 else 0.0
    feats["range_compression"] = range_5

    return feats


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


FEATURE_NAMES = [
    "orb_position", "orb_breakout", "vwap_distance", "gap_pct",
    "volume_surge", "rsi_14", "atr_norm", "session_return",
    "ret_15m", "time_of_day", "spy_session_return", "rel_vol_spy",
    "range_compression",
]
