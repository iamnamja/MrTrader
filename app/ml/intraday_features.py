"""
Intraday feature engineering for 5-minute bar models.

Features (53 total — Phase 50 added 3 session-segment features):
  ── Price / structure ──────────────────────────────────────────────────────
  orb_position        Price position within opening 30-min range (0=low, 1=high)
  orb_breakout        +1 above ORB high, -1 below ORB low, 0 inside
  orb_direction_strength  Signed distance from nearest ORB boundary / ORB range.
                          +1 = one full range above ORB high (strong breakout up),
                          -1 = one full range below ORB low (strong breakdown),
                          0 = at ORB midpoint. Captures breakout velocity.
  vwap_distance       (close - VWAP) / VWAP — signed distance from fair value
  vwap_cross_count    Number of times price crossed VWAP today
  gap_pct             Overnight gap: (open - prior_close) / prior_close
  gap_fill_pct        Fraction of opening gap already filled (0=unfilled, 1=full)
  session_hl_position (close - session_low) / (session_high - session_low)
  prev_day_high_dist  (close - prior_day_high) / close
  prev_day_low_dist   (close - prior_day_low) / close

  ── Trend / moving averages ────────────────────────────────────────────────
  ema9_dist           (close - EMA9) / close
  ema20_dist          (close - EMA20) / close
  ema_cross           (EMA9 - EMA20) / close — positive = bullish cross
  macd_hist           MACD histogram (12/26/9 EMA) normalised by close
  bb_position         Bollinger Band %B: 0=lower band, 1=upper band

  ── Momentum ──────────────────────────────────────────────────────────────
  rsi_14              RSI(14) on 5-min closes, normalised to [0, 1]
  session_return      (close - first_open) / first_open
  ret_15m             Return over last 3 bars (~15 min)
  ret_30m             Return over last 6 bars (~30 min)
  stoch_k             Stochastic %K(14), normalised to [0, 1]
  williams_r          Williams %R(14), normalised to [0, 1]

  ── Volume / order flow ───────────────────────────────────────────────────
  volume_surge        Last bar volume / 20-bar average volume
  cum_delta           Cumulative buying pressure: sum(close>open bars) / n_bars
  vol_trend           Volume EMA(10) slope (rising volume = momentum)
  obv_slope           On-balance volume EMA(10) slope (volume confirms price)

  ── Candlestick / structure ───────────────────────────────────────────────
  upper_wick_ratio    Upper wick / total range — rejection / selling pressure
  lower_wick_ratio    Lower wick / total range — support / buying pressure
  body_ratio          Candle body / total range — conviction of last bar
  consecutive_bars    Signed count of consecutive same-direction closes (+up/-down)

  ── Volatility ────────────────────────────────────────────────────────────
  atr_norm            ATR(14) / close — normalised intraday volatility
  range_compression   5-bar H-L range / close — tight = potential breakout

  ── Market context ────────────────────────────────────────────────────────
  spy_session_return  SPY return from session open (benchmark direction)
  spy_rsi_14          SPY RSI(14), normalised to [0, 1]
  rel_vol_spy         Stock 20-bar avg vol / SPY 20-bar avg vol


  ── Session timing ────────────────────────────────────────────────────────
  time_of_day         Fraction of 6.5-hr session elapsed (0=open, 1=close)
  minutes_since_open  Minutes elapsed since 09:30 open (0-390)
  is_open_session     1 if within first 30 min (open auction / gap-fill window)
  is_close_session    1 if within last 60 min (closing imbalance / MOC window)

  ── Daily vol context (from prior daily bars) ─────────────────────────────
  daily_vol_percentile  Realized vol percentile vs 52-week range [0,1]
  daily_vol_regime      Short/long vol ratio — expanding vs contracting
  daily_parkinson_vol   Parkinson high-low vol estimator (annualised)

  ── Phase 47-5: Quality / structure features ──────────────────────────────
  trend_efficiency      Net displacement / total path in bars so far (1=perfectly linear)
  green_bar_ratio       Fraction of bars that closed up (buy pressure proxy)
  above_vwap_ratio      Fraction of bars where close > rolling VWAP (persistence above fair value)
  pullback_from_high    (session_high - close) / session_high — distance from peak
  range_vs_20d_avg      Today's 5-min range vs 20-day avg range (volatility expansion)
  rel_strength_vs_spy   Stock session_return - spy_session_return (alpha vs benchmark)
  vol_x_momentum        volume_surge × session_return interaction (volume-confirmed momentum)
  gap_followthrough     gap_pct × session_return — measures if gap direction continued

  ── NIS features (Phase 64b) ──────────────────────────────────────────────
  nis_direction_score   LLM news direction [-1, 1] (point-in-time from NewsSignalCache)
  nis_materiality_score News materiality [0, 1] — how market-moving the event is
  nis_already_priced_in Already-priced-in score [0, 1] — reduces edge of stale news
  nis_sizing_mult       NIS position sizing multiplier [0.5, 2.0]
  nis_downside_risk     Downside risk score [0, 1]
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
    daily_bars: Optional[pd.DataFrame] = None,
    spy_daily_bars: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
    as_of_date=None,
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
    # Signed distance from nearest ORB boundary, normalised by ORB range.
    # Distinguishes a weak poke above ORB from a sustained trend breakout.
    orb_mid = (orb_high + orb_low) / 2.0
    feats["orb_direction_strength"] = float(
        np.clip((last_close - orb_mid) / orb_range, -2.0, 2.0)
    )

    # ── VWAP ──────────────────────────────────────────────────────────────
    typical = (highs + lows + closes) / 3.0
    cum_vol = np.cumsum(volumes)
    cum_tp_vol = np.cumsum(typical * volumes)
    vwap = cum_tp_vol[-1] / cum_vol[-1] if cum_vol[-1] > 0 else last_close
    feats["vwap_distance"] = float((last_close - vwap) / vwap) if vwap != 0 else 0.0

    # VWAP cross count: number of times closes crossed VWAP
    feats["vwap_cross_count"] = float(_vwap_cross_count(closes, highs, lows, volumes))

    # ── Gap and gap fill ──────────────────────────────────────────────────
    if prior_close and prior_close > 0:
        gap = float(first_open - prior_close)
        gap_pct = gap / prior_close
        feats["gap_pct"] = gap_pct
        # Gap fill: how much of the gap has been retraced
        if abs(gap) > 1e-6:
            if gap > 0:  # gap up — fill = price came back down toward prior_close
                filled = float(min(first_open - lows.min(), abs(gap)) / abs(gap))
            else:  # gap down — fill = price came back up toward prior_close
                filled = float(min(highs.max() - first_open, abs(gap)) / abs(gap))
            feats["gap_fill_pct"] = float(np.clip(filled, 0.0, 1.0))
        else:
            feats["gap_fill_pct"] = 1.0
    else:
        feats["gap_pct"] = 0.0
        feats["gap_fill_pct"] = 1.0

    # ── Session high/low position ─────────────────────────────────────────
    session_high = float(highs.max())
    session_low = float(lows.min())
    session_range = session_high - session_low if session_high > session_low else 1e-6
    feats["session_hl_position"] = float((last_close - session_low) / session_range)

    # ── Prior-day S/R levels ──────────────────────────────────────────────
    feats["prev_day_high_dist"] = (
        float((last_close - prior_day_high) / last_close)
        if prior_day_high and prior_day_high > 0 else 0.0
    )
    feats["prev_day_low_dist"] = (
        float((last_close - prior_day_low) / last_close)
        if prior_day_low and prior_day_low > 0 else 0.0
    )

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

    # ── RSI(14) normalised to [0, 1] ──────────────────────────────────────
    feats["rsi_14"] = _rsi(closes, 14) / 100.0

    # ── Stochastic %K(14) normalised to [0, 1] ────────────────────────────
    feats["stoch_k"] = _stochastic_k(highs, lows, closes, period=14) / 100.0

    # ── Williams %R(14) normalised to [0, 1] ─────────────────────────────
    # Raw Williams %R is [-100, 0]; we flip and normalise to [0, 1]
    feats["williams_r"] = (_williams_r(highs, lows, closes, period=14) + 100.0) / 100.0

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
    feats["cum_delta"] = float(np.sum(closes > opens) / len(closes))
    feats["vol_trend"] = _ema_slope(volumes, period=10)

    # ── OBV slope ─────────────────────────────────────────────────────────
    obv = _obv(closes, volumes)
    feats["obv_slope"] = _ema_slope(obv, period=10)

    # ── Candlestick structure (last bar) ──────────────────────────────────
    last_high = highs[-1]
    last_low = lows[-1]
    last_open = opens[-1]
    bar_range = last_high - last_low if last_high > last_low else 1e-6
    body = abs(last_close - last_open)
    upper_wick = last_high - max(last_close, last_open)
    lower_wick = min(last_close, last_open) - last_low
    feats["upper_wick_ratio"] = float(upper_wick / bar_range)
    feats["lower_wick_ratio"] = float(lower_wick / bar_range)
    feats["body_ratio"] = float(body / bar_range)
    feats["consecutive_bars"] = float(_consecutive_bars(closes))

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

    # ── Session timing ────────────────────────────────────────────────────
    # Use last bar's timestamp for actual time-of-day; fall back to bar count.
    try:
        last_ts = bars.index[-1]
        minutes_elapsed = float((last_ts.hour - 9) * 60 + (last_ts.minute - 30))
        minutes_elapsed = float(np.clip(minutes_elapsed, 0, 390))
    except Exception:
        minutes_elapsed = float(min(len(bars) * 5, 390))
    feats["time_of_day"] = minutes_elapsed / 390.0
    feats["minutes_since_open"] = minutes_elapsed
    feats["is_open_session"] = float(minutes_elapsed <= 30)    # first 30 min: gap-fill / auction
    feats["is_close_session"] = float(minutes_elapsed >= 330)  # last 60 min: MOC / closing imbalance
    # Phase 50: session segment — 0=open (60-90min), 1=mid (90-210min), 2=close (210-330min)
    if minutes_elapsed < 90:
        feats["session_segment"] = 0.0
    elif minutes_elapsed < 210:
        feats["session_segment"] = 1.0
    else:
        feats["session_segment"] = 2.0

    # ── Daily vol context ─────────────────────────────────────────────────
    if daily_bars is not None and len(daily_bars) >= 22:
        from app.ml.options_features import compute_vol_features
        d_closes = daily_bars["close"].values.astype(float)
        d_highs = daily_bars["high"].values.astype(float)
        d_lows = daily_bars["low"].values.astype(float)
        vf = compute_vol_features(d_closes, d_highs, d_lows)
        feats["daily_vol_percentile"] = vf["vol_percentile_52w"]
        feats["daily_vol_regime"] = vf["vol_regime"]
        feats["daily_parkinson_vol"] = vf["parkinson_vol"]
    else:
        feats["daily_vol_percentile"] = 0.5
        feats["daily_vol_regime"] = 1.0
        feats["daily_parkinson_vol"] = 0.0

    # ── Whale candle ──────────────────────────────────────────────────────────
    try:
        _opens = bars["open"].values.astype(float)
        _body = float(abs(closes[-1] - _opens[-1]))
        _atr_val = _atr(highs, lows, closes, 14)
        feats["whale_candle"] = 1.0 if (_atr_val > 0 and _body > 2.0 * _atr_val) else 0.0
    except Exception:
        feats["whale_candle"] = 0.0

    # ── Phase 47-5: Quality / structure features ─────────────────────────────
    # trend_efficiency: net displacement / total path (1=perfectly linear, 0=choppy)
    try:
        net = abs(closes[-1] - closes[0])
        total_path = float(np.sum(np.abs(np.diff(closes))))
        feats["trend_efficiency"] = float(net / (total_path + 1e-8))
    except Exception:
        feats["trend_efficiency"] = 0.5

    # green_bar_ratio: fraction of up-closes (cum_delta already counts up-closes vs open)
    feats["green_bar_ratio"] = float(np.mean(closes > opens))

    # above_vwap_ratio: fraction of bars where each close > running VWAP
    try:
        bar_vwaps = cum_tp_vol / np.maximum(cum_vol, 1e-8)
        feats["above_vwap_ratio"] = float(np.mean(closes > bar_vwaps))
    except Exception:
        feats["above_vwap_ratio"] = 0.5

    # pullback_from_high: (session_high - close) / session_high
    feats["pullback_from_high"] = float(
        (session_high - last_close) / session_high if session_high > 0 else 0.0
    )

    # range_vs_20d_avg: today's total H-L range vs 20-day avg
    try:
        if daily_bars is not None and len(daily_bars) >= 5:
            d_ranges = (daily_bars["high"].values[-20:] - daily_bars["low"].values[-20:]).astype(float)
            avg_daily_range = float(d_ranges.mean()) if len(d_ranges) > 0 else last_close * 0.01
            today_range = session_high - session_low
            feats["range_vs_20d_avg"] = float(today_range / (avg_daily_range + 1e-8))
        else:
            feats["range_vs_20d_avg"] = 1.0
    except Exception:
        feats["range_vs_20d_avg"] = 1.0

    # rel_strength_vs_spy: alpha vs SPY over session so far
    stock_session_ret = feats.get("session_return", 0.0)
    spy_session_ret = feats.get("spy_session_return", 0.0)
    feats["rel_strength_vs_spy"] = float(stock_session_ret - spy_session_ret)

    # vol_x_momentum: volume_surge × session_return interaction
    feats["vol_x_momentum"] = float(feats.get("volume_surge", 1.0) * stock_session_ret)

    # gap_followthrough: gap_pct × session_return — positive = gap direction held
    feats["gap_followthrough"] = float(feats.get("gap_pct", 0.0) * stock_session_ret)

    # Phase 50: session-segment interactions — model learns how edge varies by time of day
    seg = feats.get("session_segment", 0.0)
    feats["seg_x_high_dist"] = float(seg * feats.get("prev_day_high_dist", 0.0))
    feats["seg_x_atr_norm"] = float(seg * feats.get("atr_norm", 0.0))

    # ── Phase 86b: stock-relative SPY features (survive cs_normalize) ───────
    # These vary by symbol within each day, so cs_normalize preserves signal.
    # Contrast: raw SPY values (same for every symbol) → zeroed by cs_normalize.
    if spy_daily_bars is not None and len(spy_daily_bars) >= 6 and daily_bars is not None and len(daily_bars) >= 6:
        try:
            spy_d = spy_daily_bars["close"].values.astype(float)
            stk_d = daily_bars["close"].values.astype(float)
            spy_5d = float((spy_d[-1] - spy_d[-6]) / (spy_d[-6] + 1e-8))
            stk_5d = float((stk_d[-1] - stk_d[-6]) / (stk_d[-6] + 1e-8))
            feats["stock_vs_spy_5d_return"] = float(stk_5d - spy_5d)
            spy_1d = float((spy_d[-1] - spy_d[-2]) / (spy_d[-2] + 1e-8))
            stk_1d = float((stk_d[-1] - stk_d[-2]) / (stk_d[-2] + 1e-8))
            feats["stock_vs_spy_mom_ratio"] = float(stk_1d - spy_1d)
        except Exception:
            feats["stock_vs_spy_5d_return"] = 0.0
            feats["stock_vs_spy_mom_ratio"] = 0.0
    else:
        feats["stock_vs_spy_5d_return"] = 0.0
        feats["stock_vs_spy_mom_ratio"] = 0.0

    # gap_vs_spy_gap: stock's overnight gap minus SPY's — idiosyncratic gap signal
    stock_gap_pct = feats.get("gap_pct", 0.0)
    spy_gap_pct = 0.0
    if spy_bars is not None and len(spy_bars) >= 1 and spy_daily_bars is not None and len(spy_daily_bars) >= 2:
        try:
            spy_prior_close = float(spy_daily_bars["close"].values[-2])
            spy_open = float(spy_bars["open"].values[0])
            spy_gap_pct = float((spy_open - spy_prior_close) / (spy_prior_close + 1e-8))
        except Exception:
            spy_gap_pct = 0.0
    feats["gap_vs_spy_gap"] = float(stock_gap_pct - spy_gap_pct)

    # ── NIS features (Phase 64b) ─────────────────────────────────────────────
    # Point-in-time daily NIS signal — same lookup as swing model.
    # Intraday entries are short-hold (< 1 session) but news context still
    # matters: high materiality + negative direction = avoid entry.
    # NaN when no data — lets XGBoost use its learned missing-value direction
    # rather than treating default values as real signal.
    _nan = float("nan")
    _nis_missing = {
        "nis_direction_score": _nan,
        "nis_materiality_score": _nan,
        "nis_already_priced_in": _nan,
        "nis_sizing_mult": _nan,
        "nis_downside_risk": _nan,
    }
    if symbol is not None:
        try:
            from app.ml.features import _get_nis_features_pit
            feats.update(_get_nis_features_pit(symbol, as_of_date))
        except Exception as _nis_exc:
            logger.debug("NIS features failed for %s: %s", symbol, _nis_exc)
            feats.update(_nis_missing)
    else:
        feats.update(_nis_missing)

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


def _williams_r(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> float:
    """Williams %R: returns value in [-100, 0]. -100=oversold, 0=overbought."""
    if len(closes) < period:
        return -50.0
    h = float(highs[-period:].max())
    lo = float(lows[-period:].min())
    if h == lo:
        return -50.0
    return float(np.clip((h - closes[-1]) / (h - lo) * -100.0, -100.0, 0.0))


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


def _obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    obv = np.zeros(len(closes))
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def _vwap_cross_count(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> int:
    """Count how many times the close crossed the rolling VWAP."""
    typical = (highs + lows + closes) / 3.0
    cum_vol = np.cumsum(volumes)
    cum_tp_vol = np.cumsum(typical * volumes)
    vwap_series = np.where(cum_vol > 0, cum_tp_vol / cum_vol, closes)
    crosses = 0
    for i in range(1, len(closes)):
        was_above = closes[i - 1] >= vwap_series[i - 1]
        is_above = closes[i] >= vwap_series[i]
        if was_above != is_above:
            crosses += 1
    return crosses


def _consecutive_bars(closes: np.ndarray) -> float:
    """
    Signed count of consecutive same-direction closes ending at the last bar.
    Positive = consecutive up closes, negative = consecutive down closes.
    """
    if len(closes) < 2:
        return 0.0
    direction = 1 if closes[-1] >= closes[-2] else -1
    count = 1
    for i in range(len(closes) - 2, 0, -1):
        bar_dir = 1 if closes[i] >= closes[i - 1] else -1
        if bar_dir == direction:
            count += 1
        else:
            break
    return float(direction * count)


FEATURE_NAMES = [
    # Price / structure
    "orb_position", "orb_breakout", "orb_direction_strength", "vwap_distance", "vwap_cross_count",
    "gap_pct", "gap_fill_pct", "session_hl_position",
    "prev_day_high_dist", "prev_day_low_dist",
    # Trend / moving averages
    "ema9_dist", "ema20_dist", "ema_cross", "macd_hist", "bb_position",
    # Momentum
    "rsi_14", "session_return", "ret_15m", "ret_30m", "stoch_k", "williams_r",
    # Volume / order flow
    "volume_surge", "cum_delta", "vol_trend", "obv_slope",
    # Candlestick
    "upper_wick_ratio", "lower_wick_ratio", "body_ratio", "consecutive_bars",
    # Volatility
    "atr_norm", "range_compression",
    # Market context
    "spy_session_return", "spy_rsi_14", "rel_vol_spy",
    # Session timing
    "time_of_day", "minutes_since_open", "is_open_session", "is_close_session",
    # Daily vol context
    "daily_vol_percentile", "daily_vol_regime", "daily_parkinson_vol",
    # Institutional activity
    "whale_candle",
    # Phase 47-5: Quality / structure features
    "trend_efficiency", "green_bar_ratio", "above_vwap_ratio", "pullback_from_high",
    "range_vs_20d_avg", "rel_strength_vs_spy", "vol_x_momentum", "gap_followthrough",
    # Phase 50: time-of-day segmentation
    "session_segment", "seg_x_high_dist", "seg_x_atr_norm",
    # NIS features intentionally excluded from this list (Phase 1c).
    # They encode time (NaN = pre-May-2025) rather than sentiment quality, creating
    # a time-leak in walk-forward folds. Re-add after 2yr backfill via
    # backfill_stock_nis_history.py reduces NaN rate to <20%.
    # NIS is still active at PM gate layer in features.py — infrastructure preserved.
    # "nis_direction_score", "nis_materiality_score", "nis_already_priced_in",
    # "nis_sizing_mult", "nis_downside_risk",
    # Phase 86b: stock-relative SPY features (survive cs_normalize)
    "stock_vs_spy_5d_return", "stock_vs_spy_mom_ratio", "gap_vs_spy_gap",
]
