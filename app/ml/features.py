"""
Feature engineering for the ML portfolio selection model.

Accepts OHLCV DataFrames from either Alpaca (lowercase columns) or
yfinance (capitalized columns) — both are normalised internally.

Feature groups (48 total):
  1.  RSI (14, 7)                                               — price-only
  2.  MACD (line, signal, histogram)                            — price-only
  3.  EMAs (20, 50, 200) + price position flags                 — price-only
  4.  Price change                                              — price-only
  5.  52-week high / low proximity                              — price-only
  6.  Volume ratio + volume trend                               — price-only
  7.  Trend flags (5d up/down)                                  — price-only
  8.  Volatility (annualised std)                               — price-only
  9.  Momentum (5d, 20d, 60d)                                   — price-only
  10. ATR(14) normalised                                        — price-only
  11. Bollinger Band %B (20d, 2σ)                               — price-only
  12. Stochastic %K (14d)                                       — price-only
  13. ADX(14) — trend strength                                  — price-only
  14. Relative strength vs SPY (20d)                            — price-only
  15. Consecutive up/down days                                  — price-only
  16. Sentiment                                                 — caller-supplied
  17. Fundamentals (P/E, P/B, margins, revenue growth, D/E)     — yfinance
  18. Earnings (proximity days, surprise %)                     — yfinance + AV
  19. Sector momentum (sector ETF 20d return)                   — yfinance
  20. Insider activity (net buy score)                          — SEC EDGAR
  21. Macro regime (composite score)                            — FRED / existing
  22. Earnings history: 1q surprise, 2q avg, days since report  — yfinance (free)
  23. Short interest % of float                                  — yfinance (free)
  24. Short-term RS vs SPY: 5d, 10d differentials               — computed
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.indicators.technical import (
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)

logger = logging.getLogger(__name__)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case all column names so Alpaca and yfinance DataFrames work identically."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns and "adj close" in df.columns:
        df = df.rename(columns={"adj close": "close"})
    return df


# ── Pure-numpy indicator helpers (no external deps) ───────────────────────────

def _atr_norm(highs, lows, closes, period=14):
    """ATR(period) / last_close — normalised volatility."""
    if len(closes) < 2:
        return 0.0
    n = min(period, len(closes) - 1)
    trs = []
    for i in range(-n, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    atr = float(np.mean(trs)) if trs else 0.0
    return atr / closes[-1] if closes[-1] > 0 else 0.0


def _bollinger_pct_b(closes, period=20, num_std=2.0):
    """Bollinger Band %B: 0=lower band, 0.5=middle, 1=upper band."""
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    mid = float(window.mean())
    std = float(window.std(ddof=1)) if len(window) > 1 else 1e-9
    band_range = 2 * num_std * std
    if band_range < 1e-9:
        return 0.5
    return float(np.clip((closes[-1] - (mid - num_std * std)) / band_range, 0.0, 1.0))


def _stochastic_k(highs, lows, closes, period=14):
    """Stochastic %K in [0, 100]."""
    if len(closes) < period:
        return 50.0
    h = float(highs[-period:].max())
    lo = float(lows[-period:].min())
    if h == lo:
        return 50.0
    return float(np.clip((closes[-1] - lo) / (h - lo) * 100.0, 0.0, 100.0))


def _adx(highs, lows, closes, period=14):
    """
    Average Directional Index in [0, 100].
    Values above 25 indicate a trending market.
    """
    if len(closes) < period * 2:
        return 0.0
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)

    n = len(closes)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr_arr = np.zeros(n)

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr_arr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    def _smooth(arr, p):
        s = float(arr[1:p + 1].sum())
        out = [s]
        for v in arr[p + 1:]:
            s = s - s / p + v
            out.append(s)
        return np.array(out)

    sm_tr = _smooth(tr_arr, period)
    sm_plus = _smooth(plus_dm, period)
    sm_minus = _smooth(minus_dm, period)

    plus_di = np.where(sm_tr > 0, 100 * sm_plus / sm_tr, 0.0)
    minus_di = np.where(sm_tr > 0, 100 * sm_minus / sm_tr, 0.0)
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    if len(dx) < period:
        return float(dx.mean()) if len(dx) > 0 else 0.0
    adx = float(dx[-period:].mean())
    return float(np.clip(adx, 0.0, 100.0))


def _consecutive_days(closes):
    """
    Signed count of consecutive same-direction daily closes.
    Positive = consecutive up days, negative = consecutive down days.
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


def _volume_trend(volumes, period=10):
    """Volume EMA slope: (ema_last - ema_prev) / ema_prev."""
    if len(volumes) < period + 1:
        return 0.0
    k = 2.0 / (period + 1)
    ema = float(np.mean(volumes[:period]))
    prev_ema = ema
    for i, v in enumerate(volumes[period:]):
        prev_ema = ema
        ema = v * k + ema * (1 - k)
    return float((ema - prev_ema) / prev_ema) if prev_ema != 0 else 0.0


class FeatureEngineer:
    """Extract a fixed feature vector for any stock given its OHLCV history."""

    MIN_BARS = 52

    def engineer_features(
        self,
        symbol: str,
        bars: pd.DataFrame,
        sentiment: Optional[float] = None,
        sector: Optional[str] = None,
        regime_score: Optional[float] = None,
        fetch_fundamentals: bool = True,
        spy_returns: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Compute features for a single stock.

        Args:
            symbol:             Stock ticker.
            bars:               OHLCV DataFrame (Alpaca or yfinance format).
            sentiment:          Optional news sentiment score in [-1, 1].
            sector:             GICS sector string (used for ETF momentum lookup).
            regime_score:       Composite macro regime score from RegimeDetector.
            fetch_fundamentals: Set False in backtests to skip live API calls.
            spy_returns:        Optional array of SPY daily returns (same length as bars)
                                for relative-strength computation.

        Returns:
            Dict of feature_name → float, or None if insufficient data.
        """
        bars = _normalise_columns(bars)

        required = {"close", "high", "low", "volume"}
        if not required.issubset(set(bars.columns)):
            return None

        if len(bars) < self.MIN_BARS:
            return None

        prices = bars["close"].to_numpy(dtype=float)
        highs = bars["high"].to_numpy(dtype=float)
        lows = bars["low"].to_numpy(dtype=float)
        volumes = bars["volume"].to_numpy(dtype=float)
        current_price = prices[-1]

        features: Dict[str, float] = {}

        # ── 1. RSI ────────────────────────────────────────────────────────────
        rsi_14 = calculate_rsi(prices, period=14)
        rsi_7 = calculate_rsi(prices, period=7)
        features["rsi_14"] = rsi_14 if rsi_14 is not None else 50.0
        features["rsi_7"] = rsi_7 if rsi_7 is not None else 50.0

        # ── 2. MACD ───────────────────────────────────────────────────────────
        macd_result = calculate_macd(prices)
        if macd_result is not None:
            macd_line, signal_line, histogram = macd_result
            features["macd"] = macd_line
            features["macd_signal"] = signal_line
            features["macd_histogram"] = histogram
        else:
            features["macd"] = 0.0
            features["macd_signal"] = 0.0
            features["macd_histogram"] = 0.0

        # ── 3. EMAs & price position ──────────────────────────────────────────
        ema_20 = calculate_ema(prices, period=20) or current_price
        ema_50 = calculate_ema(prices, period=50) or current_price
        ema_200 = calculate_ema(prices, period=200) or current_price
        features["ema_20"] = ema_20
        features["ema_50"] = ema_50
        features["price_above_ema20"] = 1.0 if current_price > ema_20 else 0.0
        features["price_above_ema50"] = 1.0 if current_price > ema_50 else 0.0
        features["price_above_ema200"] = 1.0 if current_price > ema_200 else 0.0
        # Distance from 200-day MA — institutional trend filter
        features["dist_from_ema200"] = (
            float((current_price - ema_200) / ema_200) if ema_200 > 0 else 0.0
        )

        # ── 4. Price change ───────────────────────────────────────────────────
        prev_close = prices[-2] if len(prices) > 1 else current_price
        features["price_change_pct"] = (
            (current_price - prev_close) / prev_close if prev_close else 0.0
        )

        # ── 5. 52-week high / low ratio ───────────────────────────────────────
        lookback = min(252, len(prices))
        high_52w = max(prices[-lookback:])
        low_52w = min(prices[-lookback:])
        features["price_to_52w_high"] = current_price / high_52w if high_52w else 1.0
        features["price_to_52w_low"] = current_price / low_52w if low_52w else 1.0
        # Near 52w high = breakout candidate (< 5% away)
        features["near_52w_high"] = 1.0 if features["price_to_52w_high"] >= 0.95 else 0.0

        # ── 6. Volume ratio + trend ───────────────────────────────────────────
        avg_vol = (
            float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        )
        features["volume_ratio"] = (volumes[-1] / avg_vol) if avg_vol > 0 else 1.0
        features["volume_trend"] = _volume_trend(volumes, period=10)

        # ── 7. Trend flags ────────────────────────────────────────────────────
        lookback_5 = min(5, len(prices) - 1)
        features["uptrend"] = 1.0 if prices[-1] > prices[-(lookback_5 + 1)] else 0.0
        features["downtrend"] = 1.0 if prices[-1] < prices[-(lookback_5 + 1)] else 0.0

        # ── 8. Volatility (annualised) ────────────────────────────────────────
        returns = np.diff(prices) / prices[:-1]
        features["volatility"] = (
            float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0
        )

        # ── 9. Momentum (5d, 20d, 60d) ───────────────────────────────────────
        lookback_20 = min(20, len(prices) - 1)
        lookback_60 = min(60, len(prices) - 1)
        features["momentum_5d"] = (
            (prices[-1] - prices[-(lookback_5 + 1)]) / prices[-(lookback_5 + 1)]
            if prices[-(lookback_5 + 1)] else 0.0
        )
        features["momentum_20d"] = (
            (prices[-1] - prices[-(lookback_20 + 1)]) / prices[-(lookback_20 + 1)]
            if prices[-(lookback_20 + 1)] else 0.0
        )
        features["momentum_60d"] = (
            (prices[-1] - prices[-(lookback_60 + 1)]) / prices[-(lookback_60 + 1)]
            if prices[-(lookback_60 + 1)] else 0.0
        )

        # ── 10. ATR(14) normalised ────────────────────────────────────────────
        features["atr_norm"] = _atr_norm(highs, lows, prices, period=14)

        # ── 11. Bollinger Band %B (20d) ───────────────────────────────────────
        features["bb_position"] = _bollinger_pct_b(prices, period=20, num_std=2.0)

        # ── 12. Stochastic %K(14) normalised to [0, 1] ───────────────────────
        features["stoch_k"] = _stochastic_k(highs, lows, prices, period=14) / 100.0

        # ── 13. ADX(14) — trend strength, normalised to [0, 1] ───────────────
        features["adx_14"] = _adx(highs, lows, prices, period=14) / 100.0

        # ── 14. Relative strength vs SPY (20d return differential) ───────────
        if spy_returns is not None and len(spy_returns) >= lookback_20:
            spy_ret_20d = float(np.sum(spy_returns[-lookback_20:]))
            features["rs_vs_spy"] = features["momentum_20d"] - spy_ret_20d
        else:
            features["rs_vs_spy"] = 0.0

        # ── 15. Consecutive up/down days ──────────────────────────────────────
        features["consecutive_days"] = _consecutive_days(prices)

        # ── 16. Sentiment ─────────────────────────────────────────────────────
        features["sentiment"] = float(sentiment) if sentiment is not None else 0.0

        # ── 17. Fundamentals (yfinance) ───────────────────────────────────────
        if fetch_fundamentals:
            try:
                from app.ml.fundamental_fetcher import get_fundamentals
                fund = get_fundamentals(symbol)
                features["pe_ratio"] = fund["pe_ratio"]
                features["pb_ratio"] = fund["pb_ratio"]
                features["profit_margin"] = fund["profit_margin"]
                features["revenue_growth"] = fund["revenue_growth"]
                features["debt_to_equity"] = fund["debt_to_equity"]
                features["earnings_proximity_days"] = fund["earnings_proximity_days"]
            except Exception as exc:
                logger.debug("Fundamental features skipped for %s: %s", symbol, exc)
                features.update({
                    "pe_ratio": 0.0, "pb_ratio": 0.0, "profit_margin": 0.0,
                    "revenue_growth": 0.0, "debt_to_equity": 0.0,
                    "earnings_proximity_days": 90.0,
                })
        else:
            features.update({
                "pe_ratio": 0.0, "pb_ratio": 0.0, "profit_margin": 0.0,
                "revenue_growth": 0.0, "debt_to_equity": 0.0,
                "earnings_proximity_days": 90.0,
            })

        # ── 18. Sector ETF momentum ───────────────────────────────────────────
        if fetch_fundamentals and sector:
            try:
                from app.ml.fundamental_fetcher import get_sector_momentum
                features["sector_momentum"] = get_sector_momentum(sector)
            except Exception:
                features["sector_momentum"] = 0.0
        else:
            features["sector_momentum"] = 0.0

        # ── 19. Insider activity score (SEC EDGAR Form 4) ─────────────────────
        if fetch_fundamentals:
            try:
                from app.ml.fundamental_fetcher import get_insider_score
                features["insider_score"] = get_insider_score(symbol)
            except Exception:
                features["insider_score"] = 0.0
        else:
            features["insider_score"] = 0.0

        # ── 20. Earnings surprise (Alpha Vantage) ─────────────────────────────
        if fetch_fundamentals:
            try:
                from app.ml.fundamental_fetcher import get_earnings_surprise
                features["earnings_surprise"] = get_earnings_surprise(symbol)
            except Exception:
                features["earnings_surprise"] = 0.0
        else:
            features["earnings_surprise"] = 0.0

        # ── 21. Macro regime score ────────────────────────────────────────────
        if regime_score is not None:
            features["regime_score"] = float(regime_score)
        else:
            try:
                from app.strategy.regime_detector import RegimeDetector
                detector = RegimeDetector()
                det = detector.get_regime_detail()
                features["regime_score"] = float(det.get("composite_score", 0.5))
            except Exception:
                features["regime_score"] = 0.5

        # ── 22. Earnings history (yfinance — free) ────────────────────────────
        # PEAD (Post-Earnings Announcement Drift): stocks that beat estimates
        # continue to outperform for weeks.  Days-since captures drift window.
        if fetch_fundamentals:
            try:
                from app.ml.fundamental_fetcher import get_earnings_history
                eh = get_earnings_history(symbol)
                features["earnings_surprise_1q"] = eh["earnings_surprise_1q"]
                features["earnings_surprise_2q_avg"] = eh["earnings_surprise_2q_avg"]
                features["days_since_earnings"] = eh["days_since_earnings"]
            except Exception:
                features["earnings_surprise_1q"] = 0.0
                features["earnings_surprise_2q_avg"] = 0.0
                features["days_since_earnings"] = 90.0
        else:
            features["earnings_surprise_1q"] = 0.0
            features["earnings_surprise_2q_avg"] = 0.0
            features["days_since_earnings"] = 90.0

        # ── 23. Short interest % of float (yfinance — free) ───────────────────
        # High short interest + upward price momentum = short-squeeze fuel.
        if fetch_fundamentals:
            try:
                from app.ml.fundamental_fetcher import get_short_interest
                features["short_interest_pct"] = get_short_interest(symbol)
            except Exception:
                features["short_interest_pct"] = 0.0
        else:
            features["short_interest_pct"] = 0.0

        # ── 24. Short-term relative strength vs SPY (5d, 10d) ─────────────────
        # Stock outperforming the market over recent days = institutional buying.
        lookback_10 = min(10, len(prices) - 1)
        momentum_10d = (
            (prices[-1] - prices[-(lookback_10 + 1)]) / prices[-(lookback_10 + 1)]
            if prices[-(lookback_10 + 1)] else 0.0
        )
        if spy_returns is not None and len(spy_returns) >= lookback_5:
            spy_ret_5d = float(np.sum(spy_returns[-lookback_5:]))
            features["rs_vs_spy_5d"] = features["momentum_5d"] - spy_ret_5d
        else:
            features["rs_vs_spy_5d"] = 0.0

        if spy_returns is not None and len(spy_returns) >= lookback_10:
            spy_ret_10d = float(np.sum(spy_returns[-lookback_10:]))
            features["rs_vs_spy_10d"] = momentum_10d - spy_ret_10d
        else:
            features["rs_vs_spy_10d"] = 0.0

        return features
