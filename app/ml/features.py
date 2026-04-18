"""
Feature engineering for the ML portfolio selection model.

Accepts OHLCV DataFrames from either Alpaca (lowercase columns) or
yfinance (capitalized columns) — both are normalised internally.

Feature groups (74 total):
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
  25. FMP point-in-time: earnings surprises, analyst momentum,
      institutional ownership (8 features)                      — FMP API
  26. Volatility/options: vol percentile, vol regime, vol-of-vol,
      ATR trend, Parkinson vol (training); put/call ratio, IV ATM,
      IV premium (live inference only via yfinance)              — computed + yfinance
  27. News sentiment: 3d/7d avg sentiment, article count, momentum  — Polygon news
  28. FMP enhanced earnings: consecutive beats, revenue surprise     — FMP API
  29. Volume/price dynamics: VPT momentum, range expansion, VWAP distance — computed
  30. Daily technicals: Williams %R(14), CCI(20), price acceleration — computed
"""

import logging
from datetime import date
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
        as_of_date: Optional["date"] = None,
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

        # ── 9. Momentum (5d, 20d, 60d, 252d) ────────────────────────────────────
        lookback_20 = min(20, len(prices) - 1)
        lookback_60 = min(60, len(prices) - 1)
        lookback_252 = min(252, len(prices) - 1)
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
        # 12-month momentum (most well-documented equity factor); skip last month (reversal)
        # Uses prices[-21] as "1 month ago" to exclude short-term reversal component
        _p_12m = prices[-(lookback_252 + 1)]
        _p_1m = prices[-(min(21, len(prices) - 1) + 1)] if len(prices) > 22 else prices[-1]
        features["momentum_252d_ex1m"] = (
            (_p_1m - _p_12m) / _p_12m if _p_12m > 0 else 0.0
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

        if spy_returns is not None and len(spy_returns) >= lookback_60:
            spy_ret_60d = float(np.sum(spy_returns[-lookback_60:]))
            features["rs_vs_spy_60d"] = features["momentum_60d"] - spy_ret_60d
        else:
            features["rs_vs_spy_60d"] = 0.0

        # ── 25. FMP point-in-time fundamentals ───────────────────────────────
        # as_of_date lets training pass the window end date so we use only data
        # known at that time — eliminates the look-ahead bias that kept AUC at 0.51.
        # Live inference passes None → uses today.
        if fetch_fundamentals:
            pit_date = as_of_date if as_of_date is not None else date.today()
            try:
                from app.data.fmp_provider import get_fmp_features_at
                fmp = get_fmp_features_at(symbol, pit_date)
                features.update(fmp)
            except Exception as exc:
                logger.debug("FMP features skipped for %s: %s", symbol, exc)
                features.update({
                    "fmp_surprise_1q": 0.0,
                    "fmp_surprise_2q_avg": 0.0,
                    "fmp_days_since_earnings": 90.0,
                    "fmp_analyst_upgrades_30d": 0.0,
                    "fmp_analyst_downgrades_30d": 0.0,
                    "fmp_analyst_momentum_30d": 0.0,
                    "fmp_inst_ownership_pct": 0.0,
                    "fmp_inst_change_pct": 0.0,
                })
        else:
            features.update({
                "fmp_surprise_1q": 0.0,
                "fmp_surprise_2q_avg": 0.0,
                "fmp_days_since_earnings": 90.0,
                "fmp_analyst_upgrades_30d": 0.0,
                "fmp_analyst_downgrades_30d": 0.0,
                "fmp_analyst_momentum_30d": 0.0,
                "fmp_inst_ownership_pct": 0.0,
                "fmp_inst_change_pct": 0.0,
            })

        # ── 26. Volatility / options proxy features ───────────────────────────
        # Price-derived vol features available for all historical windows.
        # Live options features (put/call ratio, IV) added when fetch_fundamentals
        # is True and as_of_date is None (i.e. live inference, not training).
        try:
            from app.ml.options_features import compute_vol_features
            vol_feats = compute_vol_features(prices, highs, lows)
            features.update(vol_feats)
        except Exception as exc:
            logger.debug("Vol features failed for %s: %s", symbol, exc)
            features.update({
                "vol_percentile_52w": 0.5, "vol_regime": 1.0,
                "vol_of_vol": 0.0, "atr_trend": 1.0, "parkinson_vol": 0.0,
            })

        # Live options (yfinance) — only at inference time, not during training
        if fetch_fundamentals and as_of_date is None:
            try:
                from app.ml.options_features import get_live_options_features
                opt = get_live_options_features(symbol)
                features.update(opt)
            except Exception as exc:
                logger.debug("Live options failed for %s: %s", symbol, exc)
                features.update({
                    "options_put_call_ratio": 1.0,
                    "options_iv_atm": 0.0,
                    "options_iv_premium": 0.0,
                })
        else:
            features.update({
                "options_put_call_ratio": 1.0,
                "options_iv_atm": 0.0,
                "options_iv_premium": 0.0,
            })

        # ── 27. News sentiment (Polygon) ──────────────────────────────────────
        # Point-in-time during training: only articles published before as_of_date.
        # 3-day and 7-day rolling sentiment + article count + momentum (3d-7d shift).
        _news_default = {
            "news_sentiment_3d": 0.0,
            "news_sentiment_7d": 0.0,
            "news_article_count_7d": 0.0,
            "news_sentiment_momentum": 0.0,
        }
        if fetch_fundamentals:
            try:
                from app.ml.news_features import get_news_features
                news = get_news_features(symbol, as_of_date=as_of_date)
                features.update(news)
            except Exception as exc:
                logger.debug("News features failed for %s: %s", symbol, exc)
                features.update(_news_default)
        else:
            features.update(_news_default)

        # ── 28. FMP enhanced earnings features ───────────────────────────────
        _earnings_default = {
            "fmp_consecutive_beats": 0.0,
            "fmp_revenue_surprise_1q": 0.0,
        }
        if fetch_fundamentals and as_of_date is not None:
            try:
                from app.data.fmp_provider import get_earnings_history_fmp
                records = get_earnings_history_fmp(symbol)
                pit_date = as_of_date if isinstance(as_of_date, date) else date.fromisoformat(str(as_of_date))
                from datetime import datetime as _dt
                past = sorted(
                    [r for r in records if r.get("date") and
                     _dt.strptime(r["date"], "%Y-%m-%d").date() <= pit_date],
                    key=lambda r: r["date"], reverse=True
                )
                if past:
                    # Consecutive quarterly beats (surprise_pct > 0)
                    beats = 0
                    for r in past[:4]:
                        if (r.get("surprise_pct") or 0) > 0:
                            beats += 1
                        else:
                            break
                    _earnings_default["fmp_consecutive_beats"] = float(beats)
                    # Surprise magnitude of most recent quarter (signed)
                    rev_s = past[0].get("surprise_pct") or 0.0
                    _earnings_default["fmp_revenue_surprise_1q"] = float(
                        max(-2.0, min(2.0, rev_s))
                    )
            except Exception:
                pass
        features.update(_earnings_default)

        # ── 29. Volume/price dynamics ─────────────────────────────────────────
        try:
            _px = prices[-20:] if len(prices) >= 20 else prices
            _volumes = volumes[-20:] if len(volumes) >= 20 else volumes
            _highs_20 = highs[-20:] if len(highs) >= 20 else highs
            _lows_20 = lows[-20:] if len(lows) >= 20 else lows

            # Volume Price Trend: cumulative sum of volume * daily return, normalised
            if len(_px) >= 2:
                _rets = np.diff(np.log(_px))
                _vpt = float(np.sum(_volumes[1:] * _rets)) / max(float(np.mean(_volumes)), 1e-9)
            else:
                _vpt = 0.0

            # Range expansion: recent 10d H-L range vs prior 10d H-L range
            if len(_highs_20) >= 20:
                recent_range = float(_highs_20[-10:].max() - _lows_20[-10:].min())
                prior_range = float(_highs_20[:10].max() - _lows_20[:10].min())
                _range_exp = float(np.clip(recent_range / max(prior_range, 1e-9), 0, 4.0))
            else:
                _range_exp = 1.0

            # 20-day VWAP distance: avg of daily (close - vwap) / vwap
            if len(_px) >= 5 and len(_highs_20) == len(_lows_20) == len(_px):
                _typical = (_highs_20 + _lows_20 + _px) / 3.0
                _cum_tv = np.cumsum(_typical * _volumes)
                _cum_v = np.cumsum(_volumes)
                _vwap_series = np.where(_cum_v > 0, _cum_tv / _cum_v, _px)
                _vwap_dist = float(np.mean(
                    (_px - _vwap_series) / np.where(_vwap_series > 0, _vwap_series, 1)
                ))
            else:
                _vwap_dist = 0.0

            features["vpt_momentum"] = float(np.clip(_vpt, -5.0, 5.0))
            features["range_expansion"] = _range_exp
            features["vwap_distance_20d"] = float(np.clip(_vwap_dist, -0.1, 0.1))
        except Exception:
            features["vpt_momentum"] = 0.0
            features["range_expansion"] = 1.0
            features["vwap_distance_20d"] = 0.0

        # ── 30a. Sector-neutral momentum & mean-reversion ────────────────────
        # Sector-neutral: stock's 20d/60d alpha vs sector ETF (pure alpha signal)
        features["momentum_20d_sector_neutral"] = features["momentum_20d"] - features["sector_momentum"]
        features["momentum_60d_sector_neutral"] = features["momentum_60d"] - features["sector_momentum"]

        # Mean-reversion z-score: std devs above 60d rolling mean of 20d returns
        if len(prices) >= 63:
            _rolling = np.array([
                (prices[i] - prices[i - 20]) / max(prices[i - 20], 1e-6)
                for i in range(20, min(63, len(prices)))
            ])
            _rz_std = float(_rolling.std()) if len(_rolling) > 1 else 1e-6
            features["mean_reversion_zscore"] = float(
                np.clip((features["momentum_20d"] - float(_rolling.mean())) / max(_rz_std, 1e-6), -4.0, 4.0)
            )
        else:
            features["mean_reversion_zscore"] = 0.0

        # Up-day ratio: % of last 20 days closing higher than prior day
        if len(prices) >= 21:
            features["up_day_ratio_20d"] = float((np.diff(prices[-21:]) > 0).sum()) / 20.0
        else:
            features["up_day_ratio_20d"] = 0.5

        # ── 30. Daily technical indicators ───────────────────────────────────
        try:
            # Williams %R(14): -100=oversold, 0=overbought → normalise to [-1, 0]
            if len(highs) >= 14:
                _h14 = float(highs[-14:].max())
                _l14 = float(lows[-14:].min())
                _wr = float((_h14 - prices[-1]) / max(_h14 - _l14, 1e-9) * -1.0)
                features["williams_r_14"] = float(np.clip(_wr, -1.0, 0.0))
            else:
                features["williams_r_14"] = -0.5

            # CCI(20): (typical_price - SMA20) / (0.015 * mean_deviation)
            if len(prices) >= 20:
                _tp20 = (highs[-20:] + lows[-20:] + prices[-20:]) / 3.0
                _sma = float(_tp20.mean())
                _md = float(np.mean(np.abs(_tp20 - _sma)))
                _cci = float((_tp20[-1] - _sma) / max(0.015 * _md, 1e-9))
                features["cci_20"] = float(np.clip(_cci / 200.0, -1.5, 1.5))
            else:
                features["cci_20"] = 0.0

            # Price acceleration: (5d momentum) - (10d momentum)
            if len(prices) >= 11:
                _mom5 = float((prices[-1] - prices[-6]) / max(prices[-6], 1e-9))
                _mom10 = float((prices[-1] - prices[-11]) / max(prices[-11], 1e-9))
                features["price_acceleration"] = float(np.clip(_mom5 - _mom10, -0.1, 0.1))
            else:
                features["price_acceleration"] = 0.0
        except Exception:
            features["williams_r_14"] = -0.5
            features["cci_20"] = 0.0
            features["price_acceleration"] = 0.0

        return features
