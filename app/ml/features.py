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
  33. WorldQuant 101 alphas: 14 formulaic alphas (open-volume divergence,
      intraday reversal, stochastic-vol rank, momentum-reversal composites) — computed
  34. Short-term reversal: 3d, 5d, 5d vol-weighted reversal signals     — computed
  35. VRP, beta, opex, earnings drift: realized-vol spread, 252d beta,
      days-to-opex, earnings PEAD decay signal                          — computed
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
        opens = bars["open"].to_numpy(dtype=float) if "open" in bars.columns else prices
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

        # ── 29. Polygon financials (FCF margin, operating leverage, R&D intensity)
        _poly_fin_default = {
            "fcf_margin": 0.0,
            "operating_leverage": 0.0,
            "rd_intensity": 0.0,
        }
        if fetch_fundamentals:
            try:
                from app.data.polygon_financials import get_polygon_financial_features
                pit_date = as_of_date if as_of_date is not None else date.today()
                poly_fin = get_polygon_financial_features(symbol, pit_date)
                features.update(poly_fin)
            except Exception as exc:
                logger.debug("Polygon financials features skipped for %s: %s", symbol, exc)
                features.update(_poly_fin_default)
        else:
            features.update(_poly_fin_default)

        # ── 30. Volume/price dynamics ─────────────────────────────────────────
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

        # ── 30b. Quality / trend consistency features (v19) ───────────────────
        # Trend consistency: fraction of last 63 days where price > 50-day EMA
        # High consistency = institutional conviction, sustained accumulation
        if len(prices) >= 63:
            _k50 = 2.0 / 51.0
            _ema50_window = float(np.mean(prices[:50]))
            _above_count = 0
            for _i in range(50, min(len(prices), 63 + 50)):
                _ema50_window = prices[_i] * _k50 + _ema50_window * (1 - _k50)
                if prices[_i] > _ema50_window:
                    _above_count += 1
            _window_len = min(len(prices), 63 + 50) - 50
            features["trend_consistency_63d"] = float(_above_count) / max(_window_len, 1)
        else:
            features["trend_consistency_63d"] = 0.5

        # Volume-price trend confirmation: do up days have higher vol than down days?
        # Ratio > 1 = buying pressure; < 1 = distribution
        if len(prices) >= 21 and len(volumes) >= 21:
            _rets20 = np.diff(prices[-21:])
            _vols20 = volumes[-20:]
            _up_vol = float(np.mean(_vols20[_rets20 > 0])) if np.any(_rets20 > 0) else 0.0
            _down_vol = float(np.mean(_vols20[_rets20 <= 0])) if np.any(_rets20 <= 0) else 1e-6
            features["vol_price_confirmation"] = float(np.clip(_up_vol / max(_down_vol, 1e-6), 0.0, 4.0))
        else:
            features["vol_price_confirmation"] = 1.0

        # Price efficiency: fraction of gross move captured vs total path length
        # High efficiency = directional conviction; low = choppiness
        if len(prices) >= 20:
            _net = abs(prices[-1] - prices[-20])
            _path = float(np.sum(np.abs(np.diff(prices[-20:]))))
            features["price_efficiency_20d"] = float(np.clip(_net / max(_path, 1e-6), 0.0, 1.0))
        else:
            features["price_efficiency_20d"] = 0.5

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

        # ── 31. pandas-ta extended indicators ────────────────────────────────
        try:
            import pandas_ta as pta
            _df_ta = pd.DataFrame({
                "open": opens if opens is not None else prices,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            })

            # Stochastic RSI: smoother overbought/oversold than plain RSI
            _srsi = pta.stochrsi(_df_ta["close"], length=14, rsi_length=14, k=3, d=3)
            if _srsi is not None and not _srsi.empty:
                _k_col = [c for c in _srsi.columns if "STOCHRSIk" in c]
                _d_col = [c for c in _srsi.columns if "STOCHRSId" in c]
                _k = float(_srsi[_k_col[0]].iloc[-1]) if _k_col and not pd.isna(_srsi[_k_col[0]].iloc[-1]) else 50.0
                _d = float(_srsi[_d_col[0]].iloc[-1]) if _d_col and not pd.isna(_srsi[_d_col[0]].iloc[-1]) else 50.0
                features["stochrsi_k"] = float(np.clip(_k / 100.0, 0.0, 1.0))
                features["stochrsi_d"] = float(np.clip(_d / 100.0, 0.0, 1.0))
                features["stochrsi_signal"] = float(np.clip((_k - _d) / 100.0, -1.0, 1.0))
            else:
                features["stochrsi_k"] = 0.5
                features["stochrsi_d"] = 0.5
                features["stochrsi_signal"] = 0.0

            # Keltner Channels: ATR-based bands — position within channel
            _kc = pta.kc(_df_ta["high"], _df_ta["low"], _df_ta["close"], length=20, scalar=2.0)
            if _kc is not None and not _kc.empty and len(prices) > 0:
                _kcu_col = [c for c in _kc.columns if "KCUe" in c]
                _kcl_col = [c for c in _kc.columns if "KCLe" in c]
                if _kcu_col and _kcl_col:
                    _kcu = float(_kc[_kcu_col[0]].iloc[-1])
                    _kcl = float(_kc[_kcl_col[0]].iloc[-1])
                    _band = max(_kcu - _kcl, 1e-9)
                    features["keltner_position"] = float(np.clip((prices[-1] - _kcl) / _band, -0.2, 1.2))
                else:
                    features["keltner_position"] = 0.5
            else:
                features["keltner_position"] = 0.5

            # Chaikin Money Flow (20): volume-weighted buying/selling pressure [-1, 1]
            _cmf = pta.cmf(_df_ta["high"], _df_ta["low"], _df_ta["close"], _df_ta["volume"], length=20)
            if _cmf is not None and len(_cmf) > 0:
                _cmf_val = float(_cmf.iloc[-1])
                features["cmf_20"] = float(np.clip(_cmf_val if not np.isnan(_cmf_val) else 0.0, -1.0, 1.0))
            else:
                features["cmf_20"] = 0.0

            # DEMA (Double EMA, 20d): trend-following, less lag than EMA
            _dema = pta.dema(_df_ta["close"], length=20)
            if _dema is not None and len(_dema) > 0 and prices[-1] > 0:
                _dema_val = float(_dema.iloc[-1])
                features["dema_20_dist"] = float(np.clip((prices[-1] - _dema_val) / max(_dema_val, 1e-9), -0.15, 0.15))
            else:
                features["dema_20_dist"] = 0.0

        except Exception as _ta_exc:
            logger.debug("pandas-ta features failed: %s", _ta_exc)
            features.setdefault("stochrsi_k", 0.5)
            features.setdefault("stochrsi_d", 0.5)
            features.setdefault("stochrsi_signal", 0.0)
            features.setdefault("keltner_position", 0.5)
            features.setdefault("cmf_20", 0.0)
            features.setdefault("dema_20_dist", 0.0)

        # ── 32. Entry-timing & vol-expansion features ─────────────────────────
        try:
            # Vol expansion: current 10-day realized vol vs 90-day realized vol.
            # > 1 = vol expanding (breakout context); < 1 = vol contracting (consolidation).
            if len(prices) >= 91:
                _rets_all = np.diff(np.log(prices + 1e-9))
                _rv10 = float(np.std(_rets_all[-10:]) * np.sqrt(252))
                _rv90 = float(np.std(_rets_all[-90:]) * np.sqrt(252))
                features["vol_expansion"] = float(np.clip(_rv10 / max(_rv90, 1e-9), 0.2, 5.0))
            else:
                features["vol_expansion"] = 1.0

            # ADX slope: is momentum strengthening or fading?
            # Positive = trend accelerating; negative = trend weakening.
            if len(prices) >= 20 and len(highs) >= 20 and len(lows) >= 20:
                _adx_now = _adx(highs, lows, prices, period=14)
                _adx_5ago = _adx(highs[:-5], lows[:-5], prices[:-5], period=14) if len(prices) > 19 else _adx_now
                features["adx_slope"] = float(np.clip(_adx_now - _adx_5ago, -30.0, 30.0))
            else:
                features["adx_slope"] = 0.0

            # Volume surge: mean volume of last 3 bars vs 20-day avg.
            # > 1.5 = institutional participation; < 0.7 = low-conviction move.
            if len(volumes) >= 20:
                _avg_vol20 = float(np.mean(volumes[-20:]))
                _surge_vol = float(np.mean(volumes[-3:]))
                features["volume_surge_3d"] = float(np.clip(_surge_vol / max(_avg_vol20, 1e-9), 0.1, 10.0))
            else:
                features["volume_surge_3d"] = 1.0

            # Consolidation position: where is price within its recent 20-day range?
            # 0 = at 20d low (potential base); 1 = at 20d high (breakout zone).
            if len(prices) >= 20:
                _h20 = float(highs[-20:].max())
                _l20 = float(lows[-20:].min())
                _range20 = max(_h20 - _l20, 1e-9)
                features["consolidation_position"] = float(np.clip((prices[-1] - _l20) / _range20, 0.0, 1.0))
            else:
                features["consolidation_position"] = 0.5

        except Exception:
            features.setdefault("vol_expansion", 1.0)
            features.setdefault("adx_slope", 0.0)
            features.setdefault("volume_surge_3d", 1.0)
            features.setdefault("consolidation_position", 0.5)

        # ── 33. WorldQuant 101 formulaic alphas (subset, OHLCV-only) ─────────────
        # Kakushadze (2016) arxiv:1601.00991 — production-grade alpha factors.
        # Using a curated 15 that target 5-20 day momentum/reversal/volume patterns.
        try:
            _c = prices       # close prices array
            _h = highs
            _l = lows
            _o = opens if len(opens) == len(_c) else _c
            _v = volumes

            def _ts_rank(arr, n):
                """Rank of last value among trailing n values, scaled to [0,1]."""
                if len(arr) < n:
                    return 0.5
                window = arr[-n:]
                return float(np.sum(window < window[-1])) / (n - 1 + 1e-9)

            def _ts_corr(a, b, n):
                """Pearson correlation of trailing n values."""
                if len(a) < n or len(b) < n:
                    return 0.0
                x, y = a[-n:], b[-n:]
                if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                    return 0.0
                return float(np.corrcoef(x, y)[0, 1])

            def _ts_stddev(arr, n):
                return float(np.std(arr[-n:])) if len(arr) >= n else 0.0

            def _rank_arr(arr):
                """Cross-sectional rank stub — for single-stock use, returns ts_rank."""
                return _ts_rank(arr, min(len(arr), 20))

            _rets = np.diff(np.log(np.where(_c > 0, _c, 1e-9)))

            # Alpha003: -corr(rank(open), rank(volume), 10) — open-volume divergence
            wq_a3 = -_ts_corr(_o, _v, 10)

            # Alpha006: -corr(open, volume, 10) — raw open-volume momentum
            wq_a6 = -_ts_corr(_o, _v, 10) if len(_o) >= 10 else 0.0

            # Alpha012: sign(Δvol) * (-Δclose) — volume-direction reversal
            if len(_c) >= 2 and len(_v) >= 2:
                wq_a12 = float(np.sign(_v[-1] - _v[-2]) * -(_c[-1] - _c[-2]))
                wq_a12 = float(np.clip(wq_a12 / max(abs(_c[-1]), 1e-9), -0.1, 0.1))
            else:
                wq_a12 = 0.0

            # Alpha033: rank(-(1 - open/close)) — intraday reversal strength
            if len(_c) >= 5 and len(_o) >= 5 and _c[-1] > 0:
                _oc_ratios = np.array([-(1 - o / c) for o, c in zip(_o[-20:], _c[-20:]) if c > 0])
                wq_a33 = _ts_rank(_oc_ratios, min(len(_oc_ratios), 10))
            else:
                wq_a33 = 0.5

            # Alpha034: rank(1 - rank(std(ret,2)/std(ret,5))) + rank(1 - rank(Δclose))
            if len(_rets) >= 5:
                _std2 = float(np.std(_rets[-2:])) if len(_rets) >= 2 else 0.0
                _std5 = float(np.std(_rets[-5:])) if len(_rets) >= 5 else 1e-9
                _ratio = _std2 / max(_std5, 1e-9)
                _dc = _rets[-1] if len(_rets) >= 1 else 0.0
                wq_a34 = float(np.clip(1.0 - _ratio - _dc, -2.0, 2.0))
            else:
                wq_a34 = 0.0

            # Alpha035: ts_rank(vol,32) * (1 - ts_rank(high-low range, 16)) * (1 - ts_rank(ret,32))
            # High volume + low range + low return → pending breakout signal
            if len(_v) >= 32 and len(_rets) >= 32:
                _hl_range = _h - _l
                wq_a35 = (
                    _ts_rank(_v, 32)
                    * (1.0 - _ts_rank(_hl_range, 16))
                    * (1.0 - _ts_rank(_rets, 32))
                )
            else:
                wq_a35 = 0.0

            # Alpha040: -rank(std(high,10)) * corr(high,volume,10)
            if len(_h) >= 10:
                _std_h10 = _ts_stddev(_h, 10) / max(_c[-1], 1e-9)
                _corr_hv10 = _ts_corr(_h, _v, 10)
                wq_a40 = float(np.clip(-_std_h10 * _corr_hv10, -0.5, 0.5))
            else:
                wq_a40 = 0.0

            # Alpha043: (volume/adv20) * (-Δclose_7d) — volume-amplified 7d reversal
            if len(_c) >= 8 and len(_v) >= 20:
                _adv20 = float(np.mean(_v[-20:]))
                _vol_ratio = float(_v[-1]) / max(_adv20, 1)
                _delta7 = (_c[-1] - _c[-8]) / max(_c[-8], 1e-9)
                wq_a43 = float(np.clip(_vol_ratio * -_delta7, -1.0, 1.0))
            else:
                wq_a43 = 0.0

            # Alpha044: -corr(high, rank(volume), 5) — high-volume divergence signal
            wq_a44 = float(np.clip(-_ts_corr(_h, _v, 5), -1.0, 1.0)) if len(_h) >= 5 else 0.0

            # Alpha053: -delta(((close-low)-(high-close))/(close-low+ε), 9)
            # Negative momentum of money-flow ratio (stochastic-like)
            if len(_c) >= 11 and len(_h) >= 11 and len(_l) >= 11:
                def _mfr(i):
                    hl = _h[i] - _l[i]
                    return (((_c[i] - _l[i]) - (_h[i] - _c[i])) / max(hl, 1e-9))
                wq_a53 = float(np.clip(-(_mfr(-1) - _mfr(-10)), -1.0, 1.0))
            else:
                wq_a53 = 0.0

            # Alpha055: -corr(rank((close-tsmin(low,12))/(tsmax(high,12)-tsmin(low,12))), rank(vol), 6)
            # Rank-volume correlation of normalized price position — stochastic-volume divergence
            if len(_c) >= 12 and len(_h) >= 12 and len(_l) >= 12 and len(_v) >= 6:
                _ts_min_low = float(np.min(_l[-12:]))
                _ts_max_hi = float(np.max(_h[-12:]))
                _range12 = max(_ts_max_hi - _ts_min_low, 1e-9)
                _norm_pos = np.array([(_c[i] - _ts_min_low) / _range12 for i in range(-12, 0)])
                wq_a55 = float(np.clip(-_ts_corr(_norm_pos, _v[-12:], 6), -1.0, 1.0))
            else:
                wq_a55 = 0.0

            # Alpha004: -ts_rank(rank(low), 9) — low-price momentum reversal
            wq_a4 = float(1.0 - _ts_rank(_l, 9)) if len(_l) >= 9 else 0.5

            # Alpha046: acceleration of 10d momentum (decay signal)
            # Compares (delay20-delay10)/10 vs (delay10-close)/10; negative accel = reversal
            if len(_c) >= 21:
                _d20 = _c[-21]
                _d10 = _c[-11]
                _now = _c[-1]
                _accel = ((_d20 - _d10) / 10.0) - ((_d10 - _now) / 10.0)
                wq_a46 = float(np.clip(-_accel / max(abs(_now), 1e-9), -0.1, 0.1))
            else:
                wq_a46 = 0.0

            # Alpha054: intraday body vs range — gap/reversal signal
            if len(_c) >= 5 and len(_o) >= 5 and len(_h) >= 5 and len(_l) >= 5:
                _hl = _h[-1] - _l[-1]
                wq_a54 = float(np.clip((_l[-1] - _c[-1]) / max(_hl, 1e-9), -1.0, 0.0))
            else:
                wq_a54 = 0.0

            features["wq_alpha3"] = float(np.clip(wq_a3, -1.0, 1.0))
            features["wq_alpha4"] = float(np.clip(wq_a4, 0.0, 1.0))
            features["wq_alpha6"] = float(np.clip(wq_a6, -1.0, 1.0))
            features["wq_alpha12"] = wq_a12
            features["wq_alpha33"] = wq_a33
            features["wq_alpha34"] = wq_a34
            features["wq_alpha35"] = wq_a35
            features["wq_alpha40"] = wq_a40
            features["wq_alpha43"] = wq_a43
            features["wq_alpha44"] = wq_a44
            features["wq_alpha46"] = wq_a46
            features["wq_alpha53"] = wq_a53
            features["wq_alpha54"] = wq_a54
            features["wq_alpha55"] = wq_a55
        except Exception as exc:
            logger.debug("WorldQuant alphas failed: %s", exc)
            for _k in [
                "wq_alpha3", "wq_alpha4", "wq_alpha6", "wq_alpha12", "wq_alpha33",
                "wq_alpha34", "wq_alpha35", "wq_alpha40", "wq_alpha43", "wq_alpha44",
                "wq_alpha46", "wq_alpha53", "wq_alpha54", "wq_alpha55",
            ]:
                features.setdefault(_k, 0.0)

        # ── 34. Short-term reversal signals ──────────────────────────────────────
        # 5-day price reversal: prior losers tend to revert within 10 days.
        # Volume-weighted reversal is more reliable (high-vol selloffs revert more).
        try:
            if len(prices) >= 6:
                _ret5 = float((prices[-1] - prices[-6]) / max(prices[-6], 1e-9))
                features["reversal_5d"] = float(np.clip(-_ret5, -0.2, 0.2))
            else:
                features["reversal_5d"] = 0.0

            # Volume-weighted reversal: weight the 5d return by relative volume
            if len(prices) >= 6 and len(volumes) >= 20:
                _adv20 = float(np.mean(volumes[-20:]))
                _recent_vol_ratio = float(np.mean(volumes[-5:])) / max(_adv20, 1)
                features["reversal_5d_vol_weighted"] = float(
                    np.clip(-_ret5 * _recent_vol_ratio, -0.5, 0.5)
                ) if len(prices) >= 6 else 0.0
            else:
                features["reversal_5d_vol_weighted"] = 0.0

            # 3-day micro-reversal (captures even shorter noise reversion)
            if len(prices) >= 4:
                _ret3 = float((prices[-1] - prices[-4]) / max(prices[-4], 1e-9))
                features["reversal_3d"] = float(np.clip(-_ret3, -0.15, 0.15))
            else:
                features["reversal_3d"] = 0.0
        except Exception as exc:
            logger.debug("Reversal features failed: %s", exc)
            features.setdefault("reversal_5d", 0.0)
            features.setdefault("reversal_5d_vol_weighted", 0.0)
            features.setdefault("reversal_3d", 0.0)

        # ── 35. VRP, beta, opex calendar, earnings drift ─────────────────────────

        # 35a. Volatility Risk Premium: implied vol minus 20d realized vol.
        # High VRP = expensive options relative to realised movement → mean-reversion signal.
        try:
            if len(prices) >= 21:
                _log_rets = np.diff(np.log(np.where(prices > 0, prices, 1e-9)))
                _realized_vol_20d = float(np.std(_log_rets[-20:])) * np.sqrt(252)
            else:
                _realized_vol_20d = 0.0
            _iv_atm = float(features.get("options_iv_atm", 0.0))
            # VRP only meaningful when we have live IV; clip to avoid outliers
            features["vrp"] = float(np.clip(_iv_atm - _realized_vol_20d, -0.5, 0.5))
            features["realized_vol_20d"] = float(np.clip(_realized_vol_20d, 0.0, 2.0))
        except Exception as exc:
            logger.debug("VRP features failed for %s: %s", symbol, exc)
            features["vrp"] = 0.0
            features["realized_vol_20d"] = 0.0

        # 35b. Days to next monthly options expiration (3rd Friday of each month).
        # Gamma-pinning compresses stocks before expiration; post-expiry release drives moves.
        try:
            from datetime import date as _date, timedelta as _td
            _ref = as_of_date if as_of_date is not None else _date.today()

            def _next_opex(d):
                # Find 3rd Friday of current month; if past, use next month
                for month_offset in range(3):
                    if month_offset == 0:
                        y, m = d.year, d.month
                    else:
                        m = d.month + month_offset
                        y = d.year + (m - 1) // 12
                        m = ((m - 1) % 12) + 1
                    # 3rd Friday: find first Friday then +14 days
                    first_day = _date(y, m, 1)
                    days_to_friday = (4 - first_day.weekday()) % 7
                    third_friday = first_day + _td(days=days_to_friday + 14)
                    if third_friday >= d:
                        return third_friday
                return d + _td(days=7)  # fallback

            _opex = _next_opex(_ref)
            _days_to_opex = (_opex - _ref).days
            features["days_to_opex"] = float(np.clip(_days_to_opex, 0, 30))
            # Binary flag: within 3 days of expiration (gamma pin zone)
            features["near_opex"] = 1.0 if _days_to_opex <= 3 else 0.0
        except Exception as exc:
            logger.debug("Opex calendar failed for %s: %s", symbol, exc)
            features["days_to_opex"] = 15.0
            features["near_opex"] = 0.0

        # 35c. Rolling 252-day beta vs SPY.
        # Low-beta stocks have systematic edge (Betting Against Beta, Sharpe 0.594).
        # Also used as risk-adjustment context for momentum signals.
        try:
            if spy_returns is not None and len(spy_returns) >= 60 and len(prices) >= 61:
                _stock_rets = np.diff(np.log(np.where(prices > 0, prices, 1e-9)))
                _n = min(252, len(_stock_rets), len(spy_returns))
                _sr = _stock_rets[-_n:]
                _mr = spy_returns[-_n:]
                if len(_sr) == len(_mr) and np.std(_mr) > 1e-9:
                    _cov = float(np.cov(_sr, _mr)[0, 1])
                    _var_m = float(np.var(_mr))
                    _beta = float(np.clip(_cov / _var_m, -3.0, 3.0))
                else:
                    _beta = 1.0
            else:
                _beta = 1.0
            features["beta_252d"] = _beta
            # Beta deviation from 1.0: how much this stock amplifies/dampens market moves
            features["beta_deviation"] = float(np.clip(abs(_beta - 1.0), 0.0, 2.0))
        except Exception as exc:
            logger.debug("Beta features failed for %s: %s", symbol, exc)
            features["beta_252d"] = 1.0
            features["beta_deviation"] = 0.0

        # 35d. Earnings drift signal: decaying directional signal post-announcement.
        # sign(surprise) × decay(days_since) captures ongoing PEAD (post-earnings drift).
        # Fades to 0 after ~30 days, strongest in first 1-5 days after announcement.
        try:
            _surprise = float(features.get("earnings_surprise_1q", features.get("earnings_surprise", 0.0)))
            _days_since = float(features.get("days_since_earnings", features.get("fmp_days_since_earnings", 90.0)))
            _days_since = max(_days_since, 1.0)
            # Exponential decay: half-life ~7 days; zero by day 30
            _decay = float(np.exp(-_days_since / 7.0))
            features["earnings_drift_signal"] = float(np.clip(np.sign(_surprise) * _decay, -1.0, 1.0))
            # Interaction: large surprise + recent = strong drift
            features["earnings_pead_strength"] = float(np.clip(_surprise * _decay, -0.5, 0.5))
        except Exception as exc:
            logger.debug("Earnings drift failed for %s: %s", symbol, exc)
            features["earnings_drift_signal"] = 0.0
            features["earnings_pead_strength"] = 0.0

        return features
