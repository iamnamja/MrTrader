"""
Options and volatility features for the ML models.

Two tiers of data availability:

1. Price-derived volatility features (always available, used in training):
   - Realized volatility percentile vs 52-week range (IV proxy)
   - Short/long vol ratio (vol regime: expanding vs contracting)
   - Volatility-of-volatility (clustering — precedes big moves)
   - ATR trend (expanding range signals accumulation/distribution)

2. Live options features (yfinance, used in live inference only):
   - Put/call volume ratio (sentiment — high put/call = fear)
   - ATM implied volatility
   - IV percentile vs 30-day realized vol (vol premium)

Historical options data (Polygon options S3/REST) requires an upgraded
plan.  The architecture is ready; swap _fetch_options_yf() for a Polygon
call when the plan is upgraded.
"""

import logging
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Price-derived volatility features ────────────────────────────────────────

def compute_vol_features(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> Dict[str, float]:
    """
    Compute volatility-based features from OHLCV price arrays.
    All inputs are 1-D numpy arrays, most-recent value last.

    Returns dict with:
      vol_percentile_52w  — current 10d realized vol vs 52-week range [0,1]
                            (proxy for IV percentile; low = cheap options / quiet stock)
      vol_regime          — short/long vol ratio (>1 = expanding, <1 = contracting)
      vol_of_vol          — std of rolling 10d vol series (clustering signal)
      atr_trend           — ATR(5) / ATR(20) ratio (expanding range = momentum)
      parkinson_vol       — Parkinson high-low volatility estimator (more efficient than close-only)
    """
    result = {
        "vol_percentile_52w": 0.5,
        "vol_regime": 1.0,
        "vol_of_vol": 0.0,
        "atr_trend": 1.0,
        "parkinson_vol": 0.0,
    }

    n = len(prices)
    if n < 22:
        return result

    # Daily log returns
    returns = np.diff(np.log(prices))

    # 10-day realized vol (annualised)
    def _rvol(r, window):
        if len(r) < window:
            return None
        return float(np.std(r[-window:]) * np.sqrt(252))

    rv10 = _rvol(returns, 10)
    rv20 = _rvol(returns, 20)
    if rv10 is None or rv20 is None:
        return result

    # Vol percentile vs 52-week rolling window
    lookback = min(252, n - 1)
    if lookback >= 10:
        # Compute rolling 10d vol at each point over the lookback
        rv_series = [
            float(np.std(returns[max(0, i-10):i]) * np.sqrt(252))
            for i in range(10, lookback + 1)
        ]
        if rv_series:
            rv_min, rv_max = min(rv_series), max(rv_series)
            if rv_max > rv_min:
                result["vol_percentile_52w"] = float(
                    (rv10 - rv_min) / (rv_max - rv_min)
                )
            else:
                result["vol_percentile_52w"] = 0.5

    # Vol regime: short (10d) / long (60d)
    rv60 = _rvol(returns, min(60, len(returns)))
    if rv60 and rv60 > 0:
        result["vol_regime"] = float(min(3.0, rv10 / rv60))

    # Vol-of-vol: std of rolling 10d vol values (last 60 days)
    if len(returns) >= 60:
        vov_series = [
            float(np.std(returns[i:i+10]) * np.sqrt(252))
            for i in range(len(returns) - 60, len(returns) - 10)
        ]
        result["vol_of_vol"] = float(np.std(vov_series)) if vov_series else 0.0

    # ATR trend: ATR(5) / ATR(20)
    if len(highs) >= 20 and len(lows) >= 20:
        tr = np.maximum(highs[1:] - lows[1:],
             np.maximum(np.abs(highs[1:] - prices[:-1]),
                        np.abs(lows[1:] - prices[:-1])))
        atr5 = float(np.mean(tr[-5:])) if len(tr) >= 5 else None
        atr20 = float(np.mean(tr[-20:])) if len(tr) >= 20 else None
        if atr5 and atr20 and atr20 > 0:
            result["atr_trend"] = float(min(3.0, atr5 / atr20))

    # Parkinson high-low volatility (more efficient than close-only)
    if len(highs) >= 10 and len(lows) >= 10:
        hl = np.log(highs[-10:] / lows[-10:])
        result["parkinson_vol"] = float(
            np.sqrt(np.mean(hl**2) / (4 * np.log(2))) * np.sqrt(252)
        )

    return result


# ── Live options features (yfinance) ─────────────────────────────────────────

_options_cache: dict = {}
_OPTIONS_TTL = 3600  # 1 hour — options data changes intraday


def get_live_options_features(symbol: str) -> Dict[str, float]:
    """
    Fetch current options chain via yfinance and compute:
      options_put_call_ratio  — put vol / call vol across all near-term expirations
                                (>1 = bearish sentiment; <0.5 = bullish)
      options_iv_atm          — avg implied volatility of near-ATM contracts
      options_iv_premium      — IV minus recent realized vol (vol risk premium)

    Returns zeros on any failure — safe to call from live inference pipeline.
    Only used at inference time, not during historical training.
    """
    import time

    result = {
        "options_put_call_ratio": 1.0,
        "options_iv_atm": 0.0,
        "options_iv_premium": 0.0,
    }

    now = time.time()
    cached = _options_cache.get(symbol)
    if cached and now - cached[1] < _OPTIONS_TTL:
        return cached[0]

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        exps = ticker.options
        if not exps:
            _options_cache[symbol] = (result, now)
            return result

        # Use next 2 expirations for volume (most liquid)
        total_call_vol = 0.0
        total_put_vol = 0.0
        iv_samples = []

        for exp in exps[:2]:
            try:
                chain = ticker.option_chain(exp)
                calls = chain.calls.dropna(subset=["impliedVolatility"])
                puts = chain.puts.dropna(subset=["impliedVolatility"])

                total_call_vol += float(calls["volume"].fillna(0).sum())
                total_put_vol += float(puts["volume"].fillna(0).sum())

                # ATM IV: contracts within 5% of current price
                info = ticker.fast_info
                spot = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
                if spot and spot > 0:
                    atm_mask_c = (calls["strike"] - spot).abs() / spot < 0.05
                    atm_mask_p = (puts["strike"] - spot).abs() / spot < 0.05
                    atm_iv = pd.concat([
                        calls.loc[atm_mask_c, "impliedVolatility"],
                        puts.loc[atm_mask_p, "impliedVolatility"],
                    ])
                    iv_samples.extend(atm_iv.tolist())
            except Exception:
                continue

        if total_call_vol > 0:
            result["options_put_call_ratio"] = float(
                min(5.0, total_put_vol / total_call_vol)
            )

        if iv_samples:
            result["options_iv_atm"] = float(np.mean(iv_samples))

            # Vol premium: IV vs 20d realized vol
            try:
                hist = ticker.history(period="30d")
                if len(hist) >= 20:
                    rv20 = float(hist["Close"].pct_change().dropna().std() * np.sqrt(252))
                    result["options_iv_premium"] = float(
                        result["options_iv_atm"] - rv20
                    )
            except Exception:
                pass

    except Exception as exc:
        logger.debug("Live options fetch failed for %s: %s", symbol, exc)

    _options_cache[symbol] = (result, now)
    return result
