"""
Factor portfolio scorer — momentum + quality composite.

Extracted from scripts/factor_portfolio_backtest.py (Phase D deployment).
Validated Sharpe = 1.335 on 6-year walk-forward (2019-2024).

Public API:
    compute_composite_score(date, closes, bars, fundamentals, use_tier2) -> pd.Series
    select_top_n(scores, n) -> list[str]
    regime_gate_ok(spy_closes, date, vix_value, ma_window, vix_threshold) -> bool
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _zscore_cross(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score, winsorised at ±3σ."""
    mu, sig = series.mean(), series.std()
    if sig < 1e-9:
        return pd.Series(0.0, index=series.index)
    z = (series - mu) / sig
    return z.clip(-3, 3)


def _momentum_252d_ex1m(closes: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """12-month momentum excluding last month: close[-252] → close[-21]."""
    idx = closes.index.get_loc(as_of) if as_of in closes.index else None
    if idx is None or idx < 252:
        return pd.Series(dtype=float)
    c_now = closes.iloc[idx - 21]
    c_start = closes.iloc[max(0, idx - 252)]
    ret = (c_now / c_start.replace(0, np.nan)) - 1.0
    return ret.dropna()


def _price_to_52w_high(closes: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Current close / 52-week high."""
    idx = closes.index.get_loc(as_of) if as_of in closes.index else None
    if idx is None or idx < 252:
        return pd.Series(dtype=float)
    window = closes.iloc[max(0, idx - 252): idx + 1]
    ratio = closes.iloc[idx] / window.max().replace(0, np.nan)
    return ratio.dropna()


def _price_to_52w_low(closes: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Current close / 52-week low (higher = farther from panic low)."""
    idx = closes.index.get_loc(as_of) if as_of in closes.index else None
    if idx is None or idx < 252:
        return pd.Series(dtype=float)
    window = closes.iloc[max(0, idx - 252): idx + 1]
    ratio = closes.iloc[idx] / window.min().replace(0, np.nan)
    return ratio.dropna()


def _volume_trend(bars: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series:
    """20d avg volume / 60d avg volume."""
    result = {}
    for sym, df in bars.items():
        if "volume" not in df.columns or as_of not in df.index:
            continue
        idx = df.index.get_loc(as_of)
        if idx < 60:
            continue
        v20 = df["volume"].iloc[max(0, idx - 20): idx].mean()
        v60 = df["volume"].iloc[max(0, idx - 60): idx].mean()
        if v60 > 0:
            result[sym] = v20 / v60
    return pd.Series(result)


def _range_expansion(bars: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series:
    """5d ATR / 20d ATR."""
    result = {}
    for sym, df in bars.items():
        if as_of not in df.index:
            continue
        idx = df.index.get_loc(as_of)
        if idx < 20:
            continue
        try:
            tr = (df["high"] - df["low"]).abs()
            atr5 = tr.iloc[max(0, idx - 5): idx].mean()
            atr20 = tr.iloc[max(0, idx - 20): idx].mean()
            if atr20 > 0:
                result[sym] = atr5 / atr20
        except Exception:
            pass
    return pd.Series(result)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_composite_score(
    as_of: pd.Timestamp,
    closes: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    fundamentals: Optional[pd.DataFrame] = None,
    use_tier2: bool = True,
) -> pd.Series:
    """Cross-sectional composite factor score for all symbols on `as_of`.

    Weights:
        momentum_252d_ex1m ×2.0 (IR=1.99, dominant factor)
        price_to_52w_high  ×1.0
        quality factors    ×1.0 each (profit_margin, operating_margin, -pe_ratio)
        tier2 factors      ×0.5 each (price_to_52w_low, volume_trend, range_expansion,
                                       gross_margin, revenue_growth)

    Returns:
        Series indexed by symbol, higher = more attractive.
    """
    fund = fundamentals if fundamentals is not None else pd.DataFrame()
    scores: dict[str, pd.Series] = {}

    # ── Tier 1 ──
    mom = _momentum_252d_ex1m(closes, as_of)
    if not mom.empty:
        scores["momentum_252d_ex1m"] = _zscore_cross(mom) * 2.0

    p52h = _price_to_52w_high(closes, as_of)
    if not p52h.empty:
        scores["price_to_52w_high"] = _zscore_cross(p52h)

    if not fund.empty:
        for col, sign in [("profit_margin", 1), ("operating_margin", 1), ("pe_ratio", -1)]:
            if col in fund.columns:
                vals = fund[col].reindex(closes.columns).dropna()
                if not vals.empty:
                    scores[col] = _zscore_cross(vals) * sign

    # ── Tier 2 ──
    if use_tier2:
        p52l = _price_to_52w_low(closes, as_of)
        if not p52l.empty:
            scores["price_to_52w_low"] = _zscore_cross(p52l) * 0.5

        vt = _volume_trend(bars, as_of)
        if not vt.empty:
            scores["volume_trend"] = _zscore_cross(vt) * 0.5

        re = _range_expansion(bars, as_of)
        if not re.empty:
            scores["range_expansion"] = _zscore_cross(re) * 0.5

        if not fund.empty:
            for col in ("gross_margin", "revenue_growth_yoy"):
                if col in fund.columns:
                    vals = fund[col].reindex(closes.columns).dropna()
                    if not vals.empty:
                        key = "revenue_growth" if "growth" in col else col
                        scores[key] = _zscore_cross(vals) * 0.5

    if not scores:
        return pd.Series(dtype=float)

    return pd.DataFrame(scores).mean(axis=1).dropna()


def select_top_n(scores: pd.Series, n: int = 20) -> list[str]:
    """Return top-N symbols by composite score (descending)."""
    if scores.empty:
        return []
    return scores.nlargest(n).index.tolist()


def regime_gate_ok(
    spy_closes: pd.Series,
    as_of: pd.Timestamp,
    vix_value: Optional[float] = None,
    ma_window: int = 200,
    vix_threshold: float = 30.0,
) -> bool:
    """Return True if market regime allows factor portfolio to trade.

    Rules (both must pass):
        SPY > SPY.rolling(200).mean()  — bull trend filter
        VIX < 30.0                     — volatility filter (if vix_value provided)
    """
    if spy_closes.empty:
        return True  # permissive if no data

    past = spy_closes[spy_closes.index <= as_of]
    if len(past) < ma_window:
        return True  # not enough history — allow trading

    spy_last = float(past.iloc[-1])
    spy_ma = float(past.iloc[-ma_window:].mean())
    spy_ok = spy_last > spy_ma

    vix_ok = (vix_value is None) or (vix_value < vix_threshold)

    return spy_ok and vix_ok
