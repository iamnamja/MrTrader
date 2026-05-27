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

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


# WF-R5 (FIX 4): IC weights calibration cutoff. All `_V*_IC_WEIGHTS` constants
# below were derived from daily_ic.parquet rows with `date <= 2021-04-26` (one
# trading day before fold-1's earliest train_end at the default WF config:
# --total-years 5 --n-folds 3 --as-of 2026-05-27). Importers (e.g. the WF
# harness) use this constant to assert that every fold's train_end is strictly
# greater than this cutoff — otherwise the "pre-fold" weights are in-sample
# again. If you recompute weights with a different END_DATE in
# `scripts/compute_factor_ic.py`, update this constant in lockstep.
IC_WEIGHTS_CALIBRATION_END: date = date(2021, 4, 26)


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


def select_bottom_n(scores: pd.Series, n: int = 15) -> list[str]:
    """Return bottom-N symbols by composite score (ascending) — short candidates."""
    if scores.empty:
        return []
    return scores.nsmallest(n).index.tolist()


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

    # H-3: strict `<` to prevent same-day bar leak if as_of matches a bar timestamp.
    past = spy_closes[spy_closes.index < as_of]
    if len(past) < ma_window:
        return True  # not enough history — allow trading

    spy_last = float(past.iloc[-1])
    spy_ma = float(past.iloc[-ma_window:].mean())
    spy_ok = spy_last > spy_ma

    vix_ok = (vix_value is None) or (vix_value < vix_threshold)

    return spy_ok and vix_ok


class FactorPortfolioScorer:
    """Callable wrapper for use as AgentSimulator.factor_scorer.

    Implements the interface:
        (day, symbols_data, vix_history) -> [(sym, conf, direction)]

    direction is "long" (positive conf) or "short" (negative conf).
    When long_short=False, only longs are returned (legacy behaviour).

    Usage:
        scorer = FactorPortfolioScorer(top_n=20, top_n_short=15)
        sim = AgentSimulator(..., factor_scorer=scorer)
    """

    def __init__(
        self,
        top_n: int = 20,
        top_n_short: int = 15,
        long_short: bool = True,
        use_tier2: bool = True,
        vix_threshold: float = 30.0,
        spy_ma_window: int = 200,
        require_positive_momentum_days: int = 0,  # 0=disabled; >0: require stock ret>0 over N days
        require_spy_outperform_days: int = 0,     # 0=disabled; >0: require stock ret > SPY ret over N days
    ):
        self.top_n = top_n
        self.top_n_short = top_n_short
        self.long_short = long_short
        self.use_tier2 = use_tier2
        self.vix_threshold = vix_threshold
        self.spy_ma_window = spy_ma_window
        self.require_positive_momentum_days = require_positive_momentum_days
        self.require_spy_outperform_days = require_spy_outperform_days

    def __call__(
        self,
        day,
        symbols_data: dict,
        vix_history=None,
    ) -> list:
        """Score all symbols on `day`. Returns [(sym, conf, direction)].

        conf is in [0.55, 0.95] for longs; [-0.95, -0.55] for shorts.
        direction is "long" or "short".
        """
        import pandas as pd

        # Build aligned closes DataFrame
        close_cols = {}
        for sym, df in symbols_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            _day_d = day.date() if hasattr(day, "date") else day
            mask = df.index.date < _day_d if hasattr(df.index[0], "date") else df.index < pd.Timestamp(day)
            _past_raw = df.loc[mask, "close"] if mask.any() else pd.Series(dtype=float)
            past = _past_raw.iloc[:, 0] if isinstance(_past_raw, pd.DataFrame) else _past_raw
            if len(past) >= 60:
                close_cols[sym] = past

        if not close_cols:
            return []

        closes = pd.DataFrame(close_cols)
        closes.index = pd.to_datetime(closes.index)
        as_of = closes.index[-1]

        # Regime gate
        spy_series = closes.get("SPY", pd.Series(dtype=float))
        vix_val = None
        if vix_history is not None:
            try:
                # H-3: strict `<` to avoid same-day VIX bar leak
                past_vix = vix_history[vix_history.index < as_of]
                if len(past_vix) > 0:
                    vix_val = float(past_vix.iloc[-1])
            except Exception:
                pass

        regime_ok = regime_gate_ok(
            spy_series, as_of, vix_value=vix_val,
            ma_window=self.spy_ma_window, vix_threshold=self.vix_threshold,
        )
        # In bear market: suppress all trading. Bottom-N momentum shorts (beaten-down stocks)
        # reverse violently during bear market rallies, destroying the short book.
        # The strategy's edge is long momentum; sit in cash during bear markets.
        if not regime_ok:
            return []

        bars_by_sym = {
            sym: df.loc[df.index.date < _day_d] if hasattr(df.index[0], "date")
            else df.loc[df.index < pd.Timestamp(day)]
            for sym, df in symbols_data.items()
            if sym in close_cols
        }

        scores = compute_composite_score(as_of, closes, bars_by_sym, use_tier2=self.use_tier2)
        if scores.empty:
            return []

        # Optional: filter out stocks with negative N-day momentum (trending down)
        if self.require_positive_momentum_days > 0:
            n = self.require_positive_momentum_days
            eligible = [
                sym for sym in scores.index
                if sym in closes.columns and len(closes[sym].dropna()) >= n + 1
                and float(closes[sym].dropna().iloc[-1]) > float(closes[sym].dropna().iloc[-(n + 1)])
            ]
            scores = scores.loc[scores.index.isin(eligible)]
            if scores.empty:
                return []

        # Optional: filter out stocks underperforming SPY over N days (market-relative momentum)
        if self.require_spy_outperform_days > 0 and "SPY" in closes.columns:
            n = self.require_spy_outperform_days
            spy_c = closes["SPY"].dropna()
            if len(spy_c) >= n + 1:
                spy_ret = float(spy_c.iloc[-1]) / float(spy_c.iloc[-(n + 1)]) - 1
                eligible_rel = [
                    sym for sym in scores.index
                    if sym in closes.columns and len(closes[sym].dropna()) >= n + 1
                    and (float(closes[sym].dropna().iloc[-1]) / float(closes[sym].dropna().iloc[-(n + 1)]) - 1) > spy_ret
                ]
                scores = scores.loc[scores.index.isin(eligible_rel)]
                if scores.empty:
                    return []

        s_min, s_max = float(scores.min()), float(scores.max())
        s_range = max(s_max - s_min, 1e-6)

        def _conf(raw: float) -> float:
            return 0.55 + 0.40 * ((raw - s_min) / s_range)

        result = []

        # Longs: top-N (only when regime is ok)
        if regime_ok:
            top_syms = select_top_n(scores, n=self.top_n)
            result.extend(
                (sym, _conf(float(scores[sym])), "long")
                for sym in top_syms
            )

        # Shorts: bottom-N (when L/S mode enabled; excluded symbols that are longs)
        if self.long_short:
            long_set = {s for s, _, _ in result}
            bottom_syms = [
                s for s in select_bottom_n(scores, n=self.top_n_short)
                if s not in long_set
            ]
            result.extend(
                (sym, -_conf(float(scores[sym])), "short")
                for sym in bottom_syms
            )

        return result


# ── v219 IC-weighted composite scorer (Phase 88, 2026-05-25) ─────────────────
# Uses h20 IC IR from 2026-05-24 audit as weights. No ML training, no HPO.
# Features: momentum_252d_ex1m, ix_momentum_vol, price_to_52w_low/high,
#           profit_margin, operating_margin, pe_ratio, volume_trend,
#           range_expansion, vol_regime, vrp.
# WQ alphas (wq_alpha35/40/43) omitted — complex formulas; excluded weight
# redistributed proportionally to remaining features.
# Motivation: LambdaRank v218 failed to beat 60d momentum baseline. Deterministic
# IC-weighted composite avoids HPO overfitting and leverages all 17-feature audit.

# WF-C1 FIX (R4, 2026-05-27): the original V219 weights (now preserved as
# `_V219_IC_WEIGHTS_IN_SAMPLE`) were derived from a per-feature IC IR audit run
# over 2019-01-01 -> 2026-11-08 — a window that fully overlaps every WF test
# fold. That is in-sample feature selection. The replacement below uses only
# pre-fold-1 data (dates <= 2021-04-26, one day before fold 1's earliest
# train_end) from the same daily_ic.parquet (h=20, negatives clipped to 0,
# normalized). Many features had negative pre-fold IR, confirming the
# in-sample contamination of the original weights.

# Original in-sample (contaminated) weights — kept for reference/reproducibility.
_V219_IC_WEIGHTS_IN_SAMPLE: dict[str, float] = {
    "ix_momentum_vol":          2.3095,
    "momentum_252d_ex1m":       2.2334,
    "price_to_52w_low":         2.2233,
    "profit_margin":            1.2582,
    "operating_margin":         1.0727,
    "pe_ratio":                 0.8842,   # inverted: lower PE = better
    "price_to_52w_high":        0.9442,
    "volume_trend":             0.8840,
    "reversal_5d_vol_weighted": 0.6254,
    "downtrend":                0.4532,
    "range_expansion":          0.1795,
    "vol_regime":               0.2386,
    "vrp":                      0.0566,
    # wq_alpha40: 1.1834 — omitted (complex formula)
    # wq_alpha43: 0.7592 — omitted
    # wq_alpha35: 0.6437 — omitted
    # vol_percentile_52w: -0.1343 — dropped (negative IC)
}

# Pre-fold-1 IC IR weights (h=20, dates <= 2021-04-26, negatives clipped to 0).
# Source: data/diagnostics/feature_ic/20260524T230508Z/daily_ic.parquet.
_V219_IC_WEIGHTS_RAW: dict[str, float] = {
    "ix_momentum_vol":          0.0271,
    "momentum_252d_ex1m":       0.0102,
    "price_to_52w_low":         0.2449,
    "profit_margin":            0.0000,   # pre-fold IR was -0.16
    "operating_margin":         0.0000,   # pre-fold IR was -0.21
    "pe_ratio":                 0.0041,   # pre-fold IR (raw, before sign-flip)
    "price_to_52w_high":        0.0000,   # pre-fold IR was -0.10
    "volume_trend":             0.0000,   # pre-fold IR was -0.12
    "reversal_5d_vol_weighted": 0.0817,
    "downtrend":                0.0840,
    "range_expansion":          0.0000,   # pre-fold IR was -0.24
    "vol_regime":               0.0000,   # pre-fold IR was -0.09
    "vrp":                      0.0000,   # pre-fold IR was -0.12
}
_V219_TOTAL = sum(_V219_IC_WEIGHTS_RAW.values())
V219_IC_WEIGHTS: dict[str, float] = (
    {k: v / _V219_TOTAL for k, v in _V219_IC_WEIGHTS_RAW.items()}
    if _V219_TOTAL > 0
    else {k: 0.0 for k in _V219_IC_WEIGHTS_RAW}
)


def _vol_regime(closes_sym: pd.Series, as_of_idx: int, short_w: int = 21, long_w: int = 126) -> float:
    """Realized vol ratio: short / long (lower = calmer = more attractive)."""
    if as_of_idx < long_w:
        return float("nan")
    rets = closes_sym.pct_change()
    vol_short = float(rets.iloc[max(0, as_of_idx - short_w): as_of_idx].std())
    vol_long = float(rets.iloc[max(0, as_of_idx - long_w): as_of_idx].std())
    if vol_long < 1e-9:
        return float("nan")
    return vol_short / vol_long


def _reversal_5d_vw(df: pd.DataFrame, as_of_idx: int) -> float:
    """5-day volume-weighted price return (positive = recent recovery from dip)."""
    if as_of_idx < 5 or "volume" not in df.columns:
        return float("nan")
    window = df.iloc[max(0, as_of_idx - 5): as_of_idx]
    if window.empty or window["volume"].sum() < 1:
        return float("nan")
    c_now = float(df["close"].iloc[as_of_idx - 1])
    c_start = float(df["close"].iloc[max(0, as_of_idx - 5)])
    if c_start < 1e-9:
        return float("nan")
    # Negative 5d return = oversold = good for reversal (invert sign for ranking)
    return -(c_now / c_start - 1.0)


def _downtrend_score(closes_sym: pd.Series, as_of_idx: int, period: int = 20) -> float:
    """Returns 1 if price is in downtrend (below SMA20), 0 otherwise.
    For ranking: higher score = more oversold = better contrarian entry."""
    if as_of_idx < period:
        return float("nan")
    window = closes_sym.iloc[max(0, as_of_idx - period): as_of_idx]
    sma = window.mean()
    c_now = float(closes_sym.iloc[as_of_idx - 1])
    # Score = distance below SMA (positive = more oversold)
    return max(0.0, (sma - c_now) / sma)


def _vrp_score(closes_sym: pd.Series, vix_val: float, as_of_idx: int, period: int = 21) -> float:
    """Variance risk premium: implied vol (VIX) minus realized vol.
    Higher VRP = market overpaying for insurance = mean-reversion signal."""
    if as_of_idx < period or vix_val <= 0:
        return float("nan")
    rets = closes_sym.pct_change()
    realized = float(rets.iloc[max(0, as_of_idx - period): as_of_idx].std()) * (252 ** 0.5) * 100
    return vix_val - realized  # positive = VIX > realized (insurance premium)


def compute_v219_score(
    as_of: pd.Timestamp,
    closes: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    fundamentals: Optional[pd.DataFrame] = None,
    vix_val: Optional[float] = None,
) -> pd.Series:
    """IC-weighted cross-sectional composite score for Phase 88 / v219.

    Weights derived from 2026-05-24 IC audit (h20 IC IR, clipped at 0).
    Returns Series indexed by symbol, higher = more attractive.
    """
    fund = fundamentals if fundamentals is not None else pd.DataFrame()
    raw_features: dict[str, dict[str, float]] = {k: {} for k in V219_IC_WEIGHTS}

    as_of_ts = pd.Timestamp(as_of)

    for sym in closes.columns:
        if sym in ("SPY", "^VIX", "VIX"):
            continue
        col = closes[sym].dropna()
        if len(col) < 60:
            continue
        # Find index in col
        # H-3: strict `<` (defensive — callers pre-truncate, but enforce no same-day leak)
        mask = col.index < as_of_ts
        past = col[mask]
        if len(past) < 60:
            continue
        idx = len(past)

        c_past = past

        # momentum_252d_ex1m
        if idx >= 252:
            c_now = float(c_past.iloc[-21]) if idx >= 21 else float(c_past.iloc[-1])
            c_start = float(c_past.iloc[-252])
            if c_start > 1e-9:
                raw_features["momentum_252d_ex1m"][sym] = (c_now / c_start) - 1.0

        # price_to_52w_high
        if idx >= 252:
            high_52w = float(c_past.iloc[-252:].max())
            if high_52w > 1e-9:
                raw_features["price_to_52w_high"][sym] = float(c_past.iloc[-1]) / high_52w

        # price_to_52w_low
        if idx >= 252:
            low_52w = float(c_past.iloc[-252:].min())
            if low_52w > 1e-9:
                raw_features["price_to_52w_low"][sym] = float(c_past.iloc[-1]) / low_52w

        # vol_regime
        vr = _vol_regime(c_past, idx)
        if not np.isnan(vr):
            raw_features["vol_regime"][sym] = -vr  # lower vol_ratio = better (invert)

        # ix_momentum_vol = momentum × (1 - vol_ratio) — high momentum, low vol
        mom = raw_features["momentum_252d_ex1m"].get(sym, float("nan"))
        if not np.isnan(mom) and sym in raw_features["vol_regime"]:
            raw_features["ix_momentum_vol"][sym] = mom * (1.0 + raw_features["vol_regime"][sym])

        # volume_trend
        df_sym = bars.get(sym)
        if df_sym is not None and "volume" in df_sym.columns:
            sym_mask = df_sym.index < as_of_ts  # H-3: strict `<`
            sym_past = df_sym[sym_mask]
            sidx = len(sym_past)
            if sidx >= 60:
                v20 = float(sym_past["volume"].iloc[-20:].mean())
                v60 = float(sym_past["volume"].iloc[-60:].mean())
                if v60 > 0:
                    raw_features["volume_trend"][sym] = v20 / v60

            # range_expansion
            if sidx >= 20 and "high" in df_sym.columns and "low" in df_sym.columns:
                tr = (sym_past["high"] - sym_past["low"]).abs()
                atr5 = float(tr.iloc[-5:].mean())
                atr20 = float(tr.iloc[-20:].mean())
                if atr20 > 0:
                    raw_features["range_expansion"][sym] = atr5 / atr20

            # reversal_5d_vol_weighted
            rv = _reversal_5d_vw(sym_past, sidx)
            if not np.isnan(rv):
                raw_features["reversal_5d_vol_weighted"][sym] = rv

            # downtrend
            dt = _downtrend_score(sym_past["close"], sidx)
            if not np.isnan(dt):
                raw_features["downtrend"][sym] = dt

        # vrp
        if vix_val is not None:
            vp = _vrp_score(c_past, vix_val, idx)
            if not np.isnan(vp):
                raw_features["vrp"][sym] = vp

    # Fundamentals: profit_margin, operating_margin, pe_ratio (inverted)
    if not fund.empty:
        for feat, sign in [("profit_margin", 1), ("operating_margin", 1), ("pe_ratio", -1)]:
            if feat in fund.columns:
                for sym, val in fund[feat].items():
                    if sym in closes.columns and not np.isnan(val):
                        raw_features[feat][sym] = sign * val

    if not any(raw_features.values()):
        return pd.Series(dtype=float)

    # Cross-sectional z-score each feature, then weight-average
    score_df_parts: list[pd.Series] = []
    for feat, w in V219_IC_WEIGHTS.items():
        vals = pd.Series(raw_features[feat]).dropna()
        if len(vals) < 5:
            continue
        z = _zscore_cross(vals) * w
        score_df_parts.append(z)

    if not score_df_parts:
        return pd.Series(dtype=float)

    combined = pd.concat(score_df_parts, axis=1).sum(axis=1, min_count=1).dropna()
    return combined


class IcCompositeScorer:
    """IC-weighted deterministic composite scorer for Phase 88 / v219.

    No ML model. No HPO. Weights from 2026-05-24 IC audit (h20 IC IR).
    Designed for rebalance mode: top-30, 20-day cadence, regime gate + inv-vol.

    Interface matches AgentSimulator.factor_scorer:
        (day, symbols_data, vix_history) -> [(sym, score)]
    """

    def __init__(self, fmp_parquet: Optional[str] = None, fold_ic_weights: Optional[dict] = None):
        self._fundamentals: Optional[pd.DataFrame] = None
        self._fmp_path = fmp_parquet or "data/fundamentals/fmp_fundamentals_history.parquet"
        # WF-C1 R4: optional per-fold IC-weight injection. Same Option-A migration
        # path as V221 — caller may pass pre-train_end IR values; we clip & normalize.
        if fold_ic_weights is not None and len(fold_ic_weights) > 0:
            _t = sum(max(v, 0.0) for v in fold_ic_weights.values())
            self._weights = (
                {k: max(v, 0.0) / _t for k, v in fold_ic_weights.items()}
                if _t > 0
                else dict(V219_IC_WEIGHTS)
            )
        else:
            self._weights = V219_IC_WEIGHTS

    def _get_fundamentals(self, as_of: pd.Timestamp) -> pd.DataFrame:
        if self._fundamentals is None:
            try:
                raw = pd.read_parquet(self._fmp_path)
                self._fundamentals = raw
            except Exception:
                self._fundamentals = pd.DataFrame()
        if self._fundamentals.empty:
            return self._fundamentals
        # PIT-safe: filter to rows on or before as_of, take latest per symbol
        df = self._fundamentals
        date_col = next((c for c in ("date", "report_date", "filed_date") if c in df.columns), None)
        sym_col = next((c for c in ("symbol", "ticker", "sym") if c in df.columns), None)
        if date_col and sym_col:
            try:
                filtered = df[pd.to_datetime(df[date_col]) <= as_of]
                if not filtered.empty:
                    return filtered.sort_values(date_col).groupby(sym_col).last()
            except Exception:
                pass
        return df

    def __call__(self, day, symbols_data: dict, vix_history=None) -> list:
        import pandas as pd

        _day_d = day.date() if hasattr(day, "date") else day
        as_of_ts = pd.Timestamp(day)

        close_cols = {}
        for sym, df in symbols_data.items():
            if sym in ("^VIX", "VIX") or df is None or df.empty or "close" not in df.columns:
                continue
            mask = df.index.date < _day_d if hasattr(df.index[0], "date") else df.index < as_of_ts
            _past_raw = df.loc[mask, "close"] if mask.any() else pd.Series(dtype=float)
            past = _past_raw.iloc[:, 0] if isinstance(_past_raw, pd.DataFrame) else _past_raw
            if len(past) >= 60:
                close_cols[sym] = past

        if not close_cols:
            return []

        closes = pd.DataFrame(close_cols)
        closes.index = pd.to_datetime(closes.index)

        bars_past = {
            sym: symbols_data[sym].loc[
                (symbols_data[sym].index.date < _day_d)
                if hasattr(symbols_data[sym].index[0], "date")
                else (symbols_data[sym].index < as_of_ts)
            ]
            for sym in close_cols if sym in symbols_data
        }

        vix_val = None
        if vix_history is not None:
            try:
                # H-3: strict `<` to avoid same-day VIX leak
                past_vix = vix_history[vix_history.index < as_of_ts]
                if len(past_vix) > 0:
                    vix_val = float(past_vix.iloc[-1])
            except Exception:
                pass

        fund = self._get_fundamentals(as_of_ts)
        # WF-C1 R4: route through _compute_weighted_score with self._weights so
        # per-fold IC weight injection (if any) actually flows through. Note
        # _compute_weighted_score reads VIX from closes[^VIX] internally; ensure
        # VIX series is included alongside symbol closes.
        if vix_history is not None:
            try:
                vix_series = vix_history if isinstance(vix_history, pd.Series) else None
                if vix_series is not None and len(vix_series) > 0:
                    closes = closes.copy()
                    closes["^VIX"] = vix_series.reindex(closes.index).ffill()
            except Exception:
                pass
        scores = _compute_weighted_score(as_of_ts, closes, bars_past, self._weights, fundamentals=fund)

        if scores.empty:
            return []

        return sorted(
            [(sym, float(scores[sym])) for sym in scores.index if sym != "SPY"],
            key=lambda x: x[1],
            reverse=True,
        )


# ── Phase 90 — Regime-Conditional Composite (v220) ───────────────────────────
# Two-composite switch driven by market breadth (% symbols above 200d MA).
# Composite A: momentum-tilted (active when breadth > 60% — blow-off/trending regime)
# Composite B: quality-tilted (v219 weights, active otherwise)
#
# Diagnosis (Opus 4.7, 2026-05-25): v219 fails Fold 1 because quality features
# (profit_margin, operating_margin, pe_ratio) are wrong for late-cycle blow-off.
# Pure momentum baseline scores +0.60 in Fold 1 vs v219's -0.37.

# WF-C1 FIX (R4, 2026-05-27): the original V220A weights (now preserved as
# `_V220A_WEIGHTS_IN_SAMPLE`) were a hand-tuned Opus-4.7 momentum tilt designed
# AFTER inspecting the full-period IC audit (2019->2026), so they implicitly
# leak fold information through the designer's prior knowledge.
#
# We replace them with pre-fold-1 IC IR weights computed from the same
# daily_ic.parquet (dates <= 2021-04-26, h=20, negatives clipped to 0,
# normalized to sum=1), restricted to V220A's original feature set.
#
# NOTE on V220A's two-composite regime switch: after R2 also corrected V221
# (the "quality" composite used in the non-momentum regime), both composites
# now derive from the same pre-fold IC IR audit and end up momentum-tilted
# (the pre-fold-1 IR for fundamentals is negative -> clipped to 0). The
# regime-switch logic therefore offers little differentiation in this
# window; per the R4 fix-spec we document this but do NOT re-engineer the
# regime logic — that is a separate research question.
_V220A_WEIGHTS_IN_SAMPLE: dict[str, float] = {
    "ix_momentum_vol":          0.23,
    "momentum_252d_ex1m":       0.21,
    "price_to_52w_high":        0.13,
    "volume_trend":             0.08,
    "operating_margin":         0.05,
    "range_expansion":          0.05,
    "downtrend":                0.05,
    "vol_regime":               0.04,
    "reversal_5d_vol_weighted": 0.04,
    "price_to_52w_low":         0.03,
    "profit_margin":            0.03,
    "pe_ratio":                 0.03,  # inverted: lower PE = better
    "vrp":                      0.03,
}
assert abs(sum(_V220A_WEIGHTS_IN_SAMPLE.values()) - 1.0) < 1e-9, "V220A in-sample weights must sum to 1.0"

# Pre-fold-1 IC IR (h=20, dates <= 2021-04-26, negatives clipped to 0),
# restricted to V220A's original 13-feature set.
# Source: data/diagnostics/feature_ic/20260524T230508Z/daily_ic.parquet.
_V220A_WEIGHTS_RAW: dict[str, float] = {
    "ix_momentum_vol":          0.0271,
    "momentum_252d_ex1m":       0.0102,
    "price_to_52w_high":        0.0000,   # pre-fold IR was -0.10
    "volume_trend":             0.0000,   # pre-fold IR was -0.12
    "operating_margin":         0.0000,   # pre-fold IR was -0.21
    "range_expansion":          0.0000,   # pre-fold IR was -0.24
    "downtrend":                0.0840,
    "vol_regime":               0.0000,   # pre-fold IR was -0.09
    "reversal_5d_vol_weighted": 0.0817,
    "price_to_52w_low":         0.2449,
    "profit_margin":            0.0000,   # pre-fold IR was -0.16
    "pe_ratio":                 0.0041,   # pre-fold IR (raw)
    "vrp":                      0.0000,   # pre-fold IR was -0.12
}
_V220A_TOTAL = sum(_V220A_WEIGHTS_RAW.values())
V220A_WEIGHTS: dict[str, float] = (
    {k: v / _V220A_TOTAL for k, v in _V220A_WEIGHTS_RAW.items()}
    if _V220A_TOTAL > 0
    else {k: 0.0 for k in _V220A_WEIGHTS_RAW}
)

# Composite B = v219 (imported above as V219_IC_WEIGHTS)


def _compute_breadth(closes: pd.DataFrame, as_of_ts: pd.Timestamp, ma_days: int = 200) -> float:
    """Compute % of symbols with close > MA(ma_days) as of as_of_ts.

    Returns float [0, 1] or NaN if insufficient data.
    """
    above = 0
    total = 0
    for sym in closes.columns:
        if sym in ("SPY", "^VIX", "VIX"):
            continue
        col = closes[sym].dropna()
        past = col[col.index < as_of_ts]  # H-3: strict `<`
        total += 1  # BUG-3 fix: always count in denominator (short history = below MA)
        if len(past) < ma_days:
            continue
        ma_val = float(past.iloc[-ma_days:].mean())
        if float(past.iloc[-1]) > ma_val:
            above += 1
    if total < 50:
        return float("nan")
    return above / total


class IcCompositeV220Scorer:
    """Phase 90 regime-conditional composite scorer (v220).

    Switches between momentum-tilted (Composite A) and quality-tilted (Composite B = v219)
    based on market breadth signal (% symbols above 200d MA).

    Breadth > 60% (with 5pp hysteresis): use Composite A (momentum-tilted)
    Breadth < 55%: use Composite B (quality-tilted)

    This directly addresses Fold 1 (Jun 2021-May 2022) failure where quality features
    selected wrong names vs the momentum baseline (+0.60).
    """

    # Switch thresholds with hysteresis
    BREADTH_A_THRESHOLD = 0.60  # enter momentum regime
    BREADTH_B_THRESHOLD = 0.55  # exit momentum regime (5pp deadband)

    def __init__(self) -> None:
        self._fmp_fundamentals: Optional[pd.DataFrame] = None
        self._fmp_path = "data/fundamentals/fmp_fundamentals_history.parquet"
        # Regime state: True = Composite A (momentum), False = Composite B (quality)
        self._in_momentum_regime: bool = False
        # EMA of breadth (span=5 ≈ 3-day half-life)
        self._breadth_ema: Optional[float] = None

    def reset(self) -> None:
        """WF-R4: clear mutable per-fold state.

        Re-initialises stateful fields (_in_momentum_regime, _breadth_ema) to
        their __init__ defaults so that a deepcopy from a pre-warmed instance
        starts each fold from a clean state. Fundamentals cache is preserved.
        """
        self._in_momentum_regime = False
        self._breadth_ema = None

    def get_state(self) -> tuple:
        """WF-R4: snapshot of stateful fields (for post-reset assertion)."""
        return (self._in_momentum_regime, self._breadth_ema)

    def _get_fundamentals(self, as_of: pd.Timestamp) -> pd.DataFrame:
        # BUG-9 fix: load raw parquet once, filter PIT per call (was using nonexistent load_pit_fundamentals)
        if self._fmp_fundamentals is None:
            try:
                self._fmp_fundamentals = pd.read_parquet(self._fmp_path)
            except Exception:
                self._fmp_fundamentals = pd.DataFrame()
        df = self._fmp_fundamentals
        if df.empty:
            return df
        date_col = next((c for c in ("date", "report_date", "filed_date") if c in df.columns), None)
        sym_col = next((c for c in ("symbol", "ticker", "sym") if c in df.columns), None)
        if date_col and sym_col:
            try:
                filtered = df[pd.to_datetime(df[date_col]) <= as_of]
                if not filtered.empty:
                    return filtered.sort_values(date_col).groupby(sym_col).last()
            except Exception:
                pass
        return df

    def _update_breadth(self, raw_breadth: float) -> float:
        """Update EMA of breadth with hysteresis and return smoothed value."""
        if not np.isfinite(raw_breadth):
            return self._breadth_ema if self._breadth_ema is not None else 0.5
        alpha = 2.0 / (5 + 1)  # span=5 EMA
        if self._breadth_ema is None:
            self._breadth_ema = raw_breadth
        else:
            self._breadth_ema = alpha * raw_breadth + (1 - alpha) * self._breadth_ema
        return self._breadth_ema

    def __call__(self, day, symbols_data: dict, vix_history=None) -> list:
        import pandas as pd

        _day_d = day.date() if hasattr(day, "date") else day
        as_of_ts = pd.Timestamp(day)

        close_cols = {}
        for sym, df in symbols_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            mask = df.index.date < _day_d if hasattr(df.index[0], "date") else df.index < as_of_ts
            _past_raw = df.loc[mask, "close"] if mask.any() else pd.Series(dtype=float)
            past = _past_raw.iloc[:, 0] if isinstance(_past_raw, pd.DataFrame) else _past_raw
            if sym in ("^VIX", "VIX"):
                if len(past) > 0:
                    close_cols[sym] = past  # BUG-8 fix: include VIX for VRP computation
            elif len(past) >= 60:
                close_cols[sym] = past

        if not close_cols:
            return []

        closes = pd.DataFrame(close_cols)
        closes.index = pd.to_datetime(closes.index)

        # Compute breadth and update regime state
        raw_breadth = _compute_breadth(closes, as_of_ts)
        breadth_smooth = self._update_breadth(raw_breadth)

        if self._in_momentum_regime:
            if breadth_smooth < self.BREADTH_B_THRESHOLD:
                self._in_momentum_regime = False
        else:
            if breadth_smooth > self.BREADTH_A_THRESHOLD:
                self._in_momentum_regime = True

        # Select weights based on regime
        weights = V220A_WEIGHTS if self._in_momentum_regime else V219_IC_WEIGHTS

        bars_past = {
            sym: symbols_data[sym].loc[
                (symbols_data[sym].index.date < _day_d)
                if hasattr(symbols_data[sym].index[0], "date")
                else (symbols_data[sym].index < as_of_ts)
            ]
            for sym in close_cols if sym in symbols_data and sym not in ("^VIX", "VIX")
        }

        fund = self._get_fundamentals(as_of_ts)

        # Reuse compute_v219_score but override weights
        scores = _compute_weighted_score(as_of_ts, closes, bars_past, weights, fundamentals=fund)

        if scores.empty:
            return []

        return sorted(
            [(sym, float(scores[sym])) for sym in scores.index if sym != "SPY"],
            key=lambda x: x[1],
            reverse=True,
        )

    @property
    def active_regime(self) -> str:
        return "momentum" if self._in_momentum_regime else "quality"


# =============================================================================
# Phase 92 — v222 scorer: v221 base weights + v220 breadth-regime switch
# Composite A (bull market, breadth>60%): V220A_WEIGHTS (momentum tilt)
# Composite B (shock/neutral, breadth<55%): V221_IC_WEIGHTS (fundamentals ×0.30)
# =============================================================================

class IcCompositeV222Scorer:
    """Phase 92 hybrid regime-conditional scorer (v222).

    Combines v221's reduced-fundamentals base with v220's breadth-based regime switch.
    - Bull market (breadth > 60%): Composite A = momentum tilt (V220A_WEIGHTS)
    - Shock/neutral (breadth < 55%): Composite B = v221 quality weights (V221_IC_WEIGHTS)
    - 5pp hysteresis deadband, EMA-smoothed breadth (span=5)
    """

    BREADTH_A_THRESHOLD = 0.60
    BREADTH_B_THRESHOLD = 0.55

    def __init__(self) -> None:
        self._fmp_fundamentals: Optional[pd.DataFrame] = None
        self._fmp_path = "data/fundamentals/fmp_fundamentals_history.parquet"
        self._in_momentum_regime: bool = False
        self._breadth_ema: Optional[float] = None

    def reset(self) -> None:
        """WF-R4: clear mutable per-fold state (see IcCompositeV220Scorer.reset)."""
        self._in_momentum_regime = False
        self._breadth_ema = None

    def get_state(self) -> tuple:
        return (self._in_momentum_regime, self._breadth_ema)

    def _get_fundamentals(self, as_of: pd.Timestamp) -> pd.DataFrame:
        if self._fmp_fundamentals is None:
            try:
                self._fmp_fundamentals = pd.read_parquet(self._fmp_path)
            except Exception:
                self._fmp_fundamentals = pd.DataFrame()
        df = self._fmp_fundamentals
        if df.empty:
            return df
        date_col = next((c for c in ("date", "report_date", "filed_date") if c in df.columns), None)
        sym_col = next((c for c in ("symbol", "ticker", "sym") if c in df.columns), None)
        if date_col and sym_col:
            try:
                filtered = df[pd.to_datetime(df[date_col]) <= as_of]
                if not filtered.empty:
                    return filtered.sort_values(date_col).groupby(sym_col).last()
            except Exception:
                pass
        return df

    def _update_breadth(self, raw_breadth: float) -> float:
        if not np.isfinite(raw_breadth):
            return self._breadth_ema if self._breadth_ema is not None else 0.5
        alpha = 2.0 / (5 + 1)
        if self._breadth_ema is None:
            self._breadth_ema = raw_breadth
        else:
            self._breadth_ema = alpha * raw_breadth + (1 - alpha) * self._breadth_ema
        return self._breadth_ema

    def __call__(self, day, symbols_data: dict, vix_history=None) -> list:
        _day_d = day.date() if hasattr(day, "date") else day
        as_of_ts = pd.Timestamp(day)

        close_cols = {}
        for sym, df in symbols_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            mask = df.index.date < _day_d if hasattr(df.index[0], "date") else df.index < as_of_ts
            _past_raw = df.loc[mask, "close"] if mask.any() else pd.Series(dtype=float)
            past = _past_raw.iloc[:, 0] if isinstance(_past_raw, pd.DataFrame) else _past_raw
            if sym in ("^VIX", "VIX"):
                if len(past) > 0:
                    close_cols[sym] = past
            elif len(past) >= 60:
                close_cols[sym] = past

        if not close_cols:
            return []

        closes = pd.DataFrame(close_cols)
        closes.index = pd.to_datetime(closes.index)

        raw_breadth = _compute_breadth(closes, as_of_ts)
        breadth_smooth = self._update_breadth(raw_breadth)

        if self._in_momentum_regime:
            if breadth_smooth < self.BREADTH_B_THRESHOLD:
                self._in_momentum_regime = False
        else:
            if breadth_smooth > self.BREADTH_A_THRESHOLD:
                self._in_momentum_regime = True

        weights = V220A_WEIGHTS if self._in_momentum_regime else V221_IC_WEIGHTS

        bars_past = {
            sym: symbols_data[sym].loc[
                (symbols_data[sym].index.date < _day_d)
                if hasattr(symbols_data[sym].index[0], "date")
                else (symbols_data[sym].index < as_of_ts)
            ]
            for sym in close_cols if sym in symbols_data and sym not in ("^VIX", "VIX")
        }

        fund = self._get_fundamentals(as_of_ts)
        scores = _compute_weighted_score(as_of_ts, closes, bars_past, weights, fundamentals=fund)

        if scores.empty:
            return []

        return sorted(
            [(sym, float(scores[sym])) for sym in scores.index if sym != "SPY"],
            key=lambda x: x[1],
            reverse=True,
        )

    @property
    def active_regime(self) -> str:
        return "momentum" if self._in_momentum_regime else "quality"


def _compute_weighted_score(
    as_of: pd.Timestamp,
    closes: pd.DataFrame,
    bars: dict,
    weights: dict[str, float],
    fundamentals: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Compute IC-weighted composite score with arbitrary weights dict.

    Shares feature computation logic with compute_v219_score but accepts
    an externally-provided weights dict (supports both v219 and v220A).
    """
    # Re-use compute_v219_score, then re-weight
    # This avoids duplicating feature computation logic.
    # compute_v219_score returns a Series; we recompute from raw features using the given weights.

    fund = fundamentals if fundamentals is not None else pd.DataFrame()
    raw_features: dict[str, dict[str, float]] = {k: {} for k in weights}

    as_of_ts = pd.Timestamp(as_of)

    # Extract VIX value for VRP computation (PIT-safe: use last available before as_of)
    _vix_val: Optional[float] = None
    for _vix_col in ("^VIX", "VIX"):
        if _vix_col in closes.columns:
            _past_vix = closes[_vix_col].dropna()
            _past_vix = _past_vix[_past_vix.index < as_of_ts]  # H-3: strict `<`
            if len(_past_vix) > 0:
                _vix_val = float(_past_vix.iloc[-1])
            break

    for sym in closes.columns:
        if sym in ("SPY", "^VIX", "VIX"):
            continue
        col = closes[sym].dropna()
        if len(col) < 60:
            continue
        mask = col.index < as_of_ts  # H-3: strict `<`
        past = col[mask]
        if len(past) < 60:
            continue
        idx = len(past)

        # momentum_252d_ex1m
        if "momentum_252d_ex1m" in raw_features and idx >= 252:
            c_now = float(past.iloc[-21]) if idx >= 21 else float(past.iloc[-1])
            c_start = float(past.iloc[-252])
            if c_start > 1e-9:
                raw_features["momentum_252d_ex1m"][sym] = (c_now / c_start) - 1.0

        # price_to_52w_high
        if "price_to_52w_high" in raw_features and idx >= 252:
            h52 = float(past.iloc[-252:].max())
            if h52 > 1e-9:
                raw_features["price_to_52w_high"][sym] = float(past.iloc[-1]) / h52

        # price_to_52w_low
        if "price_to_52w_low" in raw_features and idx >= 252:
            l52 = float(past.iloc[-252:].min())
            if l52 > 1e-9:
                raw_features["price_to_52w_low"][sym] = float(past.iloc[-1]) / l52

        # vol_regime
        if "vol_regime" in raw_features:
            vr = _vol_regime(past, idx)
            if not np.isnan(vr):
                raw_features["vol_regime"][sym] = -vr

        # ix_momentum_vol
        if "ix_momentum_vol" in raw_features:
            mom = raw_features.get("momentum_252d_ex1m", {}).get(sym, float("nan"))
            if not np.isnan(mom) and sym in raw_features.get("vol_regime", {}):
                raw_features["ix_momentum_vol"][sym] = mom * (1.0 + raw_features["vol_regime"][sym])

        # reversal_5d_vol_weighted
        if "reversal_5d_vol_weighted" in raw_features:
            rv = _reversal_5d_vw(bars.get(sym), idx)
            if not np.isnan(rv):
                raw_features["reversal_5d_vol_weighted"][sym] = rv

        # downtrend
        if "downtrend" in raw_features:
            dt = _downtrend_score(past, idx)
            if not np.isnan(dt):
                raw_features["downtrend"][sym] = dt

        # range_expansion — 5d ATR / 20d ATR (matches compute_v219_score definition)
        if "range_expansion" in raw_features and sym in bars:
            df_b = bars[sym]
            if df_b is not None and len(df_b) >= 20 and "high" in df_b.columns and "low" in df_b.columns:
                try:
                    tr = (df_b["high"] - df_b["low"]).abs()
                    atr5 = float(tr.iloc[-5:].mean())
                    atr20 = float(tr.iloc[-20:].mean())
                    if atr20 > 1e-9:
                        raw_features["range_expansion"][sym] = atr5 / atr20
                except Exception:
                    pass

        # volume_trend — 20d/60d ratio (matches compute_v219_score definition)
        if "volume_trend" in raw_features and sym in bars:
            df_b = bars[sym]
            if df_b is not None and "volume" in df_b.columns and len(df_b) >= 60:
                try:
                    vol_20 = float(df_b["volume"].iloc[-20:].mean())
                    vol_60 = float(df_b["volume"].iloc[-60:].mean())
                    if vol_60 > 1e-9:
                        raw_features["volume_trend"][sym] = vol_20 / vol_60
                except Exception:
                    pass

        # vrp (requires VIX)
        if "vrp" in raw_features and _vix_val is not None:
            vrp_val = _vrp_score(past, _vix_val, idx)
            if not np.isnan(vrp_val):
                raw_features["vrp"][sym] = vrp_val

        # mom_63d: 3-month price momentum (intermediate, captures recent trend persistence)
        if "mom_63d" in raw_features and idx >= 63:
            c_now = float(past.iloc[-1])
            c_start = float(past.iloc[-63])
            if c_start > 1e-9:
                raw_features["mom_63d"][sym] = (c_now / c_start) - 1.0

    # Fundamentals
    if not fund.empty:
        for feat, sign in [("profit_margin", 1), ("operating_margin", 1), ("pe_ratio", -1)]:
            if feat in raw_features and feat in fund.columns:
                for sym, val in fund[feat].items():
                    if sym in closes.columns and not np.isnan(val):
                        raw_features[feat][sym] = sign * val

    if not any(raw_features.values()):
        return pd.Series(dtype=float)

    score_df_parts: list[pd.Series] = []
    for feat, w in weights.items():
        if w <= 0:
            continue
        vals = pd.Series(raw_features.get(feat, {})).dropna()
        if len(vals) < 5:
            continue
        z = _zscore_cross(vals) * w
        score_df_parts.append(z)

    if not score_df_parts:
        return pd.Series(dtype=float)

    combined = pd.concat(score_df_parts, axis=1).sum(axis=1, min_count=1).dropna()
    return combined


# =============================================================================
# Phase 91 - v221 scorer: v219 with fundamentals down-weighted 70%
# =============================================================================
#
# WF-C1 FIX (2026-05-27): The original V221 weights below ("_IN_SAMPLE") were
# derived from a per-feature IC IR audit run over 2019-01-01 -> 2026-11-08 — a
# window that fully overlaps every WF test fold (F1 2022-11-21->2023-09-02,
# F2 2024-02-20->2024-12-01, F3 2025-05-21->2026-03-02). That is in-sample
# feature selection and inflates all reported fold Sharpes.
#
# Fix: recomputed the same h=20 IC IR from the existing daily-IC parquet
# (data/diagnostics/feature_ic/20260524T230508Z/daily_ic.parquet) using only
# dates <= 2021-04-26 (one trading day before fold 1's earliest train_end).
# Negative IRs are clipped to 0 (Phase 88 design rule). Fundamentals are
# down-weighted by 0.30 as in the original v221 design.
#
# Many features had near-zero or negative pre-fold IR, confirming the
# in-sample contamination of the original weights.
# =============================================================================

# Original (in-sample, contaminated) weights — kept for reference / reproducibility
_V221_IC_WEIGHTS_IN_SAMPLE: dict[str, float] = {
    "ix_momentum_vol":          2.3095,
    "momentum_252d_ex1m":       2.2334,
    "price_to_52w_low":         2.2233,
    "profit_margin":            1.2582 * 0.30,
    "operating_margin":         1.0727 * 0.30,
    "pe_ratio":                 0.8842 * 0.30,
    "price_to_52w_high":        0.9442,
    "volume_trend":             0.8840,
    "reversal_5d_vol_weighted": 0.6254,
    "downtrend":                0.4532,
    "vol_regime":               0.2386,
    "range_expansion":          0.1795,
    "vrp":                      0.0566,
}

# Pre-fold-1 IC IR weights (h=20, dates <= 2021-04-26, negatives clipped to 0,
# pe_ratio sign inverted per Phase 88 design, fundamentals * 0.30).
# Source: data/diagnostics/feature_ic/20260524T230508Z/daily_ic.parquet
_V221_IC_WEIGHTS_RAW: dict[str, float] = {
    "ix_momentum_vol":          0.0271,
    "momentum_252d_ex1m":       0.0102,
    "price_to_52w_low":         0.2449,
    "profit_margin":            0.0000 * 0.30,  # pre-fold IR was -0.16
    "operating_margin":         0.0000 * 0.30,  # pre-fold IR was -0.21
    "pe_ratio":                 0.0000 * 0.30,  # pre-fold IR was -0.004 after sign-flip
    "price_to_52w_high":        0.0000,         # pre-fold IR was -0.10
    "volume_trend":             0.0000,         # pre-fold IR was -0.12
    "reversal_5d_vol_weighted": 0.0817,
    "downtrend":                0.0840,
    "vol_regime":               0.0000,         # pre-fold IR was -0.09
    "range_expansion":          0.0000,         # pre-fold IR was -0.24
    "vrp":                      0.0000,         # pre-fold IR was -0.12
}
_V221_TOTAL = sum(_V221_IC_WEIGHTS_RAW.values())
V221_IC_WEIGHTS: dict[str, float] = (
    {k: v / _V221_TOTAL for k, v in _V221_IC_WEIGHTS_RAW.items()}
    if _V221_TOTAL > 0
    else {k: 0.0 for k in _V221_IC_WEIGHTS_RAW}
)


class IcCompositeV221Scorer:
    """IC composite v221 scorer.

    Args:
        fold_ic_weights: Optional dict overriding the module-level V221_IC_WEIGHTS
            (e.g. for per-fold IC weights computed at runtime on pre-tr_end data
            only — Option-A migration path for the WF-C1 in-sample bias fix).
            Must have the same key structure as V221_IC_WEIGHTS. If provided,
            the dict is normalized internally so callers can pass raw IR values.
    """

    def __init__(self, fold_ic_weights: Optional[dict] = None) -> None:
        self._fmp_fundamentals = None
        self._fmp_path = "data/fundamentals/fmp_fundamentals_history.parquet"
        if fold_ic_weights is not None and len(fold_ic_weights) > 0:
            _t = sum(max(v, 0.0) for v in fold_ic_weights.values())
            if _t > 0:
                self._weights = {k: max(v, 0.0) / _t for k, v in fold_ic_weights.items()}
            else:
                self._weights = dict(V221_IC_WEIGHTS)
        else:
            self._weights = V221_IC_WEIGHTS

    def _get_fundamentals(self, as_of: pd.Timestamp) -> pd.DataFrame:
        # BUG-9 fix: load raw parquet once, filter PIT per call (was using nonexistent load_pit_fundamentals)
        if self._fmp_fundamentals is None:
            try:
                self._fmp_fundamentals = pd.read_parquet(self._fmp_path)
            except Exception:
                self._fmp_fundamentals = pd.DataFrame()
        df = self._fmp_fundamentals
        if df.empty:
            return df
        date_col = next((c for c in ("date", "report_date", "filed_date") if c in df.columns), None)
        sym_col = next((c for c in ("symbol", "ticker", "sym") if c in df.columns), None)
        if date_col and sym_col:
            try:
                filtered = df[pd.to_datetime(df[date_col]) <= as_of]
                if not filtered.empty:
                    return filtered.sort_values(date_col).groupby(sym_col).last()
            except Exception:
                pass
        return df

    def __call__(self, day, symbols_data: dict, vix_history=None) -> list:
        _day_d = day.date() if hasattr(day, "date") else day
        as_of_ts = pd.Timestamp(day)

        close_cols = {}
        for sym, df in symbols_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            mask = df.index.date < _day_d if hasattr(df.index[0], "date") else df.index < as_of_ts
            _past_raw = df.loc[mask, "close"] if mask.any() else pd.Series(dtype=float)
            past = _past_raw.iloc[:, 0] if isinstance(_past_raw, pd.DataFrame) else _past_raw
            if sym in ("^VIX", "VIX"):
                if len(past) > 0:
                    close_cols[sym] = past  # BUG-8 fix: include VIX for VRP computation
            elif len(past) >= 60:
                close_cols[sym] = past

        if not close_cols:
            return []

        closes = pd.DataFrame(close_cols)
        closes.index = pd.to_datetime(closes.index)

        bars_past = {
            sym: symbols_data[sym].loc[
                (symbols_data[sym].index.date < _day_d)
                if hasattr(symbols_data[sym].index[0], "date")
                else (symbols_data[sym].index < as_of_ts)
            ]
            for sym in close_cols if sym in symbols_data and sym not in ("^VIX", "VIX")
        }

        fund = self._get_fundamentals(as_of_ts)
        scores = _compute_weighted_score(as_of_ts, closes, bars_past, self._weights, fundamentals=fund)

        if scores.empty:
            return []

        return sorted(
            [(sym, float(scores[sym])) for sym in scores.index if sym != "SPY"],
            key=lambda x: x[1],
            reverse=True,
        )


# =============================================================================
# Phase v224 — momentum-enhanced scorer: v221 + mom_63d, reduced contrarian
# Hypothesis: v221 underperforms in 2024 bull (AI/momentum tape) because
# contrarian features (reversal, downtrend) add noise in trending markets.
# Add 3-month momentum and reduce contrarian weights to capture growth/AI trend.
# =============================================================================

# WF-C1 FIX (R4, 2026-05-27): the original V224 weights (preserved as
# `_V224_IC_WEIGHTS_IN_SAMPLE`) used full-period IC IR (2019->2026), overlapping
# every WF test fold — in-sample feature selection. The replacement below uses
# pre-fold-1 IC IR (dates <= 2021-04-26, h=20, negatives clipped to 0,
# fundamentals * 0.30 as in V224's original design, contrarian features halved
# as in V224's original design). `mom_63d` is not present in daily_ic.parquet
# (only `momentum_60d` is) so its pre-fold weight is set to 0 per the R4 spec.
_V224_IC_WEIGHTS_IN_SAMPLE: dict[str, float] = {
    "ix_momentum_vol":          2.3095,
    "momentum_252d_ex1m":       2.2334,
    "mom_63d":                  1.8000,   # NEW: 3-month momentum (recent trend)
    "price_to_52w_low":         2.2233,
    "profit_margin":            1.2582 * 0.30,
    "operating_margin":         1.0727 * 0.30,
    "pe_ratio":                 0.8842 * 0.30,
    "price_to_52w_high":        0.9442,
    "volume_trend":             0.8840,
    "reversal_5d_vol_weighted": 0.6254 * 0.50,
    "downtrend":                0.4532 * 0.50,
    "vol_regime":               0.2386,
    "range_expansion":          0.1795,
    "vrp":                      0.0566,
}

_V224_IC_WEIGHTS_RAW: dict[str, float] = {
    "ix_momentum_vol":          0.0271,
    "momentum_252d_ex1m":       0.0102,
    "mom_63d":                  0.0000,   # not in daily_ic.parquet
    "price_to_52w_low":         0.2449,
    "profit_margin":            0.0000 * 0.30,   # pre-fold IR was -0.16
    "operating_margin":         0.0000 * 0.30,   # pre-fold IR was -0.21
    "pe_ratio":                 0.0041 * 0.30,
    "price_to_52w_high":        0.0000,          # pre-fold IR was -0.10
    "volume_trend":             0.0000,          # pre-fold IR was -0.12
    "reversal_5d_vol_weighted": 0.0817 * 0.50,
    "downtrend":                0.0840 * 0.50,
    "vol_regime":               0.0000,          # pre-fold IR was -0.09
    "range_expansion":          0.0000,          # pre-fold IR was -0.24
    "vrp":                      0.0000,          # pre-fold IR was -0.12
}
_V224_TOTAL = sum(_V224_IC_WEIGHTS_RAW.values())
V224_IC_WEIGHTS: dict[str, float] = (
    {k: v / _V224_TOTAL for k, v in _V224_IC_WEIGHTS_RAW.items()}
    if _V224_TOTAL > 0
    else {k: 0.0 for k in _V224_IC_WEIGHTS_RAW}
)


class IcCompositeV224Scorer:
    """IC composite v224: v221 + 3-month momentum + reduced contrarian features.

    Changes vs v221:
    - Added mom_63d (3-month momentum, weight ~1.80) — captures AI/growth trend persistence
    - Halved reversal_5d_vol_weighted and downtrend (contrarian signals hurt in trending bull)
    """

    def __init__(self) -> None:
        self._fmp_fundamentals = None
        self._fmp_path = "data/fundamentals/fmp_fundamentals_history.parquet"

    def _get_fundamentals(self, as_of: pd.Timestamp) -> pd.DataFrame:
        if self._fmp_fundamentals is None:
            try:
                self._fmp_fundamentals = pd.read_parquet(self._fmp_path)
            except Exception:
                self._fmp_fundamentals = pd.DataFrame()
        df = self._fmp_fundamentals
        if df.empty:
            return df
        date_col = next((c for c in ("date", "report_date", "filed_date") if c in df.columns), None)
        sym_col = next((c for c in ("symbol", "ticker", "sym") if c in df.columns), None)
        if date_col and sym_col:
            try:
                filtered = df[pd.to_datetime(df[date_col]) <= as_of]
                if not filtered.empty:
                    return filtered.sort_values(date_col).groupby(sym_col).last()
            except Exception:
                pass
        return pd.DataFrame()

    def __call__(self, day, symbols_data: dict, vix_history=None) -> list:
        as_of_ts = pd.Timestamp(day)
        _day_d = as_of_ts.date() if hasattr(as_of_ts, "date") else day

        # L-4 fix: enforce len(past) >= 60 guard consistent with V221 to avoid
        # computing factor scores on symbols with insufficient history (which can
        # produce noisy/degenerate factor values that distort the rank-IC composite).
        close_cols: dict = {}
        for sym, df in symbols_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            mask = df.index.date < _day_d if hasattr(df.index[0], "date") else df.index < as_of_ts
            _past_raw = df.loc[mask, "close"] if mask.any() else pd.Series(dtype=float)
            past = _past_raw.iloc[:, 0] if isinstance(_past_raw, pd.DataFrame) else _past_raw
            if sym in ("^VIX", "VIX"):
                if len(past) > 0:
                    close_cols[sym] = past
            elif sym == "SPY":
                continue
            elif len(past) >= 60:
                close_cols[sym] = past
        if not close_cols:
            return []
        closes = pd.DataFrame(close_cols)
        closes.index = pd.to_datetime(closes.index)
        bars_past = {
            sym: symbols_data[sym].loc[
                (symbols_data[sym].index.date < _day_d)
                if hasattr(symbols_data[sym].index[0], "date")
                else (symbols_data[sym].index < as_of_ts)
            ]
            for sym in close_cols if sym in symbols_data and sym not in ("^VIX", "VIX")
        }

        fund = self._get_fundamentals(as_of_ts)
        scores = _compute_weighted_score(as_of_ts, closes, bars_past, V224_IC_WEIGHTS, fundamentals=fund)

        if scores.empty:
            return []

        return sorted(
            [(sym, float(scores[sym])) for sym in scores.index if sym != "SPY"],
            key=lambda x: x[1],
            reverse=True,
        )
