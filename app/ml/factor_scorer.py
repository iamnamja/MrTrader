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

    past = spy_closes[spy_closes.index <= as_of]
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
            past = df.loc[mask, "close"] if mask.any() else pd.Series(dtype=float)
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
                past_vix = vix_history[vix_history.index <= as_of]
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

# h20 IC IR weights, clipped to 0 for negatives (vol_percentile_52w = -0.13)
_V219_IC_WEIGHTS_RAW: dict[str, float] = {
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
_V219_TOTAL = sum(_V219_IC_WEIGHTS_RAW.values())
V219_IC_WEIGHTS: dict[str, float] = {k: v / _V219_TOTAL for k, v in _V219_IC_WEIGHTS_RAW.items()}


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
        mask = col.index <= as_of_ts
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
            sym_mask = df_sym.index <= as_of_ts
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

    def __init__(self, fmp_parquet: Optional[str] = None):
        self._fundamentals: Optional[pd.DataFrame] = None
        self._fmp_path = fmp_parquet or "data/fundamentals/fmp_fundamentals_history.parquet"

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
            past = df.loc[mask, "close"] if mask.any() else pd.Series(dtype=float)
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
                past_vix = vix_history[vix_history.index <= as_of_ts]
                if len(past_vix) > 0:
                    vix_val = float(past_vix.iloc[-1])
            except Exception:
                pass

        fund = self._get_fundamentals(as_of_ts)
        scores = compute_v219_score(as_of_ts, closes, bars_past, fundamentals=fund, vix_val=vix_val)

        if scores.empty:
            return []

        return sorted(
            [(sym, float(scores[sym])) for sym in scores.index if sym != "SPY"],
            key=lambda x: x[1],
            reverse=True,
        )
