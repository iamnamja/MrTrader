"""
Sleeve allocator — Alpha-v4 Phase 3.

Combines N validated sleeves (each a daily return series) into ONE book. This is
the architectural unlock of the portfolio-of-premia thesis: instead of hunting
one hero strategy, hold a few uncorrelated sleeves and weight them by risk, with
an OPTIONAL regime tilt.

Three schemes (each PIT, weekly-rebalanced, turnover-costed):
  * equal_capital      — 1/N constant (reference).
  * static vol-weight  — w_i ∝ 1/vol_i(t) (inverse-vol ≈ equal-risk-contribution
                         for low-correlation sleeves). THE baseline to beat.
  * regime-tilted      — static vol-weight × an A-PRIORI economic regime tilt
                         (e.g. PEAD up in BULL/risk-on, trend up in BEAR/defensive),
                         with persistence (regime must hold N days before the tilt
                         switches) + a continuous blend toward target weights
                         (no step changes -> no flicker / turnover spikes).

Discipline (the "attribute, don't amputate" / "regime must earn its complexity"
rule, §3 of the synthesis): the regime tilt is adopted ONLY if it beats static
vol-weight out-of-sample NET OF TURNOVER. Otherwise hold static vol-weight and
keep the tilt OFF. The tilt factors are economic priors declared BEFORE seeing
results — never fit to the sample.

POINT-IN-TIME: weights are computed from data through day t (trailing vol +
regime label at t) and applied to the t -> t+1 sleeve returns via shift(1).
Regime labels must themselves be PIT (the coarse3 map is — expanding-quantile VIX).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

ANN = 252


@dataclass
class AllocatorConfig:
    vol_lookback: int = 60                  # trailing window for per-sleeve realized vol
    rebalance_days: int = 5                 # weekly
    cost_bps: float = 1.0                   # allocator-level cost on sleeve-weight turnover
    vol_floor: float = 0.02                 # floor on sleeve vol (avoid blow-up weights)
    # A-PRIORI regime tilt: {regime_label: {sleeve: multiplier}}. None -> no tilt.
    # Economic priors: PEAD is risk-on (up in BULL, down in BEAR); trend is
    # defensive (up in BEAR, slight down in BULL). NEUTRAL = 1.0 (no tilt).
    regime_tilt: Optional[Dict[str, Dict[str, float]]] = None
    regime_persistence: int = 5             # regime must persist N days before tilt switches
    tilt_blend_days: int = 5                # blend weights toward target over N days (continuous)
    ann: int = ANN


# Default economic tilt (declared a-priori; PEAD risk-on, trend defensive).
DEFAULT_REGIME_TILT: Dict[str, Dict[str, float]] = {
    "BULL":    {"pead": 1.30, "trend": 0.80},
    "NEUTRAL": {"pead": 1.00, "trend": 1.00},
    "BEAR":    {"pead": 0.50, "trend": 1.30},
}


def _realized_vol(returns: pd.DataFrame, cfg: AllocatorConfig) -> pd.DataFrame:
    rv = returns.rolling(cfg.vol_lookback,
                         min_periods=max(cfg.vol_lookback // 2, 10)).std() * np.sqrt(cfg.ann)
    return rv.clip(lower=cfg.vol_floor)


def equal_capital_weights(returns: pd.DataFrame) -> pd.DataFrame:
    n = returns.shape[1]
    return pd.DataFrame(1.0 / n, index=returns.index, columns=returns.columns)


def vol_weights(returns: pd.DataFrame, cfg: AllocatorConfig) -> pd.DataFrame:
    """Inverse-vol (risk-parity) weights summing to 1, PIT.

    Warmup guard: the vol estimator needs MORE history (min_periods ~30) than the loader's
    min_deployed-days gate, so a sleeve still in vol-warmup has NaN vol. The old code set that
    sleeve's weight to 0 and handed the FULL budget to the other sleeve (doubling its exposure). On
    any row where a sleeve's vol is still NaN, fall back to EQUAL weights for that row instead."""
    rv = _realized_vol(returns, cfg)
    inv = 1.0 / rv
    w = inv.div(inv.sum(axis=1), axis=0)
    n = returns.shape[1]
    if n > 0:
        warmup_rows = rv.isna().any(axis=1)
        if warmup_rows.any():
            w.loc[warmup_rows, :] = 1.0 / n
    return w.fillna(0.0)


def _persist_regime(labels: pd.Series, n: int) -> pd.Series:
    """Hysteresis: only switch the ACTIVE regime once a new label has held for `n`
    consecutive days; otherwise keep the prior active regime. Kills flicker at
    regime boundaries (the boundary-thrash problem)."""
    lab = labels.fillna("NEUTRAL").to_numpy()
    out = lab.copy()
    active = lab[0]
    run_val, run_len = lab[0], 0
    for i in range(len(lab)):
        if lab[i] == run_val:
            run_len += 1
        else:
            run_val, run_len = lab[i], 1
        if run_len >= n:
            active = run_val
        out[i] = active
    return pd.Series(out, index=labels.index)


def apply_regime_tilt(base_weights: pd.DataFrame, regime_labels: pd.Series,
                      cfg: AllocatorConfig) -> pd.DataFrame:
    """Multiply base (vol) weights by the a-priori per-sleeve regime multiplier,
    renormalize to sum 1, then blend continuously toward the target over
    tilt_blend_days (EWMA-style) so the tilt never steps. PIT: uses the regime
    label at t (known at t)."""
    tilt = cfg.regime_tilt or DEFAULT_REGIME_TILT
    active = _persist_regime(regime_labels.reindex(base_weights.index), cfg.regime_persistence)
    # Build the per-day target multiplier matrix from the active regime.
    mult = pd.DataFrame(1.0, index=base_weights.index, columns=base_weights.columns)
    for reg, m in tilt.items():
        rows = (active == reg)
        for sleeve, factor in m.items():
            if sleeve in mult.columns:
                mult.loc[rows, sleeve] = factor
    target = base_weights * mult
    target = target.div(target.sum(axis=1), axis=0).fillna(0.0)
    # Continuous blend toward target (span = tilt_blend_days) so weights glide.
    blended = target.ewm(span=max(cfg.tilt_blend_days, 1), adjust=False).mean()
    return blended.div(blended.sum(axis=1), axis=0).fillna(0.0)


@dataclass
class BookResult:
    returns: pd.Series
    weights: pd.DataFrame
    sharpe: float
    cagr: float
    ann_vol: float
    max_drawdown: float
    calmar: float
    ann_turnover: float

    def summary(self) -> dict:
        return {"sharpe": self.sharpe, "cagr": self.cagr, "ann_vol": self.ann_vol,
                "max_drawdown": self.max_drawdown, "calmar": self.calmar,
                "ann_turnover": self.ann_turnover, "n_days": int(len(self.returns))}


def _maxdd(returns: pd.Series) -> float:
    eq = (1 + returns.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def combine(returns: pd.DataFrame, target_weights: pd.DataFrame,
            cfg: AllocatorConfig) -> BookResult:
    """Build the book return series from sleeve returns + target weights.

    Weekly rebalance (hold target between rebalances), PIT (held.shift(1) * returns),
    allocator-level turnover cost on sleeve-weight changes (shifted to align with
    the day the new weights start earning — same convention as the TSMOM sleeve)."""
    returns = returns.sort_index()
    target_weights = target_weights.reindex(returns.index).fillna(0.0)
    n = len(returns)
    is_rebal = (np.arange(n) % cfg.rebalance_days == 0)
    held = target_weights.where(pd.Series(is_rebal, index=returns.index), other=np.nan).ffill().fillna(0.0)

    dw = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    cost = dw * (cfg.cost_bps / 1e4)
    gross_ret = (held.shift(1) * returns).sum(axis=1)
    net = (gross_ret - cost.shift(1)).dropna()

    eq = (1 + net).cumprod()
    ann_vol = float(net.std() * np.sqrt(cfg.ann)) if net.std() > 0 else 0.0
    sharpe = float(net.mean() / net.std() * np.sqrt(cfg.ann)) if net.std() > 0 else 0.0
    yrs = max(len(net) / cfg.ann, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if len(eq) else 0.0
    mdd = _maxdd(net)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    ann_turnover = float(dw.reindex(net.index).sum() / yrs)
    return BookResult(returns=net, weights=held.reindex(net.index), sharpe=sharpe,
                      cagr=cagr, ann_vol=ann_vol, max_drawdown=mdd, calmar=calmar,
                      ann_turnover=ann_turnover)


def build_book(returns: pd.DataFrame, scheme: str = "vol",
               regime_labels: Optional[pd.Series] = None,
               cfg: Optional[AllocatorConfig] = None) -> BookResult:
    """Top-level: build a book under 'equal' | 'vol' | 'regime'."""
    cfg = cfg or AllocatorConfig()
    if scheme == "equal":
        w = equal_capital_weights(returns)
    elif scheme == "vol":
        w = vol_weights(returns, cfg)
    elif scheme == "regime":
        if regime_labels is None:
            raise ValueError("scheme='regime' requires regime_labels")
        w = apply_regime_tilt(vol_weights(returns, cfg), regime_labels, cfg)
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    return combine(returns, w, cfg)
