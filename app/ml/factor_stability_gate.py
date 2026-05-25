"""
Phase 89: Cross-sectional factor stability regime gate.

Measures whether the IC composite scores are currently predictive by computing
rolling Spearman rank IC between frozen rebalance scores and realized 20d forward
returns. When the rolling IC falls below threshold, factor leadership is rotating
and the gate reduces gross exposure.

Designed to catch Fold-1 (Jun 2021-May 2022 bear transition) and Fold-4
(Jun 2024-May 2025 post-election/tariff rotation) where SPY+VIX gate missed
because SPY stayed above MA200 during most of both windows.

Parameters:
    lookback_days (int): Rolling window for averaging daily ICs. Default 63 (one quarter).
    ic_threshold (float): IC boundary for on/off. Default 0.02.
    fwd_window (int): Forward return horizon matching rebalance cadence. Default 20.
    hysteresis_days (int): Consecutive days before gate state changes. Default 5.
"""
from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_factor_stability_gate(
    scores_at_rebalance: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    lookback_days: int = 63,
    ic_threshold: float = 0.02,
    fwd_window: int = 20,
    hysteresis_days: int = 5,
) -> pd.DataFrame:
    """
    Compute the factor stability gate series from frozen rebalance scores and prices.

    Args:
        scores_at_rebalance: DataFrame index=trading_date, cols=symbol.
            Each row is the IC composite score frozen at the last rebalance,
            forward-filled between rebalance dates. Must NOT be recomputed daily.
        prices: DataFrame index=trading_date, cols=symbol (adjusted close).
        lookback_days: Rolling window over daily ICs.
        ic_threshold: Boundary for gate state: IC > +threshold → 1.0,
            IC in [-threshold, +threshold] → 0.5, IC < -threshold → 0.0.
        fwd_window: Forward return horizon in trading days (should match rebalance cadence).
        hysteresis_days: Days a new gate value must persist before it takes effect.

    Returns:
        DataFrame with columns: daily_ic, realized_ic, raw_gate, gate.
        The 'gate' column is what gets multiplied into gross exposure.

    Note on look-ahead safety:
        realized_ic is shifted by fwd_window so only fully-realized IC periods
        contribute. The gate at rebalance day T uses only IC from periods ending
        at or before T-fwd_window.
    """
    # Forward return: return from t to t+fwd_window (shift back by fwd_window to align)
    fwd_ret = prices.pct_change(fwd_window).shift(-fwd_window)

    common_idx = scores_at_rebalance.index.intersection(fwd_ret.index)
    common_cols = scores_at_rebalance.columns.intersection(fwd_ret.columns)
    if len(common_cols) < 50:
        raise ValueError(
            f"Only {len(common_cols)} common symbols between scores and prices; need ≥50"
        )

    s = scores_at_rebalance.loc[common_idx, common_cols]
    r = fwd_ret.loc[common_idx, common_cols]

    # Daily cross-sectional Spearman rank IC
    daily_ic = pd.Series(index=common_idx, dtype=float, name="daily_ic")
    for dt in common_idx:
        sv = s.loc[dt].values
        rv = r.loc[dt].values
        mask = np.isfinite(sv) & np.isfinite(rv)
        if mask.sum() < 50:
            continue
        rho, _ = spearmanr(sv[mask], rv[mask])
        daily_ic.loc[dt] = rho

    # Shift by fwd_window to ensure no look-ahead: at day T we only know ICs
    # whose forward return window ended before T.
    realized_ic = (
        daily_ic
        .shift(fwd_window)
        .rolling(lookback_days, min_periods=lookback_days // 2)
        .mean()
    )

    # Three-state gate: 1.0 (on), 0.5 (reduced), 0.0 (off)
    raw_gate = pd.Series(1.0, index=realized_ic.index, name="raw_gate")
    raw_gate[realized_ic < -ic_threshold] = 0.0
    raw_gate[(realized_ic >= -ic_threshold) & (realized_ic <= ic_threshold)] = 0.5
    # Burn-in: insufficient history → default to full exposure
    raw_gate[realized_ic.isna()] = 1.0

    # Hysteresis: gate changes only after hysteresis_days consecutive agreement
    gate_vals = raw_gate.values.copy()
    current = 1.0
    streak_val = current
    streak_n = 0
    for i, v in enumerate(gate_vals):
        if v == streak_val:
            streak_n += 1
        else:
            streak_val = v
            streak_n = 1
        if streak_n >= hysteresis_days:
            current = streak_val
        gate_vals[i] = current

    gate = pd.Series(gate_vals, index=realized_ic.index, name="gate")

    return pd.DataFrame({
        "daily_ic": daily_ic,
        "realized_ic": realized_ic,
        "raw_gate": raw_gate,
        "gate": gate,
    })


class FactorStabilityGate:
    """
    Callable gate used in the rebalance loop.

    Usage:
        gate = FactorStabilityGate(scores_df, prices_df)
        mult = gate(rebalance_date)  # returns 0.0, 0.5, or 1.0
    """

    def __init__(
        self,
        scores_at_rebalance: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        lookback_days: int = 63,
        ic_threshold: float = 0.02,
        fwd_window: int = 20,
        hysteresis_days: int = 5,
    ) -> None:
        self._gate_series = compute_factor_stability_gate(
            scores_at_rebalance,
            prices,
            lookback_days=lookback_days,
            ic_threshold=ic_threshold,
            fwd_window=fwd_window,
            hysteresis_days=hysteresis_days,
        )["gate"]

    def __call__(self, day: date) -> float:
        """Return gross exposure multiplier for the given rebalance date."""
        ts = pd.Timestamp(day)
        # PIT: use gate value strictly before this day
        history = self._gate_series[self._gate_series.index < ts]
        if history.empty:
            return 1.0
        return float(history.iloc[-1])

    @property
    def gate_series(self) -> pd.Series:
        return self._gate_series
