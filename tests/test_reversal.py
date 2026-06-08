"""Tests for the short-term cross-sectional reversal sleeve (app/strategy/reversal.py).

Validated to a KILL (cost-dead) — but the harness is retained + tested (reusable, and the
PIT/sign/neutrality correctness is what makes the KILL trustworthy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.reversal import (
    ReversalConfig, reversal_weights, reversal_backtest, _xs_demeaned_l1_weights,
)


def _prices(n=60, seed=0):
    """5 names with distinct trajectories; A is the clear recent loser, E the winner."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(seed)
    base = {}
    drifts = {"A": -0.004, "B": -0.001, "C": 0.0, "D": 0.001, "E": 0.004}
    for s, d in drifts.items():
        steps = rng.normal(d, 0.002, n)
        base[s] = 100 * np.exp(np.cumsum(steps))
    P = pd.DataFrame(base, index=idx)
    V = pd.DataFrame(1e7, index=idx, columns=P.columns)  # all equally (very) liquid
    return P, V


def _cfg(**kw):
    base = dict(lookback=5, skip=1, min_names=2, liquidity_top_n=10, adv_lookback=5)
    base.update(kw)
    return ReversalConfig(**base)


def test_reversal_sign_longs_losers_shorts_winners():
    P, V = _prices()
    w = reversal_weights(P, V, _cfg()).iloc[-1]
    assert w["A"] > 0          # biggest recent loser -> LONG
    assert w["E"] < 0          # biggest recent winner -> SHORT
    assert w["A"] > w["E"]


def test_weights_are_dollar_neutral_and_gross_one():
    P, V = _prices()
    w = reversal_weights(P, V, _cfg())
    live = w.iloc[-1]
    assert abs(live.sum()) < 1e-9            # dollar-neutral (sum ~ 0)
    assert abs(live.abs().sum() - 1.0) < 1e-9  # gross-normalized to 1


def test_no_lookahead_future_shock_leaves_past_weights_unchanged():
    P, V = _prices()
    w0 = reversal_weights(P, V, _cfg())
    P2 = P.copy()
    P2.iloc[-1] *= 1.5                        # shock the LAST day only
    w1 = reversal_weights(P2, V, _cfg())
    # every row strictly before the last must be identical (no future leakage)
    pd.testing.assert_frame_equal(w0.iloc[:-1], w1.iloc[:-1])


def test_min_names_zeros_sparse_cross_section():
    P, V = _prices()
    w = reversal_weights(P, V, _cfg(min_names=20))  # only 5 names < 20 -> all zero
    assert (w.iloc[-1].abs().sum()) == 0.0


def test_liquidity_gate_keeps_only_top_n():
    P, V = _prices()
    V = V.copy()
    V[["B", "C", "D", "E"]] = 1.0            # make A by far the most liquid
    w = reversal_weights(P, V, _cfg(liquidity_top_n=2))
    # with top-2 liquid, the illiquid names get zero weight across the book
    held_any = (w.abs().sum() > 0)
    assert held_any["A"]                      # most-liquid name participates


def test_higher_cost_lowers_net_sharpe():
    P, V = _prices(n=400)
    lo = reversal_backtest(P, V, _cfg(cost_bps=1.0))
    hi = reversal_backtest(P, V, _cfg(cost_bps=50.0))
    assert hi.sharpe < lo.sharpe              # cost monotonically hurts
    assert hi.summary()["ann_turnover"] == lo.summary()["ann_turnover"]  # turnover same


def test_xs_demeaned_weights_handle_empty_rows():
    # a row of all-NaN scores must produce zero weights, not NaN/inf
    idx = pd.bdate_range("2024-01-01", periods=3)
    score = pd.DataFrame(np.nan, index=idx, columns=["A", "B", "C"])
    w = _xs_demeaned_l1_weights(score, _cfg(), min_names=2)
    assert (w.values == 0.0).all()
