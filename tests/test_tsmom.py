"""
Tests for the TSMOM trend sleeve (app/strategy/tsmom.py) — Alpha-v4 Phase 2.

Priority is POINT-IN-TIME correctness (the deep-dive's main worry for any new
backtester): a weight derived from data through day t must never depend on a
future price, and the backtest must earn day-t returns on weights known at t-1.
Plus the economic behaviors: long-flat trend direction, inverse-vol sizing,
gross cap, crisis rotation (equity down + bond up -> hold bond), and that costs
reduce returns.
"""
import numpy as np
import pandas as pd
import pytest

from app.strategy.tsmom import (
    TSMOMConfig, tsmom_signals, tsmom_weights, tsmom_backtest, realized_vol,
)


def _prices(paths: dict, n=400, start="2020-01-01"):
    idx = pd.date_range(start=start, periods=n, freq="B")
    return pd.DataFrame({k: v(np.arange(n)) for k, v in paths.items()}, index=idx)


def _cfg(**kw):
    base = dict(universe=["A", "B"], lookbacks=(21, 63), vol_lookback=30,
                rebalance_days=5, target_vol=0.10, max_weight=0.5, max_gross=1.0,
                cost_bps=2.0)
    base.update(kw)
    return TSMOMConfig(**base)


# ── PIT / no-look-ahead (the critical one) ────────────────────────────────────

def test_weights_do_not_depend_on_future_prices():
    """Changing a price at date T must leave ALL weights strictly before T
    unchanged — proves no future information leaks into past weights."""
    rng = np.random.default_rng(0)
    p = _prices({
        "A": lambda t: 100 * (1 + 0.0005 * t) + rng.normal(0, 1, len(t)).cumsum(),
        "B": lambda t: 100 * (1 - 0.0003 * t) + rng.normal(0, 1, len(t)).cumsum(),
    })
    cfg = _cfg()
    w1 = tsmom_weights(p, cfg)
    T = 300
    p2 = p.copy()
    p2.iloc[T, p2.columns.get_loc("A")] *= 1.25   # shock a FUTURE price
    w2 = tsmom_weights(p2, cfg)
    pd.testing.assert_frame_equal(w1.iloc[:T], w2.iloc[:T])


def test_backtest_return_uses_prior_day_weights():
    """Day-t portfolio return must use the weight held going INTO t (set at t-1),
    not the weight computed from t's own price. Construct a single asset that is
    flat then jumps on day t; the sleeve return on the jump day must reflect the
    pre-jump weight (which was 0 while flat), not a weight that 'saw' the jump."""
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    px = np.full(n, 100.0)
    px[120:] = 100.0          # flat through 120
    px[150] = 130.0           # one-day spike at 150 then back
    px[151:] = 100.0
    p = pd.DataFrame({"A": px, "B": np.full(n, 100.0)}, index=idx)
    # cost_bps=0 isolates the look-ahead check from the (legitimate) rebalance cost
    # that the spike-day weight change would otherwise charge.
    cfg = _cfg(universe=["A", "B"], rebalance_days=1, cost_bps=0.0)
    res = tsmom_backtest(p, cfg)
    # The flat-history weight going into day 150 is 0 (no trend), so the +30% jump
    # on day 150 must NOT be earned by the sleeve (gross return on 150 from A = 0).
    assert abs(res.returns.loc[idx[150]]) < 1e-9
    # The spike-aware weight (set at 150) instead applies to day 151's reversal:
    # the sleeve holds a long into the -23% drop -> day-151 return is negative.
    assert res.returns.loc[idx[151]] < 0.0


# ── signal direction (long-flat) ──────────────────────────────────────────────

def test_rising_series_signals_long_and_flat_drops_downtrend():
    p = _prices({
        "A": lambda t: 100 * (1.002 ** t),     # strong uptrend
        "B": lambda t: 100 * (0.998 ** t),     # strong downtrend
    })
    cfg = _cfg()
    sig = tsmom_signals(p, cfg).iloc[-1]
    assert sig["A"] > 0.9 and sig["B"] < -0.9   # ensemble agrees on direction
    w = tsmom_weights(p, cfg).iloc[-1]
    assert w["A"] > 0 and w["B"] == 0.0          # long-flat drops the downtrend


def test_allow_short_takes_negative_weight_on_downtrend():
    p = _prices({"A": lambda t: 100 * (1.002 ** t), "B": lambda t: 100 * (0.998 ** t)})
    w = tsmom_weights(p, _cfg(allow_short=True)).iloc[-1]
    assert w["B"] < 0    # shorting enabled -> short the downtrend


# ── sizing: inverse-vol + caps ────────────────────────────────────────────────

def test_inverse_vol_sizing_lower_weight_for_higher_vol():
    rng = np.random.default_rng(1)
    # Both uptrend, but A is ~3x more volatile than B -> A should get LESS weight.
    a = 100 * (1.001 ** np.arange(400)) + rng.normal(0, 3.0, 400).cumsum()
    b = 100 * (1.001 ** np.arange(400)) + rng.normal(0, 1.0, 400).cumsum()
    p = pd.DataFrame({"A": a, "B": b}, index=pd.date_range("2020-01-01", periods=400, freq="B"))
    cfg = _cfg(max_weight=10.0, max_gross=100.0)   # relax caps to isolate vol effect
    rv = realized_vol(p, cfg).iloc[-1]
    w = tsmom_weights(p, cfg).iloc[-1]
    if w["A"] > 0 and w["B"] > 0:
        assert (w["A"] < w["B"]) == (rv["A"] > rv["B"])   # higher vol -> lower weight


def test_gross_cap_respected():
    p = _prices({"A": lambda t: 100 * (1.002 ** t), "B": lambda t: 100 * (1.002 ** t)})
    cfg = _cfg(max_gross=0.6, max_weight=1.0)
    gross = tsmom_weights(p, cfg).abs().sum(axis=1)
    assert gross.max() <= 0.6 + 1e-9


# ── crisis rotation ───────────────────────────────────────────────────────────

def test_crisis_rotation_holds_bond_when_equity_falls():
    """Equity downtrends while a bond uptrends -> long-flat trend exits equity and
    holds the bond, so the sleeve is POSITIVE while equity crashes (crisis alpha)."""
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    eq = 100 * (1.0008 ** np.arange(n))
    eq[300:] = eq[299] * (0.99 ** np.arange(n - 300))    # equity crashes after 300
    bond = 100 * (1.0003 ** np.arange(n))
    bond[300:] = bond[299] * (1.004 ** np.arange(n - 300))  # bond rallies in the crash
    p = pd.DataFrame({"EQ": eq, "BD": bond}, index=idx)
    cfg = _cfg(universe=["EQ", "BD"], rebalance_days=5)
    res = tsmom_backtest(p, cfg)
    crash = res.returns[res.returns.index >= idx[320]]   # well into the crash (post-rebalance)
    assert crash.sum() > 0, "trend sleeve should be positive during the equity crash"


# ── costs ─────────────────────────────────────────────────────────────────────

def test_costs_reduce_returns():
    p = _prices({"A": lambda t: 100 * (1.001 ** t), "B": lambda t: 100 * (1.0005 ** t)})
    free = tsmom_backtest(p, _cfg(cost_bps=0.0)).returns.sum()
    costly = tsmom_backtest(p, _cfg(cost_bps=20.0)).returns.sum()
    assert costly < free


def test_initial_entry_cost_is_charged_not_orphaned():
    """Regression for the cost-timing fix: a single smooth uptrend takes ONE
    initial position (constant weight thereafter -> no further turnover), so the
    only cost is the initial entry. With the buggy alignment that cost lands on
    the dropna()'d first row and vanishes (free entry); with cost.shift(1) it is
    charged. So cost_bps=0 and cost_bps=100 must differ by ~the initial-entry cost,
    NOT be ~equal."""
    p = _prices({"A": lambda t: 100 * (1.001 ** t)})   # perfectly smooth -> constant weight
    cfg0 = _cfg(universe=["A"], cost_bps=0.0)
    cfg1 = _cfg(universe=["A"], cost_bps=100.0)         # 1% one-way
    free = tsmom_backtest(p, cfg0).returns.sum()
    costly = tsmom_backtest(p, cfg1).returns.sum()
    drag = free - costly
    assert drag > 0.002, f"initial-entry cost looks orphaned (drag={drag:.5f})"


def test_backtest_runs_and_reports_metrics():
    p = _prices({"A": lambda t: 100 * (1.001 ** t), "B": lambda t: 100 * (1.0005 ** t)})
    s = tsmom_backtest(p, _cfg()).summary()
    assert set(s) >= {"sharpe", "cagr", "ann_vol", "max_drawdown", "calmar", "avg_gross"}
    assert s["n_days"] > 100
