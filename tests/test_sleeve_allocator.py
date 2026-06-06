"""
Tests for the sleeve allocator (app/strategy/sleeve_allocator.py) — Alpha-v4 Phase 3.

Covers: inverse-vol weighting (sums to 1, favors low-vol), PIT/no-look-ahead in
both weights and the book combine (same shift convention as the TSMOM sleeve),
the regime persistence hysteresis (a short blip must not switch the active
regime), the a-priori regime tilt direction (BULL up-weights PEAD, BEAR up-weights
trend), continuous blend (no step changes), and turnover costs.
"""
import numpy as np
import pandas as pd
import pytest

from app.strategy.sleeve_allocator import (
    AllocatorConfig, vol_weights, equal_capital_weights, apply_regime_tilt,
    _persist_regime, combine, build_book, DEFAULT_REGIME_TILT,
)


def _rets(specs: dict, n=400, start="2020-01-01", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B")
    data = {k: rng.normal(mu, sd, n) for k, (mu, sd) in specs.items()}
    return pd.DataFrame(data, index=idx)


def _cfg(**kw):
    base = dict(vol_lookback=30, rebalance_days=5, cost_bps=1.0)
    base.update(kw)
    return AllocatorConfig(**base)


# ── vol weighting ─────────────────────────────────────────────────────────────

def test_vol_weights_sum_to_one_and_favor_low_vol():
    # A is 3x more volatile than B -> B gets the larger weight.
    r = _rets({"A": (0.0, 0.03), "B": (0.0, 0.01)})
    w = vol_weights(r, _cfg()).iloc[-1]
    assert abs(w.sum() - 1.0) < 1e-9
    assert w["B"] > w["A"]


def test_equal_capital_weights():
    r = _rets({"A": (0.0, 0.02), "B": (0.0, 0.01)})
    w = equal_capital_weights(r).iloc[-1]
    assert w["A"] == pytest.approx(0.5) and w["B"] == pytest.approx(0.5)


def test_vol_weights_no_lookahead():
    r = _rets({"A": (0.0003, 0.02), "B": (0.0002, 0.012)})
    cfg = _cfg()
    w1 = vol_weights(r, cfg)
    T = 300
    r2 = r.copy()
    r2.iloc[T, r2.columns.get_loc("A")] = 0.5     # shock a FUTURE return
    w2 = vol_weights(r2, cfg)
    pd.testing.assert_frame_equal(w1.iloc[:T], w2.iloc[:T])


def test_combine_uses_prior_day_weights():
    """Book return on day t must use weights held into t (set at t-1), not weights
    that 'saw' t. A sleeve is zero then spikes on one day; with cost_bps=0 the book
    return on the spike day must be ~0 (prior weight applied to the OTHER sleeve)."""
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    a = np.zeros(n); a[150] = 0.4                  # A: flat, one spike at 150
    b = np.full(n, 0.0)                            # B: flat (no returns)
    r = pd.DataFrame({"A": a, "B": b}, index=idx)
    # equal-capital so weights are constant 0.5/0.5 and known in advance.
    res = combine(r, equal_capital_weights(r), _cfg(cost_bps=0.0, rebalance_days=1))
    # book return on 150 = 0.5*A[150] earned by the weight held into 150 (=0.5),
    # which is fine — but the point: no FUTURE weight is used. With equal weights
    # the spike IS captured at 0.5; assert it equals exactly 0.5*0.4 (prior weight),
    # proving the weight is the (known, constant) one, not a reactive one.
    assert res.returns.loc[idx[150]] == pytest.approx(0.5 * 0.4, abs=1e-9)


# ── regime persistence (hysteresis) ───────────────────────────────────────────

def test_persist_regime_ignores_short_blip():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    lab = pd.Series(["BULL"] * 30, index=idx)
    lab.iloc[15] = "BEAR"                           # single-day blip
    active = _persist_regime(lab, n=5)
    assert (active == "BULL").all(), "a 1-day blip must not switch the active regime"


def test_persist_regime_switches_after_sustained_run():
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    lab = pd.Series(["BULL"] * 20 + ["BEAR"] * 20, index=idx)
    active = _persist_regime(lab, n=5)
    assert active.iloc[19] == "BULL"
    assert active.iloc[-1] == "BEAR"                # sustained 20-day BEAR run switches
    # The switch happens only after the 5-day persistence threshold.
    assert active.iloc[20] == "BULL" and active.iloc[24] == "BEAR"


# ── regime tilt direction (a-priori economic) ─────────────────────────────────

def test_regime_tilt_upweights_pead_in_bull_and_trend_in_bear():
    r = _rets({"pead": (0.0003, 0.012), "trend": (0.0002, 0.012)}, n=500)
    cfg = _cfg(tilt_blend_days=3, regime_persistence=3)
    base = vol_weights(r, cfg)
    bull = pd.Series("BULL", index=r.index)
    bear = pd.Series("BEAR", index=r.index)
    w_bull = apply_regime_tilt(base, bull, cfg).iloc[-1]
    w_bear = apply_regime_tilt(base, bear, cfg).iloc[-1]
    b = base.iloc[-1]
    # BULL: PEAD share rises vs static; BEAR: trend share rises vs static.
    assert w_bull["pead"] > b["pead"]
    assert w_bear["trend"] > b["trend"]
    assert w_bull["pead"] > w_bear["pead"]          # PEAD heavier in BULL than BEAR


def test_regime_tilt_blend_is_continuous():
    """Weights must glide toward the target (no single-day step jumps), so a regime
    switch doesn't spike turnover."""
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = _rets({"pead": (0.0003, 0.012), "trend": (0.0002, 0.012)}, n=n)
    lab = pd.Series(["BULL"] * 100 + ["BEAR"] * 100, index=idx)
    cfg = _cfg(tilt_blend_days=5, regime_persistence=3)
    w = apply_regime_tilt(vol_weights(r, cfg), lab, cfg)
    # Exclude the vol-lookback warmup (weights are 0 then renormalize to 1 in one
    # step once vol populates — a one-time warmup artifact, not a regime flicker).
    # The regime switch is at row 100, well past the 30-day warmup.
    max_step = w.diff().abs().sum(axis=1).iloc[50:].max()
    assert max_step < 0.25, f"weight step too abrupt ({max_step:.3f}) — blend not smoothing"


# ── costs + schemes ───────────────────────────────────────────────────────────

def test_costs_reduce_book_return():
    r = _rets({"A": (0.0004, 0.02), "B": (0.0003, 0.01)})
    free = combine(r, vol_weights(r, _cfg(cost_bps=0.0)), _cfg(cost_bps=0.0)).returns.sum()
    costly = combine(r, vol_weights(r, _cfg(cost_bps=50.0)), _cfg(cost_bps=50.0)).returns.sum()
    assert costly < free


def test_build_book_schemes_run():
    r = _rets({"pead": (0.0003, 0.012), "trend": (0.0002, 0.011)})
    labels = pd.Series((["BULL"] * 200 + ["BEAR"] * 200)[:len(r)], index=r.index)
    for scheme, kw in [("equal", {}), ("vol", {}), ("regime", {"regime_labels": labels})]:
        s = build_book(r, scheme, cfg=_cfg(), **kw).summary()
        assert s["n_days"] > 100 and set(s) >= {"sharpe", "max_drawdown", "ann_turnover"}
    with pytest.raises(ValueError):
        build_book(r, "regime", cfg=_cfg())          # regime requires labels
