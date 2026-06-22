"""Option A — cross-sectional sector-ETF rotation sleeve (app/research/etf_rotation.py).

Pins: 12-1 momentum is PIT, top-K selection, the dual-momentum cash filter, weight normalisation,
and warmup trimming. Uses deterministic synthetic price panels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research import etf_rotation as er


def _panel(drifts, n=1500, sd=0.012, seed=0, start="2015-01-01"):
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"E{i}": 100 * np.cumprod(1 + rng.normal(d, sd, n)) for i, d in enumerate(drifts)},
        index=idx)


def _det_panel(drifts, n=1200, start="2015-01-01"):
    # deterministic monotonic prices (no noise) → momentum ordering is EXACT, for selection tests
    idx = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame({f"E{i}": 100 * (1 + d) ** np.arange(n) for i, d in enumerate(drifts)},
                        index=idx)


# ── signal PIT ────────────────────────────────────────────────────────────────
def test_rotation_signal_is_pit_and_correct():
    p = _panel([0.0005, 0.0003], n=400)
    cfg = er.RotationConfig(lookback=252, skip=21)
    mom = er.rotation_signal(p, cfg)
    # warmup (< lookback) is NaN — no future peeking via a short window
    assert mom.iloc[:cfg.lookback].isna().all().all()
    # a known row equals price[t-skip]/price[t-lookback]-1 exactly (uses only past prices)
    t = 300
    expected = p["E0"].iloc[t - cfg.skip] / p["E0"].iloc[t - cfg.lookback] - 1.0
    assert mom["E0"].iloc[t] == pytest.approx(expected)


# ── top-K selection ───────────────────────────────────────────────────────────
def test_selects_top_k_by_momentum():
    # deterministic monotonic prices, strictly ordered drift → top-2 by momentum are exactly E0,E1
    p = _det_panel([0.0010, 0.0008, 0.0004, 0.0001, -0.0002])
    w = er.rotation_target_weights(p, er.RotationConfig(top_k=2, weight="equal", dual_momentum=False))
    row = w.iloc[-1]
    held = set(row[row > 0].index)
    assert held == {"E0", "E1"}                       # the two strongest
    assert row.sum() == pytest.approx(1.0)            # weights renormalise to 1


def test_equal_weight_splits_evenly():
    p = _panel([0.0010, 0.0008, 0.0004], n=1000)
    w = er.rotation_target_weights(p, er.RotationConfig(top_k=2, weight="equal", dual_momentum=False))
    row = w.iloc[-1]
    nz = row[row > 0]
    assert len(nz) == 2 and np.allclose(nz.values, 0.5)


# ── dual-momentum cash filter ─────────────────────────────────────────────────
def test_dual_momentum_goes_to_cash_when_all_negative():
    # all ETFs decline → 12-1 momentum negative everywhere → dual-mom holds NOTHING (cash)
    p = _panel([-0.0008, -0.0006, -0.0004], n=1000, sd=0.004)
    w_dual = er.rotation_target_weights(p, er.RotationConfig(top_k=2, dual_momentum=True))
    w_rel = er.rotation_target_weights(p, er.RotationConfig(top_k=2, dual_momentum=False))
    post = w_dual.iloc[400:]                           # well past warmup
    assert (post.sum(axis=1) == 0).all()              # dual-momentum → fully in cash
    assert (w_rel.iloc[400:].sum(axis=1) > 0).any()   # relative-only still holds the "least bad"


# ── backtest output + warmup trim ─────────────────────────────────────────────
def test_backtest_trims_warmup_and_reports_stats():
    p = _panel([0.0009, 0.0006, 0.0003, 0.0, -0.0003], n=1500)
    res = er.rotation_backtest(p, er.RotationConfig(top_k=2))
    # warmup (~lookback=252d, all-cash since momentum is NaN) trimmed → series starts well in
    assert res.returns.index.min() > p.index[200]
    assert res.returns.ne(0).any()
    assert np.isfinite(res.sharpe) and res.ann_vol > 0
    assert 0.0 <= res.cash_fraction <= 1.0 and res.avg_n_held <= 2.0


def test_degenerate_panel_no_crash():
    flat = pd.DataFrame({"A": [100.0] * 400, "B": [100.0] * 400},
                        index=pd.bdate_range("2015-01-01", periods=400))
    res = er.rotation_backtest(flat, er.RotationConfig(top_k=1))
    assert res.sharpe == 0.0                           # no edge, no crash


# ── backtest-level PIT: weights earn NEXT day, perturbing the future can't change the past ──
def test_future_perturbation_does_not_change_past_returns():
    p = _panel([0.0008, 0.0005, 0.0002], n=1000, seed=3)
    base = er.rotation_backtest(p, er.RotationConfig(top_k=2)).returns
    p2 = p.copy()
    p2.iloc[900:] *= 5.0                                # spike the FUTURE
    pert = er.rotation_backtest(p2, er.RotationConfig(top_k=2)).returns
    cut = p.index[895]                                  # strictly before the spike
    common = base.index.intersection(pert.index)
    common = common[common < cut]
    assert len(common) > 100
    assert np.allclose(base.loc[common].values, pert.loc[common].values)   # past is byte-invariant


# ── ragged history: a late-joining ETF is excluded until it has its own 252d ──
def test_ragged_history_late_joiner_excluded_until_present():
    a = _det_panel([0.0006], n=1200)                   # E0 full history
    b = _det_panel([0.0010], n=1200)
    b.columns = ["E1"]
    b.iloc[:600] = np.nan                              # E1 joins ~row 600
    p = pd.concat([a, b], axis=1)
    w = er.rotation_target_weights(p, er.RotationConfig(top_k=1, weight="equal", dual_momentum=False))
    assert w["E1"].iloc[700] == 0.0                    # < its own 252d history → not selectable
    assert w["E1"].iloc[-1] > 0.0                      # once it has history, its momentum selects it
    res = er.rotation_backtest(p, er.RotationConfig(top_k=1, weight="equal"))
    assert np.isfinite(res.returns.values).all()       # NaN pre-history never corrupts the book


# ── vol floor: a zero-vol strong-momentum name must NOT blow up the row to cash (the MAJOR fix) ──
def test_vol_floor_prevents_all_cash_on_zero_vol_name():
    idx = pd.bdate_range("2015-01-01", periods=1000)
    e0 = 100 * (1.0005) ** np.arange(1000)             # monotone up, ZERO vol → 1/0=inf w/o floor
    rng = np.random.default_rng(0)
    e1 = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, 1000))
    e2 = 100 * np.cumprod(1 + rng.normal(0.0001, 0.01, 1000))
    p = pd.DataFrame({"E0": e0, "E1": e1, "E2": e2}, index=idx)
    row = er.rotation_target_weights(p, er.RotationConfig(top_k=2, weight="inverse_vol")).iloc[-1]
    assert np.isfinite(row.values).all()
    assert row.sum() == pytest.approx(1.0)             # NOT flushed to cash by the inf weight
    assert row["E0"] > 0                               # the zero-vol winner held with a capped weight
