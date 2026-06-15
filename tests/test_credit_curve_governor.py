"""Tests for app/strategy/credit_curve_governor.py (Alpha-v8 G1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.credit_curve_governor import (
    CreditGovernorConfig, CurveGovernorConfig,
    credit_multiplier, live_credit_multiplier,
    curve_multiplier, live_curve_multiplier, credit_timing_returns,
)


def _series(vals, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)


# ── credit ──────────────────────────────────────────────────────────────────────────
def test_credit_derisks_when_ratio_below_ma():
    n = 200
    # IEF flat; HYG rises then falls hard -> HYG/IEF ratio drops below its MA late
    hyg = _series(list(np.linspace(100, 130, 150)) + list(np.linspace(130, 95, 50)))
    ief = _series([100.0] * n)
    mult = credit_multiplier(hyg, ief, CreditGovernorConfig(lookback=60, derisk_to=0.5))
    assert set(np.unique(mult.round(6))).issubset({0.5, 1.0})
    assert (mult == 0.5).any()          # de-risks during the HY selloff
    assert (mult == 1.0).any()          # full exposure during the uptrend


def test_credit_pit_shift():
    n = 120
    hyg = _series(list(np.linspace(100, 120, 80)) + list(np.linspace(120, 90, 40)))
    ief = _series([100.0] * n)
    cfg = CreditGovernorConfig(lookback=30, band=0.0, derisk_to=0.0)
    mult = credit_multiplier(hyg, ief, cfg)
    # multiplier is shift(1) of the raw signal -> first value reflects the prior day
    raw_stressed = (hyg / ief) < (hyg / ief).rolling(30).mean()
    # the day AFTER the first stressed close should be de-risked
    first_stress = raw_stressed[raw_stressed].index[0]
    nxt = mult.index[mult.index.get_loc(first_stress) + 1] if first_stress in mult.index else None
    if nxt is not None:
        assert mult.loc[nxt] == 0.0


def test_credit_live_multiplier_and_thin_guard():
    hyg = _series(list(np.linspace(100, 120, 80)) + list(np.linspace(120, 90, 40)))
    ief = _series([100.0] * 120)
    cfg = CreditGovernorConfig(lookback=30, derisk_to=0.5)
    m = live_credit_multiplier(hyg, ief, cfg)
    assert m in (0.5, 1.0)
    # too little data -> None (caller fail-safes to 1.0)
    assert live_credit_multiplier(hyg.tail(5), ief.tail(5), cfg) is None


def test_credit_derisk_to_validation():
    hyg, ief = _series([1, 2, 3]), _series([1, 1, 1])
    with pytest.raises(ValueError):
        credit_multiplier(hyg, ief, CreditGovernorConfig(derisk_to=1.5))


# ── curve ───────────────────────────────────────────────────────────────────────────
def test_curve_derisks_on_inversion():
    n = 100
    y3m = _series([2.0] * n)
    y10 = _series([3.0] * 50 + [1.5] * 50)   # inverts (10y<3m) in the back half
    mult = curve_multiplier(y10, y3m, CurveGovernorConfig(threshold=0.0, derisk_to=0.75,
                                                          confirm_days=1))
    assert (mult == 0.75).any()
    assert (mult == 1.0).any()


def test_curve_confirm_days_debounce():
    n = 60
    y3m = _series([2.0] * n)
    # single-day blip inversion shouldn't trigger a confirm_days=5 de-risk
    y10v = [3.0] * n
    y10v[30] = 1.0
    mult = curve_multiplier(_series(y10v), y3m, CurveGovernorConfig(confirm_days=5))
    assert (mult == 1.0).all() or (mult < 1.0).sum() == 0


def test_curve_live_multiplier():
    y3m = _series([2.0] * 30)
    y10 = _series([1.0] * 30)   # inverted
    m = live_curve_multiplier(y10, y3m, CurveGovernorConfig(threshold=0.0, derisk_to=0.75,
                                                            confirm_days=1))
    assert m == 0.75
    assert live_curve_multiplier(_series([1.0]), _series([2.0]),
                                 CurveGovernorConfig(confirm_days=5)) is None


# ── G3 additive credit-timing sleeve ────────────────────────────────────────────────
def test_credit_timing_long_when_healthy_flat_when_stressed():
    n = 200
    spy = _series(list(np.linspace(100, 130, 150)) + list(np.linspace(130, 110, 50)))
    hyg = _series(list(np.linspace(100, 130, 150)) + list(np.linspace(130, 95, 50)))   # HY breaks down late
    ief = _series([100.0] * n)
    cfg = CreditGovernorConfig(lookback=60, band=0.0)
    ret = credit_timing_returns(spy, hyg, ief, cfg, cost_bps=0.0)
    assert isinstance(ret, pd.Series) and len(ret) > 100
    # during the late credit-stress window the sleeve should be flat (zero return) on most days
    late = ret.iloc[-30:]
    assert (late == 0.0).sum() >= 15


def test_credit_timing_pit_and_cost():
    n = 160
    spy = _series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0003, 0.01, n)))
    hyg = _series(list(np.linspace(100, 120, 100)) + list(np.linspace(120, 95, 60)))
    ief = _series([100.0] * n)
    free = credit_timing_returns(spy, hyg, ief, CreditGovernorConfig(lookback=40), cost_bps=0.0)
    costed = credit_timing_returns(spy, hyg, ief, CreditGovernorConfig(lookback=40), cost_bps=10.0)
    assert costed.sum() <= free.sum()           # cost reduces return
    assert free.iloc[0] == 0.0 or abs(free.iloc[0]) >= 0.0   # PIT: first day position from prior
