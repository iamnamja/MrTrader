"""R0.1 risk-policy v1 tests (app/live_trading/risk_policy.py).

It's a pure data artifact, but the drawdown-ladder logic and the frozen/conservative invariants are
load-bearing for the future risk gate, so pin them.
"""
from __future__ import annotations

import dataclasses

import pytest

from app.live_trading.risk_policy import RISK_POLICY_V1, RiskPolicy, POLICY_VERSION


def test_policy_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RISK_POLICY_V1.book_vol_target_launch = 0.5    # must not be mutable in place


def test_conservative_launch_invariants():
    p = RISK_POLICY_V1
    assert p.version == POLICY_VERSION
    # launch <= steady <= hard cap; all modest
    assert p.book_vol_target_launch <= p.book_vol_target_steady <= p.book_vol_hard_cap <= 0.12
    assert p.max_drawdown_budget <= 0.25
    assert p.margin_to_equity_hard <= 0.30
    assert p.ibkr_min_cash_reserve_frac > 0.0          # Alpaca cash can't fund IBKR margin
    assert p.ramp_steps[0] == 0.0 and p.ramp_steps[-1] == 1.0


def test_drawdown_ladder_picks_deepest_breached_rung():
    p = RISK_POLICY_V1
    assert p.ladder_multiplier(0.0) == 1.0
    assert p.ladder_multiplier(-0.05) == 1.0           # above the first rung
    assert p.ladder_multiplier(-0.08) == 0.75
    assert p.ladder_multiplier(-0.10) == 0.75          # between -8 and -12 -> still 0.75
    assert p.ladder_multiplier(-0.12) == 0.50
    assert p.ladder_multiplier(-0.17) == 0.25
    assert p.ladder_multiplier(-0.25) == 0.00          # past the kill line -> flat


def test_ladder_is_monotone_non_increasing():
    p = RISK_POLICY_V1
    prev = 1.0
    for dd in [0.0, -0.05, -0.08, -0.12, -0.16, -0.20, -0.30]:
        m = p.ladder_multiplier(dd)
        assert m <= prev + 1e-12                        # gross can only fall as drawdown deepens
        prev = m


def test_to_dict_roundtrips_version():
    d = RiskPolicy().to_dict()
    assert d["version"] == POLICY_VERSION and "drawdown_ladder" in d
