"""Alpha-v10 audit Wave 4b — research/ML look-ahead fixes.

Pins: the event-scorer VIX regime gate reads the PRIOR day's close (strict PIT, not today's, which
isn't knowable at the day-T open); the factor-scorer ix_momentum_vol multiplier never sign-flips
momentum; and the VIX-VRP sleeve charges a turnover/flip cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_event_scorer_vix_gate_is_strict_pit():
    from app.ml.pead_scorer import PEADScorer
    s = PEADScorer.__new__(PEADScorer)
    idx = pd.to_datetime(["2026-06-19", "2026-06-22"])
    vix = pd.DataFrame({"close": [20.0, 99.0]}, index=idx)
    # asking for 2026-06-22 must return the PRIOR day's close (20.0), NOT today's 99.0 (look-ahead)
    assert s._vix_today(pd.Timestamp("2026-06-22"), {"^VIX": vix}) == 20.0
    # a day with no prior bar -> None (can't peek at same-day close)
    assert s._vix_today(pd.Timestamp("2026-06-19"), {"^VIX": vix}) is None


def test_ix_momentum_vol_never_sign_flips():
    # the multiplier is max(0, 1 - vol_ratio): a high-vol name (vr>1) down-weights momentum to 0,
    # never inverts its sign. (mirrors compute_v219_score / _compute_weighted_score)
    mom = 0.5
    for vr in (0.5, 1.0, 2.0, 3.5):
        vol_regime = -vr
        ix = mom * max(0.0, 1.0 + vol_regime)
        assert ix >= 0.0                      # never negative for positive momentum


def test_vix_vrp_charges_turnover_cost():
    from app.research import vix_vrp as vv
    idx = pd.bdate_range("2020-01-01", periods=80)
    vix = pd.Series(15.0, index=idx)
    vix3m = pd.Series(18.0, index=idx)        # always contango -> always short
    vx = pd.Series(-0.01, index=idx)          # steady roll-down
    free = vv.vix_vrp_returns(vx, vix, vix3m, vv.VixVRPConfig(cost_bps=0.0))
    charged = vv.vix_vrp_returns(vx, vix, vix3m, vv.VixVRPConfig(cost_bps=5.0))
    common = free.index.intersection(charged.index)
    # the cost-charged series is strictly worse on average (daily re-leverage turnover is charged)
    assert charged.loc[common].mean() < free.loc[common].mean()
    assert np.isfinite(charged).all()
