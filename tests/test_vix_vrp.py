"""P3.1 — VIX-curve VRP tests (gate logic + PIT + crash-flat behavior)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import vix_vrp as vv


def test_contango_gate_aligns_and_flags():
    idx = pd.bdate_range("2020-01-01", periods=5)
    vix = pd.Series([15, 16, 30, 28, 14], index=idx)
    vix3m = pd.Series([18, 18, 25, 26, 17], index=idx)        # day 2/3 backwardation (vix>vix3m)
    g = vv.contango_gate(vix, vix3m)
    assert list(g) == [True, True, False, False, True]


def test_short_in_contango_flat_in_backwardation_and_pit():
    idx = pd.bdate_range("2020-01-01", periods=6)
    # contango days 0-2, backwardation day 3 (a vol spike), then contango
    vix = pd.Series([15, 15, 15, 35, 15, 15], index=idx)
    vix3m = pd.Series([18, 18, 18, 25, 18, 18], index=idx)
    # VX falls in contango (roll-down) then SPIKES +40% on the backwardation day
    vx = pd.Series([0.0, -0.02, -0.02, 0.40, -0.02, -0.02], index=idx)
    cfg = vv.VixVRPConfig(vol_lookback=2, max_leverage=1.0)
    r = vv.vix_vrp_returns(vx, vix, vix3m, cfg)
    # position = -gate.shift(1): the +40% spike day is day 3; gate[day3]=False (backwardation),
    # but the position THAT day is from day2's gate (contango) -> short -> takes part of the spike;
    # crucially day4 (after the spike) is FLAT (gate[day3]=False shifted) -> no continued bleed.
    # Assert the post-spike day is ~flat (gate correctly de-risked) and the series is finite.
    assert np.isfinite(r).all()
    # the day AFTER the backwardation onset must be flat (no short into a known vol spike)
    assert abs(r.loc[idx[4]]) < 1e-9


def test_short_harvests_contango_rolldown():
    # pure contango, VX drifts down every day -> short-vol makes money
    idx = pd.bdate_range("2020-01-01", periods=120)
    vix = pd.Series(15.0, index=idx)
    vix3m = pd.Series(18.0, index=idx)                         # always contango
    vx = pd.Series(-0.01, index=idx)                          # steady roll-down
    r = vv.vix_vrp_returns(vx, vix, vix3m, vv.VixVRPConfig(vol_lookback=20))
    assert r.sum() > 0                                         # short a falling VIX future -> profit
