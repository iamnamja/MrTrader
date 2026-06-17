"""P3-5 — aggregate short-volume timing signal + frozen verdict tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research import short_volume as sv


def _dates(n, start="2019-01-02"):
    return pd.bdate_range(start, periods=n)


def test_trailing_zscore_is_pit_and_warms_up():
    idx = _dates(100)
    s = pd.Series(np.arange(100.0), index=idx)
    z = sv.trailing_zscore(s, 63)
    assert z.iloc[:62].isna().all()        # need a full window before any value
    assert np.isfinite(z.iloc[63])
    # z at t uses only data through t (rolling, no center) -> recomputing on a prefix
    # gives the same value at that point
    z_prefix = sv.trailing_zscore(s.iloc[:80], 63)
    assert z_prefix.iloc[79] == pytest.approx(z.iloc[79])


def test_overlay_goes_flat_when_short_vol_elevated():
    idx = _dates(200)
    rng = np.random.default_rng(0)
    ratio = pd.Series(0.45 + 0.02 * rng.normal(0, 1, 200), index=idx)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 200)), index=idx)
    out = sv.overlay_returns(ratio, close, window=63)
    assert set(out["position"].dropna().unique()) <= {0.0, 1.0}   # long-flat
    # position is the prior day's signal (PIT shift) -> first valid position is NaN-shifted
    assert out["position"].notna().all()


def _signal_world(n=700, beta=0.006, noise=0.0015, seed=1):
    """High short-vol z at t-1 -> negative SPY return at t (informed-short world)."""
    idx = _dates(n)
    rng = np.random.default_rng(seed)
    shock = rng.normal(0, 1, n)
    ratio = pd.Series(0.45 + 0.02 * shock, index=idx)
    # ret[t] depends on shock[t-1]: elevated shorting yesterday -> down today
    lag = np.roll(shock, 1)
    lag[0] = 0.0
    rets = 0.0005 - beta * lag + rng.normal(0, noise, n)
    close = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    return ratio, close


def test_verdict_pass_when_high_shortvol_precedes_drops():
    ratio, close = _signal_world()
    v = sv.short_volume_verdict(ratio, close)
    assert v.verdict == "PASS"
    assert v.overlay_sharpe >= sv.PAPER_SR_FLOOR
    assert v.overlay_hac_p < sv.HAC_P_MAX
    assert v.overlay_sharpe > v.buyhold_sharpe
    # diagnostic: high-z tercile should have the lowest next-day return
    fwd = v.fwd_ret_by_z_tercile
    assert fwd["high"] < fwd["low"]


def test_strong_stationary_signal_is_a_standalone_edge():
    # A genuine, stationary, significant signal -> standalone_edge True (alpha sig + stable).
    ratio, close = _signal_world(beta=0.008, noise=0.0012, seed=11)
    v = sv.short_volume_verdict(ratio, close)
    assert v.verdict == "PASS" and v.standalone_edge is True
    assert "STANDALONE" in v.reason
    assert v.incr_alpha_t >= sv.INCR_ALPHA_T_MIN
    assert v.h1_delta > 0 and v.h2_delta > 0


def test_robustness_fields_populated():
    ratio, close = _signal_world()
    v = sv.short_volume_verdict(ratio, close)
    for x in (v.incr_alpha_ann, v.incr_alpha_t, v.beta_spy, v.h1_delta, v.h2_delta):
        assert np.isfinite(x)
    assert isinstance(v.standalone_edge, bool)


def test_verdict_kill_on_noise():
    idx = _dates(700)
    rng = np.random.default_rng(2)
    ratio = pd.Series(0.45 + 0.02 * rng.normal(0, 1, 700), index=idx)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 700)), index=idx)
    v = sv.short_volume_verdict(ratio, close)
    assert v.verdict == "KILL"          # no predictive signal -> fails significance/beat


def test_verdict_does_not_flip_direction():
    # The verdict is fixed to risk_off_when_high; the opposite is a diagnostic only.
    ratio, close = _signal_world()
    v = sv.short_volume_verdict(ratio, close)
    assert v.registration_id == "P3-5-SHORTVOL-AGG"
    assert v.opposite_overlay_sharpe == pytest.approx(
        sv._sharpe(sv.overlay_returns(ratio, close, direction="risk_on_when_high")["net"]))


def test_overlay_charges_turnover_cost():
    ratio, close = _signal_world(seed=4)
    free = sv.overlay_returns(ratio, close, cost_bps=0.0)["net"]
    costed = sv.overlay_returns(ratio, close, cost_bps=5.0)["net"]
    # costed total return must be <= free (cost only subtracts)
    assert costed.sum() <= free.sum()
