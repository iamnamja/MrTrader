"""Tests for the VIX-term crash governor (app/strategy/crash_governor.py) + the overlay
evaluation path in sleeve_lab (Alpha-v7 F1b)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.crash_governor import VixTermGovernorConfig, vix_term_multiplier
from scripts.walkforward.sleeve_lab import (
    Overlay, evaluate_overlay, format_overlay_report, OverlayReport,
)


def _vix_pair(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    vix3m = pd.Series(18.0 + rng.normal(0, 1, n).cumsum() * 0.1, index=idx).clip(10, 40)
    vix = vix3m.copy()
    # force a backwardation window (vix > vix3m) in the middle
    stress = slice(120, 150)
    vix.iloc[stress] = vix3m.iloc[stress] * 1.25
    vix.iloc[:120] = vix3m.iloc[:120] * 0.92      # contango elsewhere
    vix.iloc[150:] = vix3m.iloc[150:] * 0.92
    return vix, vix3m


# ── multiplier logic + PIT ─────────────────────────────────────────────────────────
def test_multiplier_derisks_in_backwardation():
    vix, vix3m = _vix_pair()
    cfg = VixTermGovernorConfig(derisk_to=0.5, ratio_threshold=1.0)
    mult = vix_term_multiplier(vix, vix3m, cfg)
    assert set(np.unique(mult.round(6))).issubset({0.5, 1.0})
    assert (mult == 0.5).any()                  # de-risked on the stress window
    assert (mult == 1.0).any()                  # full exposure elsewhere


def test_multiplier_is_lagged_pit():
    vix, vix3m = _vix_pair()
    cfg = VixTermGovernorConfig(derisk_to=0.0, ratio_threshold=1.0)
    mult = vix_term_multiplier(vix, vix3m, cfg)
    # raw signal at close t; multiplier applies to t+1 -> first stress day is NOT yet derisked
    ratio = (vix / vix3m)
    first_stress = ratio[ratio > 1.0].index[0]
    # the multiplier ON the first stress day reflects the PRIOR day's (calm) signal => 1.0
    assert mult.loc[first_stress] == 1.0
    # the day AFTER the first stress day reflects the stress => 0.0
    nxt = mult.index[mult.index.get_loc(first_stress) + 1]
    assert mult.loc[nxt] == 0.0


def test_confirm_days_debounce():
    vix, vix3m = _vix_pair()
    loose = vix_term_multiplier(vix, vix3m, VixTermGovernorConfig(confirm_days=1))
    strict = vix_term_multiplier(vix, vix3m, VixTermGovernorConfig(confirm_days=3))
    # requiring 3 consecutive inverted closes can only REDUCE the number of de-risk days
    assert (strict < 1.0).sum() <= (loose < 1.0).sum()


def test_derisk_to_validation():
    vix, vix3m = _vix_pair()
    with pytest.raises(ValueError):
        vix_term_multiplier(vix, vix3m, VixTermGovernorConfig(derisk_to=1.5))


# ── overlay evaluation path ────────────────────────────────────────────────────────
def _book_with_crash(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    r = rng.normal(0.0004, 0.008, n)
    r[200:215] = -0.03      # a sharp drawdown window
    return pd.Series(r, index=idx)


def test_overlay_rejects_negative_multiplier():
    idx = pd.bdate_range("2020-01-02", periods=10)
    with pytest.raises(ValueError):
        Overlay("bad", pd.Series([-0.1] * 10, index=idx))


def test_evaluate_overlay_derisks_the_crash():
    base = _book_with_crash()
    # a governor that de-risks exactly over the crash window (lagged by 1 to be PIT-fair)
    mult = pd.Series(1.0, index=base.index)
    mult.iloc[201:216] = 0.0     # flat through the crash (already "as-applied")
    rep = evaluate_overlay(Overlay("test_gov", mult), base)
    assert isinstance(rep, OverlayReport)
    assert rep.with_["max_dd"] > rep.without["max_dd"]    # shallower drawdown (less negative)
    assert rep.d_max_dd > 0
    assert rep.improves_tail
    txt = format_overlay_report(rep)
    txt.encode("ascii")
    assert "VERDICT" in txt
    assert "max_dd" in rep.to_dict()["with"]


def test_evaluate_overlay_inert_when_always_full():
    base = _book_with_crash(seed=2)
    mult = pd.Series(1.0, index=base.index)   # never de-risks -> ~identical (toggle cost 0)
    rep = evaluate_overlay(Overlay("inert", mult), base, toggle_cost_bps=0.0)
    assert abs(rep.d_sharpe) < 1e-9
    assert abs(rep.d_max_dd) < 1e-9
    assert rep.derisk_fraction == 0.0
