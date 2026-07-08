"""CH3 regime-conditional diagnostic tests (scripts/ch3_regime_diagnostic.py).

Pins the pure pieces — the per-regime return profile, the per-regime correlation slicing, and the
conditional-diversifier verdict. The heavy Q1/Q2/Q3 runs are exercised by running the module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import ch3_regime_diagnostic as ch3


def _labels(index, first_half, second_half):
    n = len(index)
    return pd.Series([first_half] * (n // 2) + [second_half] * (n - n // 2), index=index)


# ── regime_conditional_corr ───────────────────────────────────────────────────
def test_regime_conditional_corr_slices_by_label(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    trend = pd.Series(np.random.default_rng(0).normal(0, 0.01, 40), index=idx)
    # BEAR half: x = -trend (corr -1) ; BULL half: x = +trend (corr +1)
    x = trend.copy()
    x.iloc[:20] = -trend.iloc[:20]
    frame = pd.DataFrame({"trend": trend, "x": x})
    monkeypatch.setattr(ch3, "_aligned_labels", lambda index: _labels(index, "BEAR", "BULL"))
    corr = ch3.regime_conditional_corr(frame, "trend")
    assert corr["BEAR"]["corr_to_base"]["x"] == pytest.approx(-1.0, abs=1e-6)
    assert corr["BULL"]["corr_to_base"]["x"] == pytest.approx(1.0, abs=1e-6)
    assert corr["ALL"]["n_days"] == 40


def test_regime_conditional_corr_thin_bucket_is_none(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    frame = pd.DataFrame({"trend": np.random.default_rng(1).normal(0, 0.01, 30),
                          "x": np.random.default_rng(2).normal(0, 0.01, 30)}, index=idx)
    # 5 BEAR (< _MIN_REGIME_DAYS) + 25 BULL
    labels = pd.Series(["BEAR"] * 5 + ["BULL"] * 25, index=idx)
    monkeypatch.setattr(ch3, "_aligned_labels", lambda index: labels)
    corr = ch3.regime_conditional_corr(frame, "trend")
    assert corr["BEAR"]["corr_to_base"] is None       # thin -> not estimated
    assert corr["BULL"]["corr_to_base"] is not None


# ── _conditional_diversifier_verdict (decorrelation AND standalone value) ─────
_CORR = {"ALL": {"corr_to_base": {"x": 0.50}},
         "BEAR": {"corr_to_base": {"x": 0.14}, "n_days": 1218},
         "BULL": {"corr_to_base": {"x": 0.77}, "n_days": 1613}}


def test_verdict_candidate_needs_decorrelate_AND_active_and_nonlosing():
    # decorrelates in BEAR (0.14) AND is active (vol ~ full) + non-losing there -> candidate
    prof = {"BEAR": {"ann_vol": 0.15, "ann_return": 0.02, "sharpe": 0.1}}
    v = ch3._conditional_diversifier_verdict(_CORR, "x", prof, full_ann_vol=0.15)
    assert v["conditional_diversifier"] is True and "CH4 CANDIDATE" in v["verdict"]


def test_verdict_closed_when_decorrelates_but_loses():
    # THE sector_rotation case: decorrelates in BEAR but LOSES there -> NOT a candidate (a different
    # way to lose is not a hedge). This is the guard the CH3 review demanded.
    prof = {"BEAR": {"ann_vol": 0.27, "ann_return": -0.34, "sharpe": -1.26}}
    v = ch3._conditional_diversifier_verdict(_CORR, "x", prof, full_ann_vol=0.20)
    assert v["conditional_diversifier"] is False and "LOSES" in v["verdict"]


def test_verdict_closed_when_decorrelates_but_flat():
    # uncorrelated-because-DEAD: sits in cash in BEAR (near-zero vol) -> mechanical decorrelation
    prof = {"BEAR": {"ann_vol": 0.01, "ann_return": 0.0, "sharpe": 0.0}}
    v = ch3._conditional_diversifier_verdict(_CORR, "x", prof, full_ann_vol=0.20)
    assert v["conditional_diversifier"] is False and "FLAT" in v["verdict"]


def test_verdict_closed_when_collinear_in_all_regimes():
    corr = {"ALL": {"corr_to_base": {"x": 0.61}},
            "BEAR": {"corr_to_base": {"x": 0.33}, "n_days": 1218},
            "BULL": {"corr_to_base": {"x": 0.81}, "n_days": 1613}}
    prof = {"BEAR": {"ann_vol": 0.15, "ann_return": 0.05, "sharpe": 0.3}}
    v = ch3._conditional_diversifier_verdict(corr, "x", prof, full_ann_vol=0.15)
    assert v["conditional_diversifier"] is False       # 0.33 not < 0.30
    assert "collinear" in v["verdict"]


# ── regime_conditional_profile ────────────────────────────────────────────────
def test_regime_conditional_profile_metrics(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    r = pd.Series(np.concatenate([np.full(30, 0.01), np.full(30, -0.005)]), index=idx)
    monkeypatch.setattr(ch3, "_aligned_labels", lambda index: _labels(index, "BULL", "BEAR"))
    prof = ch3.regime_conditional_profile(r)
    assert prof["BULL"]["n_days"] == 30 and prof["BEAR"]["n_days"] == 30
    assert prof["BULL"]["hit_rate"] == pytest.approx(1.0)     # all +0.01 days win
    assert prof["BEAR"]["hit_rate"] == pytest.approx(0.0)     # all -0.005 days lose
    assert prof["BULL"]["contribution_sum_ret"] == pytest.approx(0.30)   # 30 * 0.01
    assert prof["BEAR"]["max_drawdown_artifact"] < 0         # a losing streak draws down (artifact)
    assert prof["BULL"]["ann_vol"] is not None
