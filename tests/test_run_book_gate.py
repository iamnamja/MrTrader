"""
Tests for the Track-B runner dispatcher (scripts/run_book_gate.py `_evaluate`):
TRACKB_MODE="ruler_v2" routes to the budget-invariant appraisal gate, "book_delta"
to the legacy gate, and both return the uniform tuple the runner consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_book_gate import _evaluate
from scripts.walkforward.track_b_appraisal import TrackBAppraisalResult
from scripts.walkforward.book_gate import BookGateResult


def _idx(n):
    return pd.bdate_range("2019-01-02", periods=n)


def _base(n=800):
    rng = np.random.default_rng(0)
    return pd.Series(0.0004 + rng.normal(0, 0.008, n), index=_idx(n))


def _candidate(n=800):
    rng = np.random.default_rng(1)
    return pd.Series(0.0012 + rng.normal(0, 0.008, n), index=_idx(n))   # strong, ~uncorr


def test_evaluate_ruler_v2_routes_to_appraisal():
    res, report, crit, dlabel, dval, passed, failed = _evaluate(
        _base(), _candidate(), mode="ruler_v2", candidate_risk_budget=0.20)
    assert isinstance(res, TrackBAppraisalResult)
    assert "TRACK B v2 APPRAISAL" in report
    assert dlabel == "P(dSR>0)"
    assert "min_ir" in crit                       # TrackBAppraisalCriteria shape
    assert isinstance(passed, bool)
    report.encode("ascii")


def test_evaluate_book_delta_routes_to_legacy():
    res, report, crit, dlabel, dval, passed, failed = _evaluate(
        _base(), _candidate(), mode="book_delta", candidate_risk_budget=0.20)
    assert isinstance(res, BookGateResult)
    assert dlabel == "dSharpe"
    assert "min_sharpe_delta" in crit             # BookGateCriteria shape
    assert isinstance(passed, bool)


def test_evaluate_ruler_v2_appraisal_ir_budget_invariant():
    # The runner under ruler_v2 inherits the budget-invariance property: the appraisal
    # IR is identical across budgets (only the dSR-significance side moves).
    lo = _evaluate(_base(), _candidate(), mode="ruler_v2", candidate_risk_budget=0.05)[0]
    hi = _evaluate(_base(), _candidate(), mode="ruler_v2", candidate_risk_budget=0.25)[0]
    assert lo.appraisal_ir == hi.appraisal_ir
