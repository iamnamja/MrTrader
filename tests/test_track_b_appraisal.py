"""
Tests for Track-B v2 (Alpha-v7 Phase B, Phase 3) — scripts/walkforward/track_b_appraisal.py.

Pins the budget-invariant design: a genuine uncorrelated diversifier PASSES (high
appraisal IR + P(ΔSR>0)); a redundant high-correlation sleeve FAILS; the appraisal IR
is INVARIANT to the blend budget (the legacy ΔSharpe bar was not); the worst-regime
floor is waived for declared diversifiers but gates other components; a constant
candidate is a clean degenerate REJECT; the bootstrap is seed-reproducible; and the
legacy TRACKB_MODE/book_delta gate is untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward import track_b_appraisal as tba

CRIT = tba.TrackBAppraisalCriteria.from_retrain_config()


def _idx(n):
    return pd.bdate_range("2019-01-02", periods=n)


def _base(n=800, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(0.0004 + rng.normal(0, 0.008, n), index=_idx(n))   # SR ~0.8


def _diversifier(n=800, seed=1):
    # A GENUINE diversifier: a solid own Sharpe (~1.3), ~uncorrelated to the base book
    # → high residual-α IR and a robust book-Sharpe lift when added. (A small drift
    # vs a large vol would let the realized sample mean wander negative on an unlucky
    # seed — and the gate would then correctly reject it; that's a fixture choice, so
    # use a clearly-positive signal.)
    rng = np.random.default_rng(seed)
    return pd.Series(0.0012 + rng.normal(0, 0.008, n), index=_idx(n))   # SR ~1.3


def _redundant(base: pd.Series, seed=2):
    # Essentially the book again (high correlation, no residual alpha).
    rng = np.random.default_rng(seed)
    return base * 1.0 + pd.Series(rng.normal(0, 0.0005, len(base)), index=base.index)


# ───────────────────────── core verdicts ────────────────────────────────────

def test_genuine_diversifier_passes():
    res = tba.appraise_track_b(_base(), _diversifier(), component_type="diversifier",
                               criteria=CRIT, candidate_risk_budget=0.20, seed=0)
    assert res.passed is True and res.verdict == "PASS"
    assert res.checks["appraisal_ir"][1] is True
    assert res.checks["p_delta_sr_gt_0"][1] is True
    assert res.checks["corr_to_book"][1] is True
    assert res.regime_waived is True


def test_redundant_sleeve_fails():
    base = _base()
    res = tba.appraise_track_b(base, _redundant(base), component_type="alpha",
                               criteria=CRIT, candidate_risk_budget=0.20, seed=0)
    assert res.passed is False
    # It is the book again → fails the correlation backstop and earns ~no residual α.
    assert res.checks["corr_to_book"][1] is False
    assert res.checks["appraisal_ir"][1] is False


# ───────────────────────── budget invariance ────────────────────────────────

def test_appraisal_ir_is_budget_invariant():
    base, div = _base(), _diversifier()
    lo = tba.appraise_track_b(base, div, component_type="diversifier", criteria=CRIT,
                              candidate_risk_budget=0.05, seed=0)
    hi = tba.appraise_track_b(base, div, component_type="diversifier", criteria=CRIT,
                              candidate_risk_budget=0.25, seed=0)
    # The appraisal IR is a property of the sleeve-vs-book, NOT the blend fraction.
    assert lo.appraisal_ir == pytest.approx(hi.appraisal_ir, abs=1e-9)
    # ΔSR magnitude DOES scale with budget (bigger add → bigger book move)…
    assert hi.delta_sr_point > lo.delta_sr_point
    # …but a genuine diversifier passes across the whole budget range.
    assert lo.passed is True and hi.passed is True


# ───────────────────────── regime waiver ─────────────────────────────────────

def test_worst_regime_waived_for_diversifier():
    res = tba.appraise_track_b(_base(), _diversifier(), component_type="risk_premium",
                               criteria=CRIT, candidate_risk_budget=0.20,
                               worst_regime_sharpe=-2.0, seed=0)   # catastrophic
    assert res.regime_waived is True
    assert "worst_regime" not in res.checks            # not a gating criterion
    assert "worst_regime_report" in res.checks
    assert res.passed is True                           # waived → still passes


def test_worst_regime_gates_non_diversifier():
    res = tba.appraise_track_b(_base(), _diversifier(), component_type="alpha",
                               criteria=CRIT, candidate_risk_budget=0.20,
                               worst_regime_sharpe=-2.0, seed=0)
    assert res.regime_waived is False
    assert res.checks["worst_regime"][1] is False
    assert res.passed is False


def test_non_diversifier_without_regime_data_fails_closed():
    # Mirrors Track-A: a non-diversifier with NO regime data and no human waiver must
    # NOT silently pass — the missing backstop fails closed.
    res = tba.appraise_track_b(_base(), _diversifier(), component_type="alpha",
                               criteria=CRIT, candidate_risk_budget=0.20, seed=0)
    val, ok = res.checks["worst_regime"]
    assert np.isnan(val) and ok is False
    assert res.passed is False
    assert any("FAILING CLOSED" in w for w in res.warnings)


def test_non_diversifier_missing_regime_passes_only_with_explicit_waiver():
    res = tba.appraise_track_b(_base(), _diversifier(), component_type="alpha",
                               criteria=CRIT, candidate_risk_budget=0.20,
                               regime_waiver_approved=True, seed=0)
    assert "worst_regime" not in res.checks          # not gating under the waiver
    assert res.checks["requires_human_review"][1] is False   # surfaced as a flag
    assert res.passed is True


# ───────────────────────── guards / determinism ─────────────────────────────

def test_constant_candidate_degenerate_reject():
    base = _base()
    flat = pd.Series(np.full(len(base), 0.001), index=base.index)
    res = tba.appraise_track_b(base, flat, component_type="diversifier", criteria=CRIT)
    assert res.passed is False and "degenerate" in res.verdict.lower()


def test_out_of_budget_raises():
    with pytest.raises(ValueError, match="budget"):
        tba.appraise_track_b(_base(), _diversifier(), component_type="diversifier",
                             criteria=CRIT, candidate_risk_budget=0.99)


def test_too_little_history_raises():
    with pytest.raises(ValueError, match="aligned"):
        tba.appraise_track_b(_base(50), _diversifier(50), component_type="diversifier",
                             criteria=CRIT)


def test_bootstrap_reproducible_with_seed():
    base, div = _base(), _diversifier()
    a = tba.appraise_track_b(base, div, component_type="diversifier", criteria=CRIT,
                             candidate_risk_budget=0.20, seed=7)
    b = tba.appraise_track_b(base, div, component_type="diversifier", criteria=CRIT,
                             candidate_risk_budget=0.20, seed=7)
    assert a.p_delta_sr_gt_0 == b.p_delta_sr_gt_0
    assert a.delta_sr_ci_low == b.delta_sr_ci_low


def test_legacy_book_delta_gate_retained():
    # 2026-06-13: TRACKB_MODE flipped to "ruler_v2" (live). The legacy book_delta gate
    # is RETAINED (importable + callable) for reproducibility, exactly as mean_sharpe
    # was kept when significance went live. (The mode itself is no longer asserted to a
    # default — that is now an owner-controlled live setting.)
    import app.ml.retrain_config as rc
    assert rc.TRACKB_MODE in ("ruler_v2", "book_delta")
    from scripts.walkforward.book_gate import book_delta_gate   # still importable
    assert callable(book_delta_gate)
