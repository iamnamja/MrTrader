"""CRITICAL-1: DSR implausibility ceiling + saturation tests."""
from datetime import date

from scripts.walkforward.gates import (
    FoldResult, WalkForwardReport, deflated_sharpe_ratio,
)
from scripts.walkforward.cpcv import CPCVResult
from app.ml.retrain_config import DSR_SATURATION_P


def _fold(sharpe: float, n_obs: int = 250) -> FoldResult:
    return FoldResult(
        fold=1,
        train_start=date(2020, 1, 1), train_end=date(2022, 1, 1),
        test_start=date(2022, 1, 1), test_end=date(2023, 1, 1),
        trades=100, win_rate=0.6, sharpe=sharpe,
        max_drawdown=0.02, total_return=0.5, stop_exit_rate=0.1, n_obs=n_obs,
        # Phase 2: populate regime data so the (now-enforced) regime gate has a
        # real value to check — this test is about the human-review flag, not regime.
        regime_sharpes={"BULL": 1.0, "BEAR": 0.5, "NEUTRAL": 0.8},
    )


def _report(sharpe: float, n_folds: int = 2) -> WalkForwardReport:
    return WalkForwardReport(model_type="swing",
                             folds=[_fold(sharpe) for _ in range(n_folds)],
                             is_true_walkforward=True)  # P0-3: promotable run is true-WF


# ── WalkForwardReport ──────────────────────────────────────────────────────────
def test_dsr_saturates_at_high_sharpe():
    _, p = deflated_sharpe_ratio(5.14, 250, 250)
    assert p > DSR_SATURATION_P


def test_high_sharpe_report_saturated_and_review():
    r = _report(5.14)
    assert r.dsr_saturated() is True
    assert r.requires_human_review() is True


def test_sharpe_below_ceiling_no_review():
    r = _report(2.9)
    assert r.requires_human_review() is False


def test_sharpe_above_ceiling_boundary():
    r = _report(3.01)
    assert r.requires_human_review() is True


def test_gate_detail_human_review_key():
    r = _report(5.14)
    detail = r.gate_detail()
    assert "human_review_required" in detail
    # ok=False when review IS required
    assert detail["human_review_required"][1] is False


def test_gate_passed_ignores_human_review():
    # High-Sharpe model: gate_passed must still be True (review flag does NOT block it).
    r = _report(5.14)
    assert r.requires_human_review() is True
    assert r.gate_passed() is True


# ── CPCVResult mirror ────────────────────────────────────────────────────────────
def _cpcv(sharpe: float) -> CPCVResult:
    return CPCVResult(
        model_type="swing", n_folds=6, n_paths=2,
        path_sharpes=[sharpe] * 10,
        path_profit_factors=[2.0] * 10,
        path_calmars=[1.0] * 10,
        path_n_obs=[250] * 10,
        # Phase 2: real regime value so the now-enforced regime gate isn't the blocker.
        worst_regime_sharpe=0.5,
        is_true_walkforward=True,  # P0-3: promotable run is true-WF
    )


def test_cpcv_dsr_saturates():
    _, p = deflated_sharpe_ratio(5.14, 250, 250)
    assert p > DSR_SATURATION_P


def test_cpcv_high_sharpe_saturated_and_review():
    c = _cpcv(5.14)
    assert c.dsr_saturated() is True
    assert c.requires_human_review() is True


def test_cpcv_below_ceiling_no_review():
    assert _cpcv(2.9).requires_human_review() is False


def test_cpcv_above_ceiling_boundary():
    assert _cpcv(3.01).requires_human_review() is True


def test_cpcv_gate_detail_human_review_key():
    detail = _cpcv(5.14).gate_detail()
    assert "human_review_required" in detail
    assert detail["human_review_required"][1] is False


def test_cpcv_gate_passed_ignores_human_review():
    c = _cpcv(5.14)
    assert c.requires_human_review() is True
    assert c.gate_passed() is True
