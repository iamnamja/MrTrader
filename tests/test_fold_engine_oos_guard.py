"""Tests that FoldEngine.run() enforces the OOS guard and gates.WalkForwardReport
respects in_sample_override — the public-API path that previously had no guard."""
from __future__ import annotations

import pytest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts.walkforward.oos_guard import OOSViolation
from scripts.walkforward.gates import WalkForwardReport, FoldResult


# ---------------------------------------------------------------------------
# gates.WalkForwardReport.in_sample_override
# ---------------------------------------------------------------------------

def _make_passing_report(in_sample_override: bool) -> WalkForwardReport:
    """Build a report with metrics that would normally pass all gates."""
    from scripts.walkforward.gates import FoldResult
    r = WalkForwardReport(model_type="intraday", in_sample_override=in_sample_override)
    for _ in range(3):
        r.folds.append(FoldResult(
            fold=1, train_start=date(2022, 1, 1), train_end=date(2023, 1, 1),
            test_start=date(2023, 1, 1), test_end=date(2023, 6, 1),
            trades=100, win_rate=0.60, sharpe=2.5, max_drawdown=0.05,
            total_return=0.20, stop_exit_rate=0.1, model_version=63,
            profit_factor=1.8, calmar_ratio=1.5, n_obs=120,
        ))
    return r


def test_gates_walkforward_report_gate_passes_normally():
    r = _make_passing_report(in_sample_override=False)
    assert r.gate_passed() is True


def test_gates_walkforward_report_in_sample_override_blocks_gate():
    """in_sample_override=True must block gate_passed() regardless of metrics."""
    r = _make_passing_report(in_sample_override=True)
    assert r.gate_passed() is False


def test_gates_walkforward_report_default_override_is_false():
    r = WalkForwardReport(model_type="swing")
    assert r.in_sample_override is False


# ---------------------------------------------------------------------------
# FoldEngine.run() OOS guard
# ---------------------------------------------------------------------------

def _make_strategy(trained_through, n_days=400, days_start=None):
    model = SimpleNamespace(trained_through=trained_through)
    # all_days starts from days_start and goes forward n_days trading days
    base = days_start or date(2020, 1, 1)
    all_days = [base + timedelta(days=i) for i in range(n_days)]
    fold_result = FoldResult(
        fold=1, train_start=all_days[0], train_end=all_days[min(100, len(all_days)-1)],
        test_start=all_days[min(150, len(all_days)-1)], test_end=all_days[min(200, len(all_days)-1)],
        trades=50, win_rate=0.55, sharpe=1.5, max_drawdown=0.05,
        total_return=0.10, stop_exit_rate=0.2, model_version=1,
        profit_factor=1.5, calmar_ratio=1.0,
    )
    strategy = SimpleNamespace(
        model=model,
        model_type="intraday",
        version=1,
        all_days_sorted=all_days,
        run_fold=MagicMock(return_value=fold_result),
        fetch_data=MagicMock(),
    )
    return strategy, all_days


def test_fold_engine_raises_when_model_in_sample():
    """FoldEngine.run() must raise OOSViolation when trained_through is after test folds.
    all_days start 2020-01-01; trained_through=2022-01-01 is after all fold test dates."""
    from scripts.walkforward.engine import FoldEngine

    strategy, all_days = _make_strategy(
        trained_through=date(2022, 1, 1),
        days_start=date(2020, 1, 1),
    )

    with patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        engine = FoldEngine(strategy=strategy, purge_days=5, embargo_days=5)
        with pytest.raises(OOSViolation):
            engine.run(n_folds=3, total_days=400, allow_sacred_holdout=True)


def test_fold_engine_passes_when_trained_through_before_test():
    """FoldEngine.run() must succeed when trained_through is before all test folds.
    all_days start 2020-01-01; trained_through=2018-01-01 is well before all folds."""
    from scripts.walkforward.engine import FoldEngine

    strategy, all_days = _make_strategy(
        trained_through=date(2018, 1, 1),
        days_start=date(2020, 1, 1),
    )

    with patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        engine = FoldEngine(strategy=strategy, purge_days=5, embargo_days=5)
        report = engine.run(n_folds=3, total_days=400, allow_sacred_holdout=True)

    assert report.in_sample_override is False
    assert len(report.folds) == 3


def test_fold_engine_allow_in_sample_sets_override():
    """allow_in_sample=True must log warning but not raise; sets in_sample_override."""
    from scripts.walkforward.engine import FoldEngine

    strategy, all_days = _make_strategy(
        trained_through=date(2022, 1, 1),
        days_start=date(2020, 1, 1),
    )

    with patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        engine = FoldEngine(strategy=strategy, purge_days=0, embargo_days=0)
        report = engine.run(
            n_folds=3, total_days=400,
            allow_sacred_holdout=True, allow_in_sample=True,
        )

    assert report.in_sample_override is True
    assert report.gate_passed() is False
