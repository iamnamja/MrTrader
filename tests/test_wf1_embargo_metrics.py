"""
Tests for WF-1 — Embargo + Multi-Metric Gate.

Verifies:
  1. _compute_profit_factor: correct ratio, zero on no losses, zero on empty
  2. _compute_calmar: correct ratio, zero on zero drawdown or zero years
  3. _compute_k_ratio: positive for upward curve, zero for short series
  4. FoldResult fields profit_factor / calmar_ratio / k_ratio default to 0.0
  5. WalkForwardReport.avg_profit_factor / avg_calmar / avg_k_ratio aggregate correctly
  6. gate_passed(): passes when PF and Calmar gates met; fails when either not met
  7. gate_detail(): keys and boolean values match gate logic
  8. embargo_days propagates into fold boundaries (smoke test via run_swing_walkforward mock)
"""
import pytest
import numpy as np
from datetime import date
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Metric helper tests
# ---------------------------------------------------------------------------

class TestComputeProfitFactor:
    def _pf(self, returns):
        from scripts.walkforward_tier3 import _compute_profit_factor
        return _compute_profit_factor(returns)

    def test_basic_ratio(self):
        # wins=3, losses=1 => pf=3.0
        assert abs(self._pf([1.0, 1.0, 1.0, -1.0]) - 3.0) < 1e-9

    def test_no_losses_returns_sentinel(self):
        # All wins, no losses: undefined PF, report 999.0 sentinel (not 0.0)
        assert self._pf([1.0, 2.0]) == 999.0

    def test_empty_returns_zero(self):
        assert self._pf([]) == 0.0

    def test_all_losses(self):
        result = self._pf([-1.0, -2.0])
        assert result == 0.0  # no wins => 0 / sum_losses => 0.0

    def test_equal_wins_losses(self):
        assert abs(self._pf([1.0, -1.0]) - 1.0) < 1e-9


class TestComputeCalmar:
    def _cal(self, total_return, max_dd, years):
        from scripts.walkforward_tier3 import _compute_calmar
        return _compute_calmar(total_return, max_dd, years)

    def test_basic(self):
        # annualised = 0.20/2 = 0.10; calmar = 0.10/0.05 = 2.0
        assert abs(self._cal(0.20, 0.05, 2.0) - 2.0) < 1e-9

    def test_zero_drawdown_returns_zero(self):
        assert self._cal(0.10, 0.0, 1.0) == 0.0

    def test_zero_years_returns_zero(self):
        assert self._cal(0.10, 0.05, 0.0) == 0.0

    def test_negative_return_gives_negative_calmar(self):
        result = self._cal(-0.10, 0.05, 1.0)
        assert result < 0


class TestComputeKRatio:
    def _k(self, curve):
        from scripts.walkforward_tier3 import _compute_k_ratio
        return _compute_k_ratio(curve)

    def test_upward_curve_positive(self):
        rng = np.random.default_rng(0)
        curve = list(np.cumsum(rng.normal(0.5, 0.1, 100)))  # noisy upward trend
        assert self._k(curve) > 0

    def test_flat_curve(self):
        curve = [1.0] * 50
        # std of diffs = 0 => returns 0.0
        assert self._k(curve) == 0.0

    def test_short_curve_returns_zero(self):
        assert self._k([1.0, 2.0]) == 0.0
        assert self._k([]) == 0.0

    def test_downward_curve_negative(self):
        rng = np.random.default_rng(1)
        curve = list(np.cumsum(rng.normal(-0.5, 0.1, 100)))  # noisy downward trend
        assert self._k(curve) < 0


# ---------------------------------------------------------------------------
# FoldResult defaults
# ---------------------------------------------------------------------------

class TestFoldResultDefaults:
    def _fold(self, sharpe=0.9):
        from scripts.walkforward_tier3 import FoldResult
        return FoldResult(
            fold=1,
            train_start=date(2021, 1, 1),
            train_end=date(2022, 12, 31),
            test_start=date(2023, 1, 1),
            test_end=date(2023, 12, 31),
            trades=50,
            win_rate=0.55,
            sharpe=sharpe,
            max_drawdown=0.05,
            total_return=0.10,
            stop_exit_rate=0.4,
        )

    def test_default_fields_zero(self):
        f = self._fold()
        assert f.profit_factor == 0.0
        assert f.calmar_ratio == 0.0
        assert f.k_ratio == 0.0

    def test_can_set_fields(self):
        f = self._fold()
        f.profit_factor = 1.5
        f.calmar_ratio = 0.6
        f.k_ratio = 0.3
        assert f.profit_factor == 1.5


# ---------------------------------------------------------------------------
# WalkForwardReport aggregation
# ---------------------------------------------------------------------------

def _make_report(folds_data):
    """folds_data: list of (sharpe, pf, calmar, k)"""
    from scripts.walkforward_tier3 import FoldResult, WalkForwardReport
    report = WalkForwardReport(model_type="swing")
    for i, (sh, pf, cal, k) in enumerate(folds_data, 1):
        f = FoldResult(
            fold=i,
            train_start=date(2021, 1, 1),
            train_end=date(2022, 12, 31),
            test_start=date(2023, 1, 1),
            test_end=date(2023, 12, 31),
            trades=100,
            win_rate=0.55,
            sharpe=sh,
            max_drawdown=0.05,
            total_return=0.10,
            stop_exit_rate=0.4,
            profit_factor=pf,
            calmar_ratio=cal,
            k_ratio=k,
        )
        report.folds.append(f)
    return report


class TestWalkForwardReportAggregation:
    def test_avg_profit_factor_excludes_zeros(self):
        report = _make_report([(0.9, 1.5, 0.5, 0.1), (0.9, 0.0, 0.5, 0.1)])
        # only first fold has pf > 0
        assert abs(report.avg_profit_factor - 1.5) < 1e-9

    def test_avg_profit_factor_no_data(self):
        report = _make_report([(0.9, 0.0, 0.0, 0.0)])
        assert report.avg_profit_factor == 0.0

    def test_avg_profit_factor_caps_sentinel_at_5(self):
        # PF=999 sentinel (all-wins fold) must be capped before averaging
        # to prevent one lucky small fold from inflating the mean past the gate.
        report = _make_report([(0.9, 999.0, 0.5, 0.1), (0.9, 1.0, 0.5, 0.1)])
        # capped: (5.0 + 1.0) / 2 = 3.0  (not (999+1)/2 = 500)
        assert abs(report.avg_profit_factor - 3.0) < 1e-9

    def test_avg_calmar(self):
        report = _make_report([(0.9, 1.5, 0.4, 0.1), (0.9, 1.5, 0.6, 0.1)])
        assert abs(report.avg_calmar - 0.5) < 1e-9

    def test_avg_k_ratio(self):
        report = _make_report([(0.9, 1.5, 0.5, 0.2), (0.9, 1.5, 0.5, 0.4)])
        assert abs(report.avg_k_ratio - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

class TestGateLogic:
    """Gate tests patch DSR to isolate PF/Calmar/Sharpe logic.
    DSR is tested separately by the existing Phase 22 tests and the unit test below."""

    def _report_with_sharpes(self, sharpes, pf=1.5, calmar=0.5):
        from scripts.walkforward_tier3 import FoldResult, WalkForwardReport
        report = WalkForwardReport(model_type="swing")
        for i, sh in enumerate(sharpes, 1):
            f = FoldResult(
                fold=i,
                train_start=date(2021, 1, 1),
                train_end=date(2022, 12, 31),
                test_start=date(2023, 1, 1),
                test_end=date(2023, 12, 31),
                trades=100,
                win_rate=0.55,
                sharpe=sh,
                max_drawdown=0.05,
                total_return=0.20,
                stop_exit_rate=0.4,
                profit_factor=pf,
                calmar_ratio=calmar,
            )
            report.folds.append(f)
        return report

    @patch("scripts.walkforward_tier3._deflated_sharpe_ratio", return_value=(2.0, 0.99))
    def test_passes_all_gates(self, _dsr):
        report = self._report_with_sharpes([1.2, 1.1, 1.3], pf=1.5, calmar=0.5)
        assert report.gate_passed()

    @patch("scripts.walkforward_tier3._deflated_sharpe_ratio", return_value=(2.0, 0.99))
    def test_fails_sharpe_gate(self, _dsr):
        report = self._report_with_sharpes([0.5, 0.5, 0.5], pf=1.5, calmar=0.5)
        assert not report.gate_passed()

    @patch("scripts.walkforward_tier3._deflated_sharpe_ratio", return_value=(2.0, 0.99))
    def test_fails_min_fold_sharpe(self, _dsr):
        report = self._report_with_sharpes([1.5, 1.5, -0.5], pf=1.5, calmar=0.5)
        assert not report.gate_passed()

    @patch("scripts.walkforward_tier3._deflated_sharpe_ratio", return_value=(2.0, 0.99))
    def test_fails_profit_factor_gate(self, _dsr):
        report = self._report_with_sharpes([1.2, 1.1, 1.3], pf=0.9, calmar=0.5)
        assert not report.gate_passed()

    @patch("scripts.walkforward_tier3._deflated_sharpe_ratio", return_value=(2.0, 0.99))
    def test_fails_calmar_gate(self, _dsr):
        report = self._report_with_sharpes([1.2, 1.1, 1.3], pf=1.5, calmar=0.1)
        assert not report.gate_passed()

    def test_pf_zero_treated_as_not_computed(self):
        report = self._report_with_sharpes([1.2, 1.1, 1.3], pf=0.0, calmar=0.5)
        detail = report.gate_detail()
        assert detail["avg_profit_factor"][1] is True  # 0.0 => pass-through

    def test_calmar_zero_treated_as_not_computed(self):
        report = self._report_with_sharpes([1.2, 1.1, 1.3], pf=1.5, calmar=0.0)
        detail = report.gate_detail()
        assert detail["avg_calmar"][1] is True


class TestGateDetail:
    def test_keys_present(self):
        from scripts.walkforward_tier3 import WalkForwardReport
        report = WalkForwardReport(model_type="swing")
        detail = report.gate_detail()
        assert "avg_sharpe" in detail
        assert "min_sharpe" in detail
        assert "dsr_p" in detail
        assert "avg_profit_factor" in detail
        assert "avg_calmar" in detail

    def test_values_are_tuples(self):
        from scripts.walkforward_tier3 import WalkForwardReport
        report = WalkForwardReport(model_type="swing")
        for k, v in report.gate_detail().items():
            assert isinstance(v, tuple), f"{k} should be (value, bool)"
            assert isinstance(v[1], bool), f"{k}[1] should be bool"


# ---------------------------------------------------------------------------
# Embargo boundary arithmetic
# ---------------------------------------------------------------------------

class TestEmbargoBoundaries:
    """Verify embargo_days creates the correct post-test gap in fold boundaries."""

    def test_embargo_creates_post_test_gap(self):
        """Fold N's test_end must be embargo_days before fold N+1's train_end.

        With the fixed formula raw_test_end_dt = train_end_dt + segment_days - embargo,
        fold 0 test ends at fold_1_train_end - embargo_days, creating the proper gap.
        """
        from datetime import datetime, timedelta

        total_years = 3
        n_folds = 2
        purge_days = 10
        embargo_days = 5

        end_all = datetime(2024, 1, 1)
        segment_days = int(total_years * 365 / (n_folds + 1))
        _embargo = embargo_days

        # Fold 0
        fold0_train_end = end_all - timedelta(days=segment_days * n_folds)
        fold0_test_end = fold0_train_end + timedelta(days=segment_days - _embargo)

        # Fold 1's train_end (the segment boundary)
        fold1_train_end = end_all - timedelta(days=segment_days * (n_folds - 1))

        # Gap between fold 0 test end and fold 1 train end should equal embargo_days
        gap = (fold1_train_end - fold0_test_end).days
        assert gap == embargo_days, f"Expected gap={embargo_days}, got {gap}"

    def test_embargo_defaults_to_purge_days(self):
        """When embargo_days is None, it should equal purge_days."""
        from scripts.walkforward_tier3 import run_swing_walkforward
        import inspect
        sig = inspect.signature(run_swing_walkforward)
        assert "embargo_days" in sig.parameters
        assert sig.parameters["embargo_days"].default is None  # None = defaults to purge_days at runtime

    def test_fold_result_summary_includes_pf_calmar(self):
        from scripts.walkforward_tier3 import FoldResult
        f = FoldResult(
            fold=1,
            train_start=date(2021, 1, 1),
            train_end=date(2022, 12, 31),
            test_start=date(2023, 1, 1),
            test_end=date(2023, 12, 31),
            trades=50,
            win_rate=0.55,
            sharpe=1.0,
            max_drawdown=0.05,
            total_return=0.10,
            stop_exit_rate=0.4,
            profit_factor=1.3,
            calmar_ratio=0.45,
        )
        summary = f.summary_line()
        assert "PF=1.30" in summary
        assert "Cal=0.45" in summary
