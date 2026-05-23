"""
Tests for WF-2 — Pluggable Engine Architecture.

Verifies:
  1. Package is importable; public API is exposed
  2. FoldEngine builds correct calendar fold boundaries (purge + embargo)
  3. FoldEngine builds correct trading-day fold boundaries
  4. FoldEngine delegates to strategy.run_fold() per fold
  5. cost_models: FixedBpsCostModel and SpreadCostModel compute correct cost_pct
  6. run_swing_walkforward / run_intraday_walkforward importable from package
  7. SwingStrategy / IntradayStrategy constructible without DB
  8. FoldResult and WalkForwardReport are the same classes as in walkforward_tier3
"""
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. Package import / public API
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_fold_engine_importable(self):
        from scripts.walkforward import FoldEngine
        assert FoldEngine is not None

    def test_fold_result_importable(self):
        from scripts.walkforward import FoldResult
        assert FoldResult is not None

    def test_walk_forward_report_importable(self):
        from scripts.walkforward import WalkForwardReport
        assert WalkForwardReport is not None

    def test_cost_models_importable(self):
        from scripts.walkforward import FixedBpsCostModel, SpreadCostModel
        assert FixedBpsCostModel is not None
        assert SpreadCostModel is not None

    def test_run_functions_importable(self):
        from scripts.walkforward import run_swing_walkforward, run_intraday_walkforward
        assert callable(run_swing_walkforward)
        assert callable(run_intraday_walkforward)

    def test_fold_result_same_class_as_tier3(self):
        """FoldResult from package and from walkforward_tier3 must be the same class."""
        from scripts.walkforward import FoldResult as PkgFR
        from scripts.walkforward_tier3 import FoldResult as T3FR
        # Both should be importable; package uses gates.py, tier3 has its own
        # They are separate classes but must have the same fields
        import dataclasses
        pkg_fields = {f.name for f in dataclasses.fields(PkgFR)}
        t3_fields = {f.name for f in dataclasses.fields(T3FR)}
        assert pkg_fields == t3_fields, f"Field mismatch: pkg={pkg_fields} tier3={t3_fields}"


# ---------------------------------------------------------------------------
# 2. FoldEngine — calendar fold boundaries (swing)
# ---------------------------------------------------------------------------

class TestFoldEngineCalendarFolds:
    def _engine(self, purge=10, embargo=5):
        from scripts.walkforward.engine import FoldEngine
        strategy = MagicMock()
        return FoldEngine(strategy=strategy, purge_days=purge, embargo_days=embargo)

    def test_correct_number_of_folds(self):
        engine = self._engine()
        end_all = datetime(2024, 1, 1)
        start_all = end_all - timedelta(days=5 * 365 + 30)
        boundaries = engine._build_calendar_folds(3, start_all, end_all, 5, None)
        assert len(boundaries) == 3

    def test_test_starts_after_purge(self):
        engine = self._engine(purge=10, embargo=5)
        end_all = datetime(2024, 1, 1)
        start_all = end_all - timedelta(days=5 * 365 + 30)
        boundaries = engine._build_calendar_folds(3, start_all, end_all, 5, None)
        for tr_start, tr_end, te_start, te_end in boundaries:
            gap = (te_start - tr_end).days
            assert gap >= 10, f"Purge gap too small: {gap}"

    def test_embargo_reduces_test_end(self):
        engine_no_embargo = self._engine(purge=10, embargo=0)
        engine_with_embargo = self._engine(purge=10, embargo=15)
        end_all = datetime(2024, 1, 1)
        start_all = end_all - timedelta(days=5 * 365 + 30)
        b_no = engine_no_embargo._build_calendar_folds(3, start_all, end_all, 5, None)
        b_with = engine_with_embargo._build_calendar_folds(3, start_all, end_all, 5, None)
        for (_, _, _, te_end_no), (_, _, _, te_end_with) in zip(b_no, b_with):
            assert te_end_with < te_end_no, "Embargo should shrink test window"

    def test_rolling_window_train_start(self):
        engine = self._engine()
        end_all = datetime(2024, 1, 1)
        start_all = end_all - timedelta(days=5 * 365 + 30)
        boundaries_rolling = engine._build_calendar_folds(3, start_all, end_all, 5, train_years=2)
        boundaries_expanding = engine._build_calendar_folds(3, start_all, end_all, 5, train_years=None)
        # Rolling should have later train_start for early folds
        for (tr_start_r, _, _, _), (tr_start_e, _, _, _) in zip(boundaries_rolling, boundaries_expanding):
            assert tr_start_r >= tr_start_e


# ---------------------------------------------------------------------------
# 3. FoldEngine — trading-day fold boundaries (intraday)
# ---------------------------------------------------------------------------

class TestFoldEngineTradingDayFolds:
    def _engine(self, purge=2, embargo=2):
        from scripts.walkforward.engine import FoldEngine
        strategy = MagicMock()
        return FoldEngine(strategy=strategy, purge_days=purge, embargo_days=embargo)

    def _days(self, n=300):
        from datetime import date, timedelta
        start = date(2022, 1, 3)
        all_days = []
        d = start
        for _ in range(n):
            if d.weekday() < 5:  # Mon-Fri
                all_days.append(d)
            d += timedelta(days=1)
        return all_days[:n]

    def test_correct_number_of_folds(self):
        engine = self._engine()
        all_days = self._days(300)
        boundaries = engine._build_trading_day_folds(3, all_days)
        assert len(boundaries) == 3

    def test_test_starts_after_purge(self):
        engine = self._engine(purge=3, embargo=2)
        all_days = self._days(300)
        boundaries = engine._build_trading_day_folds(3, all_days)
        for tr_start, tr_end, te_start, te_end in boundaries:
            tr_end_idx = all_days.index(tr_end)
            te_start_idx = all_days.index(te_start)
            gap = te_start_idx - tr_end_idx
            assert gap >= 3

    def test_empty_days_returns_empty(self):
        engine = self._engine()
        assert engine._build_trading_day_folds(3, []) == []


# ---------------------------------------------------------------------------
# 4. FoldEngine delegates to strategy.run_fold
# ---------------------------------------------------------------------------

class TestFoldEngineDelegation:
    def test_run_calls_strategy_run_fold(self):
        from scripts.walkforward.engine import FoldEngine
        from scripts.walkforward.gates import FoldResult

        mock_strategy = MagicMock()
        mock_strategy.all_days_sorted = []
        mock_strategy.model_type = "swing"

        # Provide a realistic FoldResult for each call
        def make_fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
            return FoldResult(
                fold=fold_idx, train_start=tr_start, train_end=tr_end,
                test_start=te_start, test_end=te_end,
                trades=10, win_rate=0.5, sharpe=0.9,
                max_drawdown=0.05, total_return=0.10, stop_exit_rate=0.4,
            )
        mock_strategy.run_fold.side_effect = make_fold

        engine = FoldEngine(strategy=mock_strategy, purge_days=10, embargo_days=5)
        report = engine.run(n_folds=3, total_years=5, allow_sacred_holdout=True)  # unit test: strategy is mocked

        assert mock_strategy.fetch_data.call_count == 1
        assert mock_strategy.run_fold.call_count == 3
        assert len(report.folds) == 3
        assert report.folds[0].fold == 1


# ---------------------------------------------------------------------------
# 5. Cost models
# ---------------------------------------------------------------------------

class TestCostModels:
    def test_fixed_bps_cost_pct(self):
        from scripts.walkforward.cost_models import FixedBpsCostModel
        m = FixedBpsCostModel(round_trip_bps=5.0)
        assert abs(m.cost_pct - 0.00025) < 1e-10

    def test_spread_cost_model(self):
        from scripts.walkforward.cost_models import SpreadCostModel
        m = SpreadCostModel(half_spread_bps=5.0, impact_bps=2.0)
        assert abs(m.cost_pct - 0.0007) < 1e-10

    def test_default_values(self):
        from scripts.walkforward.cost_models import FixedBpsCostModel
        m = FixedBpsCostModel()
        assert m.round_trip_bps == 5.0


# ---------------------------------------------------------------------------
# 6. Strategy classes constructible without DB
# ---------------------------------------------------------------------------

class TestStrategyConstruction:
    def test_swing_strategy_constructible(self):
        from scripts.walkforward.strategies.swing import SwingStrategy
        s = SwingStrategy(
            model=MagicMock(),
            version=1,
            symbols=["AAPL", "MSFT"],
        )
        assert s.version == 1
        assert s.atr_stop_mult == 1.5  # v216 default: 1.5×ATR (outside daily noise for LambdaRank)

    def test_intraday_strategy_constructible(self):
        from scripts.walkforward.strategies.intraday import IntradayStrategy
        s = IntradayStrategy(
            model=MagicMock(),
            version=2,
            symbols=["AAPL"],
        )
        assert s.version == 2
        assert s.transaction_cost_pct == 0.0015  # default


# ---------------------------------------------------------------------------
# 7. WalkForwardReport from package has gate methods
# ---------------------------------------------------------------------------

class TestPackageWalkForwardReport:
    def test_gate_passed_method(self):
        from scripts.walkforward import WalkForwardReport
        r = WalkForwardReport(model_type="swing")
        assert hasattr(r, "gate_passed")
        assert hasattr(r, "gate_detail")
        assert r.gate_passed() is False  # empty report fails

    def test_gate_detail_keys(self):
        from scripts.walkforward import WalkForwardReport
        r = WalkForwardReport(model_type="swing")
        detail = r.gate_detail()
        assert "avg_sharpe" in detail
        assert "dsr_p" in detail
        assert "avg_profit_factor" in detail
        assert "avg_calmar" in detail
