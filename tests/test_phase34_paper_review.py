"""Tests for Phase 34: paper trading review + go/no-go gate."""
from datetime import date
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(
    model_type="swing",
    total_trades=50,
    win_rate=0.55,
    sharpe=0.8,
    max_dd=0.08,
    profit_factor=1.4,
):
    from app.backtesting.metrics import BacktestResult, Trade
    r = BacktestResult(model_type=model_type)
    r.total_trades = total_trades
    r.winning_trades = int(total_trades * win_rate)
    r.losing_trades = total_trades - r.winning_trades
    r.win_rate = win_rate
    r.sharpe_ratio = sharpe
    r.max_drawdown_pct = max_dd
    r.profit_factor = profit_factor
    r.avg_pnl_pct = 0.015
    r.avg_hold_bars = 4.0
    r.total_pnl = 500.0
    r.trades = [
        Trade(
            symbol="X", entry_date=date(2023, 1, 1), exit_date=date(2023, 1, 5),
            entry_price=100.0, exit_price=104.0, quantity=10,
            pnl=40.0, pnl_pct=0.04, hold_bars=4,
            exit_reason="TARGET", trade_type=model_type,
        )
    ]
    return r


# ── BacktestReadinessChecker ──────────────────────────────────────────────────

class TestBacktestReadinessChecker:

    def _checker(self, **kwargs):
        from app.live_trading.backtest_readiness import BacktestReadinessChecker
        return BacktestReadinessChecker(**kwargs)

    def test_ready_when_all_criteria_met(self):
        checker = self._checker()
        result = _make_result(
            total_trades=50, win_rate=0.55, sharpe=0.8,
            max_dd=0.08, profit_factor=1.4,
        )
        report = checker.evaluate(swing_result=result)
        assert report["ready"] is True
        assert len(report["blockers"]) == 0

    def test_not_ready_when_sharpe_too_low(self):
        checker = self._checker()
        result = _make_result(sharpe=0.2)
        report = checker.evaluate(swing_result=result)
        assert report["ready"] is False
        blocker_names = [b["check"] for b in report["blockers"]]
        assert "swing_sharpe" in blocker_names

    def test_not_ready_when_win_rate_too_low(self):
        checker = self._checker()
        result = _make_result(win_rate=0.30)
        report = checker.evaluate(swing_result=result)
        assert report["ready"] is False
        blocker_names = [b["check"] for b in report["blockers"]]
        assert "swing_win_rate" in blocker_names

    def test_not_ready_when_drawdown_too_high(self):
        checker = self._checker()
        result = _make_result(max_dd=0.25)
        report = checker.evaluate(swing_result=result)
        assert report["ready"] is False
        blocker_names = [b["check"] for b in report["blockers"]]
        assert "swing_max_drawdown" in blocker_names

    def test_not_ready_when_profit_factor_below_one(self):
        checker = self._checker()
        result = _make_result(profit_factor=0.85)
        report = checker.evaluate(swing_result=result)
        assert report["ready"] is False
        blocker_names = [b["check"] for b in report["blockers"]]
        assert "swing_profit_factor" in blocker_names

    def test_low_trade_count_is_warning_not_blocker(self):
        """Too few trades is a warning (data issue), not a hard failure."""
        checker = self._checker()
        result = _make_result(total_trades=10)
        report = checker.evaluate(swing_result=result)
        warning_names = [w["check"] for w in report["warnings"]]
        assert "swing_min_trades" in warning_names
        blocker_names = [b["check"] for b in report["blockers"]]
        assert "swing_min_trades" not in blocker_names

    def test_no_result_is_a_blocker(self):
        checker = self._checker()
        report = checker.evaluate()
        assert report["ready"] is False
        assert len(report["blockers"]) > 0

    def test_zero_trades_skips_metric_checks(self):
        checker = self._checker()
        result = _make_result(total_trades=0)
        result.sharpe_ratio = 0.0
        result.win_rate = 0.0
        result.max_drawdown_pct = 0.0
        result.profit_factor = 0.0
        report = checker.evaluate(swing_result=result)
        # Only min_trades check should appear (and as warning)
        check_names = [c["check"] for c in report["all_checks"]]
        assert "swing_sharpe" not in check_names

    def test_evaluates_both_models(self):
        checker = self._checker()
        swing = _make_result("swing", sharpe=0.8, win_rate=0.55)
        intraday = _make_result("intraday", sharpe=0.6, win_rate=0.50)
        report = checker.evaluate(swing_result=swing, intraday_result=intraday)
        check_names = [c["check"] for c in report["all_checks"]]
        assert any("swing" in n for n in check_names)
        assert any("intraday" in n for n in check_names)

    def test_model_results_in_report(self):
        checker = self._checker()
        result = _make_result()
        report = checker.evaluate(swing_result=result)
        assert "swing" in report["model_results"]

    def test_custom_thresholds_respected(self):
        checker = self._checker(min_sharpe=1.5)  # very strict
        result = _make_result(sharpe=0.8)
        report = checker.evaluate(swing_result=result)
        assert report["ready"] is False

    def test_check_result_to_dict(self):
        from app.live_trading.backtest_readiness import BacktestCheckResult
        c = BacktestCheckResult("test_check", True, 0.8, "All good")
        d = c.to_dict()
        assert d["check"] == "test_check"
        assert d["passed"] is True
        assert d["value"] == 0.8


# ── BacktestCheckResult ───────────────────────────────────────────────────────

class TestBacktestCheckResult:

    def test_warning_flag_default_false(self):
        from app.live_trading.backtest_readiness import BacktestCheckResult
        c = BacktestCheckResult("x", False, None, "detail")
        assert c.is_warning is False

    def test_warning_flag_set_true(self):
        from app.live_trading.backtest_readiness import BacktestCheckResult
        c = BacktestCheckResult("x", False, None, "detail", is_warning=True)
        assert c.is_warning is True


# ── Review script helpers ─────────────────────────────────────────────────────

class TestReviewScript:

    def test_enable_paper_mode_logs_audit(self):
        """_enable_paper_mode should add an AuditLog row."""
        from scripts.review_paper_trading import _enable_paper_mode

        mock_db = MagicMock()
        with patch("app.database.session.get_session", return_value=mock_db):
            _enable_paper_mode()

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_enable_paper_mode_rolls_back_on_error(self):
        from scripts.review_paper_trading import _enable_paper_mode

        mock_db = MagicMock()
        mock_db.commit.side_effect = Exception("db error")
        with patch("app.database.session.get_session", return_value=mock_db):
            _enable_paper_mode()  # should not raise

        mock_db.rollback.assert_called_once()
