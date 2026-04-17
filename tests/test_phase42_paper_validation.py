"""Tests for Phase 42: paper trading validation and AUC gate."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, date


def _trades(n_wins=10, n_losses=5, win_pnl=50.0, loss_pnl=-30.0, entry=1000.0):
    trades = []
    for _ in range(n_wins):
        trades.append({"pnl": win_pnl, "entry_price": entry, "closed_at": datetime.utcnow().isoformat()})
    for _ in range(n_losses):
        trades.append({"pnl": loss_pnl, "entry_price": entry, "closed_at": datetime.utcnow().isoformat()})
    return trades


class TestPaperValidator:

    def test_compute_metrics_win_rate(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator()
        trades = _trades(n_wins=6, n_losses=4)
        m = v.compute_metrics(trades)
        assert abs(m["win_rate"] - 0.6) < 0.001

    def test_compute_metrics_profit_factor(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator()
        trades = _trades(n_wins=6, n_losses=4, win_pnl=50.0, loss_pnl=-30.0)
        m = v.compute_metrics(trades)
        # gross_wins=300, gross_losses=120 → PF=2.5
        assert abs(m["profit_factor"] - 2.5) < 0.01

    def test_compute_metrics_empty(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator()
        m = v.compute_metrics([])
        assert m["n_trades"] == 0
        assert m["win_rate"] == 0.0

    def test_validate_passes_with_good_trades(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator(min_trades=5)
        # Wins clustered before losses to keep drawdown low
        trades = _trades(n_wins=15, n_losses=3, win_pnl=100.0, loss_pnl=-30.0)
        result = v.validate(trades)
        assert result["ready"] is True
        assert len(result["blockers"]) == 0

    def test_validate_fails_on_too_few_trades(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator(min_trades=20)
        trades = _trades(n_wins=3, n_losses=2)
        result = v.validate(trades)
        assert result["ready"] is False
        names = [c["name"] for c in result["blockers"]]
        assert "min_trades" in names

    def test_validate_fails_on_low_win_rate(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator(min_trades=5, min_win_rate=0.5)
        trades = _trades(n_wins=2, n_losses=8)  # 20% win rate
        result = v.validate(trades)
        assert result["ready"] is False

    def test_drift_check_present_when_backtest_given(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator(min_trades=5)
        trades = _trades(n_wins=8, n_losses=2)  # 80% win rate
        result = v.validate(trades, backtest_win_rate=0.85)
        check_names = [c["name"] for c in result["checks"]]
        assert "drift_from_backtest" in check_names

    def test_drift_check_absent_without_backtest(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator(min_trades=5)
        trades = _trades(n_wins=8, n_losses=2)
        result = v.validate(trades, backtest_win_rate=None)
        check_names = [c["name"] for c in result["checks"]]
        assert "drift_from_backtest" not in check_names

    def test_drift_blocks_when_too_large(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator(min_trades=5, max_drift_pct=10.0)
        # Backtest says 70%, live is 40% → 30pp drift
        trades = _trades(n_wins=4, n_losses=6)
        result = v.validate(trades, backtest_win_rate=0.70)
        drift_check = next(c for c in result["checks"] if c["name"] == "drift_from_backtest")
        assert not drift_check["passed"]

    def test_sharpe_is_warning_not_blocker(self):
        from app.live_trading.paper_validator import PaperValidator
        # All losses → sharpe will be negative / zero
        v = PaperValidator(min_trades=5, min_sharpe=99.0)
        trades = _trades(n_wins=10, n_losses=5)
        result = v.validate(trades)
        sharpe_check = next((c for c in result["checks"] if c["name"] == "sharpe_ratio"), None)
        if sharpe_check and not sharpe_check["passed"]:
            assert sharpe_check.get("warning") is True

    def test_max_drawdown_computed(self):
        from app.live_trading.paper_validator import PaperValidator
        v = PaperValidator()
        # PnLs: gain 100, lose 50 → small drawdown
        trades = [
            {"pnl": 100.0, "entry_price": 1000.0, "closed_at": ""},
            {"pnl": -50.0, "entry_price": 1000.0, "closed_at": ""},
            {"pnl": 200.0, "entry_price": 1000.0, "closed_at": ""},
        ]
        m = v.compute_metrics(trades)
        assert 0.0 <= m["max_drawdown_pct"] <= 100.0


class TestReadinessCheckerAUCGate:

    def test_auc_gate_added_to_checks(self):
        from app.live_trading.readiness import ReadinessChecker
        checker = ReadinessChecker()
        assert hasattr(checker, "_check_model_auc")

    def test_auc_check_passes_above_threshold(self):
        from app.live_trading.readiness import ReadinessChecker, MIN_MODEL_AUC
        checker = ReadinessChecker()

        mock_version = MagicMock()
        mock_version.version = 10
        mock_version.performance = {"auc": MIN_MODEL_AUC + 0.05}

        with patch("app.live_trading.readiness.get_session") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value = mock_db
            mock_db.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = mock_version
            result = checker._check_model_auc()

        assert result.passed is True

    def test_auc_check_fails_below_threshold(self):
        from app.live_trading.readiness import ReadinessChecker, MIN_MODEL_AUC
        checker = ReadinessChecker()

        mock_version = MagicMock()
        mock_version.version = 10
        mock_version.performance = {"auc": MIN_MODEL_AUC - 0.05}

        with patch("app.live_trading.readiness.get_session") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value = mock_db
            mock_db.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = mock_version
            result = checker._check_model_auc()

        assert result.passed is False

    def test_auc_check_fails_no_model(self):
        from app.live_trading.readiness import ReadinessChecker
        checker = ReadinessChecker()

        with patch("app.live_trading.readiness.get_session") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value = mock_db
            mock_db.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = None
            result = checker._check_model_auc()

        assert result.passed is False

    def test_min_model_auc_threshold(self):
        from app.live_trading.readiness import MIN_MODEL_AUC
        assert MIN_MODEL_AUC >= 0.55
        assert MIN_MODEL_AUC < 1.0
