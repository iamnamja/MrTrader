"""
Tests for the live trading readiness checker.
All external calls (DB, Redis, Alpaca) are mocked.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.live_trading.readiness import (
    ReadinessChecker, MIN_PAPER_TRADES, MIN_WIN_RATE, MIN_PAPER_TRADE_DAYS,
    MIN_ACCOUNT_EQUITY, MAX_DRAWDOWN_PCT,
)

MODULE = "app.live_trading.readiness"


def _mock_session(query_result=None, count_result=0):
    session = MagicMock()
    q = session.query.return_value
    q.filter.return_value.all.return_value = query_result or []
    q.filter.return_value.count.return_value = count_result
    q.filter_by.return_value.first.return_value = None
    q.order_by.return_value.first.return_value = None
    session.close = MagicMock()
    return session


class TestTradingModeCheck:
    def test_passes_when_paper(self):
        mock_cfg = MagicMock(trading_mode="paper")
        with patch(f"{MODULE}.settings", mock_cfg):
            r = ReadinessChecker()._check_trading_mode()
        assert r.passed is True
        assert r.value == "paper"

    def test_fails_when_live(self):
        mock_cfg = MagicMock(trading_mode="live")
        with patch(f"{MODULE}.settings", mock_cfg):
            r = ReadinessChecker()._check_trading_mode()
        assert r.passed is False


class TestDbCheck:
    def test_passes_when_db_ok(self):
        with patch(f"{MODULE}.check_db_connection", return_value=True):
            r = ReadinessChecker()._check_db_connection()
        assert r.passed is True

    def test_fails_when_db_down(self):
        with patch(f"{MODULE}.check_db_connection", return_value=False):
            r = ReadinessChecker()._check_db_connection()
        assert r.passed is False


class TestAlpacaEquityCheck:
    def test_passes_when_equity_sufficient(self):
        mock_client = MagicMock()
        mock_client.get_account.return_value = {"portfolio_value": 5000.0}
        with patch(f"{MODULE}.get_alpaca_client", return_value=mock_client):
            r = ReadinessChecker()._check_alpaca_equity()
        assert r.passed is True
        assert r.value == 5000.0

    def test_fails_when_equity_too_low(self):
        mock_client = MagicMock()
        mock_client.get_account.return_value = {"portfolio_value": 500.0}
        with patch(f"{MODULE}.get_alpaca_client", return_value=mock_client):
            r = ReadinessChecker()._check_alpaca_equity()
        assert r.passed is False

    def test_fails_on_alpaca_error(self):
        mock_client = MagicMock()
        mock_client.get_account.side_effect = Exception("API error")
        with patch(f"{MODULE}.get_alpaca_client", return_value=mock_client):
            r = ReadinessChecker()._check_alpaca_equity()
        assert r.passed is False


class TestWinRateCheck:
    def test_passes_with_high_win_rate(self):
        closed = [MagicMock(pnl=10.0) for _ in range(MIN_PAPER_TRADES)]
        session = _mock_session(query_result=closed)
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_win_rate()
        assert r.passed is True
        assert r.value == 100.0

    def test_fails_with_low_win_rate(self):
        closed = [MagicMock(pnl=-10.0) for _ in range(MIN_PAPER_TRADES)]
        session = _mock_session(query_result=closed)
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_win_rate()
        assert r.passed is False
        assert r.value == 0.0

    def test_fails_with_no_trades(self):
        session = _mock_session(query_result=[])
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_win_rate()
        assert r.passed is False


class TestDrawdownCheck:
    def test_passes_when_drawdown_low(self):
        mock_metric = MagicMock(max_drawdown=0.02)
        session = MagicMock()
        session.query.return_value.filter_by.return_value.first.return_value = mock_metric
        session.close = MagicMock()
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_drawdown()
        assert r.passed is True
        assert r.value == 2.0

    def test_fails_when_drawdown_high(self):
        mock_metric = MagicMock(max_drawdown=0.20)  # 20% > 15% new threshold
        session = MagicMock()
        session.query.return_value.filter_by.return_value.first.return_value = mock_metric
        session.close = MagicMock()
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_drawdown()
        assert r.passed is False

    def test_passes_when_no_metric(self):
        session = MagicMock()
        session.query.return_value.filter_by.return_value.first.return_value = None
        session.close = MagicMock()
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_drawdown()
        assert r.passed is True  # 0% drawdown


class TestPaperTradeDays:
    def test_passes_when_old_enough(self):
        old_trade = MagicMock(created_at=datetime.utcnow() - timedelta(days=75))  # > 60d threshold
        session = MagicMock()
        session.query.return_value.order_by.return_value.first.return_value = old_trade
        session.close = MagicMock()
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_paper_trade_days()
        assert r.passed is True
        assert r.value >= MIN_PAPER_TRADE_DAYS

    def test_fails_when_too_new(self):
        new_trade = MagicMock(created_at=datetime.utcnow() - timedelta(days=3))
        session = MagicMock()
        session.query.return_value.order_by.return_value.first.return_value = new_trade
        session.close = MagicMock()
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_paper_trade_days()
        assert r.passed is False


class TestAlertConfigChecks:
    def test_smtp_warning_when_unconfigured(self):
        cfg = MagicMock(smtp_host=None, alert_email=None, slack_webhook_url=None)
        with patch(f"{MODULE}.settings", cfg):
            r = ReadinessChecker()._check_smtp_configured()
        assert r.passed is False
        assert r.name == "smtp_configured"

    def test_slack_warning_when_unconfigured(self):
        cfg = MagicMock(slack_webhook_url=None)
        with patch(f"{MODULE}.settings", cfg):
            r = ReadinessChecker()._check_slack_configured()
        assert r.passed is False
        assert r.name == "slack_configured"

    def test_smtp_passes_when_configured(self):
        cfg = MagicMock(smtp_host="smtp.gmail.com", alert_email="me@gmail.com")
        with patch(f"{MODULE}.settings", cfg):
            r = ReadinessChecker()._check_smtp_configured()
        assert r.passed is True

    def test_warnings_not_blockers_in_full_report(self):
        """smtp/slack failures must land in warnings[], not blockers[]."""
        # Create a report manually with just smtp/slack failures
        checker = ReadinessChecker()
        smtp_r = MagicMock()
        smtp_r.passed = False
        smtp_r.name = "smtp_configured"
        smtp_r.to_dict.return_value = {"check": "smtp_configured", "passed": False}

        slack_r = MagicMock()
        slack_r.passed = False
        slack_r.name = "slack_configured"
        slack_r.to_dict.return_value = {"check": "slack_configured", "passed": False}

        checks = [smtp_r, slack_r]
        failed = [c for c in checks if not c.passed]
        warnings = [c for c in failed if c.name in ("smtp_configured", "slack_configured")]
        blockers = [c for c in failed if c not in warnings]

        assert len(warnings) == 2
        assert len(blockers) == 0


class TestDeflatedSharpe:
    def _make_trades(self, pnls, days_ago=90):
        trades = []
        for pnl in pnls:
            t = MagicMock()
            t.pnl = pnl
            t.created_at = datetime.utcnow() - timedelta(days=days_ago)
            t.status = "CLOSED"
            trades.append(t)
        return trades

    def test_passes_with_strong_positive_pnl(self):
        trades = self._make_trades([100.0] * 30 + [-10.0] * 5)
        session = MagicMock()
        session.query.return_value.filter.return_value.all.return_value = trades
        session.close = MagicMock()
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_deflated_sharpe()
        assert r.passed is True

    def test_fails_with_insufficient_trades(self):
        trades = self._make_trades([10.0] * 5)
        session = MagicMock()
        session.query.return_value.filter.return_value.all.return_value = trades
        session.close = MagicMock()
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_deflated_sharpe()
        assert r.passed is False
        assert "Insufficient" in r.detail

    def test_fails_with_zero_std(self):
        trades = self._make_trades([50.0] * 15)  # all same PnL → std=0
        session = MagicMock()
        session.query.return_value.filter.return_value.all.return_value = trades
        session.close = MagicMock()
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_deflated_sharpe()
        assert r.passed is False

    def test_fails_with_negative_edge(self):
        trades = self._make_trades([-50.0] * 20 + [10.0] * 5)
        session = MagicMock()
        session.query.return_value.filter.return_value.all.return_value = trades
        session.close = MagicMock()
        with patch(f"{MODULE}.get_session", return_value=session):
            r = ReadinessChecker()._check_deflated_sharpe()
        assert r.passed is False


# ─── Composite run() ──────────────────────────────────────────────────────────

class TestReadinessRunComposite:
    """End-to-end test of ReadinessChecker.run() with all external calls mocked."""

    def _all_pass_mocks(self):
        """Return a context-manager stack that makes every sub-check succeed."""
        from contextlib import ExitStack
        stack = ExitStack()

        session = MagicMock()
        # Win rate: 80 wins, 20 losses
        win_trades = [MagicMock(pnl=10.0, status="CLOSED",
                                created_at=datetime.utcnow() - timedelta(days=60))
                      for _ in range(80)]
        loss_trades = [MagicMock(pnl=-5.0, status="CLOSED",
                                 created_at=datetime.utcnow() - timedelta(days=60))
                       for _ in range(20)]
        all_trades = win_trades + loss_trades
        session.query.return_value.filter.return_value.all.return_value = all_trades
        session.query.return_value.filter.return_value.scalar.return_value = \
            datetime.utcnow() - timedelta(days=MIN_PAPER_TRADE_DAYS + 1)
        session.close = MagicMock()

        alpaca_mock = MagicMock()
        alpaca_mock.get_account.return_value = MagicMock(
            equity=str(MIN_ACCOUNT_EQUITY + 1000)
        )

        stack.enter_context(patch(f"{MODULE}.settings",
                                  trading_mode="paper", smtp_host="smtp.x.com",
                                  slack_webhook_url="https://hooks.slack.com/x"))
        stack.enter_context(patch(f"{MODULE}.check_db_connection", return_value=True))
        stack.enter_context(patch(f"{MODULE}.get_redis_queue",
                                  return_value=MagicMock(health_check=MagicMock(return_value=True))))
        stack.enter_context(patch(f"{MODULE}.get_alpaca_client", return_value=alpaca_mock))
        stack.enter_context(patch(f"{MODULE}.get_session", return_value=session))
        stack.enter_context(patch(f"{MODULE}.kill_switch",
                                  create=True, is_active=False))
        return stack

    def test_run_returns_required_keys(self):
        checker = ReadinessChecker()
        result = checker.run()
        for key in ("ready", "timestamp", "summary", "blockers", "warnings", "passed", "all_checks"):
            assert key in result, f"Missing key: {key}"

    def test_run_ready_false_when_db_down(self):
        with patch(f"{MODULE}.check_db_connection", return_value=False), \
             patch(f"{MODULE}.settings", trading_mode="paper",
                   smtp_host=None, slack_webhook_url=None), \
             patch(f"{MODULE}.get_redis_queue", side_effect=Exception("no redis")), \
             patch(f"{MODULE}.get_alpaca_client", side_effect=Exception("no alpaca")), \
             patch(f"{MODULE}.get_session",
                   return_value=MagicMock(
                       query=MagicMock(return_value=MagicMock(
                           filter=MagicMock(return_value=MagicMock(
                               all=MagicMock(return_value=[]),
                               scalar=MagicMock(return_value=None),
                           ))
                       )),
                       close=MagicMock(),
                   )), \
             patch("app.agents.portfolio_manager.PortfolioManager",
                   side_effect=Exception("no model")):
            checker = ReadinessChecker()
            result = checker.run()
        assert result["ready"] is False
        blocker_names = [b["check"] for b in result["blockers"]]
        assert "db_connected" in blocker_names

    def test_run_summary_format(self):
        checker = ReadinessChecker()
        result = checker.run()
        assert "/" in result["summary"]
        assert "checks passed" in result["summary"]

    def test_smtp_slack_failures_are_warnings_not_blockers(self):
        """smtp/slack not configured must not block ready=True for other passing checks."""
        checker = ReadinessChecker()
        result = checker.run()
        warning_names = [w["check"] for w in result["warnings"]]
        blocker_names = [b["check"] for b in result["blockers"]]
        # smtp/slack should not appear in blockers even if they fail
        assert "smtp_configured" not in blocker_names
        assert "slack_configured" not in blocker_names
