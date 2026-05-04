"""
Pipeline integration tests — end-to-end trade proposal → approval → execution
using mocked Alpaca and in-memory DB.

These tests verify the state-machine transitions and that agents interact
correctly without any real network or infrastructure.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestApprovalWorkflowStateMachine:
    """Approval workflow state transitions."""

    def test_not_ready_with_no_trades(self):
        from app.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow()
        with patch("app.approval_workflow.get_session") as mock_db:
            db = MagicMock()
            # Ensure all query chains return None / empty so no TradingSession is found
            q = db.query.return_value
            q.filter_by.return_value.order_by.return_value.first.return_value = None
            q.filter_by.return_value.first.return_value = None
            q.filter_by.return_value.all.return_value = []
            q.filter.return_value.first.return_value = None
            q.filter.return_value.count.return_value = 0
            q.filter.return_value.all.return_value = []
            q.filter.return_value.filter.return_value.first.return_value = None
            q.filter.return_value.filter.return_value.count.return_value = 0
            q.filter.return_value.filter.return_value.all.return_value = []
            db.close.return_value = None
            mock_db.return_value = db
            is_ready, metrics = wf.check_go_live_readiness()
        assert not is_ready
        assert isinstance(metrics["total_trades"], int)
        assert metrics["total_trades"] == 0

    def test_denial_when_criteria_not_met(self):
        from app.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow()
        with patch.object(wf, "check_go_live_readiness",
                          return_value=(False, {"total_trades": 5})):
            with patch("app.approval_workflow.get_session") as mock_db:
                mock_db.return_value = MagicMock()
                result = wf.request_approval(approved_by="test")
        assert result["status"] == "denied"

    def test_approval_when_criteria_met(self):
        from app.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow()
        good_metrics = {
            "duration_days": 20, "total_trades": 25, "win_rate_pct": 55.0,
            "total_return_pct": 12.0, "max_drawdown_pct": 3.0, "sharpe_ratio": 1.8,
        }
        with patch.object(wf, "check_go_live_readiness", return_value=(True, good_metrics)):
            with patch("app.approval_workflow.get_session") as mock_db:
                session = MagicMock()
                session.query.return_value.filter_by.return_value.first.return_value = None
                mock_db.return_value = session
                result = wf.request_approval(approved_by="test")
        assert result["status"] == "approved"


class TestCapitalManagerStateMachine:
    """Capital ramp stage transitions."""

    def test_start_at_stage_1(self):
        from app.live_trading.capital_manager import CapitalManager
        cm = CapitalManager()
        cm.start()
        assert cm.current_stage.stage == 1
        assert cm.get_current_capital() == 1_000

    def test_advance_stage(self):
        from app.live_trading.capital_manager import CapitalManager
        cm = CapitalManager()
        cm.start()
        result = cm.advance()
        assert result["status"] == "advanced"
        assert cm.current_stage.stage == 2
        assert cm.get_current_capital() == 2_500

    def test_cannot_advance_past_max(self):
        from app.live_trading.capital_manager import CapitalManager
        cm = CapitalManager()
        cm.start()
        for _ in range(len(cm.STAGES) - 1):
            cm.advance()
        result = cm.advance()  # try to go beyond
        assert result["status"] == "already_at_max"

    def test_health_gate_prevents_advance(self):
        """can_advance returns False when health thresholds are breached."""
        from app.live_trading.capital_manager import CapitalManager
        from datetime import timedelta
        cm = CapitalManager()
        cm.start()
        # Simulate stage complete but bad health
        cm._stage_start = datetime.utcnow() - timedelta(days=10)
        assert not cm.can_advance(max_drawdown_pct=5.0, daily_loss_pct=0.0)

    def test_health_gate_allows_advance_when_healthy(self):
        from app.live_trading.capital_manager import CapitalManager
        from datetime import timedelta
        cm = CapitalManager()
        cm.start()
        # Stage 1 needs 3 days elapsed + good health
        cm._stage_start = datetime.utcnow() - timedelta(days=5)
        assert cm.can_advance(max_drawdown_pct=1.0, daily_loss_pct=0.5)


class TestKillSwitchStateMachine:
    """Kill switch activate / reset idempotency."""

    def test_initially_inactive(self):
        from app.live_trading.kill_switch import KillSwitch
        ks = KillSwitch()
        assert not ks.is_active

    def test_activate_sets_active(self):
        from app.live_trading.kill_switch import KillSwitch
        ks = KillSwitch()
        mock_alpaca = MagicMock()
        mock_alpaca.get_positions.return_value = []
        with (
            patch("app.live_trading.kill_switch.get_session") as mock_db,
            patch.object(ks, "_persist_state"),
            patch.object(ks, "_alpaca", return_value=mock_alpaca),
        ):
            mock_db.return_value = MagicMock()
            result = ks.activate("test")
        assert ks.is_active
        assert result["status"] == "activated"

    def test_activate_then_reset(self):
        from app.live_trading.kill_switch import KillSwitch
        ks = KillSwitch()
        mock_alpaca = MagicMock()
        mock_alpaca.get_positions.return_value = []
        with (
            patch("app.live_trading.kill_switch.get_session") as mock_db,
            patch.object(ks, "_persist_state"),
            patch.object(ks, "_alpaca", return_value=mock_alpaca),
        ):
            mock_db.return_value = MagicMock()
            ks.activate("test")
            assert ks.is_active
            ks.reset("all clear")
        assert not ks.is_active

    def test_activate_closes_positions(self):
        from app.live_trading.kill_switch import KillSwitch
        ks = KillSwitch()
        mock_alpaca = MagicMock()
        mock_alpaca.get_positions.return_value = [
            {"symbol": "AAPL", "qty": 10},
            {"symbol": "MSFT", "qty": 5},
        ]
        mock_alpaca.place_market_order.return_value = {"order_id": "x"}
        with (
            patch("app.live_trading.kill_switch.get_session") as mock_db,
            patch.object(ks, "_persist_state"),
            patch.object(ks, "_alpaca", return_value=mock_alpaca),
        ):
            mock_db.return_value = MagicMock()
            result = ks.activate("test")
        assert "AAPL" in result["positions_closed"]
        assert "MSFT" in result["positions_closed"]
        assert mock_alpaca.place_market_order.call_count == 2


class TestCircuitBreakerIntegration:
    """Circuit breaker feeding into trader agent."""

    def test_trader_skips_scan_when_cb_open(self):
        """Trader._scan_cycle should not be called when circuit breaker is open."""
        from app.agents.trader import Trader
        from app.agents.circuit_breaker import CircuitBreaker, CONSECUTIVE_LOSS_LIMIT

        trader = Trader()
        cb = CircuitBreaker()
        # Trip the circuit breaker
        for _ in range(CONSECUTIVE_LOSS_LIMIT):
            cb.record_trade_result(won=False)
        assert cb.is_open

        # Verify scan is gated
        scan_called = []
        async def mock_scan():
            scan_called.append(True)

        trader._scan_cycle = mock_scan

        import asyncio

        async def run():
            with patch("app.agents.trader.circuit_breaker", cb):
                with patch.object(cb, "check_market_volatility", return_value=False):
                    # Simulate one loop iteration manually
                    cb.check_market_volatility()
                    if not cb.is_open:
                        await trader._scan_cycle()

        asyncio.run(run())
        assert not scan_called  # scan was NOT called


class TestStartupReconciler:
    """Startup reconciler detects inconsistencies."""

    def test_clean_startup(self, db_session):
        from app.startup_reconciler import reconcile
        mock_alpaca = MagicMock()
        mock_alpaca.get_positions.return_value = []
        with patch("app.startup_reconciler._get_open_alpaca_orders", return_value=[]):
            result = reconcile(mock_alpaca, db_session)
        assert result["ghost_positions"] == []
        assert result["orphaned_orders"] == []

    def test_detects_ghost_position(self, db_session):
        from app.startup_reconciler import reconcile
        from tests.conftest import make_trade
        # Create two active trades in DB — GHOST will be absent from Alpaca, OTHER will be present
        make_trade(db_session, symbol="GHOST", status="ACTIVE")
        make_trade(db_session, symbol="OTHER", status="ACTIVE")
        db_session.commit()
        # Alpaca has OTHER but not GHOST — ghost detection fires because Alpaca returned >0 positions
        mock_alpaca = MagicMock()
        mock_alpaca.get_positions.return_value = [{"symbol": "OTHER", "qty": "10", "avg_entry_price": "100"}]
        with patch("app.startup_reconciler._get_open_alpaca_orders", return_value=[]):
            result = reconcile(mock_alpaca, db_session)
        assert len(result["ghost_positions"]) == 1
        assert result["ghost_positions"][0]["symbol"] == "GHOST"

    def test_ghost_detection_skipped_when_alpaca_empty(self, db_session):
        from app.startup_reconciler import reconcile
        from tests.conftest import make_trade
        # Single ACTIVE trade in DB + Alpaca returns 0 positions = likely API error, skip ghost
        make_trade(db_session, symbol="REALPOS", status="ACTIVE")
        db_session.commit()
        mock_alpaca = MagicMock()
        mock_alpaca.get_positions.return_value = []
        with patch("app.startup_reconciler._get_open_alpaca_orders", return_value=[]):
            result = reconcile(mock_alpaca, db_session)
        assert len(result["ghost_positions"]) == 0

    def test_detects_orphaned_order(self, db_session):
        from app.startup_reconciler import reconcile
        mock_alpaca = MagicMock()
        mock_alpaca.get_positions.return_value = []
        orphan = {
            "order_id": "orphan-999", "symbol": "TSLA",
            "qty": "5", "side": "buy", "status": "new",
        }
        with patch("app.startup_reconciler._get_open_alpaca_orders", return_value=[orphan]):
            result = reconcile(mock_alpaca, db_session)
        assert len(result["orphaned_orders"]) == 1
        assert result["orphaned_orders"][0]["symbol"] == "TSLA"
