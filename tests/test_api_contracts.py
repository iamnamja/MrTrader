"""
API contract tests — verify every dashboard endpoint returns the correct shape
and HTTP status, using mocked external dependencies (no real DB/Redis/Alpaca).

These tests guard against accidental schema breakage as the codebase evolves.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Client factory (context-manager — keeps patches alive during requests) ────

def _default_alpaca():
    m = MagicMock()
    m.health_check.return_value = True
    m.get_account.return_value = {
        "cash": 10_000.0, "buying_power": 10_000.0,
        "portfolio_value": 10_000.0, "equity": 10_000.0,
        "account_blocked": False, "status": "ACTIVE",
    }
    m.get_positions.return_value = []
    m.get_position.return_value = None
    m.place_market_order.return_value = {"order_id": "x", "symbol": "AAPL",
                                         "qty": 1, "side": "sell", "status": "filled",
                                         "created_at": datetime.utcnow().isoformat()}
    m.get_latest_price.return_value = 150.0
    return m


def _default_redis():
    m = MagicMock()
    m.health_check.return_value = True
    return m


def _empty_db_session():
    """A mock DB session that returns empty query results."""
    session = MagicMock()
    q = session.query.return_value
    q.filter_by.return_value.first.return_value = None
    q.filter.return_value.count.return_value = 0
    q.filter.return_value.all.return_value = []
    q.filter.return_value.filter.return_value.first.return_value = None
    q.order_by.return_value.filter.return_value.limit.return_value.all.return_value = []
    q.order_by.return_value.limit.return_value.all.return_value = []
    q.filter.return_value.first.return_value = None
    return session


@contextmanager
def make_client(alpaca=None, redis=None, db_session=None):
    """Context manager that keeps all patches active during the test."""
    alp = alpaca or _default_alpaca()
    red = redis or _default_redis()
    sess = db_session or _empty_db_session()
    with (
        patch("app.integrations.get_alpaca_client", return_value=alp),
        patch("app.integrations.get_redis_queue", return_value=red),
        patch("app.database.check_db_connection", return_value=True),
        patch("app.database.session.get_session", return_value=sess),
        patch("app.api.routes.get_session", return_value=sess),
        patch("app.orchestrator.AgentOrchestrator.start", new_callable=AsyncMock),
        patch("app.orchestrator.AgentOrchestrator.stop", new_callable=AsyncMock),
        patch("app.main.init_db"),
        patch("app.startup_reconciler.reconcile", return_value={"ghost_positions": [], "orphaned_orders": []}),
    ):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ── Health / status ────────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_root_health(self):
        with make_client() as client:
            r = client.get("/health")
            assert r.status_code in (200, 503)
            body = r.json()
            assert "status" in body
            assert "version" in body

    def test_api_dashboard_health(self):
        with make_client() as client:
            r = client.get("/api/dashboard/health")
            assert r.status_code == 200
            body = r.json()
            assert "status" in body
            assert "checks" in body
            assert "trading_mode" in body

    def test_system_health(self):
        with make_client() as client:
            r = client.get("/api/dashboard/system-health")
            assert r.status_code == 200
            body = r.json()
            assert body["database"] in ("OK", "FAIL")
            assert body["redis"] in ("OK", "FAIL")
            assert body["alpaca"] in ("OK", "FAIL")
            assert body["overall"] in ("HEALTHY", "DEGRADED")


# ── Positions ──────────────────────────────────────────────────────────────────

class TestPositionsEndpoint:
    def test_empty_positions(self):
        with make_client() as client:
            r = client.get("/api/dashboard/positions")
            assert r.status_code == 200
            assert isinstance(r.json(), list)

    def test_positions_with_data(self):
        alp = _default_alpaca()
        alp.get_positions.return_value = [{
            "symbol": "AAPL", "qty": 10, "avg_entry_price": 150.0,
            "market_value": 1550.0, "unrealized_pl": 50.0,
            "unrealized_plpc": 0.033, "current_price": 155.0,
        }]
        with make_client(alpaca=alp) as client:
            r = client.get("/api/dashboard/positions")
        assert r.status_code == 200
        positions = r.json()
        assert len(positions) == 1
        p = positions[0]
        assert p["symbol"] == "AAPL"
        assert "quantity" in p
        assert "avg_price" in p

    def test_alpaca_error_returns_500(self):
        alp = _default_alpaca()
        alp.get_positions.side_effect = RuntimeError("Alpaca down")
        with make_client(alpaca=alp) as client:
            r = client.get("/api/dashboard/positions")
        assert r.status_code == 500


# ── Trades ─────────────────────────────────────────────────────────────────────

class TestTradesEndpoint:
    def test_trades_returns_list(self):
        with make_client() as client:
            r = client.get("/api/dashboard/trades")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_trades_status_filter_accepted(self):
        with make_client() as client:
            r = client.get("/api/dashboard/trades?status=CLOSED")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_trades_limit_param(self):
        with make_client() as client:
            r = client.get("/api/dashboard/trades?limit=10")
        assert r.status_code == 200

    def test_trades_schema_fields(self):
        """When trades are present the response includes signal_type."""
        from app.database.models import Trade
        trade = MagicMock(spec=Trade)
        trade.id = 1
        trade.symbol = "AAPL"
        trade.direction = "BUY"
        trade.entry_price = 150.0
        trade.exit_price = 155.0
        trade.quantity = 10
        trade.pnl = 50.0
        trade.status = "CLOSED"
        trade.signal_type = "EMA_CROSSOVER"
        trade.stop_price = 145.0
        trade.target_price = 160.0
        trade.created_at = datetime.utcnow()
        trade.closed_at = datetime.utcnow()

        sess = _empty_db_session()
        sess.query.return_value.order_by.return_value.limit.return_value.all.return_value = [trade]

        with make_client(db_session=sess) as client:
            r = client.get("/api/dashboard/trades")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["signal_type"] == "EMA_CROSSOVER"
        assert rows[0]["stop_price"] == 145.0


# ── Decisions ──────────────────────────────────────────────────────────────────

class TestDecisionsEndpoint:
    def test_decisions_returns_list(self):
        with make_client() as client:
            r = client.get("/api/dashboard/decisions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_decisions_limit_param(self):
        with make_client() as client:
            r = client.get("/api/dashboard/decisions?limit=5")
        assert r.status_code == 200


# ── Approval workflow ──────────────────────────────────────────────────────────

class TestApprovalWorkflow:
    def test_approval_status_shape(self):
        with patch("app.approval_workflow.approval_workflow") as mock_wf:
            mock_wf.check_go_live_readiness.return_value = (False, {
                "duration_days": 5, "total_trades": 10, "win_rate_pct": 45.0,
                "total_return_pct": 3.0, "max_drawdown_pct": 2.5, "sharpe_ratio": 0.8,
            })
            with make_client() as client:
                r = client.get("/api/dashboard/approval/status")
        assert r.status_code == 200
        body = r.json()
        assert "is_ready" in body
        assert "metrics" in body
        assert body["is_ready"] is False

    def test_approval_denied_when_criteria_not_met(self):
        with patch("app.approval_workflow.approval_workflow") as mock_wf:
            mock_wf.request_approval.return_value = {"status": "denied", "reason": "test"}
            with make_client() as client:
                r = client.post("/api/dashboard/approval/request-live?approved_by=test")
        assert r.status_code == 400

    def test_approval_granted_when_criteria_met(self):
        with patch("app.approval_workflow.approval_workflow") as mock_wf:
            mock_wf.request_approval.return_value = {"status": "approved", "approved_by": "test"}
            with make_client() as client:
                r = client.post("/api/dashboard/approval/request-live?approved_by=test")
        assert r.status_code == 200
        assert r.json()["status"] == "approved"


# ── Kill switch ────────────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_activate_kill_switch(self):
        with patch("app.live_trading.kill_switch.kill_switch") as mock_ks:
            mock_ks.activate.return_value = {
                "status": "activated", "reason": "test",
                "positions_closed": [], "errors": [],
                "activated_at": datetime.utcnow().isoformat(),
            }
            with make_client() as client:
                r = client.post("/api/dashboard/live/kill-switch?reason=test")
        assert r.status_code == 200
        assert r.json()["status"] == "activated"

    def test_reset_kill_switch(self):
        with patch("app.live_trading.kill_switch.kill_switch") as mock_ks:
            mock_ks.reset.return_value = None
            with make_client() as client:
                r = client.post("/api/dashboard/live/kill-switch/reset?reason=test")
        assert r.status_code == 200
        assert r.json()["status"] == "kill_switch_reset"

    def test_kill_switch_idempotent(self):
        with patch("app.live_trading.kill_switch.kill_switch") as mock_ks:
            mock_ks.activate.return_value = {
                "status": "activated", "reason": "test",
                "positions_closed": [], "errors": [],
                "activated_at": datetime.utcnow().isoformat(),
            }
            with make_client() as client:
                r1 = client.post("/api/dashboard/live/kill-switch?reason=first")
                r2 = client.post("/api/dashboard/live/kill-switch?reason=second")
        assert r1.status_code == 200
        assert r2.status_code == 200


# ── Live status ────────────────────────────────────────────────────────────────

class TestLiveStatus:
    def test_live_status_shape(self):
        with (
            patch("app.live_trading.monitoring.monitor") as mock_mon,
            patch("app.live_trading.capital_manager.capital_manager") as mock_cm,
            patch("app.live_trading.kill_switch.kill_switch") as mock_ks,
            patch("app.trading_modes.mode_manager") as mock_mm,
        ):
            mock_mon.health_check.return_value = {
                "status": "healthy", "account_value": 10000.0,
                "pnl_today_pct": 0.5, "max_drawdown_pct": 1.0,
                "alpaca_connected": True,
            }
            mock_cm.get_current_capital.return_value = 1000.0
            mock_cm.current_stage.stage = 1
            mock_ks.is_active = False
            mock_mm.mode.value = "paper"
            with make_client() as client:
                r = client.get("/api/dashboard/live/status")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "kill_switch_active" in body
        assert "trading_mode" in body

    def test_capital_stages_shape(self):
        with patch("app.live_trading.capital_manager.capital_manager") as mock_cm:
            mock_cm.get_all_stages.return_value = [
                {"stage": 1, "capital": 1000, "duration_days": 3,
                 "is_current": True, "days_elapsed": 1},
            ]
            with make_client() as client:
                r = client.get("/api/dashboard/live/capital-stages")
        assert r.status_code == 200
        stages = r.json()
        assert isinstance(stages, list)
        assert "stage" in stages[0]
        assert "capital" in stages[0]


# ── Circuit breaker unit tests ─────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_circuit_breaker_initial_closed(self):
        from app.agents.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        assert cb.is_closed
        assert not cb.is_open

    def test_trips_on_consecutive_losses(self):
        from app.agents.circuit_breaker import CircuitBreaker, CONSECUTIVE_LOSS_LIMIT
        cb = CircuitBreaker()
        for _ in range(CONSECUTIVE_LOSS_LIMIT):
            cb.record_trade_result(won=False)
        assert cb.is_open
        assert "consecutive_losses" in cb._open_reason

    def test_resets_consecutive_losses_on_win(self):
        from app.agents.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        cb.record_trade_result(won=False)
        cb.record_trade_result(won=False)
        cb.record_trade_result(won=True)
        assert cb.is_closed
        assert cb._consecutive_losses == 0

    def test_trips_on_network_errors(self):
        from app.agents.circuit_breaker import CircuitBreaker, NETWORK_ERROR_LIMIT
        cb = CircuitBreaker()
        for _ in range(NETWORK_ERROR_LIMIT + 1):
            cb.record_network_error()
        assert cb.is_open

    def test_manual_reset(self):
        from app.agents.circuit_breaker import CircuitBreaker, CONSECUTIVE_LOSS_LIMIT
        cb = CircuitBreaker()
        for _ in range(CONSECUTIVE_LOSS_LIMIT):
            cb.record_trade_result(won=False)
        assert cb.is_open
        cb.reset("test")
        assert cb.is_closed

    def test_status_dict_shape(self):
        from app.agents.circuit_breaker import CircuitBreaker
        s = CircuitBreaker().status()
        assert "is_open" in s
        assert "consecutive_losses" in s
        assert "recent_errors" in s
        assert "last_vix" in s
