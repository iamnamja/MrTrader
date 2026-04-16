"""
Unit tests for Phase 6: Dashboard API schemas, routes, and WebSocket manager.

All tests are pure-Python — no network, database, Redis, or Alpaca calls.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import (
    AgentDecisionResponse,
    DailyMetricResponse,
    DashboardSummaryResponse,
    PositionResponse,
    SystemHealthResponse,
    TradeResponse,
)
from app.api.websocket import WebSocketManager


# ─── Schema tests ─────────────────────────────────────────────────────────────

class TestSchemas:
    def test_position_response_minimal(self):
        p = PositionResponse(symbol="AAPL", quantity=10, avg_price=150.0)
        assert p.symbol == "AAPL"
        assert p.current_price is None
        assert p.pnl_unrealized is None

    def test_position_response_full(self):
        p = PositionResponse(
            symbol="MSFT", quantity=5, avg_price=300.0,
            current_price=310.0, pnl_unrealized=50.0, pnl_unrealized_pct=3.33,
        )
        assert p.pnl_unrealized == 50.0
        assert p.pnl_unrealized_pct == 3.33

    def test_trade_response(self):
        t = TradeResponse(
            id=1, symbol="AAPL", direction="BUY", entry_price=150.0,
            quantity=10, status="ACTIVE", created_at=datetime.utcnow(),
        )
        assert t.direction == "BUY"
        assert t.exit_price is None
        assert t.pnl is None

    def test_agent_decision_response(self):
        d = AgentDecisionResponse(
            id=1, agent_name="risk_manager", decision_type="TRADE_APPROVED",
            timestamp=datetime.utcnow(), reasoning={"symbol": "AAPL"},
        )
        assert d.agent_name == "risk_manager"
        assert d.reasoning["symbol"] == "AAPL"

    def test_dashboard_summary_response(self):
        s = DashboardSummaryResponse(
            timestamp=datetime.utcnow(),
            account_value=21000.0, buying_power=10000.0, cash=5000.0,
            daily_pnl=100.0, daily_pnl_pct=0.5,
            total_pnl=1000.0, total_pnl_pct=5.0,
            open_positions_count=3, trades_today_count=2,
            trading_mode="paper", system_status="healthy",
        )
        assert s.account_value == 21000.0
        assert s.trading_mode == "paper"

    def test_system_health_response(self):
        h = SystemHealthResponse(
            database="OK", redis="OK", alpaca="FAIL",
            overall="DEGRADED", timestamp=datetime.utcnow().isoformat(),
        )
        assert h.overall == "DEGRADED"

    def test_daily_metric_response(self):
        m = DailyMetricResponse(date="2026-01-15", daily_pnl=250.0, max_drawdown=0.01)
        assert m.date == "2026-01-15"

    def test_optional_fields_default_to_none(self):
        t = TradeResponse(
            id=2, symbol="GOOG", direction="BUY", entry_price=200.0,
            quantity=1, status="PENDING", created_at=datetime.utcnow(),
        )
        assert t.exit_price is None
        assert t.pnl is None
        assert t.closed_at is None


# ─── WebSocket manager ────────────────────────────────────────────────────────

class TestWebSocketManager:
    def setup_method(self):
        self.mgr = WebSocketManager()

    @pytest.mark.asyncio
    async def test_connect_adds_connection(self):
        ws = AsyncMock()
        await self.mgr.connect(ws)
        assert ws in self.mgr.active_connections
        ws.accept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self):
        ws = AsyncMock()
        await self.mgr.connect(ws)
        self.mgr.disconnect(ws)
        assert ws not in self.mgr.active_connections

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_is_safe(self):
        ws = AsyncMock()
        self.mgr.disconnect(ws)  # should not raise

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self):
        ws1, ws2 = AsyncMock(), AsyncMock()
        await self.mgr.connect(ws1)
        await self.mgr.connect(ws2)
        await self.mgr.broadcast({"type": "test"})
        ws1.send_json.assert_awaited_once()
        ws2.send_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_drops_dead_connections(self):
        ws_ok = AsyncMock()
        ws_dead = AsyncMock()
        ws_dead.send_json.side_effect = Exception("disconnected")
        await self.mgr.connect(ws_ok)
        await self.mgr.connect(ws_dead)
        await self.mgr.broadcast({"type": "test"})
        assert ws_dead not in self.mgr.active_connections
        assert ws_ok in self.mgr.active_connections

    @pytest.mark.asyncio
    async def test_send_update_includes_type_and_timestamp(self):
        ws = AsyncMock()
        await self.mgr.connect(ws)
        await self.mgr.send_update("trade_executed", {"symbol": "AAPL"})
        payload = ws.send_json.call_args[0][0]
        assert payload["type"] == "trade_executed"
        assert "timestamp" in payload
        assert payload["data"]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_broadcast_empty_connections_is_safe(self):
        await self.mgr.broadcast({"type": "ping"})  # no connections — should not raise


# ─── Route tests (via TestClient) ────────────────────────────────────────────

def _mock_alpaca():
    alpaca = MagicMock()
    alpaca.health_check.return_value = True
    alpaca.get_account.return_value = {
        "portfolio_value": 21000.0,
        "buying_power": 10000.0,
        "cash": 5000.0,
    }
    alpaca.get_positions.return_value = [
        {"symbol": "AAPL", "quantity": 10, "avg_price": 150.0,
         "current_price": 155.0, "pnl_unrealized": 50.0},
    ]
    alpaca.get_position.return_value = {
        "symbol": "AAPL", "quantity": 10, "avg_price": 150.0,
    }
    alpaca.place_market_order.return_value = {"id": "order-1"}
    return alpaca


def _make_client():
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestDashboardRoutes:
    def setup_method(self):
        self.alpaca = _mock_alpaca()
        self.client = _make_client()

    def test_system_health_endpoint(self):
        with patch("app.api.routes._alpaca", return_value=self.alpaca), \
             patch("app.api.routes._redis") as mock_redis, \
             patch("app.api.routes.check_db_connection", return_value=True):
            mock_redis.return_value.health_check.return_value = True
            r = self.client.get("/api/dashboard/system-health")
        assert r.status_code == 200
        data = r.json()
        assert "database" in data
        assert "alpaca" in data
        assert "overall" in data

    def test_positions_endpoint(self):
        with patch("app.api.routes._alpaca", return_value=self.alpaca):
            r = self.client.get("/api/dashboard/positions")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "AAPL"

    def test_trades_endpoint_empty(self):
        with patch("app.api.routes.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value = mock_db
            r = self.client.get("/api/dashboard/trades")
        assert r.status_code == 200
        assert r.json() == []

    def test_decisions_endpoint_empty(self):
        with patch("app.api.routes.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
            mock_sess.return_value = mock_db
            r = self.client.get("/api/dashboard/decisions")
        assert r.status_code == 200
        assert r.json() == []

    def test_pause_endpoint(self):
        with patch("app.orchestrator.orchestrator") as mock_orch:
            r = self.client.post("/api/dashboard/control/pause")
        assert r.status_code == 200
        assert r.json()["status"] == "trading_paused"

    def test_resume_endpoint(self):
        with patch("app.orchestrator.orchestrator") as mock_orch:
            r = self.client.post("/api/dashboard/control/resume")
        assert r.status_code == 200
        assert r.json()["status"] == "trading_resumed"

    def test_kill_switch_no_positions(self):
        alpaca = _mock_alpaca()
        alpaca.get_positions.return_value = []
        with patch("app.api.routes._alpaca", return_value=alpaca):
            r = self.client.post("/api/dashboard/control/kill-switch")
        assert r.status_code == 200
        data = r.json()
        assert data["closed"] == []

    def test_kill_switch_closes_positions(self):
        with patch("app.api.routes._alpaca", return_value=self.alpaca):
            r = self.client.post("/api/dashboard/control/kill-switch")
        assert r.status_code == 200
        assert "AAPL" in r.json()["closed"]

    def test_close_position_not_found(self):
        alpaca = _mock_alpaca()
        alpaca.get_position.return_value = None
        with patch("app.api.routes._alpaca", return_value=alpaca):
            r = self.client.post("/api/dashboard/control/close-position/AAPL")
        assert r.status_code == 404

    def test_dashboard_html_served(self):
        r = self.client.get("/dashboard")
        assert r.status_code == 200
        assert "MrTrader" in r.text
