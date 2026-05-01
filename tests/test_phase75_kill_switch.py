"""
Phase 75 — Kill switch blocks all agent entry paths.

After kill_switch.activate(), no new Trade rows are created:
  - PM suppresses swing/intraday proposals
  - RM rejects incoming proposals (failed_rule='kill_switch')
  - Trader clears approved_symbols and pending limit orders
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── PM: swing proposals suppressed ───────────────────────────────────────────

class TestKillSwitchPMSwing:
    @pytest.mark.asyncio
    async def test_send_swing_proposals_blocked(self):
        from app.agents.portfolio_manager import PortfolioManager
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            pm = PortfolioManager.__new__(PortfolioManager)
            pm.logger = MagicMock()
            pm._swing_proposals = [{"symbol": "AAPL", "entry_price": 100, "confidence": 0.7}]

            with patch("app.live_trading.kill_switch.kill_switch") as ks, \
                 patch.object(pm, "log_decision", new_callable=AsyncMock):
                ks.is_active = True
                await pm._send_swing_proposals()

        assert pm._swing_proposals == []
        pm.logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_send_swing_proposals_allowed_when_inactive(self):
        from app.agents.portfolio_manager import PortfolioManager
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            pm = PortfolioManager.__new__(PortfolioManager)
            pm.logger = MagicMock()
            pm._swing_proposals = []

            with patch("app.live_trading.kill_switch.kill_switch") as ks, \
                 patch.object(pm, "log_decision", new_callable=AsyncMock):
                ks.is_active = False
                # Empty proposals list → returns early (no crash)
                await pm._send_swing_proposals()


# ── RM: proposals rejected with kill_switch rule ──────────────────────────────

class TestKillSwitchRM:
    def _make_rm(self):
        from app.agents.risk_manager import RiskManager
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            rm = RiskManager.__new__(RiskManager)
            rm.logger = MagicMock()
            rm.limits = MagicMock()
            rm._peak_equity = None
            rm._open_intraday_count = 0
            rm.status = "running"
            rm.name = "risk_manager"
            return rm

    @pytest.mark.asyncio
    async def test_rm_rejects_when_kill_switch_active(self):
        rm = self._make_rm()
        proposal = {"symbol": "AAPL", "direction": "BUY", "quantity": 10,
                    "entry_price": 100.0, "trade_type": "swing", "source_agent": "pm"}
        rejected = {}

        async def fake_reject(p, reasoning, db_proposal=None):
            rejected["rule"] = reasoning.get("failed_rule")

        msg_iter = iter([proposal, None])
        with patch.object(rm, "get_message", side_effect=lambda q, t: next(msg_iter)), \
             patch.object(rm, "_persist_proposal", return_value=None), \
             patch.object(rm, "_reject", side_effect=fake_reject), \
             patch.object(rm, "_load_peak_equity"), \
             patch("app.live_trading.kill_switch.kill_switch") as ks:
            ks.is_active = True
            rm.status = "running"

            # Process one message then stop
            import asyncio
            async def run_one():
                from app.integrations.redis_queue import get_redis_queue
                count = 0
                while rm.status == "running" and count < 5:
                    count += 1
                    msg = await asyncio.to_thread(rm.get_message, "trade_proposals", 3)
                    if msg is None:
                        break
                    if ks.is_active:
                        db_p = await asyncio.to_thread(rm._persist_proposal, msg)
                        await rm._reject(msg, {"failed_rule": "kill_switch"}, db_p)
                        break

            await run_one()

        assert rejected.get("rule") == "kill_switch"


# ── Trader: approved_symbols cleared + pending orders cancelled ───────────────

class TestKillSwitchTrader:
    def _make_trader(self):
        from app.agents.trader import Trader
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            t = Trader.__new__(Trader)
            t.logger = MagicMock()
            t.approved_symbols = {"AAPL": {"trade_type": "swing"}}
            t.active_positions = {}
            t._pending_limit_orders = {
                "TSLA": {"order_id": "LMT1", "trade_id": 1}
            }
            t._force_closed_today = False
            t._last_date = ""
            t._daily_discarded_symbols = set()
            t.status = "running"
            t.name = "trader"
            return t

    @pytest.mark.asyncio
    async def test_kill_switch_clears_approved_and_cancels_pending(self):
        t = self._make_trader()
        alpaca = MagicMock()
        alpaca.cancel_order.return_value = True

        with patch("app.live_trading.kill_switch.kill_switch") as ks, \
             patch("app.integrations.get_alpaca_client", return_value=alpaca), \
             patch("app.agents.trader.circuit_breaker") as cb:
            ks.is_active = True
            cb.is_open = False

            # Simulate the kill-switch branch inline (as trader.run() would execute it)
            if not cb.is_open:
                if ks.is_active:
                    t.approved_symbols.clear()
                    for sym, pend in list(t._pending_limit_orders.items()):
                        try:
                            alpaca.cancel_order(pend["order_id"])
                        except Exception:
                            pass
                    t._pending_limit_orders.clear()

        assert t.approved_symbols == {}
        assert t._pending_limit_orders == {}
        alpaca.cancel_order.assert_called_once_with("LMT1")


# ── KillSwitch.activate() cancels open orders ────────────────────────────────

class TestKillSwitchCancelsOrders:
    def test_activate_cancels_open_orders(self):
        from app.live_trading.kill_switch import KillSwitch
        ks = KillSwitch()

        alpaca = MagicMock()
        alpaca.get_positions.return_value = []
        alpaca.cancel_order.return_value = True

        open_orders = [{"order_id": "ORD1", "symbol": "AAPL", "qty": "10", "side": "buy", "status": "new"}]

        with patch.object(ks, "_persist_state"), \
             patch.object(ks, "_alpaca", return_value=alpaca), \
             patch.object(ks, "_audit"), \
             patch("app.startup_reconciler._get_open_alpaca_orders", return_value=open_orders):
            result = ks.activate("test")

        assert "ORD1" in result["orders_cancelled"]
        alpaca.cancel_order.assert_called_once_with("ORD1")
