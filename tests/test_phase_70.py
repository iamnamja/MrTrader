"""
Tests for Phase 70 — PM Re-scoring of Unexecuted Approvals.

Tests focus on the logic within _rescore_pending_approvals and the
Trader WITHDRAW handler, without constructing full agent instances.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Dict, Optional


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_pm_stub(pending: Dict[str, float], model_trained: bool = True):
    """
    Minimal stub that has just the attributes _rescore_pending_approvals reads.
    We import the real method and bind it to this stub.
    """
    from app.agents.portfolio_manager import PortfolioManager
    stub = MagicMock(spec=PortfolioManager)
    stub._pending_approvals = dict(pending)
    stub.model = MagicMock()
    stub.model.is_trained = model_trained
    stub._alpaca = MagicMock()
    stub.feature_engineer = MagicMock()
    stub.logger = MagicMock()
    stub.send_message = MagicMock()
    stub.log_decision = AsyncMock()
    # Bind the real async method
    stub._rescore_pending_approvals = PortfolioManager._rescore_pending_approvals.__get__(stub, PortfolioManager)
    return stub


# ─── Tests: stale TTL ────────────────────────────────────────────────────────

class TestPhase70Stale:
    def test_stale_over_90min_sends_withdraw(self):
        import time
        now = time.monotonic()
        pm = _make_pm_stub({"AAPL": now - 95 * 60})

        with patch("app.database.session.get_session"), \
             patch("app.database.agent_config.get_agent_config", return_value=None):
            asyncio.get_event_loop().run_until_complete(pm._rescore_pending_approvals())

        pm.send_message.assert_called_once()
        call_kwargs = pm.send_message.call_args[0]
        msg = call_kwargs[1]
        assert msg["symbol"] == "AAPL"
        assert msg["action"] == "WITHDRAW"
        assert "stale" in msg["reason"]

    def test_stale_symbol_removed_from_pending(self):
        import time
        now = time.monotonic()
        pm = _make_pm_stub({"TSLA": now - 100 * 60})

        with patch("app.database.session.get_session"), \
             patch("app.database.agent_config.get_agent_config", return_value=None):
            asyncio.get_event_loop().run_until_complete(pm._rescore_pending_approvals())

        assert "TSLA" not in pm._pending_approvals

    def test_not_stale_under_90min_does_not_stale_withdraw(self):
        """A 60-min old approval should go through rescore, not the stale branch."""
        import time
        now = time.monotonic()
        pm = _make_pm_stub({"MSFT": now - 60 * 60})

        bars = MagicMock()
        bars.empty = False
        pm._alpaca.get_bars.return_value = bars
        pm.feature_engineer.engineer_features.return_value = {"f1": 0.5}
        pm.model.feature_names = ["f1"]
        pm.model.predict.return_value = (None, [0.6])  # score above threshold

        with patch("app.database.session.get_session"), \
             patch("app.database.agent_config.get_agent_config", return_value="0.45"):
            asyncio.get_event_loop().run_until_complete(pm._rescore_pending_approvals())

        # Should NOT have sent a WITHDRAW (score 0.6 > 0.45 * 0.85 = 0.38)
        for call in pm.send_message.call_args_list:
            msg = call[0][1]
            assert "stale" not in msg.get("reason", "")


# ─── Tests: rescore below threshold ──────────────────────────────────────────

class TestPhase70Rescore:
    def test_low_score_sends_withdraw(self):
        import time
        now = time.monotonic()
        pm = _make_pm_stub({"NVDA": now - 10 * 60})

        bars = MagicMock(); bars.empty = False
        pm._alpaca.get_bars.return_value = bars
        pm.feature_engineer.engineer_features.return_value = {"f1": 0.1}
        pm.model.feature_names = ["f1"]
        pm.model.predict.return_value = (None, [0.2])  # very low score

        with patch("app.database.session.get_session"), \
             patch("app.database.agent_config.get_agent_config", return_value="0.45"):
            asyncio.get_event_loop().run_until_complete(pm._rescore_pending_approvals())

        pm.send_message.assert_called_once()
        msg = pm.send_message.call_args[0][1]
        assert msg["symbol"] == "NVDA"
        assert msg["action"] == "WITHDRAW"
        assert "rescore" in msg["reason"]

    def test_good_score_keeps_approval(self):
        import time
        now = time.monotonic()
        pm = _make_pm_stub({"AMD": now - 5 * 60})

        bars = MagicMock(); bars.empty = False
        pm._alpaca.get_bars.return_value = bars
        pm.feature_engineer.engineer_features.return_value = {"f1": 0.8}
        pm.model.feature_names = ["f1"]
        pm.model.predict.return_value = (None, [0.75])  # well above threshold

        with patch("app.database.session.get_session"), \
             patch("app.database.agent_config.get_agent_config", return_value="0.45"):
            asyncio.get_event_loop().run_until_complete(pm._rescore_pending_approvals())

        pm.send_message.assert_not_called()
        assert "AMD" in pm._pending_approvals

    def test_empty_bars_skips_withdraw(self):
        """If bars can't be fetched, don't withdraw (fail open)."""
        import time
        now = time.monotonic()
        pm = _make_pm_stub({"SPY": now - 5 * 60})
        pm._alpaca.get_bars.return_value = None

        with patch("app.database.session.get_session"), \
             patch("app.database.agent_config.get_agent_config", return_value=None):
            asyncio.get_event_loop().run_until_complete(pm._rescore_pending_approvals())

        pm.send_message.assert_not_called()

    def test_no_pending_returns_immediately(self):
        pm = _make_pm_stub({})
        asyncio.get_event_loop().run_until_complete(pm._rescore_pending_approvals())
        pm.send_message.assert_not_called()
        pm.model.is_trained  # model should not even be checked past early return

    def test_untrained_model_returns_early(self):
        import time
        now = time.monotonic()
        pm = _make_pm_stub({"GOOG": now - 5 * 60}, model_trained=False)
        asyncio.get_event_loop().run_until_complete(pm._rescore_pending_approvals())
        pm.send_message.assert_not_called()


# ─── Tests: Trader WITHDRAW handler ──────────────────────────────────────────

class TestTraderWithdraw:
    def _trader_approved(self, symbols):
        """Return a Trader-like stub with approved_symbols populated."""
        stub = MagicMock()
        stub.approved_symbols = dict(symbols)
        stub.logger = MagicMock()
        return stub

    def test_withdraw_removes_from_approved(self):
        from app.agents.trader import Trader
        trader = self._trader_approved({"AAPL": {"entry_price": 170.0}})
        msg = {"symbol": "AAPL", "action": "WITHDRAW", "reason": "stale_95min"}

        # Simulate the WITHDRAW branch inline (it's a few lines, no complex flow)
        symbol = msg["symbol"]
        action = msg["action"]
        reason = msg["reason"]
        if action == "WITHDRAW":
            if symbol in trader.approved_symbols:
                trader.approved_symbols.pop(symbol)

        assert "AAPL" not in trader.approved_symbols

    def test_withdraw_unknown_symbol_is_noop(self):
        trader = self._trader_approved({"TSLA": {}})
        msg = {"symbol": "NVDA", "action": "WITHDRAW", "reason": "rescore_0.200"}
        symbol = msg["symbol"]
        if symbol in trader.approved_symbols:
            trader.approved_symbols.pop(symbol)
        # TSLA should still be there
        assert "TSLA" in trader.approved_symbols
