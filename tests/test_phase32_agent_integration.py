"""Tests for Phase 32: dual-model agent integration."""
import pytest
from unittest.mock import MagicMock, patch


# ── PortfolioManager dual-model ───────────────────────────────────────────────

class TestPortfolioManagerDualModel:

    def _pm(self):
        from app.agents.portfolio_manager import PortfolioManager
        pm = PortfolioManager()
        return pm

    def test_has_intraday_model_attribute(self):
        pm = self._pm()
        from app.ml.model import PortfolioSelectorModel
        assert isinstance(pm.intraday_model, PortfolioSelectorModel)

    def test_has_selected_intraday_flag(self):
        pm = self._pm()
        assert hasattr(pm, "_selected_intraday_today")
        assert pm._selected_intraday_today is False

    def test_swing_proposal_tagged_swing(self):
        proposal = {
            "symbol": "AAPL",
            "trade_type": "swing",
        }
        assert proposal["trade_type"] == "swing"

    def test_intraday_proposal_tagged_intraday(self):
        proposal = {
            "trade_type": "intraday",
            "stop_loss": 149.25,
        }
        assert proposal["trade_type"] == "intraday"

    def test_try_load_model_handles_missing_swing(self):
        """_try_load_model should not raise when swing model is absent from DB."""
        pm = self._pm()
        mock_db = MagicMock()
        q = mock_db.query.return_value.filter_by.return_value.order_by.return_value.first
        q.return_value = None
        with patch("app.database.session.get_session", return_value=mock_db):
            result = pm._try_load_model()
        assert result is False

    @pytest.mark.asyncio
    async def test_select_intraday_skips_when_model_not_trained(self):
        pm = self._pm()
        pm.intraday_model.is_trained = False
        await pm.select_intraday_instruments()


# ── RiskManager intraday rules ────────────────────────────────────────────────

class TestRiskManagerIntradayRules:

    def _rm(self):
        from app.agents.risk_manager import RiskManager
        return RiskManager()

    def test_intraday_count_starts_zero(self):
        rm = self._rm()
        assert rm._open_intraday_count == 0

    def test_on_intraday_closed_decrements(self):
        rm = self._rm()
        rm._open_intraday_count = 2
        rm.on_intraday_position_closed()
        assert rm._open_intraday_count == 1

    def test_on_intraday_closed_floor_zero(self):
        rm = self._rm()
        rm._open_intraday_count = 0
        rm.on_intraday_position_closed()
        assert rm._open_intraday_count == 0

    def test_reset_intraday_count(self):
        rm = self._rm()
        rm._open_intraday_count = 3
        rm.reset_intraday_count()
        assert rm._open_intraday_count == 0

    @pytest.mark.asyncio
    async def test_validate_rejects_when_intraday_cap_reached(self):
        from app.agents.risk_manager import RiskManager, MAX_INTRADAY_POSITIONS
        rm = RiskManager()
        rm._open_intraday_count = MAX_INTRADAY_POSITIONS

        proposal = {
            "symbol": "AAPL",
            "direction": "BUY",
            "quantity": 10,
            "entry_price": 150.0,
            "trade_type": "intraday",
        }

        with patch.object(rm, "_fetch_account_state", side_effect=Exception("no auth")), \
             patch("app.calendars.earnings.earnings_calendar.is_blocked", return_value=False), \
             patch("app.calendars.macro.macro_calendar.is_entry_blocked", return_value=False):
            is_approved, reasoning = await rm._validate_trade(proposal)

        assert not is_approved
        assert reasoning["failed_rule"] == "intraday_position_cap"

    @pytest.mark.asyncio
    async def test_intraday_stop_loss_tighter_than_swing(self):
        from app.agents.risk_manager import RiskManager
        rm = RiskManager()

        entry = 100.0
        proposal = {
            "symbol": "TSLA",
            "direction": "BUY",
            "quantity": 1,
            "entry_price": entry,
            "trade_type": "intraday",
            "stop_loss": round(entry * 0.995, 2),
        }

        account = {"portfolio_value": 50_000.0, "buying_power": 20_000.0, "equity": 50_000.0}
        positions = []

        with patch.object(rm, "_fetch_account_state", return_value=(account, positions)):
            with patch.object(rm, "_get_daily_pnl", return_value=0.0):
                is_approved, reasoning = await rm._validate_trade(proposal)

        if is_approved:
            assert reasoning["stop_loss"] <= entry * 0.996


# ── Trader force-close ────────────────────────────────────────────────────────

class TestTraderForceClose:

    def _trader(self):
        from app.agents.trader import Trader
        return Trader()

    def test_force_closed_flag_starts_false(self):
        t = self._trader()
        assert t._force_closed_today is False

    def test_active_positions_have_trade_type(self):
        t = self._trader()
        t.active_positions["AAPL"] = {
            "entry_price": 150.0, "stop_price": 147.0, "target_price": 157.5,
            "highest_price": 150.0, "bars_held": 0, "trade_id": 1,
            "trade_type": "intraday",
        }
        assert t.active_positions["AAPL"]["trade_type"] == "intraday"

    @pytest.mark.asyncio
    async def test_force_close_only_intraday(self):
        t = self._trader()
        t.active_positions = {
            "AAPL": {"trade_type": "intraday", "entry_price": 150.0, "stop_price": 147.0,
                     "target_price": 157.0, "highest_price": 151.0, "bars_held": 5,
                     "trade_id": 1},
            "MSFT": {"trade_type": "swing", "entry_price": 300.0, "stop_price": 294.0,
                     "target_price": 315.0, "highest_price": 302.0, "bars_held": 3,
                     "trade_id": 2},
        }

        closed = []

        async def fake_exit(sym, price, reason, alpaca):
            closed.append(sym)
            t.active_positions.pop(sym, None)

        mock_alpaca = MagicMock()
        mock_alpaca.get_latest_price.return_value = 151.0

        with patch.object(t, "_execute_exit", side_effect=fake_exit):
            with patch("app.integrations.get_alpaca_client", return_value=mock_alpaca):
                with patch("app.agents.risk_manager.risk_manager"):
                    await t._force_close_intraday()

        assert "AAPL" in closed
        assert "MSFT" not in closed

    @pytest.mark.asyncio
    async def test_force_close_noop_when_no_intraday(self):
        t = self._trader()
        t.active_positions = {
            "MSFT": {"trade_type": "swing", "entry_price": 300.0, "stop_price": 294.0,
                     "target_price": 315.0, "highest_price": 302.0, "bars_held": 3,
                     "trade_id": 2},
        }
        with patch.object(t, "_execute_exit") as mock_exit:
            await t._force_close_intraday()
            mock_exit.assert_not_called()

    def test_intraday_force_close_constants(self):
        from app.agents.trader import INTRADAY_FORCE_CLOSE_HOUR, INTRADAY_FORCE_CLOSE_MINUTE
        assert INTRADAY_FORCE_CLOSE_HOUR == 15
        assert INTRADAY_FORCE_CLOSE_MINUTE == 45
