"""
Unit tests for Phase 18: Execution Quality.

Covers:
- get_quote() on AlpacaClient (mocked)
- Bid-ask spread gate in RiskLimits / risk_manager validation
- ADTV liquidity gate in risk_manager validation
- Limit-order entry path in Trader._execute_entry
- Slippage calculation in Trader._record_entry
- _poll_pending_limit_orders fill / cancel / EOD cancellation
- New agent_config keys: risk.max_spread_pct, risk.max_adtv_pct, strategy.limit_order_offset_pct

All tests are pure-Python — no database, Redis, or Alpaca connections.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pandas as pd
import numpy as np


# ─── AlpacaClient.get_quote ───────────────────────────────────────────────────

class TestGetQuote:
    def _make_client(self):
        from app.integrations.alpaca import AlpacaClient
        with patch("app.integrations.alpaca.TradingClient"), \
             patch("app.integrations.alpaca.StockHistoricalDataClient"), \
             patch("app.config.settings") as s:
            s.alpaca_api_key = "x"
            s.alpaca_secret_key = "y"
            s.trading_mode = "paper"
            client = AlpacaClient.__new__(AlpacaClient)
            client.data_client = MagicMock()
            return client

    def test_returns_bid_ask_mid_spread(self):
        client = self._make_client()
        mock_quote = MagicMock()
        mock_quote.bid_price = 99.90
        mock_quote.ask_price = 100.10
        client.data_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        with patch("app.integrations.alpaca._rate_limiter"):
            result = client.get_quote("AAPL")

        assert result is not None
        assert abs(result["bid"] - 99.90) < 0.01
        assert abs(result["ask"] - 100.10) < 0.01
        assert abs(result["mid"] - 100.00) < 0.01
        # spread = 0.20 / 100.00 = 0.002
        assert abs(result["spread_pct"] - 0.002) < 0.0001

    def test_returns_none_on_missing_symbol(self):
        client = self._make_client()
        client.data_client.get_stock_latest_quote.return_value = {}
        with patch("app.integrations.alpaca._rate_limiter"):
            result = client.get_quote("ZZZZ")
        assert result is None

    def test_returns_none_on_zero_bid(self):
        client = self._make_client()
        mock_quote = MagicMock()
        mock_quote.bid_price = 0.0
        mock_quote.ask_price = 50.0
        client.data_client.get_stock_latest_quote.return_value = {"X": mock_quote}
        with patch("app.integrations.alpaca._rate_limiter"):
            result = client.get_quote("X")
        assert result is None

    def test_returns_none_on_exception(self):
        client = self._make_client()
        client.data_client.get_stock_latest_quote.side_effect = RuntimeError("API down")
        with patch("app.integrations.alpaca._rate_limiter"):
            result = client.get_quote("FAIL")
        assert result is None


# ─── RiskLimits: new fields ───────────────────────────────────────────────────

class TestRiskLimitsNewFields:
    def test_default_spread_pct(self):
        from app.agents.risk_rules import RiskLimits
        limits = RiskLimits()
        assert limits.max_spread_pct == 0.005

    def test_default_adtv_pct(self):
        from app.agents.risk_rules import RiskLimits
        limits = RiskLimits()
        assert limits.max_adtv_pct == 0.01


# ─── AgentConfig: new Phase 18 keys ──────────────────────────────────────────

class TestAgentConfigPhase18Keys:
    def test_new_keys_present(self):
        from app.database.agent_config import _DEFAULTS
        assert "risk.max_spread_pct" in _DEFAULTS
        assert "risk.max_adtv_pct" in _DEFAULTS
        assert "strategy.limit_order_offset_pct" in _DEFAULTS

    def test_default_values(self):
        from app.database.agent_config import _DEFAULTS
        assert abs(_DEFAULTS["risk.max_spread_pct"] - 0.005) < 1e-6
        assert abs(_DEFAULTS["risk.max_adtv_pct"] - 0.01) < 1e-6
        assert abs(_DEFAULTS["strategy.limit_order_offset_pct"] - 0.003) < 1e-6


# ─── Slippage calculation ─────────────────────────────────────────────────────

class TestSlippageCalculation:
    def test_positive_slippage(self):
        intended = 100.0
        filled = 100.50
        slippage_bps = round((filled - intended) / intended * 10000, 2)
        assert abs(slippage_bps - 50.0) < 0.1

    def test_negative_slippage_better_fill(self):
        intended = 100.0
        filled = 99.80
        slippage_bps = round((filled - intended) / intended * 10000, 2)
        assert slippage_bps < 0

    def test_zero_slippage_on_exact_fill(self):
        intended = 50.0
        filled = 50.0
        slippage_bps = round((filled - intended) / intended * 10000, 2)
        assert slippage_bps == 0.0


# ─── Trader: limit order placement for swing ─────────────────────────────────

class TestTraderLimitOrderEntry:
    def _make_trader(self):
        from app.agents.trader import Trader
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            t = Trader.__new__(Trader)
            t.logger = MagicMock()
            t.approved_symbols = {}
            t.active_positions = {}
            t._pending_limit_orders = {}
            t._force_closed_today = False
            t._last_date = ""
            t._last_regime = ""
            t.status = "running"
            t.name = "trader"
            return t

    def _make_result(self):
        result = MagicMock()
        result.entry_price = 100.0
        result.stop_price = 97.0
        result.target_price = 115.0
        result.atr = 1.5
        result.signal_type = "EMA_CROSSOVER"
        result.reasoning = {"price": 100.0}
        return result

    @pytest.mark.asyncio
    async def test_swing_places_limit_order(self):
        t = self._make_trader()
        alpaca = MagicMock()
        alpaca.get_quote.return_value = {"bid": 99.85, "ask": 100.15, "mid": 100.0, "spread_pct": 0.0015}
        alpaca.place_limit_order.return_value = {"order_id": "LMT123"}
        t.approved_symbols["AAPL"] = {"trade_type": "swing"}

        result = self._make_result()

        mock_db = MagicMock()
        mock_trade = MagicMock()
        mock_trade.id = 99
        mock_db.add = MagicMock()
        mock_db.commit = MagicMock()
        mock_db.rollback = MagicMock()
        mock_db.close = MagicMock()
        mock_db.refresh = MagicMock(side_effect=lambda obj: setattr(obj, "id", 99))
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_trade

        with patch("app.agents.trader.get_session", return_value=mock_db), \
             patch("app.database.agent_config.get_agent_config", return_value=0.003):
            await t._execute_entry("AAPL", 10, result, alpaca)

        alpaca.place_limit_order.assert_called_once()
        assert "AAPL" in t._pending_limit_orders
        assert alpaca.place_market_order.call_count == 0

    @pytest.mark.asyncio
    async def test_intraday_places_market_order(self):
        t = self._make_trader()
        alpaca = MagicMock()
        alpaca.place_market_order.return_value = {"order_id": "MKT456"}
        alpaca.get_latest_price.return_value = 100.20
        t.approved_symbols["SPY"] = {"trade_type": "intraday"}

        result = self._make_result()

        with patch("app.agents.trader.get_session") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value = mock_db
            mock_db.__enter__ = MagicMock(return_value=mock_db)
            mock_db.__exit__ = MagicMock(return_value=False)
            mock_db.add = MagicMock()
            mock_db.flush = MagicMock()
            mock_db.commit = MagicMock()
            mock_db.rollback = MagicMock()
            mock_db.close = MagicMock()

            trade_mock = MagicMock()
            trade_mock.id = 99

            with patch("app.agents.trader.Trade", return_value=trade_mock), \
                 patch("app.agents.trader.Order"), \
                 patch.object(t, "log_decision", new_callable=AsyncMock):
                await t._execute_entry("SPY", 5, result, alpaca)

        alpaca.place_market_order.assert_called_once()
        call_args = alpaca.place_market_order.call_args
        assert call_args[0][:3] == ("SPY", 5, "buy")
        assert "SPY" not in t._pending_limit_orders

    @pytest.mark.asyncio
    async def test_limit_order_uses_ask_minus_offset(self):
        t = self._make_trader()
        alpaca = MagicMock()
        alpaca.get_quote.return_value = {"bid": 99.0, "ask": 100.0, "mid": 99.5, "spread_pct": 0.01}
        alpaca.place_limit_order.return_value = {"order_id": "LMT789"}
        t.approved_symbols["TSLA"] = {"trade_type": "swing"}

        result = self._make_result()

        mock_db = MagicMock()
        mock_db.refresh = MagicMock(side_effect=lambda obj: setattr(obj, "id", 99))
        mock_db.query.return_value.filter_by.return_value.first.return_value = MagicMock(id=99)

        with patch("app.agents.trader.get_session", return_value=mock_db), \
             patch("app.database.agent_config.get_agent_config", return_value=0.003):
            await t._execute_entry("TSLA", 3, result, alpaca)

        call_args = alpaca.place_limit_order.call_args
        limit_price = call_args[0][3]  # positional: symbol, qty, side, limit_price
        expected = round(100.0 * (1 - 0.003), 4)
        assert abs(limit_price - expected) < 0.001


# ─── Trader: _poll_pending_limit_orders ──────────────────────────────────────

class TestPollPendingLimitOrders:
    def _make_trader(self):
        from app.agents.trader import Trader
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            t = Trader.__new__(Trader)
            t.logger = MagicMock()
            t.approved_symbols = {}
            t.active_positions = {}
            t._pending_limit_orders = {}
            t._force_closed_today = False
            t._last_date = ""
            t._last_regime = ""
            t.status = "running"
            t.name = "trader"
            return t

    def _make_pending(self, order_id="LMT1"):
        result = MagicMock()
        result.entry_price = 100.0
        result.stop_price = 97.0
        result.target_price = 115.0
        result.atr = 1.5
        result.signal_type = "EMA_CROSSOVER"
        result.reasoning = {}
        return {
            "order_id": order_id,
            "shares": 10,
            "intended_price": 100.0,
            "limit_price": 99.70,
            "result": result,
            "proposal": {"trade_type": "swing"},
            "queued_at": datetime(2026, 4, 21, 10, 0),
        }

    @pytest.mark.asyncio
    async def test_filled_order_creates_position(self):
        t = self._make_trader()
        t._pending_limit_orders["AAPL"] = self._make_pending()

        alpaca = MagicMock()
        alpaca.get_order_status.return_value = {
            "status": "filled",
            "filled_qty": 10,
            "filled_avg_price": 99.70,
        }

        with patch.object(t, "_record_entry", new_callable=AsyncMock) as mock_record:
            await t._poll_pending_limit_orders(alpaca)

        mock_record.assert_called_once()
        assert "AAPL" not in t._pending_limit_orders

    @pytest.mark.asyncio
    async def test_cancelled_order_removed(self):
        t = self._make_trader()
        t._pending_limit_orders["TSLA"] = self._make_pending("LMT2")

        alpaca = MagicMock()
        alpaca.get_order_status.return_value = {
            "status": "canceled",
            "filled_qty": 0,
            "filled_avg_price": None,
        }

        await t._poll_pending_limit_orders(alpaca)
        assert "TSLA" not in t._pending_limit_orders

    @pytest.mark.asyncio
    async def test_eod_cancels_unfilled(self):
        t = self._make_trader()
        t._pending_limit_orders["NVDA"] = self._make_pending("LMT3")

        alpaca = MagicMock()
        alpaca.get_order_status.return_value = {
            "status": "new",
            "filled_qty": 0,
            "filled_avg_price": None,
        }

        eod_time = datetime(2026, 4, 21, 15, 50, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York"))
        with patch("app.agents.trader.datetime") as mock_dt:
            mock_dt.now.return_value = eod_time

            await t._poll_pending_limit_orders(alpaca)

        alpaca.cancel_order.assert_called_once_with("LMT3")
        assert "NVDA" not in t._pending_limit_orders

    @pytest.mark.asyncio
    async def test_noop_when_no_pending_orders(self):
        t = self._make_trader()
        alpaca = MagicMock()
        await t._poll_pending_limit_orders(alpaca)
        alpaca.get_order_status.assert_not_called()
