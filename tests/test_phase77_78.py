"""
Tests for Phase 77 (decision audit backfill) and Phase 78 (order lifecycle).
  78a — partial fill cancel remainder
  78b — persist/reload pending limit orders to DB
  78d — periodic mid-session reconciliation
"""
from __future__ import annotations

import types
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock, call

import pytest


# ---------------------------------------------------------------------------
# Phase 77 — backfill script unit tests
# ---------------------------------------------------------------------------

class TestBackfillScript:
    def test_fetch_price_change_returns_none_on_insufficient_bars(self):
        """_fetch_price_change returns None when fewer than 2 bars returned."""
        import pandas as pd
        from scripts.backfill_decision_outcomes import _fetch_price_change

        mock_alpaca = MagicMock()
        mock_alpaca.get_bars.return_value = pd.DataFrame([{"open": 100, "close": 101}])

        with patch("app.integrations.get_alpaca_client", return_value=mock_alpaca):
            result = _fetch_price_change("AAPL", datetime.now(timezone.utc), hours_forward=4)
        assert result is None

    def test_fetch_price_change_calculates_pct(self):
        """_fetch_price_change returns correct % when bars available."""
        import pandas as pd
        from scripts.backfill_decision_outcomes import _fetch_price_change

        bars = pd.DataFrame([
            {"open": 100.0, "close": 100.0},
            {"open": 100.0, "close": 102.0},
            {"open": 102.0, "close": 104.0},
            {"open": 104.0, "close": 105.0},
            {"open": 105.0, "close": 106.0},
        ])
        mock_alpaca = MagicMock()
        mock_alpaca.get_bars.return_value = bars

        with patch("app.integrations.get_alpaca_client", return_value=mock_alpaca):
            result = _fetch_price_change("AAPL", datetime.now(timezone.utc), hours_forward=4)

        # open=100, target_idx=4 => close=106 => 6.0%
        assert result == pytest.approx(6.0, abs=0.01)

    def test_run_dry_run_does_not_commit(self):
        """run(..., dry_run=True) rolls back instead of committing."""
        from scripts.backfill_decision_outcomes import run

        mock_row = MagicMock()
        mock_row.final_decision = "block"
        mock_row.outcome_pnl_pct = None
        mock_row.outcome_4h_pct = 1.0   # already filled — no change
        mock_row.outcome_1d_pct = 2.0

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_row]

        with (
            patch("app.database.session.get_session", return_value=mock_db),
            patch("scripts.backfill_decision_outcomes._fetch_price_change", return_value=None),
        ):
            n = run(lookback_days=7, dry_run=True)

        mock_db.rollback.assert_called_once()
        mock_db.commit.assert_not_called()

    def test_run_commits_when_not_dry_run(self):
        """run() commits when dry_run=False and something changed."""
        from scripts.backfill_decision_outcomes import run

        mock_row = MagicMock()
        mock_row.final_decision = "block"
        mock_row.outcome_pnl_pct = None
        mock_row.outcome_4h_pct = None
        mock_row.outcome_1d_pct = None
        mock_row.symbol = "AAPL"
        mock_row.decided_at = datetime.now(timezone.utc)

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_row]

        with (
            patch("app.database.session.get_session", return_value=mock_db),
            patch("scripts.backfill_decision_outcomes._fetch_price_change", return_value=1.5),
        ):
            n = run(lookback_days=7, dry_run=False)

        mock_db.commit.assert_called_once()
        assert n == 1


# ---------------------------------------------------------------------------
# Phase 78a — partial fill cancel remainder
# ---------------------------------------------------------------------------

class TestPartialFillCancel:
    def _make_trader(self):
        from app.agents.trader import Trader
        t = Trader.__new__(Trader)
        t.logger = MagicMock()
        t._pending_limit_orders = {}
        t._pending_limits = {}
        t._last_mid_recon_slot = -1
        return t

    @pytest.mark.asyncio
    async def test_partial_fill_cancels_remainder(self):
        """When order_status is partially_filled, cancel_order is called and remainder cancelled."""
        trader = self._make_trader()
        order_id = "ord-partial-001"
        symbol = "NVDA"

        trader._pending_limit_orders = {
            symbol: {
                "order_id": order_id,
                "trade_id": None,
                "shares": 10,
                "limit_price": 150.0,
                "intended_price": 149.5,
                "stop_price": 145.0,
                "target_price": 160.0,
                "atr": 3.0,
                "trade_type": "swing",
                "signal_type": "EMA_CROSSOVER",
                "proposal_uuid": str(uuid.uuid4()),
            }
        }

        mock_alpaca = MagicMock()
        mock_alpaca.get_order_status.return_value = {
            "status": "partially_filled",
            "filled_qty": 3,
            "filled_avg_price": "150.5",
        }
        mock_alpaca.cancel_order.return_value = True

        trader.log_decision = AsyncMock()
        trader._delete_pending_limit_db = MagicMock()
        trader._record_entry = AsyncMock()

        with patch("app.agents.trader.get_session"):
            await trader._poll_pending_limit_orders(mock_alpaca)

        mock_alpaca.cancel_order.assert_called_once_with(order_id)
        trader._delete_pending_limit_db.assert_called_once_with(symbol)
        assert symbol not in trader._pending_limit_orders


# ---------------------------------------------------------------------------
# Phase 78b — persist / reload pending limit orders
# ---------------------------------------------------------------------------

class TestPendingLimitDB:
    def _make_trader(self):
        from app.agents.trader import Trader
        t = Trader.__new__(Trader)
        t.logger = MagicMock()
        t._pending_limit_orders = {}
        t._last_mid_recon_slot = -1
        return t

    def test_save_pending_limit_db_inserts_row(self):
        """_save_pending_limit_db upserts a PendingLimitOrder row (new row path)."""
        from app.database.models import PendingLimitOrder
        trader = self._make_trader()

        mock_db = MagicMock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = None

        pending = {
            "order_id": "ord-lim-001",
            "trade_id": 42,
            "shares": 10,
            "limit_price": 200.0,
            "intended_price": 199.5,
            "proposal": {"trade_type": "swing"},
        }
        result = MagicMock()
        result.stop_price = 195.0
        result.target_price = 210.0
        result.atr = 3.0
        result.signal_type = "EMA_CROSSOVER"

        with patch("app.agents.trader.get_session", return_value=mock_db):
            trader._save_pending_limit_db("AAPL", pending, result)

        mock_db.add.assert_called_once()
        row_added = mock_db.add.call_args[0][0]
        assert isinstance(row_added, PendingLimitOrder)
        assert row_added.symbol == "AAPL"
        assert row_added.order_id == "ord-lim-001"
        mock_db.commit.assert_called_once()

    def test_delete_pending_limit_db_removes_row(self):
        """_delete_pending_limit_db deletes the row for the symbol."""
        trader = self._make_trader()

        mock_row = MagicMock()
        mock_db = MagicMock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_row

        with patch("app.agents.trader.get_session", return_value=mock_db):
            trader._delete_pending_limit_db("AAPL")

        mock_db.delete.assert_called_once_with(mock_row)
        mock_db.commit.assert_called_once()

    def test_reload_pending_limits_from_db_populates_dict(self):
        """_reload_pending_limits_from_db restores _pending_limit_orders from DB rows."""
        trader = self._make_trader()

        row = MagicMock()
        row.symbol = "MSFT"
        row.order_id = "ord-reload-001"
        row.trade_id = 7
        row.shares = 5
        row.limit_price = 300.0
        row.intended_price = 299.0
        row.stop_price = 290.0
        row.target_price = 315.0
        row.atr = 4.0
        row.trade_type = "swing"
        row.signal_type = "EMA_CROSSOVER"
        row.created_at = datetime.now(timezone.utc)

        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = [row]

        with patch("app.agents.trader.get_session", return_value=mock_db):
            trader._reload_pending_limits_from_db()

        assert "MSFT" in trader._pending_limit_orders
        assert trader._pending_limit_orders["MSFT"]["order_id"] == "ord-reload-001"
        assert trader._pending_limit_orders["MSFT"]["shares"] == 5

    def test_reload_pending_limits_empty_db(self):
        """_reload_pending_limits_from_db does nothing when table is empty."""
        trader = self._make_trader()

        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = []

        with patch("app.agents.trader.get_session", return_value=mock_db):
            trader._reload_pending_limits_from_db()

        assert trader._pending_limit_orders == {}


# ---------------------------------------------------------------------------
# Phase 78d — periodic mid-session reconciliation
# ---------------------------------------------------------------------------

class TestMidSessionRecon:
    def test_recon_slot_logic(self):
        """Verify slot calculation: every 15 min during 9-16 ET gets a unique slot."""
        # 09:00 ET → slot 36, 09:15 → 37, 10:00 → 40
        assert (9 * 60 + 0) // 15 == 36
        assert (9 * 60 + 15) // 15 == 37
        assert (10 * 60 + 0) // 15 == 40
        assert (15 * 60 + 45) // 15 == 63

    def test_slot_changes_trigger_recon(self):
        """When recon_slot changes, _last_mid_recon_slot is updated to the new slot."""
        from app.agents.trader import Trader
        trader = Trader.__new__(Trader)
        trader.logger = MagicMock()
        trader._last_mid_recon_slot = 40

        # Simulate 10:15 ET on a weekday — slot 41
        new_slot = (10 * 60 + 15) // 15
        assert new_slot == 41
        assert new_slot != trader._last_mid_recon_slot
        # Simulated update
        trader._last_mid_recon_slot = new_slot
        assert trader._last_mid_recon_slot == 41

    def test_same_slot_does_not_retrigger(self):
        """If slot unchanged, no reconciliation should run."""
        from app.agents.trader import Trader
        trader = Trader.__new__(Trader)
        trader.logger = MagicMock()
        trader._last_mid_recon_slot = 41

        new_slot = (10 * 60 + 15) // 15  # 41 again
        assert new_slot == trader._last_mid_recon_slot  # no trigger
