"""Tests for reconciler hardening: write_exit_price, _is_broker_view_trusted,
ghost state machine, and provenance-aware untracked rules.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.startup_reconciler import (
    write_exit_price,
    _is_broker_view_trusted,
    EXIT_PRICE_SOURCES,
    _LEGITIMATE_EXIT_SOURCES,
    RECONCILE_GHOST_PENDING,
    RECONCILE_GHOST_UNRESOLVED,
    GHOST_MIN_DETECTIONS,
    GHOST_MIN_ELAPSED,
    GHOST_MAX_PENDING_HOURS,
    UNTRACKED_REACTIVATE_AFTER,
    RECONCILER_EXIT_REASONS,
)


# ── write_exit_price ──────────────────────────────────────────────────────────

class TestWriteExitPrice:
    def _trade(self, exit_price=None, exit_price_source=None):
        t = MagicMock()
        t.id = 1
        t.exit_price = exit_price
        t.exit_price_source = exit_price_source
        return t

    def test_writes_price_and_provenance(self):
        t = self._trade()
        result = write_exit_price(t, 155.25, source="alpaca_fill", order_id="abc123")
        assert result is True
        assert t.exit_price == 155.25
        assert t.exit_price_source == "alpaca_fill"
        assert t.exit_order_id == "abc123"
        assert t.exit_price_written_by == "reconciler"
        assert t.exit_price_written_at is not None

    def test_rejects_unknown_source(self):
        t = self._trade()
        result = write_exit_price(t, 100.0, source="magic_source")
        assert result is False
        assert t.exit_price is None

    def test_immutability_guard_blocks_overwrite_of_legitimate_exit(self):
        t = self._trade(exit_price=150.0, exit_price_source="alpaca_fill")
        result = write_exit_price(t, 999.0, source="fallback_bar_close")
        assert result is False
        assert t.exit_price == 150.0  # unchanged

    def test_immutability_guard_allows_overwrite_of_non_legitimate_exit(self):
        t = self._trade(exit_price=150.0, exit_price_source="legacy_unknown")
        result = write_exit_price(t, 155.0, source="alpaca_fill")
        assert result is True
        assert t.exit_price == 155.0

    def test_force_overrides_immutability_guard(self):
        t = self._trade(exit_price=150.0, exit_price_source="alpaca_fill")
        result = write_exit_price(t, 160.0, source="manual", force=True)
        assert result is True
        assert t.exit_price == 160.0
        assert t.exit_price_source == "manual"

    def test_allows_write_when_no_existing_exit(self):
        t = self._trade(exit_price=None, exit_price_source=None)
        result = write_exit_price(t, 123.45, source="fallback_bar_close")
        assert result is True

    def test_none_existing_source_is_not_legitimate(self):
        # None is not in _LEGITIMATE_EXIT_SOURCES — should allow overwrite
        t = self._trade(exit_price=100.0, exit_price_source=None)
        result = write_exit_price(t, 105.0, source="alpaca_fill")
        assert result is True


# ── _is_broker_view_trusted ───────────────────────────────────────────────────

class TestIsBrokerViewTrusted:
    def test_non_empty_positions_always_trusted(self):
        alpaca = MagicMock()
        result = _is_broker_view_trusted(alpaca, {"AAPL": {}})
        assert result is True
        alpaca.trading_client.get_account.assert_not_called()

    def test_empty_positions_trusted_when_equity_equals_cash(self):
        alpaca = MagicMock()
        acct = MagicMock()
        acct.equity = "10000.00"
        acct.cash = "10000.00"
        alpaca.trading_client.get_account.return_value = acct
        result = _is_broker_view_trusted(alpaca, {})
        assert result is True

    def test_empty_positions_distrusted_when_equity_above_cash(self):
        alpaca = MagicMock()
        acct = MagicMock()
        acct.equity = "10500.00"
        acct.cash = "10000.00"
        alpaca.trading_client.get_account.return_value = acct
        result = _is_broker_view_trusted(alpaca, {})
        assert result is False

    def test_empty_positions_trusted_when_account_call_fails(self):
        alpaca = MagicMock()
        alpaca.trading_client.get_account.side_effect = Exception("API error")
        result = _is_broker_view_trusted(alpaca, {})
        assert result is True  # fail-open: trust the empty snapshot


# ── Constants sanity ──────────────────────────────────────────────────────────

class TestConstants:
    def test_legitimate_sources_subset_of_all_sources(self):
        assert _LEGITIMATE_EXIT_SOURCES.issubset(EXIT_PRICE_SOURCES)

    def test_legacy_unknown_not_legitimate(self):
        assert "legacy_unknown" not in _LEGITIMATE_EXIT_SOURCES

    def test_fallback_bar_close_not_legitimate(self):
        assert "fallback_bar_close" not in _LEGITIMATE_EXIT_SOURCES

    def test_reconciler_exit_reasons_non_empty(self):
        assert len(RECONCILER_EXIT_REASONS) >= 2

    def test_ghost_min_detections_at_least_2(self):
        assert GHOST_MIN_DETECTIONS >= 2

    def test_ghost_min_elapsed_at_least_10_min(self):
        assert GHOST_MIN_ELAPSED >= timedelta(minutes=10)
