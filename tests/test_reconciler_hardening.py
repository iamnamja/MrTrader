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
    def test_non_empty_complete_snapshot_trusted(self):
        # A non-empty snapshot whose market value accounts for the implied held value is trusted.
        alpaca = MagicMock()
        acct = MagicMock()
        acct.equity = "60000.00"
        acct.cash = "10000.00"   # implied held = 50k
        alpaca.trading_client.get_account.return_value = acct
        assert _is_broker_view_trusted(alpaca, {"AAPL": {"market_value": 50000.0}}) is True

    def test_non_empty_partial_snapshot_distrusted(self):
        # A non-empty but INCOMPLETE snapshot (MV << implied held) is distrusted — don't ghost-close
        # real positions that are simply missing from a partial API response.
        alpaca = MagicMock()
        acct = MagicMock()
        acct.equity = "60000.00"
        acct.cash = "10000.00"   # implied held = 50k
        alpaca.trading_client.get_account.return_value = acct
        assert _is_broker_view_trusted(alpaca, {"AAPL": {"market_value": 1000.0}}) is False

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


# ── validate_target_stop / write_target_stop ──────────────────────────────────

from app.startup_reconciler import (  # noqa: E402
    validate_target_stop,
    write_target_stop,
    TARGET_STOP_BOUNDS,
)


class TestValidateTargetStop:
    def test_long_legitimate_swing(self):
        ok, why = validate_target_stop(100.0, 106.0, 98.0, "BUY", "swing")
        assert ok, why

    def test_long_target_at_entry_rejected(self):
        ok, why = validate_target_stop(100.0, 100.0, 98.0, "BUY", "swing")
        assert not ok
        assert "target" in why.lower()

    def test_long_target_below_entry_rejected(self):
        ok, why = validate_target_stop(100.0, 95.0, 98.0, "BUY", "swing")
        assert not ok

    def test_long_runaway_target_rejected(self):
        # AVGO real-world bug: $413 entry, $1,993 target → ~382% above
        ok, why = validate_target_stop(413.0, 1993.0, 400.0, "BUY", "swing")
        assert not ok
        assert "exceeds max" in why or "50%" in why

    def test_long_nem_runaway_target_rejected(self):
        # NEM real-world bug: $108 entry, $217 target → ~101% above
        ok, why = validate_target_stop(108.0, 217.0, 105.0, "BUY", "swing")
        assert not ok

    def test_long_stop_above_entry_rejected(self):
        ok, why = validate_target_stop(100.0, 106.0, 101.0, "BUY", "swing")
        assert not ok
        assert "stop" in why.lower()

    def test_long_runaway_stop_rejected(self):
        ok, _ = validate_target_stop(100.0, 106.0, 50.0, "BUY", "swing")
        assert not ok

    def test_short_legitimate(self):
        ok, why = validate_target_stop(100.0, 94.0, 103.0, "SELL_SHORT", "swing")
        assert ok, why

    def test_short_target_above_entry_rejected(self):
        ok, _ = validate_target_stop(100.0, 106.0, 103.0, "SELL_SHORT", "swing")
        assert not ok

    def test_short_stop_below_entry_rejected(self):
        ok, _ = validate_target_stop(100.0, 94.0, 98.0, "SELL_SHORT", "swing")
        assert not ok

    def test_intraday_tight_target_accepted(self):
        # Intraday targets can be as low as 0.3% — must not false-positive
        ok, why = validate_target_stop(100.0, 100.35, 99.80, "BUY", "intraday")
        assert ok, why

    def test_intraday_tight_stop_accepted(self):
        ok, why = validate_target_stop(100.0, 100.50, 99.80, "BUY", "intraday")
        assert ok, why

    def test_swing_tight_target_rejected(self):
        # 0.2% target on swing is below swing min — should reject
        ok, _ = validate_target_stop(100.0, 100.20, 98.0, "BUY", "swing")
        assert not ok

    def test_only_target_supplied_ok(self):
        ok, _ = validate_target_stop(100.0, 106.0, None, "BUY", "swing")
        assert ok

    def test_only_stop_supplied_ok(self):
        ok, _ = validate_target_stop(100.0, None, 98.0, "BUY", "swing")
        assert ok

    def test_zero_entry_rejected(self):
        ok, _ = validate_target_stop(0.0, 1.0, 0.5, "BUY", "swing")
        assert not ok

    def test_negative_target_rejected(self):
        ok, _ = validate_target_stop(100.0, -5.0, 98.0, "BUY", "swing")
        assert not ok

    def test_bounds_constant_has_swing_and_intraday(self):
        assert "swing" in TARGET_STOP_BOUNDS
        assert "intraday" in TARGET_STOP_BOUNDS


class TestWriteTargetStop:
    def _trade(self, entry=100.0, direction="BUY", trade_type="swing"):
        t = MagicMock()
        t.id = 42
        t.symbol = "TEST"
        t.entry_price = entry
        t.direction = direction
        t.trade_type = trade_type
        t.target_price = None
        t.stop_price = None
        return t

    def test_writes_valid_target_only(self):
        t = self._trade()
        assert write_target_stop(t, target_price=106.0, written_by="test") is True
        assert t.target_price == 106.0
        assert t.stop_price is None

    def test_writes_valid_stop_only(self):
        t = self._trade()
        assert write_target_stop(t, stop_price=98.0, written_by="test") is True
        assert t.stop_price == 98.0
        assert t.target_price is None

    def test_writes_both(self):
        t = self._trade()
        assert write_target_stop(t, target_price=106.0, stop_price=98.0) is True
        assert t.target_price == 106.0
        assert t.stop_price == 98.0

    def test_rejects_runaway_target_no_write(self):
        t = self._trade(entry=413.0)
        t.target_price = 420.0  # pre-existing sane value
        result = write_target_stop(t, target_price=1993.0, written_by="test")
        assert result is False
        # Must NOT silently overwrite — pre-existing value preserved
        assert t.target_price == 420.0

    def test_rejects_inverted_long_target(self):
        t = self._trade()
        assert write_target_stop(t, target_price=95.0) is False
        assert t.target_price is None

    def test_no_op_when_both_none(self):
        t = self._trade()
        assert write_target_stop(t) is False
