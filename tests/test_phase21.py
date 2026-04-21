"""
Unit tests for Phase 21: Compliance & Regulatory Guardrails.

Covers:
- PDT tracking: day trade count, 5-business-day window, blocking at 2 trades
- Wash sale: flagging re-entry within 30 days of a loss close
- Settlement: unsettled cash, settled buying power
- Symbol halt list: halt, resume, is_symbol_halted
"""
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch


def _fresh_tracker():
    from app.agents.compliance import ComplianceTracker
    return ComplianceTracker()


# ─── PDT Rule ─────────────────────────────────────────────────────────────────

class TestPdtRule:
    def test_no_trades_not_blocked(self):
        ct = _fresh_tracker()
        blocked, msg = ct.is_pdt_blocked(account_equity=10_000)
        assert not blocked
        assert "0/" in msg

    def test_one_trade_not_blocked(self):
        ct = _fresh_tracker()
        ct.record_day_trade("AAPL")
        blocked, _ = ct.is_pdt_blocked(10_000)
        assert not blocked

    def test_two_trades_blocks_intraday(self):
        ct = _fresh_tracker()
        ct.record_day_trade("AAPL")
        ct.record_day_trade("TSLA")
        blocked, msg = ct.is_pdt_blocked(10_000)
        assert blocked
        assert "PDT limit" in msg

    def test_high_equity_skips_pdt(self):
        ct = _fresh_tracker()
        ct.record_day_trade("AAPL")
        ct.record_day_trade("TSLA")
        # equity above $25k threshold — PDT rules don't apply
        blocked, msg = ct.is_pdt_blocked(30_000)
        assert not blocked
        assert "PDT check skipped" in msg

    def test_day_trade_count_today(self):
        ct = _fresh_tracker()
        assert ct.day_trade_count_today() == 0
        ct.record_day_trade("NVDA")
        assert ct.day_trade_count_today() == 1

    def test_window_count_includes_today(self):
        ct = _fresh_tracker()
        ct.record_day_trade("AAPL")
        ct.record_day_trade("TSLA")
        assert ct.day_trade_count_window() >= 2

    def test_old_trades_outside_window_excluded(self):
        ct = _fresh_tracker()
        # Inject a trade 10 business days ago (outside 5-day window)
        old_date = (date.today() - timedelta(days=14)).isoformat()
        ct._day_trades[old_date] = ["AAPL"]
        assert ct.day_trade_count_window() == 0


# ─── Wash Sale ────────────────────────────────────────────────────────────────

class TestWashSale:
    def test_no_loss_close_is_clear(self):
        ct = _fresh_tracker()
        flagged, msg = ct.check_wash_sale("AAPL")
        assert not flagged
        assert "clear" in msg

    def test_recent_loss_close_flags(self):
        ct = _fresh_tracker()
        ct.record_loss_close("AAPL", date.today() - timedelta(days=10))
        flagged, msg = ct.check_wash_sale("AAPL")
        assert flagged
        assert "WASH SALE WARNING" in msg

    def test_loss_close_outside_window_is_clear(self):
        ct = _fresh_tracker()
        ct.record_loss_close("AAPL", date.today() - timedelta(days=31))
        flagged, _ = ct.check_wash_sale("AAPL")
        assert not flagged

    def test_wash_sale_on_boundary_day_30(self):
        ct = _fresh_tracker()
        ct.record_loss_close("TSLA", date.today() - timedelta(days=30))
        flagged, _ = ct.check_wash_sale("TSLA")
        assert flagged  # day 30 is still within window (days_since <= 30)

    def test_different_symbol_not_affected(self):
        ct = _fresh_tracker()
        ct.record_loss_close("AAPL", date.today())
        flagged, _ = ct.check_wash_sale("TSLA")
        assert not flagged


# ─── Settlement (Reg T) ───────────────────────────────────────────────────────

class TestSettlement:
    def test_no_unsettled_cash_initially(self):
        ct = _fresh_tracker()
        assert ct.unsettled_cash() == 0.0

    def test_sale_proceeds_are_unsettled(self):
        ct = _fresh_tracker()
        ct.record_sale_proceeds(5_000.0)  # settles tomorrow
        assert ct.unsettled_cash() == 5_000.0

    def test_settled_buying_power_reduces_by_unsettled(self):
        ct = _fresh_tracker()
        ct.record_sale_proceeds(3_000.0)
        bp = ct.settled_buying_power(10_000.0)
        assert bp == 7_000.0

    def test_settled_buying_power_never_negative(self):
        ct = _fresh_tracker()
        ct.record_sale_proceeds(20_000.0)
        bp = ct.settled_buying_power(10_000.0)
        assert bp == 0.0

    def test_sweep_removes_past_settled_entries(self):
        ct = _fresh_tracker()
        # Inject an already-settled entry (settle date = yesterday)
        yesterday = date.today() - timedelta(days=1)
        ct._unsettled.append((yesterday, 1_000.0))
        ct.sweep_settled()
        assert ct.unsettled_cash() == 0.0

    def test_future_settle_not_swept(self):
        ct = _fresh_tracker()
        ct.record_sale_proceeds(2_000.0)
        ct.sweep_settled()
        assert ct.unsettled_cash() == 2_000.0  # settles tomorrow, not yet swept


# ─── Symbol Halt List ─────────────────────────────────────────────────────────

class TestSymbolHalt:
    def test_new_symbol_not_halted(self):
        ct = _fresh_tracker()
        halted, msg = ct.is_symbol_halted("AAPL")
        assert not halted
        assert "not halted" in msg

    def test_halt_blocks_symbol(self):
        ct = _fresh_tracker()
        ct.halt_symbol("AAPL", "regulatory review")
        halted, msg = ct.is_symbol_halted("AAPL")
        assert halted
        assert "regulatory review" in msg

    def test_resume_unblocks_symbol(self):
        ct = _fresh_tracker()
        ct.halt_symbol("AAPL", "test")
        ct.resume_symbol("AAPL")
        halted, _ = ct.is_symbol_halted("AAPL")
        assert not halted

    def test_halted_symbols_property(self):
        ct = _fresh_tracker()
        ct.halt_symbol("NVDA", "sec investigation")
        ct.halt_symbol("TSLA", "halt test")
        hs = ct.halted_symbols
        assert "NVDA" in hs and "TSLA" in hs

    def test_status_dict_structure(self):
        ct = _fresh_tracker()
        s = ct.status()
        assert "pdt_day_trades_today" in s
        assert "unsettled_cash" in s
        assert "halted_symbols" in s
        assert "wash_sale_symbols" in s
