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
    # Equity in 25k-26k band: above PDT threshold (rules don't apply by count)
    # but below circuit breaker (always blocked). Tests the circuit breaker path.
    # For count-logic tests, mock PDT_CIRCUIT_BREAKER to be below test equity.
    _EQUITY_ABOVE_PDT = 26_500   # above circuit breaker → "PDT check skipped"

    def test_no_trades_not_blocked(self):
        ct = _fresh_tracker()
        blocked, msg = ct.is_pdt_blocked(account_equity=self._EQUITY_ABOVE_PDT)
        assert not blocked
        assert "PDT check skipped" in msg

    def test_one_trade_high_equity_not_blocked(self):
        ct = _fresh_tracker()
        ct.record_day_trade("AAPL")
        blocked, _ = ct.is_pdt_blocked(self._EQUITY_ABOVE_PDT)
        assert not blocked

    def test_two_trades_circuit_breaker_takes_precedence(self):
        # With circuit breaker at $26k, the count-based block (25k-26k band) is
        # superseded — circuit breaker fires first. Verify circuit breaker blocks
        # regardless of trade count.
        ct = _fresh_tracker()
        ct.record_day_trade("AAPL")
        ct.record_day_trade("TSLA")
        blocked, msg = ct.is_pdt_blocked(25_800)  # below $26k circuit breaker
        assert blocked
        assert "circuit breaker" in msg

    def test_circuit_breaker_blocks_below_26k(self):
        ct = _fresh_tracker()
        # No trades at all — blocked purely by circuit breaker
        blocked, msg = ct.is_pdt_blocked(25_500)
        assert blocked
        assert "circuit breaker" in msg

    def test_high_equity_skips_pdt(self):
        ct = _fresh_tracker()
        ct.record_day_trade("AAPL")
        ct.record_day_trade("TSLA")
        # equity above $26k circuit breaker — PDT rules don't apply
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


# ─── PDT load idempotency ─────────────────────────────────────────────────────

class TestPdtLoadIdempotency:
    def test_load_day_trades_from_db_is_idempotent(self):
        """Calling load_day_trades_from_db twice must not double-count PDT trades."""
        from datetime import datetime
        ct = _fresh_tracker()
        today = date.today()

        class _FakeTrade:
            status = "CLOSED"
            closed_at = datetime.combine(today, datetime.min.time().replace(hour=15))
            created_at = datetime.combine(today, datetime.min.time().replace(hour=9))
            entry_price = 100.0
            exit_price = 105.0
            symbol = "AAPL"

        db = MagicMock()
        db.query.return_value.filter.return_value.filter.return_value.all.return_value = [_FakeTrade()]

        ct.load_day_trades_from_db(db)
        count_after_first = ct.day_trade_count_window()
        ct.load_day_trades_from_db(db)
        count_after_second = ct.day_trade_count_window()

        assert count_after_first == count_after_second, (
            f"Double-call inflated PDT count: {count_after_first} vs {count_after_second}"
        )


# ─── Business-day settlement ──────────────────────────────────────────────────

class TestBusinessDaySettlement:
    def test_friday_sale_settles_monday_not_saturday(self):
        """T+1 settlement should skip weekends (Friday → Monday)."""
        ct = _fresh_tracker()
        # Find next Friday
        today = date.today()
        days_to_friday = (4 - today.weekday()) % 7  # 4=Friday
        if days_to_friday == 0:
            days_to_friday = 0  # today is Friday
        friday = today + timedelta(days=days_to_friday)
        assert friday.weekday() == 4  # sanity

        ct.record_sale_proceeds(1_000.0, trade_date=friday)
        # The settle entry should be for Monday, not Saturday
        settle_date = ct._unsettled[-1][0]
        assert settle_date.weekday() != 5, "Settle date landed on Saturday — business-day logic failed"
        assert settle_date.weekday() != 6, "Settle date landed on Sunday — business-day logic failed"
        assert settle_date == friday + timedelta(days=3), (
            f"Expected Monday {friday + timedelta(days=3)}, got {settle_date}"
        )


# ─── recompute_partial_pnl ────────────────────────────────────────────────────

class TestRecomputePartialPnl:
    def _make_order(self, filled_price, filled_qty, order_type="PARTIAL_EXIT"):
        o = MagicMock()
        o.filled_price = filled_price
        o.filled_qty = filled_qty
        o.order_type = order_type
        o.status = "FILLED"
        return o

    def test_long_partial_pnl_correct(self):
        from app.database.models import recompute_partial_pnl, Order
        orders = [self._make_order(110.0, 50)]
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = orders
        result = recompute_partial_pnl(db, trade_id=1, entry_price=100.0, direction="BUY")
        assert result == pytest.approx(500.0)  # (110-100)*50

    def test_short_partial_pnl_correct(self):
        from app.database.models import recompute_partial_pnl, Order
        orders = [self._make_order(90.0, 50)]
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = orders
        result = recompute_partial_pnl(db, trade_id=1, entry_price=100.0, direction="SELL_SHORT")
        assert result == pytest.approx(500.0)  # (100-90)*50

    def test_no_partial_orders_returns_zero(self):
        from app.database.models import recompute_partial_pnl
        db = MagicMock()
        db.query.return_value.filter.return_value.filter.return_value.filter.return_value.all.return_value = []
        result = recompute_partial_pnl(db, trade_id=1, entry_price=100.0, direction="BUY")
        assert result == 0.0

    def test_multiple_partial_orders_accumulate(self):
        from app.database.models import recompute_partial_pnl
        orders = [self._make_order(105.0, 30), self._make_order(108.0, 20)]
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = orders
        result = recompute_partial_pnl(db, trade_id=1, entry_price=100.0, direction="BUY")
        assert result == pytest.approx(30 * 5.0 + 20 * 8.0)  # 150 + 160 = 310


# ─── RiskLimits.from_db None coalesce ────────────────────────────────────────

class TestRiskLimitsFromDb:
    def test_none_db_values_fall_back_to_defaults(self):
        """from_db must not store None for any field when DB returns None."""
        from app.agents.risk_rules import RiskLimits
        db = MagicMock()
        with patch("app.database.agent_config.get_agent_config", return_value=None):
            limits = RiskLimits.from_db(db)
        defaults = RiskLimits()
        assert limits.MAX_POSITION_SIZE_PCT == defaults.MAX_POSITION_SIZE_PCT
        assert limits.MAX_DAILY_LOSS_PCT == defaults.MAX_DAILY_LOSS_PCT
        assert limits.MAX_PORTFOLIO_HEAT_PCT == defaults.MAX_PORTFOLIO_HEAT_PCT
        # All float fields must be non-None (no TypeError on downstream comparisons)
        for field in ("MAX_POSITION_SIZE_PCT", "MAX_SECTOR_CONCENTRATION_PCT",
                      "MAX_DAILY_LOSS_PCT", "MAX_ACCOUNT_DRAWDOWN_PCT",
                      "MAX_PORTFOLIO_HEAT_PCT", "NORMAL_VOLATILITY_ATR_RATIO",
                      "STOP_LOSS_BASE_PCT", "max_spread_pct", "max_adtv_pct",
                      "max_correlation", "max_portfolio_beta", "high_beta_threshold",
                      "max_factor_concentration"):
            assert getattr(limits, field) is not None, f"{field} should not be None"

    def test_db_exception_returns_defaults(self):
        """from_db must return hardcoded defaults if DB access raises."""
        from app.agents.risk_rules import RiskLimits
        db = MagicMock()
        with patch("app.database.agent_config.get_agent_config", side_effect=RuntimeError("DB down")):
            limits = RiskLimits.from_db(db)
        assert isinstance(limits, RiskLimits)
        assert limits.MAX_POSITION_SIZE_PCT == RiskLimits().MAX_POSITION_SIZE_PCT
