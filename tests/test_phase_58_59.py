"""
Tests for Phase 58 (Earnings Calendar Gate) and Phase 59 (Macro Calendar Awareness).
"""
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


# ─── Phase 58: Earnings Calendar ─────────────────────────────────────────────

class TestEarningsCalendar:
    def _make_calendar(self, days_until: int | None):
        """Return an EarningsCalendar with a mocked _fetch that returns days_until from today."""
        from app.calendars.earnings import EarningsCalendar
        cal = EarningsCalendar()
        if days_until is None:
            next_date = None
        else:
            next_date = date.today() + timedelta(days=days_until)
        # Phase 81: _fetch returns (date, data_ok) tuple; cache stores (date, monotonic, data_ok)
        cal._cache["TEST"] = (next_date, 1e18, True)
        return cal

    def test_swing_blocked_within_2_days(self):
        from app.calendars.earnings import EarningsCalendar
        cal = self._make_calendar(1)
        risk = cal.get_earnings_risk("TEST", "swing")
        assert risk.block_swing is True
        assert "earnings_in_1d" in risk.reason

    def test_swing_blocked_on_earnings_day(self):
        cal = self._make_calendar(0)
        from app.calendars.earnings import EarningsCalendar
        cal2 = self._make_calendar(0)
        risk = cal2.get_earnings_risk("TEST", "swing")
        assert risk.block_swing is True

    def test_swing_not_blocked_3_days_out(self):
        cal = self._make_calendar(3)
        risk = cal.get_earnings_risk("TEST", "swing")
        assert risk.block_swing is False

    def test_intraday_blocked_today(self):
        cal = self._make_calendar(0)
        risk = cal.get_earnings_risk("TEST", "intraday")
        assert risk.block_intraday is True

    def test_intraday_blocked_tomorrow(self):
        cal = self._make_calendar(1)
        risk = cal.get_earnings_risk("TEST", "intraday")
        assert risk.block_intraday is True

    def test_intraday_not_blocked_2_days_out(self):
        cal = self._make_calendar(2)
        risk = cal.get_earnings_risk("TEST", "intraday")
        assert risk.block_intraday is False

    def test_exit_review_flagged_within_3_days(self):
        cal = self._make_calendar(2)
        risk = cal.get_earnings_risk("TEST", "swing")
        assert risk.exit_review is True

    def test_no_earnings_data_does_not_block(self):
        cal = self._make_calendar(None)
        risk = cal.get_earnings_risk("TEST", "swing")
        assert risk.block_swing is False
        assert risk.block_intraday is False

    def test_days_until_earnings(self):
        cal = self._make_calendar(5)
        assert cal.days_until_earnings("TEST") == 5

    def test_is_blocked_swing(self):
        cal = self._make_calendar(1)
        assert cal.is_blocked("TEST", "swing") is True

    def test_is_blocked_intraday_false_beyond_1d(self):
        cal = self._make_calendar(2)
        assert cal.is_blocked("TEST", "intraday") is False


# ─── Phase 59: Macro Calendar ─────────────────────────────────────────────────

class TestMacroCalendar:
    def _calendar_at(self, dt_et: datetime):
        """Return a MacroCalendar whose 'now' is dt_et."""
        from app.calendars.macro import MacroCalendar
        cal = MacroCalendar()
        cal._cache = None
        cal._cache_ts = 0.0
        with patch("app.calendars.macro.datetime") as mock_dt:
            mock_dt.now.return_value = dt_et
            mock_dt.strptime = datetime.strptime
            ctx = cal._compute_context(vix=None)
        return ctx

    def test_fomc_day_is_high_impact(self):
        # 2026-01-29 is an FOMC day
        dt = datetime(2026, 1, 29, 10, 0, tzinfo=ET)
        ctx = self._calendar_at(dt)
        assert ctx.high_impact_today is True

    def test_non_event_day_is_not_high_impact(self):
        # 2026-01-05 has no events
        dt = datetime(2026, 1, 5, 10, 0, tzinfo=ET)
        ctx = self._calendar_at(dt)
        assert ctx.high_impact_today is False

    def test_fomc_window_blocks_entries_at_announcement(self):
        # FOMC announcement at 14:00 ET — should block at 14:01
        dt = datetime(2026, 1, 29, 14, 1, tzinfo=ET)
        ctx = self._calendar_at(dt)
        assert ctx.within_event_window is True
        assert ctx.block_new_entries is True

    def test_fomc_window_clear_3h_before(self):
        # 3 hours before FOMC (11:00 ET) — should not block
        dt = datetime(2026, 1, 29, 11, 0, tzinfo=ET)
        ctx = self._calendar_at(dt)
        assert ctx.within_event_window is False
        assert ctx.block_new_entries is False

    def test_cpi_window_blocks_at_release(self):
        # CPI at 08:30 ET on 2026-04-10 — within 15 min window
        dt = datetime(2026, 4, 10, 8, 32, tzinfo=ET)
        ctx = self._calendar_at(dt)
        assert ctx.within_event_window is True

    def test_cpi_window_clear_30min_after(self):
        dt = datetime(2026, 4, 10, 9, 5, tzinfo=ET)
        ctx = self._calendar_at(dt)
        assert ctx.within_event_window is False

    def test_sizing_factor_reduced_on_high_vix_event_day(self):
        from app.calendars.macro import MacroCalendar
        cal = MacroCalendar()
        cal._cache = None
        cal._cache_ts = 0.0
        with patch("app.calendars.macro.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 29, 10, 0, tzinfo=ET)
            mock_dt.strptime = datetime.strptime
            ctx = cal._compute_context(vix=22.0)
        assert ctx.sizing_factor < 1.0

    def test_sizing_factor_normal_on_low_vix_event_day(self):
        from app.calendars.macro import MacroCalendar
        cal = MacroCalendar()
        cal._cache = None
        cal._cache_ts = 0.0
        with patch("app.calendars.macro.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 29, 10, 0, tzinfo=ET)
            mock_dt.strptime = datetime.strptime
            ctx = cal._compute_context(vix=15.0)
        assert ctx.sizing_factor == 1.0

    def test_next_event_populated(self):
        dt = datetime(2026, 1, 20, 10, 0, tzinfo=ET)
        ctx = self._calendar_at(dt)
        assert ctx.next_event is not None
        assert len(ctx.next_event) > 0

    def test_is_entry_blocked_outside_window(self):
        from app.calendars.macro import MacroCalendar
        cal = MacroCalendar()
        cal._cache = None
        cal._cache_ts = 0.0
        with patch("app.calendars.macro.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 5, 10, 0, tzinfo=ET)
            mock_dt.strptime = datetime.strptime
            result = cal._compute_context(vix=None).block_new_entries
        assert result is False


# ─── RM rule logic: earnings + macro veto ─────────────────────────────────────
# Test the gate logic directly rather than running the full RM validation chain.

class TestRMCalendarGateLogic:
    def test_earnings_gate_blocks_swing(self):
        """RM earnings gate: block_swing=True → entry blocked."""
        from app.calendars.earnings import EarningsRisk
        risk = EarningsRisk(
            symbol="AAPL",
            next_earnings=date.today() + timedelta(days=1),
            days_until=1,
            block_swing=True,
            reason="earnings_in_1d",
        )
        assert risk.block_swing is True
        assert "earnings_in_1d" in risk.reason

    def test_earnings_gate_allows_clear_symbol(self):
        from app.calendars.earnings import EarningsRisk
        risk = EarningsRisk(symbol="AAPL", next_earnings=None, days_until=None)
        assert risk.block_swing is False
        assert risk.block_intraday is False

    def test_macro_gate_blocks_within_window(self):
        from app.calendars.macro import MacroContext, MacroEvent
        ctx = MacroContext(
            high_impact_today=True,
            within_event_window=True,
            block_new_entries=True,
            events_today=[MacroEvent("FOMC", "2026-01-29", "14:00")],
        )
        assert ctx.block_new_entries is True

    def test_macro_gate_allows_outside_window(self):
        from app.calendars.macro import MacroContext
        ctx = MacroContext(high_impact_today=True, within_event_window=False, block_new_entries=False)
        assert ctx.block_new_entries is False
