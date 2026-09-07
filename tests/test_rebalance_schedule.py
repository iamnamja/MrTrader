"""2026-09-07 — a holiday on the rebalance weekday cancelled the week instead of delaying it.

The weekly jobs were pinned to a calendar weekday and fail-closed on a market holiday with
no fallthrough, so Labor Day 2026-09-07 put 14 calendar days between the 08-31 and 09-14
rebalances. The frozen CH0a baseline (trend book CPCV mean_sharpe 0.7009) rebalances on a
5-TRADING-day grid that never skips a holiday week — `is_rebal = (np.arange(n) % 5 == 0)`
over a trading-day index — so the live skip was a divergence from the validated cadence,
recurring 4-5x a year.
"""
from datetime import date, timedelta

import pytest

from app.live_trading.rebalance_schedule import is_rebalance_day, week_anchor

MON = 0
FRI = 4


class TestWeekAnchor:
    def test_anchor_is_inside_todays_own_week(self):
        # Tue 2026-09-08 -> Monday of that week
        assert week_anchor(date(2026, 9, 8), MON) == date(2026, 9, 7)
        assert week_anchor(date(2026, 9, 11), MON) == date(2026, 9, 7)

    def test_anchor_never_crosses_into_another_week(self):
        """Anchoring inside the week is what stops a fallthrough colliding with the next
        week's anchor."""
        for offset in range(7):
            d = date(2026, 9, 7) + timedelta(days=offset)
            a = week_anchor(d, MON)
            assert 0 <= (d - a).days < 7


class TestNormalWeek:
    def test_runs_on_the_anchor(self):
        due, why = is_rebalance_day(date(2026, 9, 14), MON)   # ordinary Monday
        assert due and "anchor day" in why

    @pytest.mark.parametrize("d", [
        date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18),
    ])
    def test_does_not_run_again_later_in_the_week(self, d):
        due, why = is_rebalance_day(d, MON)
        assert not due
        assert "already" in why or "turn was" in why


class TestHolidayFallthrough:
    """Labor Day 2026: Mon 09-07 is a holiday, so the turn moves to Tue 09-08."""

    def test_anchor_holiday_does_not_run(self):
        due, why = is_rebalance_day(date(2026, 9, 7), MON)
        assert not due and "not a trading day" in why

    def test_runs_on_the_next_trading_day(self):
        due, why = is_rebalance_day(date(2026, 9, 8), MON)
        assert due
        assert "holiday" in why

    @pytest.mark.parametrize("d", [
        date(2026, 9, 9), date(2026, 9, 10), date(2026, 9, 11),
    ])
    def test_does_not_keep_firing_for_the_rest_of_the_week(self, d):
        """THE TRAP a naive 'next trading day' rule falls into: Wed, Thu and Fri all pass
        the market-open check too, so it would rebalance four times."""
        due, _ = is_rebalance_day(d, MON)
        assert not due

    def test_exactly_one_run_per_week_across_a_holiday_week(self):
        week = [date(2026, 9, 7) + timedelta(days=i) for i in range(7)]
        assert sum(1 for d in week if is_rebalance_day(d, MON)[0]) == 1

    def test_exactly_one_run_per_week_across_an_ordinary_week(self):
        week = [date(2026, 9, 14) + timedelta(days=i) for i in range(7)]
        assert sum(1 for d in week if is_rebalance_day(d, MON)[0]) == 1


class TestEveryMondayHolidayOfTheYear:
    """The recurrence is the point: these land on a Monday and each one used to cancel a
    week. Dates are derived from the exchange calendar rather than hardcoded."""

    def _monday_holidays(self, year):
        from app.live_trading.exchange_calendar import holidays
        return sorted(h for h in holidays(year) if h.weekday() == MON)

    def test_there_are_several_each_year(self):
        assert len(self._monday_holidays(2026)) >= 3

    def test_each_one_rolls_to_the_next_trading_day(self):
        for h in self._monday_holidays(2026):
            assert is_rebalance_day(h, MON)[0] is False
            nxt = h + timedelta(days=1)
            due, why = is_rebalance_day(nxt, MON)
            assert due, f"{h} should have rolled to {nxt}: {why}"

    def test_the_gap_stays_one_week_not_two(self):
        """The defect restated as a measurement: consecutive turns must stay ~7 days."""
        turns = []
        d = date(2026, 1, 1)
        while d < date(2027, 1, 1):
            if is_rebalance_day(d, MON)[0]:
                turns.append(d)
            d += timedelta(days=1)
        gaps = [(b - a).days for a, b in zip(turns, turns[1:])]
        assert max(gaps) <= 8, f"a gap of {max(gaps)} days remains"


class TestFridayAnchorDoesNotRollIntoTheNextWeek:
    """A Friday-anchored week whose Friday is a holiday is SKIPPED rather than rolled into
    Monday, because Monday already carries its own week's anchor — rolling would give that
    week two rebalances. Not the live config, but the guard has to hold."""

    def test_good_friday_week_does_not_roll_forward(self):
        from app.live_trading.exchange_calendar import holidays

        good_friday = next(h for h in holidays(2026) if h.weekday() == FRI)
        assert is_rebalance_day(good_friday, FRI)[0] is False
        monday_after = good_friday + timedelta(days=3)
        due, why = is_rebalance_day(monday_after, FRI)
        assert not due, f"rolled into the next week: {why}"

    def test_that_week_simply_has_no_turn(self):
        from app.live_trading.exchange_calendar import holidays

        good_friday = next(h for h in holidays(2026) if h.weekday() == FRI)
        monday = good_friday - timedelta(days=4)
        week = [monday + timedelta(days=i) for i in range(7)]
        assert sum(1 for d in week if is_rebalance_day(d, FRI)[0]) == 0


class TestOutageIsNotCaughtUp:
    """Deliberately OUT of scope: a rebalance missed because the app was down is not
    retried. That would trade on an intent formed under unknown conditions, possibly days
    stale, and needs its own decision. A missed trading day still reads as missed."""

    def test_a_trading_day_anchor_consumes_the_week_even_if_nothing_ran(self):
        anchor = date(2026, 9, 14)
        assert is_rebalance_day(anchor, MON)[0] is True
        # the app being down on the 14th does not license a run on the 15th
        assert is_rebalance_day(date(2026, 9, 15), MON)[0] is False


class TestCallersStillGuardOnMarketOpen:
    """This helper decides WHICH DAY, never whether the market is open — the static
    holiday list cannot know about an unscheduled closure, so the Alpaca-clock check
    stays as the fail-closed authority at each call site."""

    @pytest.mark.parametrize("fn", [
        "_trigger_trend_rebalance", "_trigger_cash_rebalance", "_verify_enforce_rebalance",
    ])
    def test_each_job_keeps_its_clock_check(self, fn):
        import inspect
        from app.orchestrator import AgentOrchestrator

        src = inspect.getsource(getattr(AgentOrchestrator, fn))
        assert "is_rebalance_day" in src
        assert "get_clock" in src
        assert "is_open" in src
