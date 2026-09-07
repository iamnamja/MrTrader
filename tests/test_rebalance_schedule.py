"""2026-09-07 — a holiday on the rebalance weekday cancelled the week instead of delaying it.

The weekly jobs were pinned to a calendar weekday and fail-closed on a market holiday with
no fallthrough, so Labor Day 2026-09-07 put 14 calendar days between the 08-31 and 09-14
rebalances. The frozen CH0a baseline (trend book CPCV mean_sharpe 0.7009) rebalances on a
5-TRADING-day grid that never skips a holiday week — `is_rebal = (np.arange(n) % 5 == 0)`
over a trading-day index — so the live skip was a divergence from the validated cadence,
recurring 4-5x a year.

The central invariant these pin: THE WEEK'S TURN IS TAKEN WHEN WE ACTUALLY REBALANCED,
not when the calendar says we could have.
"""
from datetime import date, timedelta

import pytest

from app.live_trading.exchange_calendar import holidays, is_trading_day
from app.live_trading.rebalance_schedule import (
    is_rebalance_day, week_anchor, week_turn,
)

MON = 0
FRI = 4


def _simulate(year: int, target_weekday: int = MON, decline: set = frozenset()):
    """Walk a year the way the orchestrator does: ask, and record a rebalance when it runs.

    `decline` = days the market-open guard refuses even though the calendar calls them
    trading days (unscheduled closure, transient clock error).
    """
    turns, last = [], None
    d = date(year, 1, 1)
    while d < date(year + 1, 1, 1):
        if d.weekday() < 5:                      # cron fires Mon-Fri
            due, _ = is_rebalance_day(d, target_weekday, last_rebalance=last)
            if due and is_trading_day(d) and d not in decline:
                turns.append(d)
                last = d
        d += timedelta(days=1)
    return turns


class TestWeekAnchorAndTurn:
    def test_anchor_is_inside_todays_own_week(self):
        assert week_anchor(date(2026, 9, 8), MON) == date(2026, 9, 7)
        assert week_anchor(date(2026, 9, 11), MON) == date(2026, 9, 7)

    def test_turn_is_the_anchor_when_it_trades(self):
        assert week_turn(date(2026, 9, 14), MON) == date(2026, 9, 14)

    def test_turn_rolls_forward_past_a_monday_holiday(self):
        assert week_turn(date(2026, 9, 7), MON) == date(2026, 9, 8)

    def test_turn_never_leaves_the_anchors_own_week(self):
        for offset in range(7):
            d = date(2026, 9, 7) + timedelta(days=offset)
            monday = d - timedelta(days=d.weekday())
            t = week_turn(d, MON)
            assert monday <= t <= monday + timedelta(days=4)


class TestHolidayFallthrough:
    """Labor Day 2026: Mon 09-07 is a holiday, so the turn moves to Tue 09-08."""

    LAST = date(2026, 8, 31)

    def test_anchor_holiday_does_not_run(self):
        due, why = is_rebalance_day(date(2026, 9, 7), MON, last_rebalance=self.LAST)
        assert not due

    def test_runs_on_the_next_trading_day(self):
        due, why = is_rebalance_day(date(2026, 9, 8), MON, last_rebalance=self.LAST)
        assert due and "turn" in why

    @pytest.mark.parametrize("d", [
        date(2026, 9, 9), date(2026, 9, 10), date(2026, 9, 11),
    ])
    def test_does_not_fire_again_once_the_week_is_taken(self, d):
        """THE TRAP a naive 'next trading day' rule falls into: Wed, Thu and Fri all pass
        the market-open check too, so it would rebalance four times."""
        due, why = is_rebalance_day(d, MON, last_rebalance=date(2026, 9, 8))
        assert not due and "already rebalanced" in why


class TestTheWeekIsSpentByTradingNotByTheCalendar:
    """The finding that a calendar-only rule re-created the bug it was fixing."""

    def test_a_declined_anchor_is_retried_later_that_week(self):
        """Anchor Mon 09-14 IS a trading day, but the rebalance was declined (clock error,
        or an unscheduled closure the static list does not know: NYSE 2025-01-09,
        2018-12-05, Sandy 2012). Inferring 'we must have traded' would skip the week."""
        due, why = is_rebalance_day(
            date(2026, 9, 15), MON, last_rebalance=date(2026, 9, 8))
        assert due and "no rebalance was recorded" in why

    def test_a_taken_anchor_is_not_retried(self):
        due, _ = is_rebalance_day(
            date(2026, 9, 15), MON, last_rebalance=date(2026, 9, 14))
        assert not due

    def test_without_a_record_it_degrades_conservatively(self):
        """A failed lookup must not license extra rebalances — the safe direction is the
        old calendar-only behaviour."""
        due, why = is_rebalance_day(date(2026, 9, 15), MON, last_rebalance=None)
        assert not due and "no rebalance record" in why

    def test_without_a_record_the_holiday_roll_still_works(self):
        due, _ = is_rebalance_day(date(2026, 9, 8), MON, last_rebalance=None)
        assert due


class TestSimulatedYear:
    """The defect restated as a measurement, over a realistic run."""

    @pytest.mark.parametrize("year", [2026, 2027, 2028])
    def test_no_gap_exceeds_eight_days(self, year):
        turns = _simulate(year)
        gaps = [(b - a).days for a, b in zip(turns, turns[1:])]
        assert max(gaps) <= 8, f"{year}: a {max(gaps)}-day gap remains"

    @pytest.mark.parametrize("year", [2026, 2027, 2028])
    def test_exactly_one_turn_per_iso_week(self, year):
        turns = _simulate(year)
        weeks = [t.isocalendar()[:2] for t in turns]
        assert len(weeks) == len(set(weeks)), "some week took two turns"

    def test_recovers_the_rebalances_the_old_rule_discarded(self):
        """Old rule = calendar weekday AND trading day, no fallthrough."""
        def old(year):
            return [d for d in _all_days(year)
                    if d.weekday() == MON and is_trading_day(d)]

        def _all_days(year):
            d = date(year, 1, 1)
            while d < date(year + 1, 1, 1):
                yield d
                d += timedelta(days=1)

        for year in (2026, 2027):
            assert len(_simulate(year)) > len(old(year))

    def test_an_unscheduled_closure_does_not_cost_the_week(self):
        """Decline the anchor the way a real outage would, and the week still gets a turn."""
        anchor = date(2026, 9, 14)
        turns = _simulate(2026, decline={anchor})
        wk = anchor.isocalendar()[:2]
        taken = [t for t in turns if t.isocalendar()[:2] == wk]
        assert len(taken) == 1 and taken[0] > anchor


class TestMondayHolidaysSpecifically:
    def _monday_holidays(self, year):
        return sorted(h for h in holidays(year) if h.weekday() == MON)

    def test_there_are_several_each_year(self):
        assert len(self._monday_holidays(2026)) >= 3

    def test_each_one_rolls_to_the_next_trading_day(self):
        for h in self._monday_holidays(2026):
            assert is_rebalance_day(h, MON, last_rebalance=h - timedelta(days=7))[0] is False
            assert week_turn(h, MON) > h
            assert is_trading_day(week_turn(h, MON))


class TestFridayAnchorRollsBackwards:
    """A Friday-anchored week whose Friday is a holiday rolls BACK to Thursday. Rolling
    forward would land on the next Monday, which already carries its own week's anchor and
    would give that week two turns; rolling backwards has no such collision, so skipping
    the week (an earlier cut) was strictly worse. `weekday` is live-tunable from the DB
    with no redeploy, so this is reachable without a code change."""

    def _good_friday(self, year):
        return next(h for h in holidays(year) if h.weekday() == FRI)

    def test_rolls_back_to_thursday(self):
        gf = self._good_friday(2027)
        turn = week_turn(gf, FRI)
        assert turn == gf - timedelta(days=1)
        assert is_trading_day(turn)

    def test_does_not_roll_into_the_next_week(self):
        gf = self._good_friday(2027)
        monday_after = gf + timedelta(days=3)
        assert week_turn(monday_after, FRI) != gf - timedelta(days=1)

    def test_friday_anchor_keeps_one_turn_per_week(self):
        turns = _simulate(2027, target_weekday=FRI)
        weeks = [t.isocalendar()[:2] for t in turns]
        assert len(weeks) == len(set(weeks))

    def test_friday_anchor_has_no_fourteen_day_gap(self):
        turns = _simulate(2027, target_weekday=FRI)
        gaps = [(b - a).days for a, b in zip(turns, turns[1:])]
        assert max(gaps) <= 8


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
        assert "get_clock" in src and "is_open" in src

    @pytest.mark.parametrize("fn", [
        "_trigger_trend_rebalance", "_trigger_cash_rebalance", "_verify_enforce_rebalance",
    ])
    def test_each_job_consults_the_rebalance_record(self, fn):
        import inspect
        from app.orchestrator import AgentOrchestrator

        src = inspect.getsource(getattr(AgentOrchestrator, fn))
        assert "last_rebalance_date" in src
        assert "last_rebalance=" in src


class TestLastRebalanceLookupIsSafe:
    def test_returns_none_and_does_not_raise_when_the_store_is_broken(self, monkeypatch):
        import app.live_trading.back_validation as bv

        def _boom(*a, **k):
            raise RuntimeError("db gone")
        monkeypatch.setattr(bv, "_conn", _boom)
        assert bv.last_rebalance_date() is None
