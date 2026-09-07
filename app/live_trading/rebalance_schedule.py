"""When the weekly rebalance actually runs — holiday-aware, stateless.

WHY THIS EXISTS (2026-09-07). The weekly jobs were pinned to a CALENDAR weekday and
fail-closed on a market holiday, with no fallthrough:

    if today.weekday() != target_weekday:
        return                      # and nothing ever catches it up

So a Monday holiday did not delay the rebalance by a day — it cancelled it for the week.
Labor Day 2026-09-07 put 14 calendar days between the 08-31 and 09-14 rebalances, and
Monday holidays recur 4-5x a year (MLK, Washington's Birthday, Memorial Day, Juneteenth,
Independence Day, Labor Day can all land on a Monday).

THE ARGUMENT IS FIDELITY, NOT TASTE. The frozen CH0a baseline that produced the trend
book's CPCV mean_sharpe 0.7009 rebalances on a TRADING-DAY grid:

    # app/strategy/tsmom.py
    is_rebal = (np.arange(n) % cfg.rebalance_days == 0)     # rebalance_days = 5

`n` indexes the daily close panel, which contains trading days only — so a holiday week
simply has four trading days and the 5-trading-day cadence never breaks. The live book
skipping a week is a cadence the validated edge was never tested at (~9 trading days
instead of 5), several times a year. This restores live to the construction that was
actually validated.

STATELESS BY DESIGN. "First trading day on-or-after the anchor, within the anchor's own
week" is decidable from the calendar alone, so there is no "have we already rebalanced
this week?" flag to persist, drift, or contradict the DB. It also gives the once-per-week
guarantee for free: on the Wednesday after a Monday holiday the loop below finds Tuesday
was a trading day and refuses — without which a naive "next trading day" rule would
rebalance again every remaining day of the week.

WHAT THIS DELIBERATELY DOES NOT DO: catch up a rebalance missed because the APP was down
(the 2026-09-05 reboot, say). That is a different risk — it would trade on an intent
formed under unknown conditions, possibly days stale — and it needs its own decision. A
missed trading day still reads as missed here.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Tuple

from app.live_trading.exchange_calendar import is_trading_day


def week_anchor(today: date, target_weekday: int) -> date:
    """The configured weekday within `today`'s own ISO week.

    Anchoring inside the week is what keeps the fallthrough from ever crossing into the
    next week and colliding with its anchor.
    """
    monday = today - timedelta(days=today.weekday())
    return monday + timedelta(days=target_weekday)


def is_rebalance_day(today: date, target_weekday: int) -> Tuple[bool, str]:
    """Should the weekly rebalance run today? Returns (verdict, human-readable reason).

    True exactly when `today` is the FIRST trading day on or after this week's anchor.

    - anchor is a trading day        -> runs on the anchor, as before
    - anchor is a holiday            -> runs on the next trading day THAT WEEK
    - a later day in the same week   -> False (the anchor week already had its turn)
    - no trading day left that week  -> False (see the Friday note below)

    A Friday-anchored week whose Friday is a holiday is SKIPPED rather than rolled into
    Monday, because Monday already carries its own week's anchor and rolling would give
    that week two rebalances. With the live config (`pm.trend_rebalance_weekday=0`) the
    roll always stays inside the week, so this is a correctness guard, not a live path.

    This decides WHICH DAY. It does not decide whether the market is open — the callers
    keep their Alpaca-clock check as the final fail-closed authority, because this
    calendar is a static holiday list and cannot know about an unscheduled closure.
    """
    anchor = week_anchor(today, target_weekday)

    if today < anchor:
        return False, f"before this week's anchor ({anchor})"
    if not is_trading_day(today):
        return False, f"{today} is not a trading day"

    # Any trading day between the anchor and today means the turn has already come and
    # gone this week — this is the once-per-week guarantee.
    probe = anchor
    while probe < today:
        if is_trading_day(probe):
            return False, (f"this week's turn was {probe} (anchor {anchor}); "
                           f"not rebalancing again")
        probe += timedelta(days=1)

    if today == anchor:
        return True, f"anchor day ({anchor})"
    return True, (f"anchor {anchor} was a holiday — first trading day after it "
                  f"({today})")
