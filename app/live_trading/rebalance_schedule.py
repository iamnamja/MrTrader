"""When the weekly rebalance actually runs — holiday-aware.

WHY THIS EXISTS (2026-09-07). The weekly jobs were pinned to a CALENDAR weekday and
fail-closed on a market holiday, with no fallthrough:

    if today.weekday() != target_weekday:
        return                      # and nothing ever catches it up

So a Monday holiday did not delay the rebalance by a day — it cancelled it for the week.
Labor Day 2026-09-07 put 14 calendar days between the 08-31 and 09-14 rebalances, and
Monday holidays recur 4-5x a year.

THE ARGUMENT IS FIDELITY, NOT TASTE. The frozen CH0a baseline that produced the trend
book's CPCV mean_sharpe 0.7009 rebalances on a TRADING-DAY grid:

    # app/strategy/tsmom.py
    is_rebal = (np.arange(n) % cfg.rebalance_days == 0)     # rebalance_days = 5

`n` indexes the daily close panel, which contains trading days only — so a holiday week
simply has four trading days and the 5-trading-day cadence never breaks. The live book
skipping a week ran a cadence the validated edge was never tested at. Measured:
2026 old = 48 turns with four 14-day gaps, new = 52 turns with a maximum gap of 8 days.

    THE WEEK'S TURN IS TAKEN WHEN WE ACTUALLY REBALANCED — NOT WHEN THE CALENDAR SAYS
    WE COULD HAVE.

That distinction is the whole correctness argument. An earlier cut inferred "the turn has
passed" from the static calendar alone, which silently re-created the bug: if the anchor
was a trading day but the rebalance was DECLINED that morning — a transient Alpaca clock
error, or an unscheduled closure absent from the static holiday list (NYSE 2025-01-09
Carter funeral, 2018-12-05, Sandy 2012) — the next day still read "this week's turn was
Monday" and skipped, producing exactly the 14-day gap this module exists to remove. So the
authority is `back_validation.last_rebalance_date()`, which records only genuine live
rebalances. The calendar is the fallback when that lookup is unavailable.

A consequence worth stating plainly: because the test is "did we rebalance", a week missed
because the APP WAS DOWN on the anchor day is also picked up on the next trading day of
that week. That is a widening from the first draft of this change, and it is deliberate —
the rebalance recomputes its weights from current data at run time, and every existing
gate (per-name enforce, whole-book, reconciliation, market-open) still applies, so there
is no stale-intent risk that would justify carrying the miss for a full extra week.

The roll never crosses a week boundary in either direction, so no week can take two turns.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional, Tuple

from app.live_trading.exchange_calendar import is_trading_day


def week_anchor(today: date, target_weekday: int) -> date:
    """The configured weekday within `today`'s own ISO week."""
    monday = today - timedelta(days=today.weekday())
    return monday + timedelta(days=target_weekday)


def _week_trading_days(today: date) -> List[date]:
    monday = today - timedelta(days=today.weekday())
    return [d for d in (monday + timedelta(days=i) for i in range(5)) if is_trading_day(d)]


def week_turn(today: date, target_weekday: int) -> Optional[date]:
    """The single day this week's rebalance belongs on, or None if the week has none.

    Preference order, all inside the anchor's own week:
      1. the anchor itself, when it trades
      2. the next trading day AFTER it   (Monday holiday -> Tuesday)
      3. the last trading day BEFORE it  (Friday holiday -> Thursday)

    Rule 3 exists because rolling a Friday anchor FORWARD would land on the next Monday,
    which already carries its own week's anchor and would give that week two turns.
    Rolling backwards has no such collision, so skipping the week outright — which an
    earlier cut did — was strictly worse than the available alternative. `weekday` is
    live-tunable from the DB with no redeploy, so this is reachable without a code change.
    """
    anchor = week_anchor(today, target_weekday)
    trading = _week_trading_days(today)
    if not trading:
        return None
    after = [d for d in trading if d >= anchor]
    if after:
        return after[0]
    return trading[-1]


def is_rebalance_day(
    today: date,
    target_weekday: int,
    last_rebalance: Optional[date] = None,
) -> Tuple[bool, str]:
    """Should the weekly rebalance run today? Returns (verdict, human-readable reason).

    `last_rebalance` is the date we last ACTUALLY rebalanced (see
    `back_validation.last_rebalance_date`). Pass it whenever it is available: it is what
    lets a declined anchor be retried later in the same week. When it is None the rule
    degrades to the conservative calendar-only behaviour — a missed anchor consumes the
    week — which is the safe direction for a failed lookup.

    This decides WHICH DAY. It never decides whether the market is open: the callers keep
    their Alpaca-clock check as the final fail-closed authority, because a static holiday
    list cannot know about an unscheduled closure.
    """
    turn = week_turn(today, target_weekday)
    if turn is None:
        return False, f"no trading day in the week of {today}"
    if today < turn:
        return False, f"before this week's turn ({turn})"
    if not is_trading_day(today):
        return False, f"{today} is not a trading day"

    monday = today - timedelta(days=today.weekday())

    if last_rebalance is not None:
        if last_rebalance >= monday:
            return False, (f"already rebalanced this week on {last_rebalance}")
        if today == turn:
            return True, f"this week's turn ({turn}); last was {last_rebalance}"
        return True, (f"this week's turn was {turn} and no rebalance was recorded "
                      f"(last {last_rebalance}) — retrying on {today}")

    # No record available: fall back to the calendar. Conservative by design — it cannot
    # tell a declined anchor from a taken one, so it treats the turn as spent.
    if today != turn:
        return False, (f"this week's turn was {turn}; no rebalance record available to "
                       f"justify a retry")
    return True, f"this week's turn ({turn}); no rebalance record available"
