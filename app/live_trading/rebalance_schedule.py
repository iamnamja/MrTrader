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

    THE WEEK'S TURN IS TAKEN WHEN THE JOB RAN — NOT WHEN THE CALENDAR SAYS IT COULD
    HAVE, AND NOT WHEN SOME OTHER JOB TRADED.

Both halves of that were learned the hard way.

A first cut inferred "the turn has passed" from the static calendar alone, which silently
re-created the bug: if the anchor was a trading day but the job was DECLINED that morning
— a transient Alpaca clock error, or an unscheduled closure absent from the static holiday
list (NYSE 2025-01-09 Carter funeral, 2018-12-05, Sandy 2012) — the next day still read
"this week's turn was Monday" and skipped, giving back the 14-day gap.

A second cut then used the TREND SLEEVE'S trade record as the authority for all three
weekly jobs, which is worse: trend writes its row at 09:45, so cash (09:50) and the
enforce-verify (11:07) both read "already rebalanced this week" and skipped — every week,
forever. It also conflated "the job ran" with "the job traded": no row is written when a
rebalance is HELD by a gate, when the target book is all-cash, or in shadow mode, so those
weeks would have re-run the sleeve every remaining day.

So the state is PER JOB and records RUNNING, not trading. Each job marks its turn as soon
as the market-open gate passes and BEFORE it does any work, which makes the guarantee
at-most-once-per-week: a crash mid-rebalance loses that week rather than risking a second
pass over partially-placed orders. A job that never got past the market-open gate has not
taken its turn and is retried the next trading day of the same week.

The roll never crosses a week boundary in either direction, so no week can take two turns.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from app.live_trading.exchange_calendar import is_trading_day

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("MRTRADER_BACKVAL_DB",
                              str(_ROOT / "data" / "back_validation.db")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS weekly_turn (
    job        TEXT PRIMARY KEY,
    week_start TEXT NOT NULL,
    taken_on   TEXT NOT NULL,
    taken_at   TEXT NOT NULL
);
"""

# Job identities. Separate rows because these are three INDEPENDENT weekly turns that
# happen to share an anchor — collapsing them is what disabled cash and the verify.
JOB_TREND = "trend_rebalance"
JOB_CASH = "cash_rebalance"
JOB_ENFORCE_VERIFY = "enforce_verify"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=10)
    c.execute("PRAGMA journal_mode=WAL;")
    c.executescript(_SCHEMA)
    return c


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def turn_taken_this_week(job: str, today: date) -> Optional[bool]:
    """Has `job` already taken its turn in `today`'s week?

    None means UNKNOWN (the store could not be read) — the caller then degrades to the
    conservative calendar-only rule rather than guessing either way.
    """
    try:
        with _conn() as c:
            row = c.execute("SELECT week_start FROM weekly_turn WHERE job = ?",
                            (job,)).fetchone()
        if not row or not row[0]:
            return False
        return date.fromisoformat(str(row[0])[:10]) >= week_start(today)
    except Exception as exc:      # noqa: BLE001 - scheduling must never break on this
        log.warning("weekly_turn read failed for %s (%s) — falling back to the "
                    "calendar-only rule", job, exc)
        return None


def mark_turn_taken(job: str, today: date) -> bool:
    """Record that `job` took its turn today. Never raises.

    Called AFTER the market-open gate and BEFORE the work, so the guarantee is
    at-most-once per week: a crash part-way through a rebalance loses the week rather
    than risking a second pass over partially-placed orders.
    """
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO weekly_turn(job, week_start, taken_on, taken_at) "
                "VALUES (?,?,?,?) ON CONFLICT(job) DO UPDATE SET "
                "week_start=excluded.week_start, taken_on=excluded.taken_on, "
                "taken_at=excluded.taken_at",
                (job, week_start(today).isoformat(), today.isoformat(),
                 datetime.now(timezone.utc).isoformat()),
            )
        return True
    except Exception as exc:      # noqa: BLE001
        log.error("weekly_turn write failed for %s on %s (%s) — the job may run again "
                  "this week", job, today, exc)
        return False


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
    turn_taken: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Should this job run today? Returns (verdict, human-readable reason).

    `turn_taken` is THIS JOB'S own weekly state from `turn_taken_this_week()`:
      False -> the job has not run this week; it may take the turn, or retry a declined one
      True  -> already ran this week; skip
      None  -> unknown (store unreadable) -> degrade to the calendar-only rule, which
               treats a passed turn as spent. The safe direction for a failed read is
               fewer rebalances, not more.

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

    if turn_taken:
        return False, "already ran this week"

    if turn_taken is None:
        # No state: fall back to the calendar. It cannot tell a declined turn from a taken
        # one, so it treats the turn as spent once the day has passed.
        if today != turn:
            return False, (f"this week's turn was {turn}; no run record available to "
                           f"justify a retry")
        return True, f"this week's turn ({turn}); no run record available"

    if today == turn:
        return True, f"this week's turn ({turn})"
    return True, (f"this week's turn was {turn} but the job did not run then — "
                  f"retrying on {today}")
