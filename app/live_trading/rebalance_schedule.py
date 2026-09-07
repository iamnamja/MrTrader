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

So the state is PER JOB and records RUNNING, not trading. Each job CLAIMS its turn as soon
as the market-open gate passes and BEFORE it does any work, and stands down if the claim
is refused. The claim is a conditional upsert, so it is a real mutex rather than a
check-then-write with a network call in the middle: at-most-once per week holds even with
two orchestrator processes on the same store, and a crash mid-rebalance loses that week
rather than risking a second pass over partially-placed orders. A job that never got past
the market-open gate has not claimed its turn and is retried the next trading day of the
same week.

CASH AND THE ENFORCE-VERIFY FOLLOW TREND, THEY DO NOT DECIDE INDEPENDENTLY. Cash parks
what the trend rebalance left idle, and the verify checks the rebalance that just ran, so
both gate on "trend claimed its turn TODAY" as well as their own. Letting them decide the
day for themselves means the verify can email a false ATTENTION about a rebalance that has
not happened yet — leaving the real one unverified, including the un-backfillable CH0b
capture — and cash can sweep into T-bills on a day trend did not rebalance.

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


class TurnStoreUnavailable(RuntimeError):
    """The weekly_turn store could not be read.

    Distinct from "never claimed" ON PURPOSE. Collapsing both into None made an unreadable
    store indistinguishable from a job that has not run yet, and the two demand opposite
    responses: a follower must stand down when it cannot tell whether the leader ran, but
    must run when it can tell the leader has.
    """


def turn_taken_on(job: str) -> Optional[date]:
    """The date `job` last claimed a turn, or None if it never has.

    Raises TurnStoreUnavailable if the store cannot be read — callers must decide, rather
    than silently receiving a value that reads as "never".
    """
    try:
        with _conn() as c:
            row = c.execute("SELECT taken_on FROM weekly_turn WHERE job = ?",
                            (job,)).fetchone()
    except Exception as exc:      # noqa: BLE001
        raise TurnStoreUnavailable(str(exc)) from exc
    return date.fromisoformat(str(row[0])[:10]) if row and row[0] else None


def leader_ran_this_week(leader: str, today: date) -> bool:
    """Has `leader` claimed a turn in `today`'s week, on or before today?

    The follower gate. Deliberately WEEK-scoped, not day-scoped: an earlier cut required
    `turn_taken_on(leader) == today`, which made the follower's own retry path unreachable
    — if the verify's 11:07 clock call threw on the day trend rebalanced fine, that week's
    verify (with its un-backfillable CH0b capture) was simply lost, because Tue-Fri could
    never satisfy `== today` again. Week scope keeps the ordering guarantee (`<= today`)
    while restoring the retry.

    It also decouples the two live-tunable weekday keys: `pm.cash_rebalance_weekday` need
    not equal `pm.trend_rebalance_weekday`. Under the day-scoped rule, cash=1 with trend=0
    left the cash sleeve permanently and silently dead.

    Raises TurnStoreUnavailable rather than guessing.
    """
    day = turn_taken_on(leader)
    if day is None:
        return False
    return week_start(today) <= day <= today


def claim_turn(job: str, today: date) -> bool:
    """Atomically claim `job`'s turn for `today`'s week. True only if THIS caller won it.

    THE CLAIM IS THE MUTEX, which is why it is a conditional upsert and not a plain write.
    A read-then-write pair leaves a window — SELECT, an Alpaca network round-trip, then an
    unconditional write — in which a second orchestrator process passes the same check and
    both trade. That is not hypothetical here: a second copy of this system pointed at the
    same account was one `docker start` away as recently as 2026-09-02 (DECISIONS). The
    `WHERE weekly_turn.week_start < excluded.week_start` clause makes the database the
    arbiter, and rowcount says who won.

    A False return therefore means EITHER someone already holds this week's turn OR the
    store could not be written. Both must stop the caller: an unclaimable turn that still
    traded would re-run every remaining day of the week — five live rebalances — with one
    ERROR line a day as the only symptom. Callers treat False as fail-closed.
    """
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO weekly_turn(job, week_start, taken_on, taken_at) "
                "VALUES (?,?,?,?) ON CONFLICT(job) DO UPDATE SET "
                "week_start=excluded.week_start, taken_on=excluded.taken_on, "
                "taken_at=excluded.taken_at "
                "WHERE weekly_turn.week_start < excluded.week_start",
                (job, week_start(today).isoformat(), today.isoformat(),
                 datetime.now(timezone.utc).isoformat()),
            )
            won = cur.rowcount > 0
        if not won:
            log.info("weekly_turn: %s already claimed for the week of %s — standing down",
                     job, week_start(today))
        return won
    except Exception as exc:      # noqa: BLE001
        log.error("weekly_turn claim FAILED for %s on %s (%s) — standing down rather than "
                  "running unclaimed", job, today, exc)
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
