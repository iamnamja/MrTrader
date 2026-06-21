"""
kill_switch_state.py — Alpha-v10 R0.4: the cross-venue KILL-SWITCH STATE MACHINE.

A single state governing the whole book across venues (the roadmap's design). Shadow / not-wired in
R0.4: it models the state + the can-I-trade questions; nothing reads it on the live order path yet
(that wiring is R0.5/R1). Distinct from the existing app-level `kill_switch.py` flag — this is the
richer state machine the portfolio brain will adopt.

States (increasing severity):
  NORMAL            — full trading.
  HALT_NEW_RISK     — no risk-INCREASING orders; reductions/exits allowed (the dead-man default).
  CANCEL_ONLY       — cancel open orders; no new orders.
  FLATTEN_NON_CORE  — flatten everything except the core (e.g. trend); reduce-only otherwise.
  FLATTEN_ALL       — cancel + flatten all venues.
  MANUAL_LOCK       — locked; only a human can move it off.

Design rules:
  * Risk-increasing orders are allowed ONLY in NORMAL.
  * The dead-man / a stale heartbeat / a reconciliation FAIL_CLOSED AUTO-escalates to HALT_NEW_RISK
    (never auto-flatten — a flaky watchdog must not liquidate the book). Full flatten is operator-
    confirmed except explicit hard rules.
  * Auto-transitions can only ESCALATE (raise severity). De-escalation requires an explicit human
    `manual=True` (mirrors "fail-closed; a human re-arms").
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

NORMAL = "NORMAL"
HALT_NEW_RISK = "HALT_NEW_RISK"
CANCEL_ONLY = "CANCEL_ONLY"
FLATTEN_NON_CORE = "FLATTEN_NON_CORE"
FLATTEN_ALL = "FLATTEN_ALL"
MANUAL_LOCK = "MANUAL_LOCK"

# severity order (index = severity; higher = more restrictive)
_SEVERITY = [NORMAL, HALT_NEW_RISK, CANCEL_ONLY, FLATTEN_NON_CORE, FLATTEN_ALL, MANUAL_LOCK]
# the max severity an AUTO (non-manual) trigger may escalate to — a flaky watchdog must NOT be able
# to liquidate the book; any FLATTEN_* / MANUAL_LOCK requires an explicit human (manual=True).
_MAX_AUTO_STATE = CANCEL_ONLY


def severity(state: str) -> int:
    return _SEVERITY.index(state)


@dataclass(frozen=True)
class KillEvent:
    ts: float
    prior: str
    new: str
    reason: str
    actor: str
    manual: bool


class KillSwitch:
    """In-memory kill-switch state machine (shadow). The live version persists to Postgres
    `control_flags` and is read on the order path; this models the logic + transition rules."""

    def __init__(self, state: str = NORMAL, *, clock=time.time):
        if state not in _SEVERITY:
            raise ValueError(f"unknown kill state {state!r}")
        self._state = state
        self._events: List[KillEvent] = []
        self._last_heartbeat: float = clock()
        self._clock = clock

    @property
    def state(self) -> str:
        return self._state

    @property
    def events(self) -> List[KillEvent]:
        return list(self._events)

    def can_increase_risk(self) -> bool:
        """Risk-increasing (new long/short) orders allowed ONLY in NORMAL."""
        return self._state == NORMAL

    def allows_reduce_only(self) -> bool:
        """Risk-reducing orders allowed unless fully locked."""
        return self._state in (NORMAL, HALT_NEW_RISK, CANCEL_ONLY,
                               FLATTEN_NON_CORE, FLATTEN_ALL)

    def requires_flatten(self) -> bool:
        return self._state in (FLATTEN_NON_CORE, FLATTEN_ALL)

    def set_state(self, new: str, *, reason: str, actor: str, manual: bool = False) -> bool:
        """Transition. AUTO (manual=False) transitions can only ESCALATE severity; de-escalation
        (or moving off MANUAL_LOCK) requires manual=True. Returns True if the state changed."""
        if new not in _SEVERITY:
            raise ValueError(f"unknown kill state {new!r}")
        if self._state == MANUAL_LOCK and not manual:
            return False                                  # only a human leaves MANUAL_LOCK
        if not manual and severity(new) < severity(self._state):
            return False                                  # auto cannot de-escalate
        if not manual and severity(new) > severity(_MAX_AUTO_STATE):
            return False                                  # auto cannot reach FLATTEN_*/MANUAL_LOCK
        if new == self._state:
            return False
        self._events.append(KillEvent(self._clock(), self._state, new, reason, actor, manual))
        self._state = new
        return True

    # --- auto safety triggers (escalate to HALT_NEW_RISK only; never auto-flatten) ---
    def heartbeat(self) -> None:
        self._last_heartbeat = self._clock()

    def dead_man_check(self, *, max_stale_sec: float) -> bool:
        """If the heartbeat is stale, AUTO-escalate to HALT_NEW_RISK (not flatten). Returns True if
        it escalated."""
        if self._clock() - self._last_heartbeat > max_stale_sec:
            return self.set_state(HALT_NEW_RISK, reason="dead-man: stale heartbeat",
                                  actor="watchdog")
        return False

    def on_reconciliation_fail(self) -> bool:
        """A reconciliation FAIL_CLOSED auto-escalates to HALT_NEW_RISK."""
        return self.set_state(HALT_NEW_RISK, reason="reconciliation FAIL_CLOSED", actor="reconciler")
