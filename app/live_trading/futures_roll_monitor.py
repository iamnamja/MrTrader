"""
futures_roll_monitor.py — Alpha-v10 R1.3: delivery-safety monitor for held futures.

Hardens the "bot down for a day or two near roll time" scenario. Three pieces, all driven by the
`futures_roll_policy` roll dates (nothing here trades):

  1. `catch_up_on_restart(held, today)` — on daemon startup, return the futures whose roll is DUE/overdue
     so they roll IMMEDIATELY, not at the next weekly rebalance cadence (an outage over the roll date
     must not leave a position sitting in an expiring contract until next Monday).
  2. Downtime-buffered urgency tiers — `APPROACHING` pre-alerts a few trading days BEFORE the roll date
     (so an outage window can be planned around it), and `CRITICAL` escalates a safety margin BEFORE the
     hard delivery floor (so even a multi-day outage can't silently cross it).
  3. `delivery_risk_alerts` / `notify_delivery_risk` — page the owner (CATASTROPHIC `futures_delivery_risk`
     → the off-box snitch webhook) when a held physical future nears its FND/last-trade floor unrolled.

Defense-in-depth: this sits on top of the EARLY hybrid roll (roll on OI-crossover, weeks before FND) and
the hard floor in `futures_roll_policy`; IBKR's own pre-delivery auto-liquidation is the last resort only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from app.live_trading import exchange_calendar as ec
from app.live_trading import futures_roll_policy as rp

log = logging.getLogger(__name__)

# Urgency tiers (ordered by _RANK).
OK = "ok"
APPROACHING = "approaching"      # within DOWNTIME_ALERT_BDAYS of the roll date — plan around outages
ROLL_DUE = "roll_due"            # today >= recommended roll — roll now (catch-up rolls these)
CRITICAL = "critical"            # within CRITICAL_MARGIN_BDAYS of the hard delivery floor — page owner
_RANK = {OK: 0, APPROACHING: 1, ROLL_DUE: 2, CRITICAL: 3}

DOWNTIME_ALERT_BDAYS = 3         # pre-alert this many trading days before the roll date
CRITICAL_MARGIN_BDAYS = 2        # escalate this many trading days BEFORE the hard delivery floor


def _business_days_between(a: date, b: date) -> int:
    """Signed TRADING-day count from `a` to `b` (positive if b is after a) — HOLIDAY-AWARE, so
    `days_to_floor` reflects real trading days, not raw weekdays across a holiday."""
    return ec.trading_days_between(a, b)


@dataclass(frozen=True)
class RollAssessment:
    root: str
    settlement: str
    delivery_month: Optional[str]
    qty: Any
    today: date
    recommended: Optional[date]      # roll date (futures_roll_policy)
    floor: Optional[date]            # hard delivery-risk deadline = earliest of {FND, last-trade cap}
    urgency: str
    roll_due: bool                   # today >= recommended
    days_to_floor: Optional[int]     # trading days from today to floor (negative = PAST the floor)

    @property
    def is_alert(self) -> bool:
        return self.urgency == CRITICAL

    def payload(self) -> Dict[str, Any]:
        return {
            "root": self.root, "settlement": self.settlement, "delivery_month": self.delivery_month,
            "qty": self.qty, "today": self.today.isoformat(),
            "recommended": self.recommended.isoformat() if self.recommended else None,
            "floor": self.floor.isoformat() if self.floor else None,
            "days_to_floor": self.days_to_floor, "urgency": self.urgency,
        }


def assess(root: str, today: date, *, contract_month: Optional[str] = None,
           last_trade: Optional[str] = None, qty: Any = None,
           liquidity_roll: Optional[date] = None) -> RollAssessment:
    """Urgency assessment for ONE held futures contract."""
    rd = rp.compute_roll_dates(root, contract_month=contract_month, last_trade=last_trade,
                               liquidity_roll=liquidity_roll)
    rec = rd.recommended
    floor_cands = [d for d in (rd.fnd_floor, rd.last_trade_cap) if d is not None]
    floor = min(floor_cands) if floor_cands else None
    roll_due = bool(rec is not None and today >= rec)

    urgency = OK
    if floor is not None and today >= rp._minus_business_days(floor, CRITICAL_MARGIN_BDAYS):
        urgency = CRITICAL                                   # near/at the delivery floor — escalate
    elif roll_due:
        urgency = ROLL_DUE
    elif rec is not None and today >= rp._minus_business_days(rec, DOWNTIME_ALERT_BDAYS):
        urgency = APPROACHING
    # FAIL-SAFE: a physically-delivered position with NO computable floor (both contract_month AND
    # last_trade missing/malformed) must NEVER read OK — escalate so it pages rather than silently
    # riding into delivery. Uses settlement()'s PHYSICAL default (an unknown root also assumes delivery).
    if floor is None and rp.settlement(root) == rp.PHYSICAL:
        urgency = CRITICAL
    return RollAssessment(
        root=str(root).upper(), settlement=rp.settlement(root), delivery_month=contract_month, qty=qty,
        today=today, recommended=rec, floor=floor, urgency=urgency, roll_due=roll_due,
        days_to_floor=(_business_days_between(today, floor) if floor is not None else None))


def assess_all(held: List[Dict[str, Any]], today: date) -> List[RollAssessment]:
    """Assess every held future. `held`: dicts with {root, contract_month, last_trade, qty?,
    liquidity_roll?}. A malformed row is skipped (logged), never crashes the monitor."""
    out: List[RollAssessment] = []
    for h in held:
        try:
            out.append(assess(h["root"], today, contract_month=h.get("contract_month"),
                              last_trade=h.get("last_trade"), qty=h.get("qty"),
                              liquidity_roll=h.get("liquidity_roll")))
        except Exception as e:  # noqa: BLE001 — a monitor must degrade, never crash the daemon
            log.warning("futures_roll_monitor: could not assess %r: %s", h, e)
    return out


def catch_up_on_restart(held: List[Dict[str, Any]], today: date) -> List[RollAssessment]:
    """Futures whose roll is DUE/overdue (or CRITICAL) — to roll IMMEDIATELY on daemon startup rather
    than waiting for the next weekly rebalance. This is the core outage-over-the-roll-date protection."""
    due = [a for a in assess_all(held, today) if _RANK[a.urgency] >= _RANK[ROLL_DUE]]
    if due:
        log.warning("futures_roll_monitor: %d held future(s) roll-DUE on restart — roll now: %s",
                    len(due), ", ".join(f"{a.root}({a.urgency})" for a in due))
    return due


def delivery_risk_alerts(held: List[Dict[str, Any]], today: date) -> List[RollAssessment]:
    """CRITICAL assessments — a held future at/near its delivery floor unrolled (page the owner)."""
    return [a for a in assess_all(held, today) if a.is_alert]


def notify_delivery_risk(alerts: List[RollAssessment], notifier=None) -> int:
    """Enqueue a CATASTROPHIC `futures_delivery_risk` alert per CRITICAL assessment (→ off-box snitch).
    Best-effort; never raises. Returns the count enqueued."""
    if not alerts:
        return 0
    if notifier is None:
        try:
            from app.notifications import notifier as notifier  # noqa: PLW0127
        except Exception:  # noqa: BLE001 — truly never raise: a failed import must not crash the daemon
            log.warning("futures_roll_monitor: notifier import failed — delivery alerts NOT sent")
            return 0
    n = 0
    for a in alerts:
        try:
            notifier.enqueue("futures_delivery_risk", a.payload(),
                             dedup_key=f"delivery_{a.root}_{a.delivery_month}_{a.today.isoformat()}")
            n += 1
        except Exception:  # noqa: BLE001
            log.warning("futures_roll_monitor: delivery alert enqueue failed for %s", a.root,
                        exc_info=True)
    return n
