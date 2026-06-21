"""
reconciliation.py — Alpha-v10 R0.4: reconciliation-before-trade (the #1 safety gate).

"The broker is reality; the DB is memory." Before any trade cycle, the brain compares the DB's
EXPECTED positions/cash against the broker's ACTUAL (normalized) truth and FAILS CLOSED on a
material, unexplained mismatch — because the cost of skipping a weekly rebalance is one week of
slightly-stale exposure, while the cost of trading on a phantom position is unbounded.

Shadow / report-only in R0.4: it computes the verdict; it does not (yet) gate live orders (that is
R0.5). Resolution policy (when wired): within tolerance -> overwrite the DB toward broker reality,
log, proceed; material/unexplained -> FAIL_CLOSED (hold current, alert).

Tolerances (risk-policy-derived): equities/futures per-instrument qty delta MUST be 0 after
excluding known-pending orders; cash within max($cash_abs, cash_bps * NAV).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.live_trading.broker_adapter import CanonicalPosition

MATCH = "MATCH"
RESOLVED = "RESOLVED_WITHIN_TOLERANCE"
FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True)
class PositionBreak:
    instrument_id: str
    venue: str
    expected_qty: float
    actual_qty: float
    delta: float


@dataclass(frozen=True)
class ReconciliationResult:
    status: str                              # MATCH / RESOLVED_WITHIN_TOLERANCE / FAIL_CLOSED
    position_breaks: List[PositionBreak] = field(default_factory=list)
    cash_break: Optional[float] = None       # |expected_cash - actual_cash| if over tolerance
    notes: List[str] = field(default_factory=list)

    @property
    def ok_to_trade(self) -> bool:
        return self.status in (MATCH, RESOLVED)


def _ckey(venue: str, iid: str) -> tuple:
    return (venue, iid)


def reconcile(expected_qty: Dict[tuple, float],
              actual_positions: List[CanonicalPosition],
              *,
              expected_cash: Optional[float] = None,
              actual_cash: Optional[float] = None,
              nav: Optional[float] = None,
              pending_qty: Optional[Dict[tuple, float]] = None,
              qty_tol: float = 1e-6,
              cash_abs_tol: float = 100.0,
              cash_bps_tol: float = 5.0) -> ReconciliationResult:
    """Compare DB-expected vs broker-actual. Keyed by **(venue, instrument_id)** so the same
    canonical instrument held on two venues never collides (an id-only key could silently drop a
    cross-venue position and report a false MATCH — the one failure this gate must never have).

    `expected_qty` / `pending_qty`: {(venue, instrument_id): signed_qty} (pending = orders
    sent-but-unfilled, added to expected first). Actual quantities are SUMMED per key (duplicate
    lots don't overwrite). A position present on ONE side only is a break. Any qty delta beyond
    `qty_tol` (fractional-share precision) is MATERIAL -> FAIL_CLOSED. Cash beyond max(abs, bps*NAV)
    is a break; if cash inputs are omitted the status carries a "cash NOT checked" note (a live
    caller must supply them).
    """
    pending = pending_qty or {}
    actual_qty: Dict[tuple, float] = {}
    for p in actual_positions:
        k = _ckey(p.venue, p.instrument_id)
        actual_qty[k] = actual_qty.get(k, 0.0) + float(p.quantity)   # SUM, never overwrite

    eff_expected: Dict[tuple, float] = {k: float(v) for k, v in expected_qty.items()}
    for k, q in pending.items():
        eff_expected[k] = eff_expected.get(k, 0.0) + float(q)

    breaks: List[PositionBreak] = []
    for k in set(eff_expected) | set(actual_qty):
        exp = float(eff_expected.get(k, 0.0))
        act = float(actual_qty.get(k, 0.0))
        if abs(exp - act) > qty_tol:
            venue, iid = k
            breaks.append(PositionBreak(instrument_id=iid, venue=venue,
                                        expected_qty=exp, actual_qty=act, delta=act - exp))

    cash_break = None
    cash_checked = expected_cash is not None and actual_cash is not None
    if cash_checked:
        tol = max(cash_abs_tol, (cash_bps_tol / 1e4) * (nav or 0.0))
        diff = abs(expected_cash - actual_cash)
        if diff > tol:
            cash_break = diff

    notes: List[str] = []
    if breaks:
        notes.append(f"{len(breaks)} position break(s); positions/futures must reconcile EXACTLY "
                     "(excl. known-pending) -> FAIL_CLOSED.")
    if cash_break is not None:
        notes.append(f"cash mismatch ${cash_break:,.0f} beyond tolerance -> FAIL_CLOSED.")
    if not cash_checked:
        notes.append("cash NOT checked (no cash inputs) — a live caller MUST supply expected/actual cash.")
    if not breaks and cash_break is None:
        notes.append("broker reality matches DB intent (within tolerance).")

    status = FAIL_CLOSED if (breaks or cash_break is not None) else MATCH
    return ReconciliationResult(status=status, position_breaks=breaks, cash_break=cash_break,
                                notes=notes)
