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

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.live_trading import instrument_master as im
from app.live_trading.broker_adapter import CanonicalPosition

log = logging.getLogger(__name__)

MATCH = "MATCH"
RESOLVED = "RESOLVED_WITHIN_TOLERANCE"
FAIL_CLOSED = "FAIL_CLOSED"

# Rollout modes (mirrors the whole-book gate). shadow = log/email only; enforce = HOLD fail-closed.
SHADOW = "shadow"
ENFORCE = "enforce"
OFF = "off"


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

    `expected_qty` / `pending_qty`: {(venue, instrument_id): signed_qty}. `expected` = positions the
    DB believes are HELD (ACTIVE). `pending` = in-flight working orders (the DB's PENDING_FILL rows).
    A working order means the broker's actual qty may legitimately be ANYWHERE on the unfilled→filled
    spectrum, so a break fires only when `actual` falls OUTSIDE the closed band
    `[expected, expected + pending]` (per key, sign-aware) by more than `qty_tol`. With no pending
    this collapses to the exact point check `|expected - actual| <= qty_tol` (backward-compatible),
    so a genuine phantom (DB held, broker flat, no working order) or orphan still breaks. Actual
    quantities are SUMMED per key (duplicate / cross-sleeve lots don't overwrite). Cash beyond
    max(abs, bps*NAV) is a break; if cash inputs are omitted the status carries a "cash NOT checked"
    note (a live caller must supply them).
    """
    pending = pending_qty or {}
    actual_qty: Dict[tuple, float] = {}
    for p in actual_positions:
        k = _ckey(p.venue, p.instrument_id)
        actual_qty[k] = actual_qty.get(k, 0.0) + float(p.quantity)   # SUM, never overwrite

    exp_held: Dict[tuple, float] = {k: float(v) for k, v in expected_qty.items()}

    breaks: List[PositionBreak] = []
    for k in set(exp_held) | set(actual_qty) | set(pending):
        held = float(exp_held.get(k, 0.0))
        pend = float(pending.get(k, 0.0))
        act = float(actual_qty.get(k, 0.0))
        # An in-flight working order makes any fill level in [held, held+pend] legitimate. Break
        # only when actual is OUTSIDE that closed band (± tol). pend==0 -> band collapses to {held}
        # -> exact point check (a genuine phantom/orphan still breaks).
        lo, hi = (held, held + pend) if pend >= 0 else (held + pend, held)
        if act < lo - qty_tol or act > hi + qty_tol:
            venue, iid = k
            # report the nearer band edge as "expected" so the break delta is the true shortfall
            edge = lo if act < lo else hi
            breaks.append(PositionBreak(instrument_id=iid, venue=venue,
                                        expected_qty=edge, actual_qty=act, delta=act - edge))

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


# HELD = positions the DB believes we currently hold. PENDING = in-flight working orders (sent,
# not yet confirmed filled). They are modelled SEPARATELY: held drives the exact match; pending
# widens the tolerated band to [held, held+pending] so a just-placed-but-unfilled order is NOT a
# false break (and a filled-but-not-yet-recorded order is still within the band).
HELD_STATUSES = ("ACTIVE",)
PENDING_STATUSES = ("PENDING_FILL",)


def _db_signed_by_status(db, statuses) -> Dict[tuple, float]:
    """Signed qty per (venue, instrument_id) for Trade rows in the given statuses. direction
    BUY -> +qty, SELL_SHORT -> -qty; summed per key (partial / cross-sleeve lots add). All current
    live positions are Alpaca-venue. Lazy import so this module stays import-light; equality-filter
    per status (no in_()) so the query stays scoped SQL-side (not a full-table load)."""
    from app.database.models import Trade
    out: Dict[tuple, float] = {}
    rows = []
    for status in statuses:
        rows.extend(db.query(Trade).filter_by(status=status).all())
    for t in rows:
        sym = (t.symbol or "").upper()
        if not sym:
            continue
        iid = im.lookup(im.ALPACA, sym) or sym
        qty = float(t.quantity or 0)
        signed = -qty if str(t.direction).upper() == "SELL_SHORT" else qty
        k = _ckey(im.ALPACA, iid)
        out[k] = out.get(k, 0.0) + signed
    return out


# ── live-path assembly + the shadow-first before-trade gate (Alpha-v10 H1) ─────────
def db_expected_positions(db) -> Dict[tuple, float]:
    """The DB's HELD book = ACTIVE Trade rows -> signed qty per (venue, instrument_id)."""
    return _db_signed_by_status(db, HELD_STATUSES)


def db_pending_positions(db) -> Dict[tuple, float]:
    """The DB's in-flight working orders = PENDING_FILL Trade rows -> signed qty per (venue,
    instrument_id). Passed as `pending_qty` so a just-placed-but-unfilled order is tolerated within
    the [held, held+pending] band instead of firing a false orphan/phantom break."""
    return _db_signed_by_status(db, PENDING_STATUSES)


def alpaca_actual_positions(raw_positions) -> List[CanonicalPosition]:
    """Broker truth -> canonical positions for the reconciler (only venue/instrument_id/quantity are
    used by reconcile()). `raw_positions` are the Alpaca client position dicts already fetched by the
    caller (no extra API call)."""
    out: List[CanonicalPosition] = []
    for p in raw_positions or []:
        sym = (p.get("symbol") or "").upper()
        if not sym:
            continue
        iid = im.lookup(im.ALPACA, sym) or sym
        try:
            qty = float(p.get("qty") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            price = float(p.get("current_price") or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        try:
            mv = float(p.get("market_value") or (qty * price))
        except (TypeError, ValueError):
            mv = qty * price
        out.append(CanonicalPosition(
            instrument_id=iid, venue=im.ALPACA, broker_symbol=sym, asset_class=im.EQUITY,
            quantity=qty, price=price, multiplier=1.0, currency="USD",
            market_value=mv, notional=abs(qty) * price, mapped=im.lookup(im.ALPACA, sym) is not None))
    return out


def shadow_reconcile_before_trade(db, raw_alpaca_positions, *, nav: Optional[float] = None,
                                  extra_actual: Optional[List[CanonicalPosition]] = None,
                                  expected: Optional[Dict[tuple, float]] = None,
                                  mode: str = SHADOW, label: str = "",
                                  notifier=None) -> ReconciliationResult:
    """FAIL-SAFE pre-trade reconciliation a sleeve calls BEFORE placing: builds DB-expected vs
    broker-actual (whole book; `extra_actual` lets a caller add other-venue positions, e.g. IBKR),
    reconciles, logs (+ emails on FAIL_CLOSED), returns the result. The CALLER acts on
    `.ok_to_trade` only in ENFORCE mode.

    `expected` overrides the DB-expected book (default: the whole Alpaca book via
    `db_expected_positions`). A single-venue caller (e.g. the IBKR futures executor) passes a
    venue-scoped expected map so it doesn't drag the *other* venue's book in as phantom breaks.

    NEVER raises into the rebalance. On an internal error it returns FAIL_CLOSED (+ an error note) —
    a reconciliation that cannot even run means "I can't confirm the broker is reality", which must
    HOLD in enforce (and is logged, harmlessly, in shadow). Cash is not yet modelled DB-side, so the
    position book is the v1 check (the #1 phantom-position risk); the result carries the cash note."""
    try:
        # Default whole-Alpaca-book path: split held (ACTIVE) vs in-flight (PENDING_FILL) so an
        # unfilled working order is tolerated, not a false break. An explicit `expected` (e.g. the
        # venue-scoped IBKR futures caller) supplies its own book and no DB pending.
        if expected is None:
            expected = db_expected_positions(db)
            pending = db_pending_positions(db)
        else:
            expected = dict(expected)
            pending = None
        actual = alpaca_actual_positions(raw_alpaca_positions)
        if extra_actual:
            actual = list(actual) + list(extra_actual)
        result = reconcile(expected, actual, nav=nav, pending_qty=pending)
        if not result.ok_to_trade:
            log.warning("[reconcile:%s mode=%s] WOULD-HOLD%s: status=%s breaks=%s %s",
                        label, mode, ("" if mode == ENFORCE else " (shadow: not blocking)"),
                        result.status,
                        [(b.venue, b.instrument_id, b.expected_qty, b.actual_qty)
                         for b in result.position_breaks],
                        "; ".join(result.notes))
            if notifier is not None:
                try:
                    notifier.enqueue("reconciliation_break", dedup_key=f"reconciliation_break:{label}:{result.status}", payload={
                        "label": label, "mode": mode, "status": result.status,
                        "position_breaks": [
                            {"venue": b.venue, "instrument_id": b.instrument_id,
                             "expected_qty": b.expected_qty, "actual_qty": b.actual_qty}
                            for b in result.position_breaks],
                        "cash_break": result.cash_break, "notes": result.notes})
                except Exception:
                    log.debug("reconcile: notify failed", exc_info=True)
        else:
            log.info("[reconcile:%s mode=%s] OK: %s", label, mode, "; ".join(result.notes))
        return result
    except Exception as exc:  # noqa: BLE001 — must never break a live rebalance; fail-CLOSED in enforce
        log.warning("[reconcile:%s] error -> FAIL_CLOSED (fail-safe): %s", label, exc, exc_info=True)
        return ReconciliationResult(status=FAIL_CLOSED, notes=[f"reconciliation error: {exc}"])
