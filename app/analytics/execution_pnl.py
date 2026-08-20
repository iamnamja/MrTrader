"""Per-execution realized P&L for the order blotter (2026-08-20).

The Executions tab shows actual broker orders, including the weekly rebalance resizings that never
create a DB Trade row — so there was nowhere to read "what did this fill actually make?". A fill on
a scaled position has no P&L of its own until you know the basis it was sold against, which is only
recoverable by replaying the symbol's whole fill history.

METHOD — FIFO lot matching
--------------------------
Each buy pushes a lot; each sell consumes the oldest lots first, realizing
``qty * (fill_price - lot_price)`` per lot consumed. Shorts are symmetric: a sell with no open
position opens a short lot and the covering buy realizes ``qty * (lot_price - fill_price)``. A fill
that exceeds the open position closes it and opens the remainder on the other side.

FIFO was chosen over moving-average cost after measuring both against the live account:

  * FIFO reproduces Alpaca's own ``avg_entry_price`` EXACTLY (0.0000 across all 8 open positions);
    moving-average does not. So this tab agrees with the Positions tab.
  * Only FIFO's realized total reconciles to account equity. Both conventions satisfy
    ``realized = (sells - buys) + remaining_basis``, but with the same fills that gives
    FIFO $978.01 vs moving-average $908.67, and the account's own
    ``equity - deposits - unrealized - fees`` implies $977.93 (an 8-cent rounding gap over 310
    fills). The $69 difference is purely where each convention draws the realized/unrealized line.
  * FIFO is also the US tax-reporting standard, so the numbers mean something outside this UI.

WHY IT CAN BE TRUSTED — the self-check
--------------------------------------
Replaying every fill must land on the position the broker actually reports. Any symbol whose replay
disagrees has history we cannot see, and every fill for it is returned with ``realized_pnl=None``
and ``pnl_basis="incomplete_history"`` rather than a plausible-looking wrong number.

Two traps this survived, both found against live data:

  * ``MP`` had ``buy qty 152 -> filled 117, status=canceled`` — a partially-filled CANCELED order
    that genuinely moved the position by 117 shares. Selecting fills by ``status == "filled"``
    silently drops it; the correct predicate is ``filled_qty > 0``, which `iter_fills` enforces.
  * A FIFO replay that ignores shorts silently strands lots and overstates realized P&L (it read
    $1,009.85 instead of $978.01). Short handling is not optional here — the book has a
    ``quality_short`` selector in its history.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

QTY_TOL = 1e-9

# Per-fill P&L provenance, so the UI can distinguish "nothing to realize" from "cannot know".
BASIS_REALIZED = "realized"              # a closing fill matched against known lots
BASIS_OPENING = "opening"                # opening/adding — realizes nothing by definition
BASIS_INCOMPLETE = "incomplete_history"  # replay disagreed with the broker: basis unknowable


def iter_fills(orders: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Orders that actually moved the position, oldest-first.

    Selects on ``filled_qty > 0`` rather than ``status == 'filled'``: a partially-filled order that
    was later CANCELED still traded its filled portion, and dropping it corrupts every subsequent
    cost basis for that symbol.
    """
    fills = [
        o for o in orders
        if float(o.get("filled_qty") or 0) > 0 and o.get("filled_avg_price") is not None
    ]
    # filled_at is the true execution time; fall back to submitted_at if the broker omitted it.
    return sorted(fills, key=lambda o: (o.get("filled_at") or o.get("submitted_at") or ""))


def _replay_symbol(fills: List[Dict[str, Any]]) -> tuple[Dict[str, Optional[float]], float]:
    """FIFO replay for ONE symbol's fills (already oldest-first).

    Returns ``(realized_by_order_id, final_position)``. `final_position` is what the caller checks
    against broker truth before trusting any of the numbers. Lots are signed and always share the
    sign of the open position: positive = long, negative = short.
    """
    realized: Dict[str, Optional[float]] = {}
    lots: Deque[List[float]] = deque()      # [signed_qty, price], oldest first

    for f in fills:
        qty = float(f["filled_qty"])
        price = float(f["filled_avg_price"])
        signed = qty if f.get("side") == "buy" else -qty
        oid = str(f.get("order_id") or id(f))

        pnl = 0.0
        matched = False
        remaining = signed

        # Consume opposing lots oldest-first while this fill still has size and the front lot
        # points the other way.
        while abs(remaining) > QTY_TOL and lots and (lots[0][0] > 0) != (remaining > 0):
            lot_qty, lot_price = lots[0]
            take = min(abs(remaining), abs(lot_qty))
            # Long lot closed by a sell: gain above basis. Short lot closed by a buy: gain below.
            pnl += take * (price - lot_price) if lot_qty > 0 else take * (lot_price - price)
            matched = True

            lot_qty -= take if lot_qty > 0 else -take
            remaining += take if remaining < 0 else -take
            if abs(lot_qty) <= QTY_TOL:
                lots.popleft()
            else:
                lots[0][0] = lot_qty

        # Anything left opens (or extends) a position on this fill's side.
        if abs(remaining) > QTY_TOL:
            lots.append([remaining, price])

        realized[oid] = pnl if matched else None

    return realized, sum(lot[0] for lot in lots)


def compute_realized_pnl(
    orders: Iterable[Dict[str, Any]],
    broker_positions: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Realized P&L per order id, keyed by order_id.

    `orders` must be the FULL retrievable history — passing only a recent page produces a wrong
    basis for any symbol whose opening buys fall outside it. `broker_positions` maps symbol -> live
    qty; when supplied, any symbol whose replay disagrees is reported as `incomplete_history`
    instead of guessing.
    """
    fills = iter_fills(orders)
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for f in fills:
        by_symbol.setdefault(f["symbol"], []).append(f)

    out: Dict[str, Dict[str, Any]] = {}
    for symbol, sym_fills in by_symbol.items():
        realized, final_pos = _replay_symbol(sym_fills)

        trustworthy = True
        if broker_positions is not None:
            expected = float(broker_positions.get(symbol, 0.0))
            if abs(final_pos - expected) > 1e-6:
                trustworthy = False
                log.info(
                    "execution_pnl: %s replay %.6f != broker %.6f — reporting incomplete_history",
                    symbol, final_pos, expected,
                )

        for f in sym_fills:
            oid = str(f.get("order_id") or id(f))
            if not trustworthy:
                out[oid] = {"realized_pnl": None, "realized_pct": None,
                            "pnl_basis": BASIS_INCOMPLETE}
                continue
            pnl = realized.get(oid)
            if pnl is None:
                out[oid] = {"realized_pnl": None, "realized_pct": None,
                            "pnl_basis": BASIS_OPENING}
            else:
                # % return on the capital that fill released, so small and large exits compare.
                proceeds = float(f["filled_qty"]) * float(f["filled_avg_price"])
                basis_amt = proceeds - pnl
                pct = (pnl / basis_amt * 100.0) if abs(basis_amt) > 1e-9 else None
                out[oid] = {"realized_pnl": pnl, "realized_pct": pct,
                            "pnl_basis": BASIS_REALIZED}
    return out


def attach_realized_pnl(
    page: List[Dict[str, Any]],
    full_history: Iterable[Dict[str, Any]],
    broker_positions: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Attach realized P&L to the orders being displayed, computing basis from `full_history`."""
    pnl = compute_realized_pnl(full_history, broker_positions)
    for o in page:
        info = pnl.get(str(o.get("order_id")))
        if info is None:
            info = {"realized_pnl": None, "realized_pct": None, "pnl_basis": BASIS_OPENING}
        o["realized_pnl"] = info["realized_pnl"]
        o["realized_pct"] = info["realized_pct"]
        o["pnl_basis"] = info["pnl_basis"]
    return page
