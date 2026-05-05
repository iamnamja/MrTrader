"""
Startup reconciler — detects and heals inconsistencies between the DB and
Alpaca on every application startup.

Three classes of problem:
  1. Ghost positions   — ACTIVE in DB but NOT in Alpaca (position closed externally)
  2. Pending fills     — PENDING_FILL in DB; check Alpaca order status and promote/cancel
  3. Untracked         — in Alpaca but no DB Trade at all (limit filled during restart)

The reconciler NEVER modifies Alpaca state; it only updates DB records and
writes AuditLog entries so humans can review.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Minimum number of Alpaca positions required before we trust a "ghost" detection.
# If Alpaca returns fewer positions than this threshold AND we have many ACTIVE DB
# trades, it's likely an API error rather than genuine position closure — skip ghost
# marking to avoid false positives that generate duplicate records on next restart.
_MIN_POSITIONS_TO_TRUST_GHOST = 1


def reconcile(alpaca, db_session) -> Dict[str, Any]:
    """
    Run reconciliation. Returns a summary dict.

    Args:
        alpaca:      AlpacaClient instance
        db_session:  SQLAlchemy session
    """
    from app.database.models import Trade, Order, AuditLog

    result: Dict[str, Any] = {
        "ghost_positions": [],
        "pending_fills_resolved": [],
        "untracked_positions": [],
        "orphaned_orders": [],
    }

    # Fetch Alpaca state once — used by all checks below
    try:
        alpaca_positions: Dict[str, Any] = {p["symbol"]: p for p in alpaca.get_positions()}
    except Exception as exc:
        logger.error("Could not fetch Alpaca positions — reconciliation skipped: %s", exc)
        return result

    active_trades = db_session.query(Trade).filter_by(status="ACTIVE").all()
    n_active_db = len(active_trades)
    n_alpaca_positions = len(alpaca_positions)

    # ── Fix 2: Guard against empty/truncated Alpaca response ─────────────────────
    # If Alpaca returns 0 positions but we have ACTIVE trades in DB, it's almost
    # certainly an API error. Marking everything as ghost would cause a flood of
    # duplicate synthetic records on the next restart. Skip ghost detection entirely.
    api_trustworthy = not (n_alpaca_positions == 0 and n_active_db > 0)
    if not api_trustworthy:
        logger.warning(
            "Reconciliation: Alpaca returned 0 positions but DB has %d ACTIVE trade(s) — "
            "skipping ghost detection to avoid false positives (likely API error)",
            n_active_db,
        )

    # ── 1. Ghost positions ─────────────────────────────────────────────────────
    if api_trustworthy:
        try:
            for trade in active_trades:
                if trade.symbol not in alpaca_positions:
                    logger.warning(
                        "GHOST POSITION: Trade#%d %s is ACTIVE in DB but not in Alpaca",
                        trade.id, trade.symbol,
                    )
                    result["ghost_positions"].append({
                        "trade_id": trade.id, "symbol": trade.symbol,
                        "entry_price": trade.entry_price, "quantity": trade.quantity,
                    })
                    trade.status = "RECONCILE_GHOST"
                    db_session.add(AuditLog(
                        action="RECONCILE_GHOST_POSITION",
                        details={
                            "trade_id": trade.id, "symbol": trade.symbol,
                            "reason": "Active in DB but no matching Alpaca position on startup",
                            "detected_at": datetime.utcnow().isoformat(),
                        },
                        timestamp=datetime.utcnow(),
                    ))

            # Close stale RECONCILE_GHOST records for symbols no longer in Alpaca.
            # Without this, old ghost rows accumulate across restarts and block the
            # "untracked" check from creating fresh ACTIVE records on later restarts.
            stale_ghosts = db_session.query(Trade).filter_by(status="RECONCILE_GHOST").all()
            for ghost in stale_ghosts:
                if ghost.symbol not in alpaca_positions:
                    exit_price, exit_order_id = _lookup_sell_fill(alpaca, ghost.symbol, ghost.quantity)
                    ghost.status = "CLOSED"
                    ghost.exit_reason = "reconcile_ghost_expired"
                    ghost.closed_at = datetime.utcnow()
                    if exit_price is not None:
                        ghost.exit_price = exit_price
                        ghost.pnl = round((exit_price - float(ghost.entry_price or 0)) * (ghost.quantity or 0), 2)
                        logger.info(
                            "Closing stale RECONCILE_GHOST Trade#%d %s — exit=$%.4f pnl=$%.2f (order %s)",
                            ghost.id, ghost.symbol, exit_price, ghost.pnl or 0, exit_order_id or "?",
                        )
                    else:
                        logger.info(
                            "Closing stale RECONCILE_GHOST Trade#%d %s (no Alpaca sell found)",
                            ghost.id, ghost.symbol,
                        )
        except Exception as exc:
            logger.error("Ghost position check failed: %s", exc)

    # ── 2. Pending fills — resolve via Alpaca order status ────────────────────
    # PENDING_FILL trades have an alpaca_order_id; check whether they filled,
    # were cancelled, or are still open (leave those — Trader will poll them).
    # Also check CANCELLED trades with real Alpaca order IDs — the order may have
    # filled before the DB was updated (restart race condition).
    try:
        cancelled_with_id = [
            t for t in db_session.query(Trade).filter_by(status="CANCELLED").all()
            if t.alpaca_order_id and not t.alpaca_order_id.startswith("order-")
        ]
        for trade in cancelled_with_id:
            try:
                order_status = alpaca.get_order_status(trade.alpaca_order_id)
            except Exception:
                continue
            if order_status is None:
                continue
            status_str = str(order_status.get("status", "")).lower()
            filled_qty = int(order_status.get("filled_qty") or 0)
            filled_price = order_status.get("filled_avg_price")
            if status_str in ("filled", "partially_filled") and filled_qty > 0 and filled_price:
                # The order actually filled — this is a cancel/fill mismatch
                # Mark it as superseded; the position is tracked elsewhere (RECONCILED row)
                logger.warning(
                    "CANCEL/FILL MISMATCH: Trade#%d %s marked CANCELLED but Alpaca shows FILLED "
                    "@ $%s — marking RECONCILE_SUPERSEDED",
                    trade.id, trade.symbol, filled_price,
                )
                trade.status = "RECONCILE_SUPERSEDED"
                trade.status_reason = "Order filled on Alpaca but DB marked CANCELLED; position tracked separately"
                db_session.add(AuditLog(
                    action="RECONCILE_CANCEL_FILL_MISMATCH",
                    details={
                        "trade_id": trade.id, "symbol": trade.symbol,
                        "alpaca_order_id": trade.alpaca_order_id,
                        "filled_price": float(filled_price), "filled_qty": filled_qty,
                        "reason": "CANCELLED in DB but FILLED on Alpaca — restart race condition",
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))
        pending_trades = db_session.query(Trade).filter_by(status="PENDING_FILL").all()
        for trade in pending_trades:
            if not trade.alpaca_order_id:
                # No order ID recorded — order placement may have failed; cancel it
                logger.warning(
                    "PENDING_FILL Trade#%d %s has no alpaca_order_id — marking CANCELLED",
                    trade.id, trade.symbol,
                )
                trade.status = "CANCELLED"
                continue

            try:
                order_status = alpaca.get_order_status(trade.alpaca_order_id)
            except Exception as exc:
                logger.warning("Could not fetch order status for Trade#%d %s: %s",
                               trade.id, trade.symbol, exc)
                continue

            if order_status is None:
                continue

            status_str = str(order_status.get("status", "")).lower()
            filled_qty = int(order_status.get("filled_qty") or 0)
            filled_price = order_status.get("filled_avg_price")

            if status_str in ("filled", "partially_filled") and filled_qty > 0 and filled_price:
                filled_price = float(filled_price)
                trade.status = "ACTIVE"
                trade.entry_price = filled_price
                trade.quantity = filled_qty
                trade.highest_price = filled_price
                # Record the fill in the Order table
                db_order = Order(
                    trade_id=trade.id,
                    order_type="ENTRY",
                    order_id=trade.alpaca_order_id,
                    status="FILLED",
                    filled_price=filled_price,
                    filled_qty=filled_qty,
                    intended_price=trade.entry_price,
                    slippage_bps=round(
                        (filled_price - trade.entry_price) / trade.entry_price * 10000, 2
                    ) if trade.entry_price > 0 else 0.0,
                )
                db_session.add(db_order)
                logger.info(
                    "PENDING_FILL resolved: Trade#%d %s filled x%d @ $%.4f",
                    trade.id, trade.symbol, filled_qty, filled_price,
                )
                result["pending_fills_resolved"].append({
                    "trade_id": trade.id, "symbol": trade.symbol,
                    "filled_qty": filled_qty, "filled_price": filled_price,
                })
                db_session.add(AuditLog(
                    action="RECONCILE_PENDING_FILL_RESOLVED",
                    details={
                        "trade_id": trade.id, "symbol": trade.symbol,
                        "filled_qty": filled_qty, "filled_price": filled_price,
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))

            elif status_str in ("canceled", "expired", "rejected"):
                trade.status = "CANCELLED"
                logger.info(
                    "PENDING_FILL Trade#%d %s order %s — marking CANCELLED",
                    trade.id, trade.symbol, status_str,
                )

            # else: still open — leave as PENDING_FILL; Trader will resume polling

    except Exception as exc:
        logger.error("Pending fill resolution failed: %s", exc)

    # ── 3. Untracked Alpaca positions ─────────────────────────────────────────
    # In Alpaca but no ACTIVE/PENDING_FILL DB record.
    # Fix 3: Before creating a synthetic record, check if a recent trade already
    # exists for this symbol (within 7 days). If so, reuse or skip — prevents
    # duplicate records when the same position gets reconciled across multiple restarts.
    try:
        live_statuses = ("ACTIVE", "PENDING_FILL")
        tracked_symbols = {
            t.symbol for t in db_session.query(Trade)
            .filter(Trade.status.in_(live_statuses)).all()
        }
        lookback_cutoff = datetime.utcnow() - timedelta(days=7)

        for symbol, pos in alpaca_positions.items():
            if symbol in tracked_symbols:
                continue

            qty = int(float(pos.get("qty") or pos.get("quantity") or 0))
            avg = float(pos.get("avg_entry_price") or pos.get("avg_price") or 0)

            # Fix 3: Check for a recent trade record for this symbol.
            # If one exists within the last 7 days that is CLOSED/CANCELLED/RECONCILE_GHOST,
            # it means the position was recently tracked and then closed — the closure may
            # have happened externally (stop hit, EOD cancel) without the DB being updated.
            # Reactivate the most recent record instead of creating a new duplicate.
            recent_trade = (
                db_session.query(Trade)
                .filter(
                    Trade.symbol == symbol,
                    Trade.created_at >= lookback_cutoff,
                    Trade.status.in_(("CLOSED", "CANCELLED", "RECONCILE_GHOST")),
                )
                .order_by(Trade.id.desc())
                .first()
            )

            if recent_trade and abs(float(recent_trade.entry_price or 0) - avg) / max(avg, 1) < 0.05:
                # Entry prices within 5% — same position, reactivate
                logger.warning(
                    "UNTRACKED POSITION: %s x%d @ $%.2f — reactivating Trade#%d (was %s) "
                    "instead of creating duplicate",
                    symbol, qty, avg, recent_trade.id, recent_trade.status,
                )
                recent_trade.status = "ACTIVE"
                recent_trade.quantity = qty
                recent_trade.exit_price = None
                recent_trade.pnl = None
                recent_trade.closed_at = None
                recent_trade.exit_reason = None
                recent_trade.highest_price = avg
                result["untracked_positions"].append({
                    "symbol": symbol, "qty": qty, "avg_price": avg,
                    "action": "reactivated", "trade_id": recent_trade.id,
                })
                db_session.add(AuditLog(
                    action="RECONCILE_REACTIVATED_TRADE",
                    details={
                        "trade_id": recent_trade.id, "symbol": symbol,
                        "qty": qty, "avg_price": avg,
                        "reason": "Reactivated existing trade instead of creating duplicate",
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))
            else:
                # Genuinely new/untracked — create a fresh synthetic record
                logger.warning(
                    "UNTRACKED POSITION: %s x%d @ $%.2f — creating RECONCILED trade",
                    symbol, qty, avg,
                )
                result["untracked_positions"].append({
                    "symbol": symbol, "qty": qty, "avg_price": avg, "action": "created",
                })
                stop_price = round(avg * 0.98, 2) if avg > 0 else None
                target_price = round(avg * 1.06, 2) if avg > 0 else None
                placeholder = Trade(
                    symbol=symbol, direction="BUY", entry_price=avg,
                    quantity=qty, status="ACTIVE", signal_type="RECONCILED",
                    trade_type="swing", stop_price=stop_price, target_price=target_price,
                )
                db_session.add(placeholder)
                db_session.add(AuditLog(
                    action="RECONCILE_UNTRACKED_POSITION",
                    details={
                        "symbol": symbol, "qty": qty, "avg_price": avg,
                        "reason": "Alpaca position with no DB Trade — likely filled during restart",
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))
    except Exception as exc:
        logger.error("Untracked position check failed: %s", exc)

    # ── 4. Orphaned open Alpaca orders ────────────────────────────────────────
    # Open orders in Alpaca with no DB record at all — log only (Trader will
    # pick them up via _pending_limit_orders once restarted, or they'll expire).
    try:
        open_alpaca_orders = _get_open_alpaca_orders(alpaca)
        from app.database.models import Order as OrderModel
        db_order_ids = {
            row.order_id for row in db_session.query(OrderModel.order_id).all()
            if row.order_id
        }
        pending_fill_order_ids = {
            t.alpaca_order_id for t in db_session.query(Trade)
            .filter(Trade.status == "PENDING_FILL").all()
            if t.alpaca_order_id
        }
        for ao in open_alpaca_orders:
            oid = ao["order_id"]
            if oid not in db_order_ids and oid not in pending_fill_order_ids:
                logger.warning(
                    "ORPHANED ORDER: Alpaca order %s (%s %s x%s) not in DB",
                    oid, ao["side"], ao["symbol"], ao["qty"],
                )
                result["orphaned_orders"].append(ao)
                db_session.add(AuditLog(
                    action="RECONCILE_ORPHANED_ORDER",
                    details={**ao, "reason": "Open order in Alpaca but not in DB on startup",
                             "detected_at": datetime.utcnow().isoformat()},
                    timestamp=datetime.utcnow(),
                ))
    except Exception as exc:
        logger.error("Orphaned order check failed: %s", exc)

    try:
        db_session.commit()
    except Exception as exc:
        logger.error("Reconciler commit failed: %s", exc)
        db_session.rollback()

    n_ghosts = len(result["ghost_positions"])
    n_resolved = len(result["pending_fills_resolved"])
    n_untracked = len(result["untracked_positions"])
    n_orphans = len(result["orphaned_orders"])

    if any([n_ghosts, n_resolved, n_untracked, n_orphans]):
        logger.warning(
            "Startup reconciliation: %d ghost(s), %d pending fills resolved, "
            "%d untracked position(s), %d orphaned order(s)",
            n_ghosts, n_resolved, n_untracked, n_orphans,
        )
    else:
        logger.info("Startup reconciliation: clean — no issues found")

    return result


def _lookup_sell_fill(alpaca, symbol: str, qty) -> tuple:
    """Return (filled_avg_price, order_id) for the most recent SELL fill for symbol, or (None, None)."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100)
        orders = alpaca.trading_client.get_orders(req)
        target_qty = int(qty or 0)
        for o in orders:
            if o.symbol != symbol:
                continue
            if "SELL" not in str(o.side).upper():
                continue
            if "FILLED" not in str(o.status).upper():
                continue
            filled_qty = int(o.filled_qty or 0)
            if target_qty > 0 and abs(filled_qty - target_qty) > max(5, target_qty * 0.2):
                continue
            if o.filled_avg_price:
                return float(o.filled_avg_price), str(o.id)
    except Exception as exc:
        logger.debug("_lookup_sell_fill failed for %s: %s", symbol, exc)
    return None, None


def _get_open_alpaca_orders(alpaca) -> List[Dict[str, Any]]:
    """Return list of open (pending/new) orders from Alpaca."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = alpaca.trading_client.get_orders(req)
        return [
            {
                "order_id": str(o.id),
                "symbol":   o.symbol,
                "qty":      str(o.qty),
                "side":     str(o.side),
                "status":   str(o.status),
            }
            for o in orders
        ]
    except Exception as exc:
        logger.error("Could not fetch open Alpaca orders: %s", exc)
        return []
