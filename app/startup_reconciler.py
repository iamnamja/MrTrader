"""
Startup reconciler — detects and logs inconsistencies between the DB and
Alpaca on every application startup.

Two classes of problem:
  1. Ghost positions  — in DB (status=ACTIVE) but NOT in Alpaca
  2. Orphaned orders  — in Alpaca (open) but NOT in the DB orders table

The reconciler NEVER modifies Alpaca state; it only updates DB records and
writes AuditLog entries so humans can review.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def reconcile(alpaca, db_session) -> Dict[str, Any]:
    """
    Run reconciliation.  Returns a summary dict.

    Args:
        alpaca:      AlpacaClient instance
        db_session:  SQLAlchemy session

    Returns:
        {"ghost_positions": [...], "orphaned_orders": [...]}
    """
    from app.database.models import Trade, Order, AuditLog

    result: Dict[str, Any] = {"ghost_positions": [], "orphaned_orders": [], "untracked_positions": []}

    # ── 1. Ghost positions ─────────────────────────────────────────────────────
    try:
        active_trades = db_session.query(Trade).filter_by(status="ACTIVE").all()
        alpaca_positions: Dict[str, Any] = {
            p["symbol"]: p for p in alpaca.get_positions()
        }

        for trade in active_trades:
            if trade.symbol not in alpaca_positions:
                logger.warning(
                    "GHOST POSITION: Trade#%d %s is ACTIVE in DB but not in Alpaca",
                    trade.id, trade.symbol,
                )
                result["ghost_positions"].append({
                    "trade_id": trade.id,
                    "symbol":   trade.symbol,
                    "entry_price": trade.entry_price,
                    "quantity":    trade.quantity,
                })
                # Mark as reconciled (not closed — human should verify)
                trade.status = "RECONCILE_GHOST"
                db_session.add(AuditLog(
                    action="RECONCILE_GHOST_POSITION",
                    details={
                        "trade_id":    trade.id,
                        "symbol":      trade.symbol,
                        "reason":      "Active in DB but no matching Alpaca position on startup",
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))
    except Exception as exc:
        logger.error("Ghost position check failed: %s", exc)

    # ── 2. Untracked Alpaca positions (in Alpaca but no DB Trade record) ─────────
    # Happens when a limit order fills after a uvicorn restart wipes _pending_limit_orders.
    try:
        live_statuses = ("ACTIVE", "RECONCILE_GHOST")
        active_db_symbols = {
            t.symbol for t in db_session.query(Trade)
            .filter(Trade.status.in_(live_statuses)).all()
        }
        for symbol, pos in alpaca_positions.items():
            if symbol not in active_db_symbols:
                qty = int(float(pos.get("qty") or pos.get("quantity") or 0))
                avg = float(pos.get("avg_entry_price") or pos.get("avg_price") or 0)
                logger.warning(
                    "UNTRACKED POSITION: %s x%d @ $%.2f is in Alpaca but has no ACTIVE DB Trade",
                    symbol, qty, avg,
                )
                result["untracked_positions"].append({"symbol": symbol, "qty": qty, "avg_price": avg})
                # Create a placeholder Trade with default stop/target (2% stop, 6% target)
                # Trader will refine these via generate_signal() on next reconcile cycle
                stop_price = round(avg * 0.98, 2) if avg > 0 else None
                target_price = round(avg * 1.06, 2) if avg > 0 else None
                placeholder = Trade(
                    symbol=symbol,
                    direction="BUY",
                    entry_price=avg,
                    quantity=qty,
                    status="ACTIVE",
                    signal_type="RECONCILED",
                    trade_type="swing",
                    stop_price=stop_price,
                    target_price=target_price,
                )
                db_session.add(placeholder)
                db_session.add(AuditLog(
                    action="RECONCILE_UNTRACKED_POSITION",
                    details={
                        "symbol": symbol, "qty": qty, "avg_price": avg,
                        "reason": "Alpaca position with no DB Trade — likely a limit order filled after restart",
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))
    except Exception as exc:
        logger.error("Untracked position check failed: %s", exc)

    # ── 3. Orphaned Alpaca orders ──────────────────────────────────────────────
    try:
        # Get all open orders from Alpaca
        open_alpaca_orders = _get_open_alpaca_orders(alpaca)
        # Get all order IDs in DB
        db_order_ids = {
            row.order_id
            for row in db_session.query(Order.order_id).all()
            if row.order_id
        }

        for ao in open_alpaca_orders:
            if ao["order_id"] not in db_order_ids:
                logger.warning(
                    "ORPHANED ORDER: Alpaca order %s (%s %s x%s) not in DB",
                    ao["order_id"], ao["side"], ao["symbol"], ao["qty"],
                )
                result["orphaned_orders"].append(ao)
                db_session.add(AuditLog(
                    action="RECONCILE_ORPHANED_ORDER",
                    details={
                        **ao,
                        "reason":      "Open order in Alpaca but not in DB on startup",
                        "detected_at": datetime.utcnow().isoformat(),
                    },
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
    n_orphans = len(result["orphaned_orders"])
    n_untracked = len(result["untracked_positions"])
    if n_ghosts or n_orphans or n_untracked:
        logger.warning(
            "Startup reconciliation: %d ghost position(s), %d orphaned order(s), %d untracked position(s)",
            n_ghosts, n_orphans, n_untracked,
        )
    else:
        logger.info("Startup reconciliation: clean — no issues found")

    return result


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
