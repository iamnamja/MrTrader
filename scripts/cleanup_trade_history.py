"""
Phase A: One-time trade history cleanup script.

Fixes three categories of dirty data:
  1. TSLA dev/test ghost trades — fake order IDs, never real positions
  2. Superseded RECONCILED tracks — same position re-tracked under a new ML_RANK row;
     the old RECONCILED row has no real exit and should be annotated as superseded
  3. Cancelled/fill mismatches — trades marked CANCELLED whose Alpaca orders actually
     filled; annotate with the superseding RECONCILED row
  4. Back-fill exit_price and PnL for CLOSED trades that have neither

Run once:
    python scripts/cleanup_trade_history.py

Dry-run (prints changes, touches nothing):
    python scripts/cleanup_trade_history.py --dry-run
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
from typing import Optional

# Make sure app is importable from project root
sys.path.insert(0, ".")

from app.database.session import get_session
from app.database.models import Trade, Order, AuditLog
from app.integrations.alpaca import get_alpaca_client
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def build_alpaca_order_map(client) -> dict:
    """Pull last 200 Alpaca orders and index by order_id and symbol+side."""
    all_orders = {}
    fills_by_symbol: dict[str, list] = {}
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=200)
        orders = client.trading_client.get_orders(req)
        for o in orders:
            all_orders[str(o.id)] = o
            sym = o.symbol
            side = str(o.side)
            if sym not in fills_by_symbol:
                fills_by_symbol[sym] = []
            fills_by_symbol[sym].append(o)
    except Exception as e:
        log(f"WARNING: Could not fetch Alpaca orders: {e}")
    return all_orders, fills_by_symbol


def find_sell_order(fills_by_symbol: dict, symbol: str, approx_qty: int):
    """Find a SELL fill for a symbol near a given qty."""
    for o in fills_by_symbol.get(symbol, []):
        side_str = str(o.side).upper()
        status_str = str(o.status).upper()
        is_sell = "SELL" in side_str
        is_filled = "FILLED" in status_str
        if is_sell and is_filled:
            filled_qty = int(o.filled_qty or 0)
            if abs(filled_qty - approx_qty) <= max(5, approx_qty * 0.2):
                return o
    return None


def run(dry_run: bool = False) -> None:
    db = get_session()
    client = get_alpaca_client()
    now_utc = datetime.utcnow()

    log(f"Starting trade history cleanup (dry_run={dry_run})")
    alpaca_order_map, fills_by_symbol = build_alpaca_order_map(client)
    log(f"Loaded {len(alpaca_order_map)} Alpaca orders")

    changes: list[str] = []

    # ── 1. TSLA dev/test ghost trades ─────────────────────────────────────────
    # IDs 49-54: fake alpaca_order_ids starting with "order-", entry $200/$110
    TSLA_TEST_IDS = [49, 50, 51, 52, 53, 54]
    log(f"\n== Group 1: TSLA test trades (ids {TSLA_TEST_IDS}) ==")
    for tid in TSLA_TEST_IDS:
        t = db.query(Trade).filter(Trade.id == tid).first()
        if t is None:
            log(f"  id={tid}: not found (already deleted?)")
            continue
        alpaca_id = t.alpaca_order_id or ""
        if not alpaca_id.startswith("order-") and alpaca_id in alpaca_order_map:
            log(f"  id={tid} {t.symbol}: alpaca_order_id is VALID — skipping deletion for safety")
            continue
        child_orders = db.query(Order).filter(Order.trade_id == tid).count()
        log(f"  DELETING id={tid} {t.symbol} status={t.status} entry=${t.entry_price} "
            f"alpaca_id={alpaca_id} ({child_orders} child orders)")
        changes.append(f"DELETE trade id={tid} {t.symbol} (dev/test ghost)")
        if not dry_run:
            db.add(AuditLog(
                action="CLEANUP_DELETE_TEST_TRADE",
                details={
                    "trade_id": tid, "symbol": t.symbol,
                    "reason": "Dev/test ghost trade — fake alpaca_order_id",
                    "entry_price": t.entry_price,
                    "alpaca_order_id": alpaca_id,
                    "cleaned_at": now_utc.isoformat(),
                },
                timestamp=now_utc,
            ))
            # Delete child Order rows first (NOT NULL FK)
            db.query(Order).filter(Order.trade_id == tid).delete()
            db.delete(t)

    # ── 2. Superseded RECONCILED tracks ───────────────────────────────────────
    # These CLOSED RECONCILED rows represent the same position that was later
    # re-tracked under an ML_RANK row. The "close" was a re-tracking event,
    # not an actual sale. We annotate them so they're not mistaken for real exits.
    #
    # Pattern: CLOSED RECONCILED row with no exit_price, for a symbol that has
    # a newer ML_RANK row covering the same position period.
    SUPERSEDED_IDS = {
        42: "JBLU id=56 ML_RANK ACTIVE",   # JBLU re-tracked May 3
        44: "ZS id=60 ML_RANK ACTIVE",     # ZS re-tracked May 3
        45: "TNDM id=58 ML_RANK CLOSED",   # TNDM re-tracked May 3
        46: "ASH id=55 ML_RANK ACTIVE",    # ASH re-tracked May 3
        47: "VFC id=59 ML_RANK ACTIVE",    # VFC re-tracked May 3
        48: "MP id=57 ML_RANK CLOSED",     # MP re-tracked May 3
    }
    log(f"\n== Group 2: Superseded RECONCILED tracks ==")
    for tid, successor in SUPERSEDED_IDS.items():
        t = db.query(Trade).filter(Trade.id == tid).first()
        if t is None:
            log(f"  id={tid}: not found")
            continue
        if t.exit_price is not None:
            log(f"  id={tid} {t.symbol}: has exit_price=${t.exit_price} — already clean, skipping")
            continue
        log(f"  ANNOTATING id={tid} {t.symbol} status={t.status} -> superseded by {successor}")
        changes.append(f"ANNOTATE trade id={tid} {t.symbol} as RECONCILE_SUPERSEDED")
        if not dry_run:
            t.status = "RECONCILE_SUPERSEDED"
            t.pnl = 0.0
            db.add(AuditLog(
                action="CLEANUP_SUPERSEDED_RETRACK",
                details={
                    "trade_id": tid, "symbol": t.symbol,
                    "reason": "RECONCILED track superseded by re-tracking under ML_RANK row",
                    "successor": successor,
                    "cleaned_at": now_utc.isoformat(),
                },
                timestamp=now_utc,
            ))

    # ── 3. CANCELLED trades whose Alpaca orders actually filled ───────────────
    # These are orphaned limit-order entries: the order filled on Alpaca but a
    # restart prevented the DB from being updated. The position is now tracked
    # under a RECONCILED row. Annotate the CANCELLED row as superseded.
    CANCEL_FILL_MISMATCHES = {
        61: ("LAC",  67, "RECONCILED ACTIVE"),
        62: ("BAX",  66, "RECONCILED ACTIVE"),
        63: ("RHI",  68, "RECONCILED ACTIVE"),
        64: ("ALB",  65, "RECONCILED CLOSED"),
    }
    log(f"\n== Group 3: Cancelled/fill mismatches ==")
    for tid, (symbol, successor_id, successor_desc) in CANCEL_FILL_MISMATCHES.items():
        t = db.query(Trade).filter(Trade.id == tid).first()
        if t is None:
            log(f"  id={tid}: not found")
            continue
        alpaca_id = t.alpaca_order_id or ""
        alpaca_order = alpaca_order_map.get(alpaca_id)
        alpaca_status = str(alpaca_order.status).lower() if alpaca_order else "not_found"
        log(f"  ANNOTATING id={tid} {symbol} db=CANCELLED alpaca={alpaca_status} "
            f"-> superseded by id={successor_id} {successor_desc}")
        changes.append(f"ANNOTATE trade id={tid} {symbol} as superseded by id={successor_id}")
        if not dry_run:
            t.status = "RECONCILE_SUPERSEDED"
            db.add(AuditLog(
                action="CLEANUP_CANCEL_FILL_MISMATCH",
                details={
                    "trade_id": tid, "symbol": symbol,
                    "alpaca_order_id": alpaca_id,
                    "alpaca_status": alpaca_status,
                    "reason": "Order filled on Alpaca but DB marked CANCELLED due to restart; "
                              "position tracked under successor trade",
                    "successor_trade_id": successor_id,
                    "cleaned_at": now_utc.isoformat(),
                },
                timestamp=now_utc,
            ))

    # ── 4. Back-fill exit_price and PnL for CLOSED trades missing both ────────
    log(f"\n== Group 4: Back-fill exit_price/PnL for CLOSED trades ==")
    closed_missing = db.query(Trade).filter(
        Trade.status == "CLOSED",
        Trade.exit_price.is_(None),
        Trade.pnl.is_(None),
    ).all()

    for t in closed_missing:
        # Skip ones being deleted (Group 1) or annotated as superseded (Group 2)
        if t.id in TSLA_TEST_IDS or t.id in SUPERSEDED_IDS:
            continue

        # Try to find a sell order on Alpaca
        sell_order = find_sell_order(fills_by_symbol, t.symbol, t.quantity or 0)
        if sell_order and sell_order.filled_avg_price:
            exit_px = float(sell_order.filled_avg_price)
            qty = t.quantity or int(sell_order.filled_qty or 0)
            pnl = (exit_px - float(t.entry_price or 0)) * qty
            if "sell" not in str(sell_order.side).lower():
                pnl = -pnl
            log(f"  BACKFILL id={t.id} {t.symbol}: exit_price=${exit_px:.4f} pnl=${pnl:.2f} "
                f"(from Alpaca order {str(sell_order.id)[:8]})")
            changes.append(f"BACKFILL exit_price/pnl for trade id={t.id} {t.symbol}")
            if not dry_run:
                t.exit_price = exit_px
                t.pnl = round(pnl, 2)
                db.add(AuditLog(
                    action="CLEANUP_BACKFILL_PNL",
                    details={
                        "trade_id": t.id, "symbol": t.symbol,
                        "exit_price": exit_px, "pnl": round(pnl, 2),
                        "source": "alpaca_order_history",
                        "alpaca_order_id": str(sell_order.id),
                        "cleaned_at": now_utc.isoformat(),
                    },
                    timestamp=now_utc,
                ))
        else:
            log(f"  CANNOT BACKFILL id={t.id} {t.symbol} status={t.status} "
                f"qty={t.quantity} entry=${t.entry_price} — no matching Alpaca sell order found")

    # ── Commit ─────────────────────────────────────────────────────────────────
    log(f"\n== Summary: {len(changes)} changes ==")
    for c in changes:
        log(f"  {c}")

    if dry_run:
        log("\nDRY RUN — no changes written. Re-run without --dry-run to apply.")
        db.rollback()
    else:
        try:
            db.commit()
            log("\nAll changes committed successfully.")
        except Exception as e:
            db.rollback()
            log(f"ERROR: Commit failed: {e}")
            raise
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up dirty trade history")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without applying")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
