"""
Admin tool: manually close a trade with a specified exit price.
Use when reconciler marks RECONCILE_GHOST_UNRESOLVED and no Alpaca fill found.

Usage:
    python scripts/admin_force_close.py --trade-id 108 --price 110.50 \
        --reason "manually verified closed via broker"

Add --confirm to actually commit (without it, the script does a dry-run only).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# Ensure project root on path when invoked from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.database.session import get_session
from app.database.models import Trade, AuditLog
from app.startup_reconciler import write_exit_price


def _fmt(t: Trade) -> str:
    return (
        f"Trade#{t.id}  {t.symbol}  dir={t.direction}  qty={t.quantity}  "
        f"entry={t.entry_price}  exit={t.exit_price}  "
        f"exit_source={getattr(t, 'exit_price_source', None)}  "
        f"status={t.status}  closed_at={t.closed_at}  pnl={t.pnl}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Manually force-close a trade (admin).")
    p.add_argument("--trade-id", type=int, required=True)
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--reason", type=str, required=True)
    p.add_argument("--confirm", action="store_true",
                   help="Actually commit. Without this flag, runs as a dry-run.")
    args = p.parse_args()

    actor = f"admin:{os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'}"
    db = get_session()
    try:
        trade = db.query(Trade).filter(Trade.id == args.trade_id).first()
        if trade is None:
            print(f"ERROR: Trade#{args.trade_id} not found.", file=sys.stderr)
            return 2

        print("BEFORE:")
        print(" ", _fmt(trade))

        is_short = (trade.direction or "BUY") == "SELL_SHORT"
        entry = float(trade.entry_price or 0)
        qty = trade.quantity or 0
        new_pnl = round(
            ((entry - args.price) if is_short else (args.price - entry)) * qty, 2,
        )

        wrote = write_exit_price(
            trade, args.price, source="manual",
            order_id=None, written_by=actor, force=True,
        )
        if not wrote:
            print("ERROR: write_exit_price refused even with force=True (invalid source?)",
                  file=sys.stderr)
            return 3

        trade.status = "CLOSED"
        trade.exit_reason = "manual_admin_force_close"
        trade.closed_at = datetime.utcnow()
        trade.pnl = new_pnl
        trade.status_reason = args.reason

        db.add(AuditLog(
            action="ADMIN_FORCE_CLOSE",
            details={
                "trade_id": trade.id, "symbol": trade.symbol,
                "exit_price": args.price, "pnl": new_pnl,
                "actor": actor, "reason": args.reason,
                "ts": datetime.utcnow().isoformat(),
            },
            timestamp=datetime.utcnow(),
        ))

        print("AFTER (pending commit):")
        print(" ", _fmt(trade))

        if not args.confirm:
            db.rollback()
            print("\nDRY-RUN — pass --confirm to commit.")
            return 0

        db.commit()
        print("\nCOMMITTED.")
        return 0
    except Exception as exc:
        db.rollback()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
