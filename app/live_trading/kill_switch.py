"""
Kill switch — emergency trading halt.

Closes every open position via Alpaca market orders and records the event
in the audit log.  Idempotent: safe to call multiple times.
"""
import logging
from datetime import datetime
from typing import Dict, Any, List

from app.database.session import get_session
from app.database.models import AuditLog

logger = logging.getLogger(__name__)


_CFG_KS_ACTIVE = "kill_switch.active"


class KillSwitch:
    """One-button emergency stop: close all positions and halt new trades."""

    def __init__(self):
        self._active = False

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    def load_state(self) -> bool:
        """Restore kill-switch state from DB on startup."""
        try:
            from app.database.config_store import get_config
            db = get_session()
            try:
                val = get_config(db, _CFG_KS_ACTIVE)
                if val is not None:
                    self._active = bool(val)
                    if self._active:
                        logger.warning("Kill switch restored as ACTIVE from persisted state")
                    return True
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Could not restore kill switch state: %s", exc)
        return False

    def _persist_state(self):
        try:
            from app.database.config_store import set_config
            db = get_session()
            try:
                set_config(db, _CFG_KS_ACTIVE, self._active, "Kill switch active flag")
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Could not persist kill switch state: %s", exc)

    # ── Actions ───────────────────────────────────────────────────────────────

    def activate(self, reason: str = "Manual activation") -> Dict[str, Any]:
        """
        Close every open position immediately, flag the switch as active,
        and write an audit log entry.
        """
        logger.critical("KILL SWITCH ACTIVATED — reason: %s", reason)
        self._active = True
        self._persist_state()

        alpaca = self._alpaca()
        closed: List[str] = []
        errors: List[Dict] = []

        try:
            positions = alpaca.get_positions()
        except Exception as exc:
            logger.error("Could not fetch positions during kill switch: %s", exc)
            positions = []

        for pos in positions:
            symbol = pos["symbol"]
            qty = int(pos.get("quantity") or pos.get("qty", 0))
            try:
                alpaca.place_market_order(symbol, qty, "sell")
                closed.append(symbol)
                logger.warning("KS: closed %s x%d", symbol, qty)
            except Exception as exc:
                logger.error("KS: failed to close %s — %s", symbol, exc)
                errors.append({"symbol": symbol, "error": str(exc)})

        # Cancel all open orders so no pending limits can fill after halt
        cancelled_orders: List[str] = []
        try:
            from app.startup_reconciler import _get_open_alpaca_orders
            open_orders = _get_open_alpaca_orders(alpaca)
            for o in open_orders:
                try:
                    alpaca.cancel_order(o["order_id"])
                    cancelled_orders.append(o["order_id"])
                    logger.warning("KS: cancelled open order %s (%s %s)", o["order_id"], o["side"], o["symbol"])
                except Exception as exc:
                    logger.error("KS: failed to cancel order %s — %s", o["order_id"], exc)
        except Exception as exc:
            logger.error("KS: could not fetch open orders for cancellation: %s", exc)

        result = {
            "status": "activated",
            "reason": reason,
            "positions_closed": closed,
            "orders_cancelled": cancelled_orders,
            "errors": errors,
            "activated_at": datetime.utcnow().isoformat(),
        }

        self._audit(result)
        return result

    def reset(self, reason: str = "Manual reset"):
        """Re-enable trading after reviewing and resolving the issue."""
        self._active = False
        self._persist_state()
        logger.warning("Kill switch RESET — reason: %s", reason)
        self._audit({"status": "reset", "reason": reason,
                     "reset_at": datetime.utcnow().isoformat()},
                    action="KILL_SWITCH_RESET")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _alpaca():
        from app.integrations import get_alpaca_client
        return get_alpaca_client()

    @staticmethod
    def _audit(details: Dict, action: str = "KILL_SWITCH_ACTIVATED"):
        db = get_session()
        try:
            db.add(AuditLog(action=action, details=details, timestamp=datetime.utcnow()))
            db.commit()
        except Exception as exc:
            logger.error("Audit log write failed: %s", exc)
        finally:
            db.close()


# Module-level singleton
kill_switch = KillSwitch()
