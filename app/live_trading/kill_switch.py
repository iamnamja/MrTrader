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


class KillSwitch:
    """One-button emergency stop: close all positions and halt new trades."""

    def __init__(self):
        self._active = False

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Actions ───────────────────────────────────────────────────────────────

    def activate(self, reason: str = "Manual activation") -> Dict[str, Any]:
        """
        Close every open position immediately, flag the switch as active,
        and write an audit log entry.
        """
        logger.critical("KILL SWITCH ACTIVATED — reason: %s", reason)
        self._active = True

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

        result = {
            "status": "activated",
            "reason": reason,
            "positions_closed": closed,
            "errors": errors,
            "activated_at": datetime.utcnow().isoformat(),
        }

        self._audit(result)
        return result

    def reset(self, reason: str = "Manual reset"):
        """Re-enable trading after reviewing and resolving the issue."""
        self._active = False
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
