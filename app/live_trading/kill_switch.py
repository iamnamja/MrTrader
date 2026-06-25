"""
Kill switch — emergency trading halt.

Closes every open position via Alpaca market orders and records the event
in the audit log.  Idempotent: safe to call multiple times.
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any

from app.database.session import get_session
from app.database.models import AuditLog

logger = logging.getLogger(__name__)


_CFG_KS_ACTIVE = "kill_switch.active"
# Monotonic sequence bumped on every activate()/reset(). load_state() uses it to tell an
# AUTHORITATIVE state change (a real cross-process reset, higher seq) apart from a STALE value left
# by our OWN failed persist (same/older seq) — so the 3s state-sync can never un-arm an active switch
# on a stale read, yet a genuine web-triggered reset still propagates to the daemon.
_CFG_KS_SEQ = "kill_switch.seq"


def _running_under_pytest() -> bool:
    """Detect pytest so the kill switch never writes to the production DB / audit log
    during tests, even if a test forgets to patch _persist_state.

    Defense-in-depth: a stray test that activates the real KillSwitch singleton must NOT
    leave kill_switch.active=True in the persistent configuration store. Delegates to the
    shared detector, which is subprocess-safe (env-var primary) — the previous local check
    keyed on os.environ["_"] (a unix-ism rarely set on Windows) and broke across process
    boundaries, the same fragility that let a test app-boot leak into the live log.
    """
    from app.utils.runtime import is_test_mode
    return is_test_mode()


class KillSwitch:
    """One-button emergency stop: close all positions and halt new trades."""

    def __init__(self):
        self._active = False
        # Last state-sequence this process has applied/persisted. A load_state() downgrade
        # (active->inactive) is honored ONLY when the DB carries a strictly higher seq.
        self._seq = 0
        # True once we have SUCCESSFULLY flattened in the current halt episode. Guards against a
        # repeat activate() (manual + watchdog, or a re-trigger) re-submitting closes on an already
        # flat book — a double-sell risk. Reset on reset() (new episode).
        self._flattened = False

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
                seq = get_config(db, _CFG_KS_SEQ)
                db_seq = seq if isinstance(seq, int) and not isinstance(seq, bool) else 0
                if val is not None:
                    # STRICT bool check — NEVER coerce via bool(). The genuine activate()/reset()
                    # path always persists a real JSON bool, so a non-bool value here is malformed
                    # (corrupted/legacy row, or a mocked config store under test). bool() would turn
                    # ANY non-empty such value — the string "false", a MagicMock, etc. — into True
                    # and spuriously HALT live trading. Treat malformed as "leave state unchanged".
                    if not isinstance(val, bool):
                        logger.warning(
                            "Kill switch persisted state is non-bool (%r, type=%s) — leaving the "
                            "in-memory state unchanged (refuse to halt OR un-arm on a malformed value)",
                            val, type(val).__name__,
                        )
                        return True
                    # ASYMMETRIC apply:
                    #  • upgrade to ACTIVE (False->True): ALWAYS honor — fail-safe, a persisted halt
                    #    must bind even on a legacy/seq-less row.
                    #  • downgrade to INACTIVE (True->False): honor ONLY if the DB carries a strictly
                    #    NEWER seq (an authoritative cross-process reset). A same/older seq means the
                    #    DB is stale (e.g. our own activate() persist failed) — refuse to un-arm.
                    if val and not self._active:
                        self._active = True
                        logger.warning("Kill switch restored as ACTIVE from persisted state")
                    elif (not val) and self._active:
                        if db_seq > self._seq:
                            self._active = False
                            logger.warning("Kill switch un-armed from persisted state "
                                           "(authoritative reset, seq %d > %d)", db_seq, self._seq)
                        else:
                            logger.warning(
                                "Kill switch DB=inactive but no NEWER seq (db_seq=%d <= %d) — "
                                "refusing to un-arm an active switch (stale/failed-persist guard); "
                                "only reset() clears it", db_seq, self._seq)
                    # advance our applied-seq watermark to the DB's if it is ahead
                    if db_seq > self._seq:
                        self._seq = db_seq
                    return True
            finally:
                db.close()
        except Exception as exc:
            # FAIL-CLOSED on a read error: we cannot CONFIRM the persisted state, and a kill switch
            # that was genuinely activated (persisted True) must not be silently dropped at boot
            # (which would start trading ENABLED). Retry briefly; if still unreadable, HALT as a
            # precaution and require an explicit operator reset after verifying.
            for _attempt in range(3):
                try:
                    from app.database.config_store import get_config as _gc
                    _db = get_session()
                    try:
                        _val = _gc(_db, _CFG_KS_ACTIVE)
                    finally:
                        _db.close()
                    self._active = bool(_val) if isinstance(_val, bool) else False
                    if self._active:
                        logger.warning("Kill switch restored as ACTIVE from persisted state (retry)")
                    return True
                except Exception:
                    time.sleep(0.2 * (_attempt + 1))
            logger.critical("Kill switch state UNREADABLE at boot (%s) — HALTING as a precaution "
                            "(fail-closed); reset manually after verifying the DB.", exc)
            self._active = True
            return False
        return False

    def _persist_state(self) -> bool:
        """Persist the active flag, then READ IT BACK to verify (retrying on failure).

        A SILENT persist failure is dangerous: the control daemon's 3 s state-sync calls load_state()
        which overwrites the in-memory flag from the DB, so if activate() set active=True but the write
        never landed, the DB still holds False and the kill switch would un-arm itself within ~3 s.
        Verifying the write (and screaming on total failure) closes that hole. Returns True iff the
        persisted value was confirmed to match."""
        if _running_under_pytest():
            logger.debug("Skipping kill switch DB persist — pytest detected")
            return True
        from app.database.config_store import set_config, get_config
        for attempt in range(3):
            try:
                db = get_session()
                try:
                    # Bump a monotonic seq ALONGSIDE the active flag so a peer/state-sync can tell
                    # this authoritative change apart from a stale value. Seed from max(in-memory,
                    # DB) so concurrent writers don't regress the counter.
                    cur = get_config(db, _CFG_KS_SEQ)
                    cur_seq = cur if isinstance(cur, int) and not isinstance(cur, bool) else 0
                    new_seq = max(self._seq, cur_seq) + 1
                    set_config(db, _CFG_KS_ACTIVE, self._active, "Kill switch active flag")
                    set_config(db, _CFG_KS_SEQ, new_seq, "Kill switch state sequence (monotonic)")
                    check = get_config(db, _CFG_KS_ACTIVE)
                    check_seq = get_config(db, _CFG_KS_SEQ)
                finally:
                    db.close()
                if (isinstance(check, bool) and check == self._active
                        and check_seq == new_seq):
                    self._seq = new_seq            # only advance on a CONFIRMED write
                    return True
                logger.warning("Kill switch persist read-back mismatch (got %r/seq %r, want %r/seq %d) "
                               "attempt %d/3", check, check_seq, self._active, new_seq, attempt + 1)
            except Exception as exc:
                logger.warning("Kill switch persist attempt %d/3 failed: %s", attempt + 1, exc)
            time.sleep(0.2 * (attempt + 1))
        logger.critical("Kill switch state could NOT be persisted after retries (want active=%s) — "
                        "halt is IN-MEMORY ONLY; cross-process state-sync may not see it. INVESTIGATE.",
                        self._active)
        return False

    # ── Actions ───────────────────────────────────────────────────────────────

    def activate(self, reason: str = "Manual activation") -> Dict[str, Any]:
        """
        Close every open position immediately, flag the switch as active,
        and write an audit log entry.
        """
        was_active = self._active
        self._active = True
        persist_ok = self._persist_state()

        # Re-activation guard: if we are ALREADY active AND already flattened this episode, do NOT
        # re-submit closes. Two activate() calls close together (operator + dead-man watchdog, or a
        # re-trigger) would otherwise each liquidate the same shares — a double-sell that flips the
        # book net short. Only the first successful flatten runs; a prior FAILED flatten can still be
        # retried (because _flattened is only set on success).
        if was_active and self._flattened:
            logger.warning("Kill switch re-activation (already active + flattened) — reason: %s; "
                           "NOT re-flattening", reason)
            result = {"status": "already_active", "reason": reason, "positions_closed": [],
                      "errors": [], "persisted": persist_ok, "flatten_ok": True,
                      "activated_at": datetime.utcnow().isoformat()}
            self._audit(result)
            return result

        logger.critical("KILL SWITCH ACTIVATED — reason: %s", reason)
        if not persist_ok:
            logger.critical("KILL SWITCH active in-memory but NOT persisted — investigate immediately.")

        # Flatten via the out-of-band primitive: ONE atomic close_all_positions(cancel_orders=True)
        # — cancels every open order AND liquidates every position in a single broker call, correctly
        # covering shorts (no negative-qty bug) with no per-symbol double-fill race.
        alpaca = self._alpaca()
        from app.live_trading.emergency_flatten import flatten_alpaca
        flat = flatten_alpaca(execute=True, alpaca=alpaca)
        flatten_ok = bool(flat.get("ok"))
        if flatten_ok:
            self._flattened = True
        closed = [p.get("symbol") for p in flat.get("positions", []) if p.get("symbol")]

        result = {
            "status": "activated",
            "reason": reason,
            "positions_closed": closed if flatten_ok else [],
            "flatten_ok": flatten_ok,
            "errors": list(flat.get("errors", [])),
            "persisted": persist_ok,
            "activated_at": datetime.utcnow().isoformat(),
        }

        self._audit(result)
        return result

    def reset(self, reason: str = "Manual reset"):
        """Re-enable trading after reviewing and resolving the issue."""
        self._active = False
        self._flattened = False   # new episode — a future activate() must flatten again
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
        if _running_under_pytest():
            logger.debug("Skipping kill switch audit log write — pytest detected")
            return
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
