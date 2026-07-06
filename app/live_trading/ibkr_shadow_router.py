"""
ibkr_shadow_router.py — Alpha-v10 R1.1: SHADOW routing of the live ETF/cash rebalance onto IBKR.

For each order the live sleeves place on Alpaca, this reconstructs the order the IBKR adapter WOULD
build (venue-neutral OrderIntent -> IBKR Stock contract + MarketOrder) in SHADOW mode — it builds,
validates, logs, and compares against the Alpaca order, and **places nothing on IBKR**. This is where
whole-share rounding and any instrument-master gap (an unmapped symbol fails closed) surface cheaply,
weekly, on real rebalance data — before a single real IBKR order at the R1.2 cutover.

Guardrails (why this is safe to call from the LIVE path):
  - **Gated OFF by default** (`ibkr.shadow_routing` agent-config flag) — fully inert until the owner
    turns it on; instantly reversible.
  - **Cannot reach the gateway.** A shadow `place()` returns before any connection dispatch, so this
    needs no IBKR connection at all. The connection stub here RAISES if anyone ever tries to dispatch —
    so R1.1 is structurally incapable of sending an IBKR order.
  - **Never raises into the live loop.** Per-order failures are caught and recorded; the caller's
    Alpaca placement is completely unaffected.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class _ShadowOnlyConn:
    """A connection that can never dispatch — guarantees the shadow router never hits the gateway.
    A shadow place()/cancel() returns before calling this; if code ever changes to dispatch, it
    FAILS LOUD instead of silently placing a live IBKR order."""

    def is_connected(self) -> bool:
        return False

    def call(self, thunk, *, timeout=None):
        raise RuntimeError("ibkr_shadow_router: dispatch attempted — the R1.1 shadow router must "
                           "NEVER reach the gateway (place nothing). This is a bug.")


def is_enabled(db, override: Optional[bool] = None) -> bool:
    """Owner flag `ibkr.shadow_routing` (default OFF). `override` forces the decision (tests)."""
    if override is not None:
        return bool(override)
    try:
        from app.database.agent_config import get_agent_config as g
        return str(g(db, "ibkr.shadow_routing") or "").strip().lower() in ("1", "true", "on", "yes")
    except Exception:  # noqa: BLE001 — a config read must never break the rebalance
        return False


def route_shadow(items: List[Dict[str, Any]], *, sleeve: str, client_ref=None, db=None,
                 enabled: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Reconstruct each live Alpaca ETF/cash order as the IBKR order it WOULD build, in SHADOW, and
    return a comparison list. The value here is a weekly **reconstruction / instrument-mapping check**
    on real rebalance data (an unmapped symbol fails closed and is recorded) — whole-share rounding is
    identical on both sides for whole-share ETFs, so that only really bites for futures lots (R1.3).

    `items`: the sleeve's own order dicts ({symbol, side, qty}; optional {price, sec_type}).
    `client_ref`: optional callable(item)->str to derive the idempotency ref per item.
    Returns [] (no-op) when the flag is off / no items / the dep or setup is unavailable. Places NOTHING
    on IBKR. **Self-enforcing: never raises** (so a caller need not wrap it — the sleeves do anyway)."""
    if not is_enabled(db, enabled) or not items:
        return []
    try:
        import ib_insync  # noqa: F401 — probe ONCE so a missing dep reports distinctly, not as N mapping gaps
    except Exception:  # noqa: BLE001
        log.warning("ibkr shadow-route [%s]: ib_insync unavailable — skipped (live path unaffected)", sleeve)
        return []
    try:
        from app.live_trading.writable_ibkr_adapter import WritableIBKRAdapter
        from app.live_trading.writable_broker_adapter import OrderIntent
        adapter = WritableIBKRAdapter(_ShadowOnlyConn(), mode="shadow")
    except Exception as e:  # noqa: BLE001 — self-enforce the no-raise contract at the setup boundary
        log.warning("ibkr shadow-route [%s]: setup failed — skipped (live path unaffected): %s", sleeve, e)
        return []

    rows: List[Dict[str, Any]] = []
    for it in items:
        sym = side = None
        try:
            sym, side, qty = it["symbol"], it["side"], int(it["qty"])
            ref = client_ref(it) if callable(client_ref) else it.get("client_ref")
            intent = OrderIntent(venue="IBKR", instrument_id=sym, sec_type=it.get("sec_type", "ETF"),
                                 side=side, quantity=qty, client_ref=ref, est_price=it.get("price"))
            res = adapter.place(intent)                 # SHADOW: builds + logs, places nothing
            ibkr_qty = res.raw.get("qty")
            rows.append({"symbol": sym, "side": side, "alpaca_qty": qty, "ibkr_qty": ibkr_qty,
                         "match": (ibkr_qty == qty), "status": res.accepted_status})
        except Exception as e:  # noqa: BLE001 — fail-closed PER ITEM; one bad order can't blind the batch
            log.warning("ibkr shadow-route %s %s failed (recorded, live path unaffected): %s",
                        side, sym, e)
            rows.append({"symbol": sym, "side": side, "alpaca_qty": it.get("qty"),
                         "ibkr_qty": None, "match": False, "status": f"error:{type(e).__name__}"})
    n_match = sum(1 for r in rows if r["match"])
    n_err = len(rows) - n_match
    log.info("ibkr SHADOW-ROUTE [%s]: %d order(s) -> %d reconstruct-match Alpaca, %d unmappable/error "
             "(PLACED NOTHING on IBKR)", sleeve, len(rows), n_match, n_err)
    return rows
