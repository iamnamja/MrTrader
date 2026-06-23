"""
futures_sleeve.py — Alpha-v10 P2.3: the IBKR futures rebalance EXECUTOR (shadow-first, default-off).

The order-construction half of the IBKR futures path: it computes target lots → order deltas → run-id
order refs, runs them through the SAME safety machinery the live trend sleeve uses (kill-switch →
reconciliation-before-trade → whole-book gate), then LOGS the would-be orders and writes shadow audit
rows. It PLACES NOTHING — there is no order-placement code path here, by design.

Inert by four independent layers:
  1. `ibkr.enabled` (default false) — the read adapter never connects.
  2. `ibkr.futures_enabled` (default false) — this executor is DORMANT (returns without running).
  3. `ibkr.trading_mode` (default "shadow") — computes + logs would-be orders; never places.
  4. NO write API exists — the read-only adapter has a compile-time no-order guard, and this module
     imports no placement method. Even with every flag flipped, the worst case is a logged warning.

Deferred to R1 (owner-gated: needs TWS Read-Only API OFF + owner-present + the live paper gateway):
real order placement, fills capture, `whatIfOrder` margin preview, roll-order emission, the live
target-weights signal path (today this consumes an EXPLICIT stub weight source, NOT a live signal),
and scheduler wiring. See docs/reference/P2_IBKR_EXECUTION_DESIGN.md §6.
"""
from __future__ import annotations

import json
import logging
from datetime import date as _date
from typing import Any, Dict, List, Optional

from app.live_trading import instrument_master as im

log = logging.getLogger(__name__)

CODE_VERSION = "p2.3"
STRATEGY_ID = "futures_book"


def _truthy(val: Any) -> bool:
    return str(val).strip().lower() == "true"


def _stub_target_weights(db) -> Dict[str, float]:
    """The shadow weight source: a JSON blob `ibkr.futures_target_weights_json` of
    {instrument_id: signed_weight}. This is an EXPLICIT STUB, **not a live signal** — there is no
    runtime signal-compute path for `futures_book` yet (it lives only in app/research); building the
    live `carry`/`xsmom` weight extraction + PIT data feed is R1. Empty/invalid → {} (no orders)."""
    from app.database.agent_config import get_agent_config
    try:
        raw = get_agent_config(db, "ibkr.futures_target_weights_json")
        if not raw:
            return {}
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
        out: Dict[str, float] = {}
        for k, v in (data or {}).items():
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if w:
                out[str(k)] = w
        if out:
            log.info("ibkr futures: STUB weights (NOT a live signal — P2.3 order-construction shadow"
                     " only): %s", out)
        return out
    except Exception:
        log.debug("ibkr futures: stub-weights parse failed -> {}", exc_info=True)
        return {}


def _ibkr_symbol(iid: str) -> Optional[str]:
    inst = im.get(iid)
    return inst.broker_symbol(im.IBKR) if inst else None


def run_futures_rebalance(db=None, *, force: bool = False, adapter=None) -> Dict[str, Any]:
    """Compute (and SHADOW-log) one IBKR futures rebalance. Never raises; places nothing.

    force=True bypasses the `ibkr.futures_enabled` dormancy gate (manual shadow dry-run); the
    kill-switch is always honored. `adapter` may be injected (tests); else a read-only IBKR adapter
    is built from config and connected. Returns a summary dict for logging/tests.
    """
    summary: Dict[str, Any] = {"status": "ok", "venue": im.IBKR, "mode": None,
                               "would_orders": [], "nav": None, "run_id": None}
    own_db = db is None
    if own_db:
        from app.database.session import get_session
        db = get_session()

    own_adapter = adapter is None
    try:
        from app.database.agent_config import get_agent_config

        enabled = (_truthy(get_agent_config(db, "ibkr.enabled"))
                   and _truthy(get_agent_config(db, "ibkr.futures_enabled")))
        if not enabled and not force:
            log.info("ibkr futures: DORMANT (ibkr.enabled / ibkr.futures_enabled false) — no run")
            summary["status"] = "dormant"
            return summary

        trading_mode = str(get_agent_config(db, "ibkr.trading_mode") or "shadow").strip().lower()
        summary["mode"] = trading_mode

        # ── Kill switch (fail-closed) ──
        from app.live_trading.kill_switch import kill_switch
        if kill_switch.is_active:
            log.warning("ibkr futures: kill switch ACTIVE — no run")
            summary["status"] = "blocked"
            summary["block_reason"] = "kill_switch"
            return summary

        # ── Connect the READ-ONLY adapter (no order capability exists) ──
        if adapter is None:
            from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter
            adapter = IBKRReadOnlyAdapter.from_config(db)
        health = adapter.connect()
        if not getattr(health, "connected", False):
            log.warning("ibkr futures: adapter not connected — fail-closed: %s",
                        getattr(health, "detail", ""))
            summary["status"] = "failed"
            summary["block_reason"] = "ibkr_not_connected"
            return summary

        # ── Verify-on-connect: DROP any CRITICAL-mismatch instrument (the #1-killer guard) ──
        try:
            mismatches = adapter.verify_contracts() or []
        except Exception:
            log.warning("ibkr futures: verify_contracts failed — fail-closed (drop all)", exc_info=True)
            mismatches = None
        if mismatches is None:
            summary["status"] = "failed"
            summary["block_reason"] = "verify_failed"
            return summary
        blocked_iids = {m.instrument_id for m in mismatches if getattr(m, "critical", False)}
        summary["verify_blocked"] = sorted(blocked_iids)

        # ── Broker truth: NAV + positions (canonical, venue=IBKR) ──
        account = adapter.get_account()
        nav = float(getattr(account, "nav", 0.0) or 0.0)
        summary["nav"] = nav
        if nav <= 0:
            log.warning("ibkr futures: NAV<=0 — fail-closed")
            summary["status"] = "failed"
            summary["block_reason"] = "nav_unavailable"
            return summary
        positions: List = adapter.get_positions() or []
        broker_lots = {p.instrument_id: int(round(p.quantity)) for p in positions}
        prices = {p.instrument_id: float(p.price) for p in positions if p.price and p.price > 0}

        # ── Target weights (STUB), minus any verify-blocked instrument ──
        weights = {k: v for k, v in _stub_target_weights(db).items() if k not in blocked_iids}

        # ── Size: target lots -> order deltas -> run-id order refs ──
        from app.live_trading.futures_sizing import target_lots, futures_order_deltas
        lots, sizing_log = target_lots(weights, nav, prices)
        summary["target_lots"] = lots
        summary["sizing_log"] = sizing_log
        deltas = futures_order_deltas(lots, broker_lots)

        from app.live_trading.order_ids import futures_run_id, futures_order_ref
        signal_date = _date.today().isoformat()
        config_hash = futures_run_id("cfg", json.dumps(weights, sort_keys=True),
                                     trading_mode, "", "")  # stable hash of the run's inputs
        run_id = futures_run_id(STRATEGY_ID, signal_date, signal_date, config_hash, CODE_VERSION)
        summary["run_id"] = run_id
        would_orders = [
            {**d, "order_ref": futures_order_ref(run_id, d["instrument_id"], d["side"])}
            for d in deltas
        ]
        summary["would_orders"] = would_orders

        # ── H1 reconciliation-before-trade (reuse; IBKR positions via extra_actual) ──
        try:
            recon_mode = str(get_agent_config(db, "pm.reconciliation_mode") or "shadow").strip().lower()
        except Exception:
            recon_mode = "shadow"
        if recon_mode != "off":
            try:
                from app.live_trading import reconciliation as _recon
                from app.notifications import notifier as _rnotifier
                # IBKR-SCOPED: pass expected={} (no DB-modelled futures book yet) so this does NOT
                # drag the Alpaca book in as phantom "missing" breaks — it compares ONLY the IBKR
                # slice. Any held IBKR future is truthfully surfaced as a break until an R1 task wires
                # the futures book into the DB. raw_alpaca=[] keeps the Alpaca actual side empty too.
                recon_result = _recon.shadow_reconcile_before_trade(
                    db, [], nav=nav, extra_actual=positions, expected={}, mode=recon_mode,
                    label="futures", notifier=_rnotifier)
                summary["recon_mode"] = recon_mode
                summary["recon_status"] = recon_result.status
            except Exception:
                log.debug("ibkr futures: reconciliation wiring failed", exc_info=True)

        # ── R0.5 whole-book gate (reuse; venue=IBKR) ──
        try:
            gate_mode = str(get_agent_config(db, "pm.whole_book_gate_mode") or "shadow").strip().lower()
        except Exception:
            gate_mode = "shadow"
        try:
            from app.live_trading import whole_book_gate as wbg
            from app.notifications import notifier as _notifier
            cur_raw = [{"symbol": _ibkr_symbol(p.instrument_id) or p.broker_symbol,
                        "qty": p.quantity, "current_price": p.price} for p in positions]
            gate_intents = [{"symbol": _ibkr_symbol(o["instrument_id"]), "side": o["side"],
                             "qty": o["qty"]} for o in would_orders
                            if _ibkr_symbol(o["instrument_id"])]
            dropped = [o["instrument_id"] for o in would_orders
                       if not _ibkr_symbol(o["instrument_id"])]
            if dropped:
                log.warning("ibkr futures: %d order(s) have no IBKR symbol — excluded from the gate "
                            "view (cannot be sized/placed anyway): %s", len(dropped), dropped)
            gate_prices = {_ibkr_symbol(iid): px for iid, px in prices.items() if _ibkr_symbol(iid)}
            gate_verdict = wbg.shadow_gate_from_intents(
                cur_raw, gate_intents, gate_prices, nav, mode=gate_mode, venue=im.IBKR,
                label="futures", notifier=_notifier)
            summary["gate_mode"] = gate_mode
            summary["gate_allow"] = bool(gate_verdict.allow)
            summary["gate_breaches"] = list(gate_verdict.breaches)
        except Exception:
            log.debug("ibkr futures: whole-book gate wiring failed", exc_info=True)

        # ── SHADOW: log + audit, place NOTHING (no placement path exists in P2.3) ──
        from app.database import decision_audit
        for o in would_orders:
            try:
                decision_audit.write_decision(
                    symbol=o["instrument_id"], strategy=STRATEGY_ID, final_decision="enter",
                    block_reason="shadow", direction=o["side"].upper(),
                    price_at_decision=prices.get(o["instrument_id"]))
            except Exception:
                log.debug("ibkr futures: audit write failed", exc_info=True)
        if trading_mode != "shadow":
            log.warning("ibkr futures: trading_mode=%s requested but NO order path exists in P2.3 — "
                        "deferred to R1; placing NOTHING", trading_mode)
        log.info("ibkr futures SHADOW: would place %d order(s) (run_id=%s) — none sent",
                 len(would_orders), run_id)
        summary["placed"] = 0
        return summary
    except Exception as exc:  # noqa: BLE001 — must never raise out of the executor
        log.warning("ibkr futures: run failed (fail-safe): %s", exc, exc_info=True)
        summary["status"] = "failed"
        summary["block_reason"] = f"error: {exc}"
        return summary
    finally:
        if own_adapter and adapter is not None:
            try:
                adapter.disconnect()
            except Exception:
                log.debug("ibkr futures: disconnect failed", exc_info=True)
        if own_db:
            try:
                db.close()
            except Exception:
                pass


# Compile-time guard: this executor must NOT expose/import any order-PLACEMENT method (P2.3 is
# order-construction + shadow only). Placement is R1, on a separate write adapter.
for _forbidden in ("place_order", "submit_order", "placeOrder", "place_market_order"):
    assert _forbidden not in globals(), \
        f"futures_sleeve must place no orders in P2.3 (found {_forbidden})"
