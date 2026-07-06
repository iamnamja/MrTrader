"""
writable_ibkr_adapter.py — Alpha-v10 R1.0c: the writable IBKR adapter (SHADOW-first, inert-by-default).

Implements the R1.0a write surface (`place`/`cancel`/`get_open_orders`) for IBKR, running ON the R1.0b
`IBKRConnectionManager` (every IB op dispatched onto its dedicated loop thread). Reads delegate to the
read-only mapping (`IBKRReadOnlyAdapter`) bound to the manager's live `ib` inside the dispatch — so the
careful single-managed-account / notional-recompute fail-closed logic is reused, not duplicated.

Two safety layers keep this inert until the owner is ready:
  1. `mode="shadow"` (DEFAULT): `place()`/`cancel()` build + validate the order, LOG the would-be action,
     and place NOTHING. `mode="live"` (explicit, R1.1+) actually calls `ib.placeOrder`.
  2. The Gateway's Read-Only API stays ON, so even a `mode="live"` place is rejected broker-side until the
     owner flips Read-Only OFF (the R1.1/R1.2 cutover step).

The #1 futures killer (wrong contract multiplier) is guarded: the contract multiplier comes ONLY from
`instrument_master` (never a caller value), and an unmapped instrument FAILS CLOSED (no order built).

`preview()` is `whatIfOrder` margin (an IBKR-specific extra, not on the Protocol). Async fill capture
(`execDetails`/`commissionReport` → `FillEvent`) lands in R1.0c-2, when a live place can actually fill.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.live_trading import instrument_master as im
from app.live_trading.broker_adapter import AccountState, BrokerHealth, CanonicalPosition
from app.live_trading.ibkr_connection import IBKRConnectionManager
from app.live_trading.writable_broker_adapter import MarginImpact, OrderIntent, OrderResult

log = logging.getLogger(__name__)


class WritableIBKRAdapter:
    """Writable IBKR adapter — shadow-first, read-capable, runs on an IBKRConnectionManager."""

    venue = "IBKR"

    def __init__(self, conn: IBKRConnectionManager, *, mode: str = "shadow"):
        self._conn = conn
        self._mode = str(mode).lower()          # "shadow" (default, places nothing) | "live" (R1.1+)

    # ── read side (Protocol) — dispatched onto the manager's loop thread ──────────
    def health(self) -> BrokerHealth:
        conn = self._conn.is_connected()
        return BrokerHealth(self.venue, connected=conn, clock_ok=conn,
                            detail="ok" if conn else "not connected")

    def normalize_instrument(self, broker_symbol: str):
        return im.lookup(self.venue, broker_symbol)

    def get_account(self) -> AccountState:
        from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter
        return self._conn.call(lambda ib: IBKRReadOnlyAdapter(ib=ib).get_account())

    def get_positions(self) -> List[CanonicalPosition]:
        from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter
        return self._conn.call(lambda ib: IBKRReadOnlyAdapter(ib=ib).get_positions())

    # ── contract / order construction (multiplier ONLY from instrument_master) ────
    def _instrument(self, instrument_id: str):
        inst = im.get(instrument_id)
        if inst is None:
            raise ValueError(f"WritableIBKRAdapter: unmapped instrument {instrument_id!r} — "
                             f"fail-closed (no contract/order built)")
        return inst

    @staticmethod
    def _ensure_event_loop():
        # eventkit (ib_insync's dependency) calls asyncio.get_event_loop() at IMPORT time; on Python
        # 3.12 that RAISES "no current event loop" if none is set on this thread — e.g. after another
        # code path called asyncio.run() (which resets the current loop to None). place()/preview()
        # run on the CALLER thread, which may be in exactly that state, so ensure a loop before the
        # lazy import. Harmless (no-op) when a loop already exists.
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

    def _build_contract(self, inst):
        # Lazy import so the module loads without ib_insync / a gateway.
        self._ensure_event_loop()
        from ib_insync import Future, Stock
        sym = inst.broker_symbol(self.venue) or inst.root or inst.instrument_id
        if inst.sec_type == "FUT":
            # THE #1-killer guard: multiplier comes ONLY from the verified instrument master, never a
            # caller/param. (Front-month qualification via reqContractDetails is R1.0c-2 — shadow needs none.)
            # FAIL CLOSED, never open: a missing/zero/fractional multiplier must raise, NOT emit an empty
            # string (which would let IBKR substitute the contract's default multiplier — a silent
            # wrong-multiplier path, the exact failure this guard exists to prevent).
            m = inst.multiplier
            if not m or float(m) != int(m):
                raise ValueError(f"WritableIBKRAdapter: FUT {inst.instrument_id} has invalid multiplier "
                                 f"{m!r} — fail-closed (multiplier must be a whole, nonzero number)")
            return Future(symbol=sym, exchange=inst.exchange or "", currency=inst.currency,
                          multiplier=str(int(m)))
        return Stock(symbol=sym, exchange="SMART", currency=inst.currency)

    def _build_order(self, intent: OrderIntent):
        self._ensure_event_loop()
        from ib_insync import MarketOrder
        if intent.order_type != "MARKET":
            raise NotImplementedError(f"WritableIBKRAdapter: only MARKET in R1.0c (got {intent.order_type})")
        if intent.quantity != int(intent.quantity):
            raise ValueError(f"WritableIBKRAdapter: non-integer quantity {intent.quantity} "
                             f"(whole shares/lots only — caller must round)")
        order = MarketOrder(str(intent.side).upper(), abs(int(intent.quantity)),
                            tif=(intent.tif or "DAY"))
        if intent.client_ref:
            order.orderRef = intent.client_ref     # idempotency link (passed VERBATIM, never re-derived)
        return order

    # ── write surface ─────────────────────────────────────────────────────────────
    def place(self, intent: OrderIntent) -> OrderResult:
        if intent.venue != self.venue:
            raise ValueError(f"WritableIBKRAdapter: intent.venue={intent.venue!r} != IBKR")
        inst = self._instrument(intent.instrument_id)
        contract = self._build_contract(inst)       # raises fail-closed if unmapped/bad
        order = self._build_order(intent)
        shadow_raw: Dict[str, Any] = {
            "symbol": getattr(contract, "symbol", None), "sec_type": inst.sec_type,
            "action": order.action, "qty": order.totalQuantity, "multiplier": inst.multiplier,
            # contract_multiplier = what's ACTUALLY bound onto the IBKR contract (a str, "" for STK).
            # Recorded separately from the master echo so a stringification regression is auditable.
            "contract_multiplier": getattr(contract, "multiplier", None),
            "exchange": getattr(contract, "exchange", None), "client_ref": intent.client_ref,
        }
        if self._mode != "live":
            log.info("ibkr SHADOW would place: %s %s x%s (%s, mult=%s, ref=%s) — PLACED NOTHING",
                     order.action, shadow_raw["symbol"], order.totalQuantity, inst.sec_type,
                     inst.multiplier, intent.client_ref)
            return OrderResult(broker_order_id=None, accepted_status="shadow",
                               raw={"shadow": True, **shadow_raw})
        # LIVE (R1.1+, owner-gated + Read-Only-OFF): actually place on the dedicated loop thread.
        trade = self._conn.call(lambda ib: ib.placeOrder(contract, order))
        return OrderResult(
            broker_order_id=str(getattr(trade.order, "orderId", "") or "") or None,
            accepted_status=str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").lower(),
            idempotent_reuse=False, raw={"live": True, **shadow_raw})

    def cancel(self, broker_order_id: str) -> OrderResult:
        if self._mode != "live":
            log.info("ibkr SHADOW would cancel order %s — DID NOTHING", broker_order_id)
            return OrderResult(broker_order_id=broker_order_id, accepted_status="shadow")
        ok = self._conn.call(lambda ib: self._cancel_live(ib, broker_order_id))
        return OrderResult(broker_order_id=broker_order_id,
                           accepted_status="canceling" if ok else "cancel_failed")

    @staticmethod
    def _cancel_live(ib, broker_order_id: str) -> bool:
        for t in ib.openTrades():
            if str(getattr(t.order, "orderId", "")) == str(broker_order_id):
                ib.cancelOrder(t.order)
                return True
        return False

    def get_open_orders(self) -> List[Dict[str, Any]]:
        try:
            trades = self._conn.call(lambda ib: ib.reqAllOpenOrdersAsync())
            return [{
                "order_id": str(getattr(t.order, "orderId", "") or ""),
                "symbol": getattr(t.contract, "symbol", None),
                "action": getattr(t.order, "action", None),
                "qty": getattr(t.order, "totalQuantity", None),
                "status": str(getattr(getattr(t, "orderStatus", None), "status", "") or "").lower(),
                "order_ref": getattr(t.order, "orderRef", None),
            } for t in (trades or [])]
        except Exception as e:  # noqa: BLE001 — read must degrade
            log.warning("ibkr get_open_orders failed: %s", e)
            return []

    def preview(self, intent: OrderIntent) -> MarginImpact:
        """Pre-trade `whatIfOrder` margin preview (does NOT place). Read-only-safe if the gateway allows
        what-if under Read-Only API; if not, it degrades to ok=False (surfaces the block honestly)."""
        inst = self._instrument(intent.instrument_id)
        contract = self._build_contract(inst)
        order = self._build_order(intent)

        def _f(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None
        try:
            st = self._conn.call(lambda ib: ib.whatIfOrderAsync(contract, order))
            init_m = _f(getattr(st, "initMarginChange", None))
            maint_m = _f(getattr(st, "maintMarginChange", None))
            # ok only if margins actually came back — the gateway's Read-Only API returns an EMPTY
            # OrderState (Error 321) rather than raising, so `st is not None` alone would over-report.
            return MarginImpact(
                init_margin=init_m, maint_margin=maint_m,
                buying_power_after=_f(getattr(st, "equityWithLoanAfter", None)),
                ok=(init_m is not None or maint_m is not None),
                raw={"commission": _f(getattr(st, "commission", None)),
                     "maxCommission": _f(getattr(st, "maxCommission", None))})
        except Exception as e:  # noqa: BLE001 — surface a failed preview honestly, don't crash
            log.warning("ibkr whatIfOrder preview failed: %s", e)
            return MarginImpact(None, None, None, ok=False, raw={"error": str(e)})
