"""
writable_broker_adapter.py — Alpha-v10 R1.0a: the WRITE side of the broker abstraction.

The execution seam every venue plugs into. R0.3 shipped the READ-only `BrokerAdapter`
(broker_adapter.py, structurally can't trade); this adds the ORDER surface so the brain can place
through ONE canonical interface regardless of venue — the foundation for the all-IBKR migration
(docs/reference/R1_ALL_IBKR_MIGRATION_PLAN_2026-07-06.md).

R1.0a scope: the canonical types + the write Protocol + a **`WritableAlpacaAdapter`** that wraps the
EXISTING `alpaca.place_market_order` byte-identically (so H3 fat-finger + H6 idempotent-reuse are
preserved). It is **INERT** — nothing routes through it yet; the live sleeves still call Alpaca
directly. Validating the seam on the working venue FIRST (before the IBKR adapter, R1.0c) is the
low-risk order. `place()` returns an ACK only; fills flow through a SEPARATE async `FillEvent`
capture (never the `place()` return) — per the Opus R1 design review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from app.live_trading.broker_adapter import BrokerAdapter


@dataclass(frozen=True)
class OrderIntent:
    """Venue-neutral order request. `instrument_id` is canonical; the adapter resolves the broker
    symbol/contract. `quantity` = shares (equities) or lots (futures). `client_ref` = the idempotency
    key (the SLEEVE computes it — the adapter passes it through verbatim; see R1 design §3.6)."""
    venue: str
    instrument_id: str
    sec_type: str                       # STK | ETF | FUT
    side: str                           # BUY | SELL
    quantity: float
    order_type: str = "MARKET"          # only MARKET in R1.0
    tif: str = "DAY"
    client_ref: Optional[str] = None
    est_price: Optional[float] = None   # so the venue's pre-trade notional guard can check a MKT order


@dataclass(frozen=True)
class OrderResult:
    """ACK only — NEVER a fill. A market order's fill qty/price are unknowable at submit on every
    venue (Alpaca returns accepted/0/None; IBKR fills arrive async). Fills are a separate FillEvent."""
    broker_order_id: Optional[str]
    accepted_status: str
    idempotent_reuse: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FillEvent:
    """Async fill capture — the SEPARATE channel (execDetails/commissionReport on IBKR; order poll on
    Alpaca). Defined here as the contract; the async wiring lands with the venue adapters (R1.0c+)."""
    client_ref: Optional[str]
    instrument_id: str
    filled_qty: float
    avg_price: Optional[float]
    commission: Optional[float] = None
    ts: Optional[str] = None


@runtime_checkable
class WritableBrokerAdapter(BrokerAdapter, Protocol):
    """Read side (BrokerAdapter) + the order surface. `place()` returns an ACK; fills are async."""
    def place(self, intent: OrderIntent) -> OrderResult: ...
    def cancel(self, broker_order_id: str) -> OrderResult: ...
    def get_open_orders(self) -> List[Dict[str, Any]]: ...


class WritableAlpacaAdapter:
    """R1.0a: writable Alpaca adapter — wraps the EXISTING `AlpacaClient.place_market_order` so the
    H3 fat-finger backstop + H6 duplicate-client_order_id idempotent-reuse are preserved byte-for-byte.
    INERT: no sleeve routes through it yet. Equities/ETFs, MARKET only (Alpaca is the equity venue)."""

    venue = "ALPACA"

    def __init__(self, client=None):
        # Lazy default so importing this module never constructs a live client (mirrors the read adapter).
        if client is None:
            from app.integrations.alpaca import AlpacaClient
            client = AlpacaClient()
        self._client = client

    # ── read side (delegates to the same client; kept minimal for the Protocol) ──
    def health(self):
        from app.live_trading.broker_adapter import AlpacaReadOnlyAdapter
        return AlpacaReadOnlyAdapter(self._client).health()

    def get_account(self):
        from app.live_trading.broker_adapter import AlpacaReadOnlyAdapter
        return AlpacaReadOnlyAdapter(self._client).get_account()

    def get_positions(self):
        from app.live_trading.broker_adapter import AlpacaReadOnlyAdapter
        return AlpacaReadOnlyAdapter(self._client).get_positions()

    def normalize_instrument(self, broker_symbol: str):
        from app.live_trading.broker_adapter import AlpacaReadOnlyAdapter
        return AlpacaReadOnlyAdapter(self._client).normalize_instrument(broker_symbol)

    # ── write side ──
    def place(self, intent: OrderIntent) -> OrderResult:
        if intent.order_type != "MARKET":
            raise NotImplementedError(f"WritableAlpacaAdapter: only MARKET in R1.0 (got {intent.order_type})")
        if intent.sec_type not in ("STK", "ETF"):
            raise NotImplementedError(f"WritableAlpacaAdapter: equities only (got sec_type={intent.sec_type})")
        # FAIL LOUD on a non-whole quantity rather than silently truncating (IBKR equities also can't
        # do fractional; the futures adapter deals in integer lots) — the caller must round explicitly.
        if intent.quantity != int(intent.quantity):
            raise ValueError(f"WritableAlpacaAdapter: non-integer quantity {intent.quantity} "
                             f"(whole shares only — caller must round before place())")
        # For Alpaca equities the canonical instrument_id IS the broker symbol. The idempotency key
        # (client_ref) is passed through VERBATIM — the adapter must not re-derive it (R1 design §3.6).
        res = self._client.place_market_order(
            symbol=intent.instrument_id,
            quantity=int(intent.quantity),
            side=str(intent.side).lower(),
            client_order_id=intent.client_ref,
            est_price=intent.est_price,
        )
        return OrderResult(
            broker_order_id=res.get("order_id"),
            accepted_status=str(res.get("status", "")),
            idempotent_reuse=bool(res.get("idempotent_reuse", False)),
            raw=dict(res),                        # copy so the frozen result can't be mutated via .raw
        )

    def cancel(self, broker_order_id: str) -> OrderResult:
        ok = self._client.cancel_order(broker_order_id)
        return OrderResult(broker_order_id=broker_order_id,
                           accepted_status="canceled" if ok else "cancel_failed")

    def get_open_orders(self) -> List[Dict[str, Any]]:
        # Reuse the new broker order-list read (added for the Executions blotter); open only.
        return self._client.get_orders(limit=200, status="open")
