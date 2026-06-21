"""
broker_adapter.py — Alpha-v10 R0.3: the BrokerAdapter abstraction + a READ-ONLY Alpaca adapter.

The seam that lets the portfolio brain speak one canonical language across venues. R0.3 ships only
the READ side (account / positions / normalize) — `place_order` is deliberately ABSENT so the shadow
book-state layer (R0.4) is structurally incapable of trading. Order placement + the IBKR adapter are
R1, behind the live-paper gateway + the whole-book risk gate.

Venue specifics (broker symbols, multipliers, margin semantics) are resolved via the instrument
master; the brain sees only `CanonicalPosition` / `AccountState`. "The broker is reality" — these
read the live broker, normalized; the DB is memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

from app.live_trading import instrument_master as im


@dataclass(frozen=True)
class CanonicalPosition:
    instrument_id: str          # canonical (None-safe: see `mapped`)
    venue: str
    broker_symbol: str
    asset_class: str
    quantity: float
    price: float                # current/mark price
    multiplier: float
    currency: str
    market_value: float         # signed market value in base currency
    notional: float             # |qty| * price * multiplier (gross notional)
    mapped: bool                # False if the broker symbol had no instrument-master entry

    @property
    def is_cash_equivalent(self) -> bool:
        return im.is_cash_equivalent(self.instrument_id) if self.mapped else False


@dataclass(frozen=True)
class AccountState:
    venue: str
    nav: float                  # net liquidation value / equity
    cash: float
    buying_power: float
    settled_cash: Optional[float] = None
    margin_used: Optional[float] = None          # None for a cash (non-margin) account
    margin_available: Optional[float] = None
    maintenance_margin: Optional[float] = None


@dataclass(frozen=True)
class BrokerHealth:
    venue: str
    connected: bool
    clock_ok: bool              # market clock reachable / not stale
    detail: str = ""


@runtime_checkable
class BrokerAdapter(Protocol):
    """Read side of the broker abstraction (R0.3). Order methods (place/cancel/flatten) are added in
    R1. Implementations normalize each venue's truth into canonical objects BEFORE the brain sees it."""
    venue: str

    def health(self) -> BrokerHealth: ...
    def get_account(self) -> AccountState: ...
    def get_positions(self) -> List[CanonicalPosition]: ...
    def normalize_instrument(self, broker_symbol: str) -> Optional[str]: ...


_READ_ONLY_METHODS = frozenset({"get_clock", "get_account", "get_positions", "get_position"})


class _ReadOnlyClientProxy:
    """Wraps a broker client and exposes ONLY whitelisted READ methods; any other attribute
    (e.g. `submit_order`) raises. Defense-in-depth so a caller holding the adapter cannot reach a
    trade-capable method via the underlying client."""
    def __init__(self, client):
        object.__setattr__(self, "_c", client)

    def __getattr__(self, name):
        if name in _READ_ONLY_METHODS:
            return getattr(object.__getattribute__(self, "_c"), name)
        raise AttributeError(f"read-only adapter: '{name}' is not permitted in R0.3 (no trading)")

    def __setattr__(self, name, value):
        raise AttributeError("read-only client proxy is immutable")


class AlpacaReadOnlyAdapter:
    """Wraps the existing AlpacaClient (behind a read-only proxy), exposing ONLY the read side as
    canonical objects. No order methods exist on this class AND the held client is proxied to read
    methods only — structurally incapable of trading (R0.3 shadow safety)."""
    venue = im.ALPACA

    def __init__(self, client=None):
        if client is None:
            from app.integrations.alpaca import AlpacaClient
            client = AlpacaClient()
        self._client = _ReadOnlyClientProxy(client)

    def health(self) -> BrokerHealth:
        try:
            clock = self._client.get_clock()
            return BrokerHealth(self.venue, connected=True, clock_ok=clock is not None,
                                detail="ok" if clock is not None else "clock unavailable")
        except Exception as e:  # noqa: BLE001 — health must never raise
            return BrokerHealth(self.venue, connected=False, clock_ok=False, detail=str(e))

    def normalize_instrument(self, broker_symbol: str) -> Optional[str]:
        return im.lookup(self.venue, broker_symbol)

    def get_account(self) -> AccountState:
        a = self._client.get_account()
        return AccountState(
            venue=self.venue,
            nav=float(a.get("equity", a.get("portfolio_value", 0.0))),
            cash=float(a.get("cash", 0.0)),
            buying_power=float(a.get("buying_power", 0.0)),
            # do NOT alias settled_cash to total cash (unsettled T+1 cash would look settled); the
            # real settled field is wired in R1. Alpaca paper here is a cash/long account -> no margin.
            settled_cash=None,
            margin_used=None, margin_available=None, maintenance_margin=None)

    def get_positions(self) -> List[CanonicalPosition]:
        out: List[CanonicalPosition] = []
        for p in self._client.get_positions():
            sym = p["symbol"]
            iid = self.normalize_instrument(sym)
            inst = im.get(iid) if iid else None
            mult = inst.multiplier if inst else 1.0
            asset_class = inst.asset_class if inst else im.EQUITY
            qty = float(p["qty"])
            price = float(p.get("current_price", 0.0))
            mv = float(p.get("market_value", qty * price * mult))
            out.append(CanonicalPosition(
                instrument_id=iid or sym, venue=self.venue, broker_symbol=sym,
                asset_class=asset_class, quantity=qty, price=price, multiplier=mult,
                currency="USD", market_value=mv, notional=abs(qty) * price * mult,
                mapped=iid is not None))
        return out


# Compile-time-ish guard: the read-only adapter must NOT expose order methods (R0.3 shadow safety).
for _forbidden in ("place_order", "submit_order", "cancel_order", "flatten_all", "liquidate_all"):
    assert not hasattr(AlpacaReadOnlyAdapter, _forbidden), \
        f"AlpacaReadOnlyAdapter must be read-only in R0.3 (found {_forbidden})"
