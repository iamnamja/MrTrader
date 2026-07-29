"""venue_reads.py — R1.2 Phase 2: venue-aware READS for the live sleeves.

Phase 1 (`execution_router.py`) made order PLACEMENT venue-aware — the trend/cash sleeves place
through the canonical `WritableBrokerAdapter` selected by `pm.{sleeve}_venue`. But every READ in the
same functions (current positions, NAV, gross-cap sum, reconciliation snapshot, idempotent re-derive)
still went to the raw Alpaca client. After a cutover that is the silent-wrong-venue hazard: orders
route to IBKR while sizing/reconciliation keep reading Alpaca.

`VenueReader` closes that gap. It returns the **Alpaca position/account DICT shape** the sleeves
already consume, so the sleeve bodies barely change:

  - `venue == "alpaca"` (DEFAULT) → delegate to the raw `AlpacaClient` methods **byte-for-byte**
    (`get_positions()` / `get_account()` / `get_position(sym)`). Identical to pre-Phase-2.
  - `venue == "ibkr"` → read canonical `CanonicalPosition` / `AccountState` off the IBKR read adapter
    and normalize to the same dict keys. **INERT until an owner flips the venue (Phase 3):** while the
    venue stays `alpaca`, the IBKR branch never executes. Its connection lifecycle is completed in
    Phase 3 (needs a live gateway) — until then an IBKR read fails CLOSED if it can't confirm reality,
    which the sleeves already turn into a `positions_unavailable` / `nav_unavailable` HOLD.

Fail-safe by construction (mirrors `execution_router`): `resolve_venue` returns 'alpaca' on any
unknown/blank/error venue, so a typo can never silently read an unintended venue.

Dict shape (the canonical → Alpaca-dict contract):
  position → {"symbol", "qty", "market_value", "current_price"}
  account  → {"equity", "cash", "buying_power", "portfolio_value"}
"""
from __future__ import annotations

from typing import Callable, List, Optional

from app.live_trading.execution_router import ALPACA, IBKR, resolve_venue


# ── canonical → Alpaca-dict normalizers ────────────────────────────────────────────
def _pos_to_dict(p) -> dict:
    """A canonical `CanonicalPosition` → the Alpaca position-dict shape the sleeves index.
    `market_value` uses the canonical signed market value (book_state's qty*price*mult trap is already
    guarded inside the IBKR adapter's canonical build)."""
    return {
        "symbol": p.broker_symbol,
        "qty": p.quantity,
        "market_value": p.market_value,
        "current_price": p.price,
    }


def _acct_to_dict(a) -> dict:
    """A canonical `AccountState` → the Alpaca account-dict shape. `nav` is the equity/portfolio_value
    equivalent; both keys map to it so any caller reading either sees the same number."""
    return {
        "equity": a.nav,
        "portfolio_value": a.nav,
        "cash": a.cash,
        "buying_power": a.buying_power,
    }


class VenueReader:
    """Venue-aware read facade returning the Alpaca position/account dict shape. Constructed via
    `get_venue_reader`; not instantiated directly by callers."""

    def __init__(self, venue: str, *, alpaca_client=None,
                 ibkr_provider: Optional[Callable[[], object]] = None):
        self.venue = venue
        self._alpaca = alpaca_client
        self._ibkr_provider = ibkr_provider
        self._ibkr = None  # lazily built canonical read adapter (ibkr branch only)

    # -- alpaca fast paths are byte-identical to the raw client ----------------------
    def _ibkr_adapter(self):
        if self._ibkr is None:
            if self._ibkr_provider is None:
                raise ConnectionError("venue_reads: no IBKR read provider wired (fail-closed)")
            self._ibkr = self._ibkr_provider()
        return self._ibkr

    def get_positions(self) -> List[dict]:
        if self.venue == ALPACA:
            return self._alpaca.get_positions() or []
        return [_pos_to_dict(p) for p in (self._ibkr_adapter().get_positions() or [])]

    def get_account(self) -> dict:
        if self.venue == ALPACA:
            return self._alpaca.get_account()
        return _acct_to_dict(self._ibkr_adapter().get_account())

    def get_position(self, symbol: str) -> Optional[dict]:
        """Single-symbol read (cash sleeve idempotent-reuse re-derive). Alpaca has a native
        `get_position`; IBKR has no single-symbol portfolio call, so filter the venue book."""
        if self.venue == ALPACA:
            return self._alpaca.get_position(symbol)
        sym = (symbol or "").upper()
        for p in self.get_positions():
            if str(p.get("symbol", "")).upper() == sym:
                return p
        return None


def _default_ibkr_provider(db) -> Callable[[], object]:
    """Phase-2 IBKR read provider: a lazily-connected `IBKRReadOnlyAdapter.from_config(db)` (mirrors
    the futures sleeve). Connects on demand and fails CLOSED (ConnectionError) if it can't — the
    sleeves turn that into a HOLD.

    ⚠️ TODO(R1.2 Phase 3 — needs a live gateway to test; do NOT flip a venue to ibkr before these):
      1. DISCONNECT lifecycle. This adapter is never disconnected (the futures sleeve disconnects in a
         `finally`). Wire connect/disconnect through the IBKRConnectionManager so a per-rebalance read
         can't strand the socket.
      2. DISTINCT clientId. `ibkr.client_id` (default 1) is shared by the futures read adapter, the
         writable execution adapter, AND both venue readers — concurrent connects in one cron window
         collide (Gateway rejects a duplicate clientId → that sleeve fail-closes/HOLDs every cycle).
         Allocate a distinct clientId per consumer.
      3. PORTFOLIO-SYNC fail-closed. `ib.portfolio()` is an async local cache; a connected-but-not-yet-
         synced session returns `[]`, which `_current_*_positions` would read as "genuinely flat" →
         a full re-buy of the sleeve (fail-OPEN). Assert the account-update subscription completed
         (non-empty accountValues / an explicit sync flag) before trusting an empty portfolio.
      4. MULTI-ASSET gross-cap. `get_positions()` returns the WHOLE IBKR book incl. futures; the trend
         gross-cap sums `market_value` (≈ daily P&L for futures → understated → fail-OPEN). Filter to
         equity/ETF (or use book_state notional) before the gross-cap sum on a shared IBKR account.
    """
    def _build():
        from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter
        adapter = IBKRReadOnlyAdapter.from_config(db)
        try:
            adapter.connect()
        except Exception:  # noqa: BLE001 — swallow here; the health check below is the fail-closed gate
            pass
        # Fail-closed + LOUD on a bad connect (mirrors the futures sleeve's health check) rather than
        # returning an unhealthy adapter whose first read raises an opaque ConnectionError.
        if not adapter.health().connected:
            raise ConnectionError("venue_reads: IBKR read adapter failed to connect (fail-closed)")
        return adapter
    return _build


def get_venue_reader(db, sleeve: str, *, alpaca_client=None,
                     ibkr_provider: Optional[Callable[[], object]] = None) -> VenueReader:
    """Return the `VenueReader` for `sleeve` ('trend' | 'cash'). Venue from `resolve_venue`
    (default/fail-safe 'alpaca'). `alpaca_client` should be the client the sleeve already holds (so the
    alpaca path reuses the same connection). `ibkr_provider` is injectable for tests."""
    venue = resolve_venue(db, sleeve)
    if venue == IBKR and ibkr_provider is None:
        ibkr_provider = _default_ibkr_provider(db)
    return VenueReader(venue, alpaca_client=alpaca_client, ibkr_provider=ibkr_provider)


__all__ = ["VenueReader", "get_venue_reader"]
