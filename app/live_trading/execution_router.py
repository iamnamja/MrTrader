"""execution_router.py — R1.2: pick the live execution venue per sleeve.

The trend/cash sleeves place through the canonical `WritableBrokerAdapter` seam instead of calling
the Alpaca client directly, so the book can be cut over to IBKR **by config** (`pm.{sleeve}_venue`)
with Alpaca as the instant rollback.

  - `alpaca` (DEFAULT) → `WritableAlpacaAdapter` — wraps `AlpacaClient.place_market_order`
    byte-for-byte (H3 fat-finger + H6 idempotent-reuse preserved). Byte-identical to pre-R1.2.
  - `ibkr` → `WritableIBKRAdapter(mode="live")` on a `from_config` connection. INERT until an owner
    both flips the venue AND turns the gateway Read-Only API off (R1.0c-2b / R1.2 Phase 3); the
    connect/disconnect + async-fill lifecycle is wired in Phase 3 (needs a live gateway to test).

Fail-safe by construction: `resolve_venue` returns 'alpaca' on any unknown/blank/error value (a live
order must never route to an unintended venue by a typo), and `get_execution_adapter` RAISES on an
unrecognized venue (a live order must never go to an unresolved venue).
"""
from __future__ import annotations

ALPACA = "alpaca"
IBKR = "ibkr"
_VALID = (ALPACA, IBKR)


def resolve_venue(db, sleeve: str) -> str:
    """The configured execution venue for `sleeve` ('trend' | 'cash'); default 'alpaca'.
    Fail-safe: unknown/blank/error → 'alpaca'."""
    try:
        from app.database.agent_config import get_agent_config
        v = str(get_agent_config(db, f"pm.{sleeve}_venue") or ALPACA).strip().lower()
    except Exception:  # noqa: BLE001
        return ALPACA
    return v if v in _VALID else ALPACA


def get_execution_adapter(venue: str, *, alpaca_client=None, db=None):
    """Return the `WritableBrokerAdapter` for `venue`. Raises `ValueError` on an unknown venue."""
    v = str(venue).strip().lower()
    if v == ALPACA:
        from app.live_trading.writable_broker_adapter import WritableAlpacaAdapter
        return WritableAlpacaAdapter(alpaca_client)
    if v == IBKR:
        from app.live_trading.writable_ibkr_adapter import WritableIBKRAdapter
        from app.live_trading.ibkr_connection import IBKRConnectionManager
        return WritableIBKRAdapter(IBKRConnectionManager.from_config(db), mode="live")
    raise ValueError(f"execution_router: unknown venue {venue!r} (expected one of {_VALID})")
