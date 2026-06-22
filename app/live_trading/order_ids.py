"""
order_ids.py — Alpha-v10 H6: deterministic per-order idempotency keys (the client_order_id scheme).

A retry of the SAME intended order reuses the SAME key, so the broker dedups it (no double-fill) and
`AlpacaClient.place_market_order` reuses the existing order idempotently instead of erroring. One
scheme, one place — so sleeves can't drift. P2.3 (IBKR) extends this with a run-id variant
(strategy + signal_date + rebalance_ts + config_hash + code_version) for the futures order path.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Optional


def idempotency_key(sleeve: str, symbol: str, *, side: Optional[str] = None,
                    day: Optional[str] = None) -> str:
    """`{sleeve}-{YYYYMMDD}-{SYMBOL}[-{side}]` — stable across a same-day retry, distinct per
    (sleeve, day, symbol[, side]). Byte-for-byte matches the historical trend/cash schemes
    (`trend-{stamp}-{sym}`, `cash-{stamp}-{sym}-{side}`) so it's a pure centralization, no behaviour
    change. One intended order per (sleeve, day, symbol[, side]) per rebalance."""
    stamp = day or _date.today().strftime("%Y%m%d")
    base = f"{sleeve}-{stamp}-{symbol}"
    return f"{base}-{side}" if side else base
