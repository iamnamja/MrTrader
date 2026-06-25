"""
order_ids.py — Alpha-v10 H6: deterministic per-order idempotency keys (the client_order_id scheme).

A retry of the SAME intended order reuses the SAME key, so the broker dedups it (no double-fill) and
`AlpacaClient.place_market_order` reuses the existing order idempotently instead of erroring. One
scheme, one place — so sleeves can't drift. P2.3 (IBKR) extends this with a run-id variant
(strategy + signal_date + rebalance_ts + config_hash + code_version) for the futures order path.
"""
from __future__ import annotations

import hashlib
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


# ── P2.3 (IBKR futures): run-id idempotency (orderRef) ────────────────────────
def futures_run_id(strategy_id: str, signal_date: str, rebalance_ts: str,
                   config_hash: str, code_version: str) -> str:
    """A deterministic short id for one futures rebalance RUN (spec §3:
    strategy_id + signal_date + rebalance_ts + config_hash + code_version). The same inputs always
    hash to the same id, so a crash-retry of the SAME run reuses the SAME order refs (the broker
    dedups → no double-send). Hashed (not concatenated) to stay within IBKR's `orderRef` length."""
    raw = "|".join(str(x) for x in (strategy_id, signal_date, rebalance_ts, config_hash, code_version))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def futures_order_ref(run_id: str, instrument_id: str, side: str) -> str:
    """`{run_id}-{instrument_id}-{side}` — the per-order IBKR `orderRef`. One intended order per
    (run, instrument, side); stable across a retry of the same run."""
    return f"{run_id}-{instrument_id}-{side}"


def requote_order_id(trade_id, symbol: str, generation) -> str:
    """Deterministic client_order_id for a RE-QUOTED / ESCALATED entry limit order.

    Keyed on the TRADE + the requote GENERATION, so a lost-response retry of the SAME generation
    reuses the SAME key (the broker dedups → no orphaned duplicate live order resting untracked),
    while each new requote/escalation is a DISTINCT order. `generation` is the requote count
    (0, 1, 2, …) or 'esc' for the final escalation. Hashed + short to stay within the broker's
    client_order_id length limit; distinct prefix ('rq') from market/exit keys so it can't collide."""
    disc = str(trade_id) if trade_id else (symbol or "")
    h = hashlib.sha1(disc.encode("utf-8")).hexdigest()[:10]
    return f"rq{str(generation)[:4]}-{h}"


def exit_order_id(trade_id, symbol: str, phase: str) -> str:
    """Deterministic client_order_id for an EXIT order (H6, exit side).

    Keyed on the TRADE (NOT on the calendar day) so a lost-response retry of the SAME exit ALWAYS
    reuses the SAME key and the broker dedups it (no double-sell / no accidental flip) — even if the
    retry crosses midnight (a day-stamp would have produced a new key → a re-sell window). A
    genuinely distinct exit — e.g. an intraday re-entry of the same symbol, which has a NEW trade_id
    — gets a DISTINCT key and is not wrongly deduped to the prior exit. `phase` ('partial'/'full')
    keeps a partial exit and the later full exit of the same trade from colliding. Hashed + short so
    it stays well within the broker's client_order_id length limit."""
    disc = str(trade_id) if trade_id else (symbol or "")
    h = hashlib.sha1(disc.encode("utf-8")).hexdigest()[:10]
    return f"x{str(phase)[:4]}-{h}"
