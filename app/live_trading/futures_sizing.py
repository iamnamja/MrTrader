"""
futures_sizing.py — Alpha-v10 P2.3: pure futures order-construction math (the multiplier-killer).

`target_lots()` turns target WEIGHTS (signed fraction of NAV) into integer contract LOTS using the
contract multiplier pulled ONLY from the verify-on-connect-confirmed `instrument_master` — a caller
can NEVER hand in a wrong multiplier (there is no such parameter), which is the panel's #1-futures-
killer mitigation. `futures_order_deltas()` diffs target vs broker lots into signed order intents
(reductions-toward-flat first, mirroring the trend sleeve so margin is freed before it's spent).

PURE: no I/O, no broker, no config; it constructs intents and places NOTHING. The live order path
(placement, fills, margin preview) is R1 — this module is the order-construction half only.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from app.live_trading import instrument_master as im

log = logging.getLogger(__name__)


def _round_half_away(x: float) -> int:
    """Round half AWAY from zero (deterministic; avoids banker's-rounding surprises in lot counts)."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def target_lots(target_weights: Dict[str, float], nav: float, prices: Dict[str, float], *,
                min_lots: int = 1,
                max_lots_per_market: Optional[int] = None,
                asset_class_notional_cap: Optional[float] = None,
                ) -> Tuple[Dict[str, int], List[dict]]:
    """Signed target LOTS per instrument_id from signed target weights.

    lots = round_half_away(weight * NAV / (price * multiplier)), with the multiplier taken ONLY from
    `instrument_master.get(iid).multiplier` (the verified source — there is no multiplier parameter).

    Skips: an instrument with no master entry, a non-positive price, or |lots| < `min_lots` (dust;
    spec §2 "< 1 lot"). Clamps |lots| to `max_lots_per_market`. Enforces an optional per-asset-class
    notional cap (fraction of NAV) by proportionally scaling that class's lots and re-rounding.

    Returns (lots, log) — `log` records every skip/clamp/scale for the audit trail. Pure: no I/O.
    """
    log_entries: List[dict] = []
    if not nav or nav <= 0:
        return {}, [{"reason": "non_positive_nav", "nav": nav}]

    lots: Dict[str, int] = {}
    for iid, w in (target_weights or {}).items():
        if not w:
            continue
        inst = im.get(iid)
        if inst is None:
            log_entries.append({"instrument_id": iid, "reason": "unmapped_instrument"})
            continue
        price = float(prices.get(iid, 0.0) or 0.0)
        if price <= 0:
            log_entries.append({"instrument_id": iid, "reason": "missing_price"})
            continue
        mult = float(inst.multiplier)
        if mult <= 0:
            log_entries.append({"instrument_id": iid, "reason": "bad_multiplier", "multiplier": mult})
            continue
        raw = float(w) * float(nav) / (price * mult)
        n = _round_half_away(raw)
        if max_lots_per_market is not None and abs(n) > int(max_lots_per_market):
            n = int(math.copysign(int(max_lots_per_market), n))
            log_entries.append({"instrument_id": iid, "reason": "clamped_max_lots", "lots": n})
        if abs(n) < int(min_lots):
            if n != 0:
                log_entries.append({"instrument_id": iid, "reason": "dust_skipped",
                                    "raw_lots": round(raw, 4)})
            continue
        lots[iid] = n

    if asset_class_notional_cap is not None and lots:
        lots = _apply_asset_class_cap(lots, nav, prices, float(asset_class_notional_cap),
                                      min_lots, log_entries)
    return lots, log_entries


def _apply_asset_class_cap(lots: Dict[str, int], nav: float, prices: Dict[str, float],
                           cap_frac: float, min_lots: int, log_entries: List[dict]) -> Dict[str, int]:
    """Scale each asset class's lots down proportionally if its notional exceeds cap_frac * NAV."""
    by_class: Dict[str, List[str]] = {}
    notional: Dict[str, float] = {}
    for iid, n in lots.items():
        inst = im.get(iid)
        cls = inst.asset_class if inst else im.FUTURE
        price = float(prices.get(iid, 0.0) or 0.0)
        mult = float(inst.multiplier) if inst else 1.0
        notional[iid] = abs(n) * price * mult
        by_class.setdefault(cls, []).append(iid)

    out: Dict[str, int] = dict(lots)
    cap_dollars = cap_frac * float(nav)
    for cls, iids in by_class.items():
        total = sum(notional[i] for i in iids)
        if total <= cap_dollars + 1e-9 or total <= 0:
            continue
        scale = cap_dollars / total
        for iid in iids:
            scaled = _round_half_away(out[iid] * scale)
            if abs(scaled) < int(min_lots):
                log_entries.append({"instrument_id": iid, "reason": "notional_cap_zeroed",
                                    "asset_class": cls})
                out.pop(iid, None)
            elif scaled != out[iid]:
                log_entries.append({"instrument_id": iid, "reason": "notional_cap_scaled",
                                    "asset_class": cls, "lots": scaled})
                out[iid] = scaled
    return out


def futures_order_deltas(target_lots_map: Dict[str, int],
                         broker_lots: Dict[str, int]) -> List[dict]:
    """Diff target vs broker lots into signed order intents. A position on one side only is a full
    open or full exit. Reductions-toward-flat are ordered FIRST (free margin before spending it),
    mirroring `trend_sleeve.compute_trend_deltas`. Zero deltas are dropped. Pure: places nothing."""
    deltas: List[dict] = []
    for iid in sorted(set(target_lots_map) | set(broker_lots)):
        tgt = int(target_lots_map.get(iid, 0))
        cur = int(broker_lots.get(iid, 0))
        d = tgt - cur
        if d == 0:
            continue
        reduces = abs(tgt) < abs(cur)            # moving toward (or to) flat
        deltas.append({
            "instrument_id": iid,
            "side": "buy" if d > 0 else "sell",
            "qty": abs(d),
            "target": tgt,
            "current": cur,
            "reduces_exposure": reduces,
            "is_full_exit": tgt == 0 and cur != 0,
        })
    # reductions/exits first, then opens/increases (stable within each group)
    deltas.sort(key=lambda x: (not x["reduces_exposure"], x["instrument_id"]))
    return deltas
