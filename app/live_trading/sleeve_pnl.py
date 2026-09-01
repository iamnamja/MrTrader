"""Per-sleeve daily P&L reconstruction (2026-09-01).

WHY THIS EXISTS
---------------
CH5's stated purpose is "let the live-forward scorecard accrue", but `trend_daily` held 14 rows
with `realized_pnl` / `unrealized_pnl` / `daily_pnl` / `cumulative_pnl` ALL NULL. The tracker was
built to accept P&L; the only caller (`trend_sleeve.run_trend_rebalance`) never passed any, and it
ran weekly rather than daily. Three months of live paper trading recorded no P&L at all.

THE ACCOUNTING MODEL
--------------------
`record_daily` treats `unrealized_pnl` as a LEVEL (mark minus cost basis) and derives
``daily_pnl = realized_today + Δ(unrealized level)``, which is the standard construction that
avoids re-counting an open gain every day. So this module must produce, per date:

  * `realized`   — P&L banked by fills THAT day (FIFO)
  * `unrealized` — the LEVEL of open-position gain at that day's close

Validated against the live account before being written: over all history,
``equity = deposits + FIFO_realized + unrealized + fees`` reconciles to **$0.68** on a $101k
account across 310+ fills. That identity is the correctness bar for anything here.

SLEEVE ATTRIBUTION
------------------
Fills carry a `client_order_id` like ``trend-20260831-DBC`` or ``cash-20260831-SGOV-buy``, so the
prefix names the sleeve. A few historical fills carry a raw UUID (manual/one-off orders — e.g. a
2026-06-17 QQQ sell), so we fall back to `classify_sleeve` on the symbol rather than dropping them
into an unattributed bucket that would silently break the reconciliation.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

QTY_TOL = 1e-9
SLEEVES = ("trend", "cash")


def sleeve_of_fill(fill: Dict[str, Any], db=None) -> str:
    """Which sleeve a fill belongs to: client_order_id prefix, else symbol classification."""
    coid = str(fill.get("client_order_id") or "")
    head = coid.split("-")[0].strip().lower()
    if head in SLEEVES:
        return head
    try:
        from app.startup_reconciler import classify_sleeve
        return (classify_sleeve(fill.get("symbol"), db) or "unknown").lower()
    except Exception:  # noqa: BLE001 — attribution must never break the reconstruction
        return "unknown"


def _fill_date(fill: Dict[str, Any]) -> str:
    return str(fill.get("filled_at") or fill.get("submitted_at") or "")[:10]


def _apply_fill(lots: Dict[str, Deque[List[float]]], fill: Dict[str, Any]) -> float:
    """Apply one fill to the FIFO lot book IN PLACE; return the realized P&L it banked.

    Shared by the warm-up phase (which discards the return, since pre-window realized P&L is not
    part of the series) and the main loop (which accumulates it) — one implementation so the two
    can never drift.
    """
    sym = fill["symbol"]
    qty = float(fill["filled_qty"])
    px = float(fill["filled_avg_price"])
    signed = qty if fill.get("side") == "buy" else -qty
    book = lots.setdefault(sym, deque())

    realized = 0.0
    remaining = signed
    # Close against opposing lots, oldest first.
    while abs(remaining) > QTY_TOL and book and (book[0][0] > 0) != (remaining > 0):
        lot_qty, lot_px = book[0]
        take = min(abs(remaining), abs(lot_qty))
        realized += take * (px - lot_px) if lot_qty > 0 else take * (lot_px - px)
        lot_qty -= take if lot_qty > 0 else -take
        remaining += take if remaining < 0 else -take
        if abs(lot_qty) <= QTY_TOL:
            book.popleft()
        else:
            book[0][0] = lot_qty
    if abs(remaining) > QTY_TOL:
        book.append([remaining, px])
    return realized


def compute_daily_pnl(
    fills: Iterable[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
    *,
    sleeve: Optional[str] = None,
    db=None,
) -> List[Dict[str, Any]]:
    """Per-date ``{date, realized, unrealized, position_value, cost_basis, positions}``.

    `fills` are blotter dicts (oldest-first is not required — they are sorted here).
    `snapshots` are ``{"date": str, "prices": {sym: px}}`` ascending; a date with no snapshot is
    simply not emitted. `sleeve` filters attribution; None means every fill.

    FIFO lots are carried across dates, so `unrealized` is a LEVEL, matching what `record_daily`
    expects. Dates with no fills still emit a row — the open book re-marks at the new prices, which
    is exactly the daily P&L the scorecard needs.
    """
    rows = sorted(
        (f for f in fills
         if float(f.get("filled_qty") or 0) > 0 and f.get("filled_avg_price") is not None
         and (sleeve is None or sleeve_of_fill(f, db) == sleeve)),
        key=lambda f: (_fill_date(f), str(f.get("filled_at") or "")),
    )

    lots: Dict[str, Deque[List[float]]] = {}
    i = 0
    out: List[Dict[str, Any]] = []
    if not snapshots:
        return out

    # WARM-UP: fills up to AND INCLUDING the first snapshot date establish the opening book (lots
    # + cost basis) but contribute NO realized P&L to the series. Two distinct errors this avoids:
    #
    #   * Without any warm-up, row 0 absorbs every historical fill and books the entire pre-window
    #     realized P&L as day one — measured at +$1,051.95 all-time against a true window move of
    #     +$210.56.
    #   * Warming up only STRICTLY before the start date still books day-0's own fills, which had
    #     already executed when that day's NAV was recorded — double-counting them against the
    #     baseline (worth -$52.61 here, traced to a 2026-06-17 QQQ sell).
    #
    # The first snapshot is an OPENING BALANCE SHEET, not a P&L day: it fixes the position, the
    # cost basis and the unrealized seed. Measurement starts the following day.
    _window_start = str(snapshots[0].get("date"))[:10]
    while i < len(rows) and _fill_date(rows[i]) <= _window_start:
        _apply_fill(lots, rows[i])
        i += 1

    for snap in snapshots:
        d = str(snap.get("date"))[:10]
        prices = snap.get("prices") or {}
        realized = 0.0

        # Apply every fill up to and including this snapshot date.
        while i < len(rows) and _fill_date(rows[i]) <= d:
            realized += _apply_fill(lots, rows[i])
            i += 1

        # Mark the open book at this date's close.
        position_value = 0.0
        cost_basis = 0.0
        positions: Dict[str, float] = {}
        unpriced: List[str] = []
        for sym, book in lots.items():
            q = sum(lot[0] for lot in book)
            if abs(q) <= QTY_TOL:
                continue
            positions[sym] = q
            cost_basis += sum(lot[0] * lot[1] for lot in book)
            px = prices.get(sym)
            if px is None:
                unpriced.append(sym)          # cannot mark -> surfaced, never guessed
                continue
            position_value += q * float(px)

        if unpriced:
            log.warning("sleeve_pnl %s %s: %d held symbol(s) unpriced in snapshot: %s",
                        sleeve or "all", d, len(unpriced), ",".join(sorted(unpriced)))

        out.append({
            "date": d,
            "realized": realized,
            # Only meaningful when every held symbol is priced; otherwise the level is partial.
            "unrealized": (position_value - cost_basis) if not unpriced else None,
            "position_value": position_value,
            "cost_basis": cost_basis,
            "positions": positions,
            "unpriced": unpriced,
        })

    return out


def load_snapshots(db_path=None) -> List[Dict[str, Any]]:
    """Daily snapshots from back_validation, ascending, with trend + cash prices MERGED.

    The two price sets are stored in separate columns on purpose (the drift metric iterates every
    key of `prices`), but for marking a book they are simply one price map.
    """
    import json
    import sqlite3
    from app.live_trading.back_validation import DB_PATH as _BV
    path = str(db_path or _BV)
    out: List[Dict[str, Any]] = []
    try:
        c = sqlite3.connect(path)
        have = {r[1] for r in c.execute("PRAGMA table_info(trend_backval_daily)")}
        cols = "trade_date, nav, prices" + (", cash_prices" if "cash_prices" in have else "")
        for row in c.execute(f"SELECT {cols} FROM trend_backval_daily ORDER BY trade_date"):
            prices = json.loads(row[2]) if row[2] else {}
            if len(row) > 3 and row[3]:
                prices = {**prices, **json.loads(row[3])}
            out.append({"date": str(row[0])[:10], "nav": row[1], "prices": prices})
    except Exception:  # noqa: BLE001
        log.exception("sleeve_pnl.load_snapshots failed")
    return out


def record_daily_pnl(asof: Optional[str] = None, *, alpaca=None, db=None) -> Dict[str, Any]:
    """Compute today's per-sleeve P&L and write it to the trend/cash trackers.

    Called from the 16:15 EOD job right after the back_validation snapshot, so the snapshot it
    marks against is the one just written. Returns a small summary; never raises — a scorecard
    write must not be able to fail the EOD job.
    """
    out: Dict[str, Any] = {"asof": asof, "written": {}, "error": None}
    try:
        if alpaca is None:
            from app.integrations.alpaca import AlpacaClient
            alpaca = AlpacaClient()
        snaps = load_snapshots()
        if not snaps:
            out["error"] = "no snapshots"
            return out
        if asof:
            snaps = [s for s in snaps if s["date"] <= str(asof)[:10]]
        fills = alpaca.get_all_orders()

        from app.live_trading import cash_tracker, trend_tracker
        for sleeve, tracker in (("trend", trend_tracker), ("cash", cash_tracker)):
            rows = daily_pnl_series(compute_daily_pnl(fills, snaps, sleeve=sleeve, db=db))
            if not rows:
                continue
            last = rows[-1]
            if last.get("unrealized") is None:
                log.warning("sleeve_pnl: %s %s not marked (unpriced holdings) — P&L not written",
                            sleeve, last["date"])
                continue
            tracker.record_daily(
                last["date"],
                realized_pnl=float(last["realized"]),
                unrealized_pnl=float(last["unrealized"]),
                # Write the series' OWN daily/cumulative rather than letting the tracker
                # re-derive them: its derivation anchors on the prior DB row and treats a
                # missing anchor as unrealized=0, which books an inherited opening level as
                # day-one P&L.
                daily_pnl_override=(float(last["daily"]) if last.get("daily") is not None else None),
                cumulative_pnl_override=float(last["cumulative"]),
            )
            out["written"][sleeve] = {
                "date": last["date"], "realized": round(float(last["realized"]), 2),
                "unrealized": round(float(last["unrealized"]), 2),
                "daily": (round(float(last["daily"]), 2) if last.get("daily") is not None else None),
                "cumulative": round(float(last["cumulative"]), 2),
            }
        log.info("sleeve P&L recorded: %s", out["written"])
    except Exception as exc:  # noqa: BLE001 — never fail the EOD job over a scorecard write
        out["error"] = str(exc)
        log.exception("sleeve_pnl.record_daily_pnl failed (swallowed)")
    return out


def daily_pnl_series(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add ``daily`` and ``cumulative`` using the tracker's convention:
    ``daily = realized + Δ(unrealized level)``. Rows whose level is unknown carry `daily=None`
    and do NOT advance the cumulative — a gap must not be silently absorbed as a gain."""
    # SEED the baseline from the FIRST marked row, not 0.0. The series inherits an already-open
    # book whose unrealized level was earned BEFORE tracking began; seeding at zero books that
    # entire pre-existing level as day-one P&L. Measured: the trend sleeve opened this window at
    # -199.33 unrealized, so a zero seed reported +13.30 cumulative where the true window P&L was
    # +212.63 — the whole starting level, silently mis-attributed to the first day. This is the
    # same level-vs-delta trap record_daily guards against, occurring at the series boundary.
    first_marked = next((r for r in rows if r.get("unrealized") is not None), None)
    prior_unreal = float(first_marked["unrealized"]) if first_marked else 0.0
    cum = 0.0
    out = []
    for r in rows:
        u = r.get("unrealized")
        if u is None:
            # Unmarked day: cannot compute a delta. Carry cumulative unchanged rather than
            # absorbing the gap as a gain, and leave prior_unreal alone so the next marked day
            # measures against the last KNOWN level (a multi-day move, correctly attributed).
            out.append({**r, "daily": None, "cumulative": cum})
            continue
        daily = float(r["realized"]) + (float(u) - prior_unreal)
        cum += daily
        prior_unreal = float(u)
        out.append({**r, "daily": daily, "cumulative": cum})
    return out
