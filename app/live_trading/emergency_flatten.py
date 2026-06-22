"""
emergency_flatten.py — Alpha-v10 H4: the OUT-OF-BAND broker-only flatten.

The "close everything now" primitive that works even if the brain is wedged or down. It speaks ONLY
to the broker (no DB / Redis / orchestrator dependency): cancel all open orders + liquidate all
positions. Use when reconciliation/kill-switch/the app itself can't be trusted — the broker is the
one source you can always reach.

DRY-RUN by default (`execute=False` only reports what it WOULD do). The caller (CLI/watchdog) must
pass `execute=True` to actually liquidate. The live book is on Alpaca; the IBKR flatten is R1 — TWS
Read-Only API blocks order placement/cancellation until trading is enabled, so an IBKR flatten can't
run yet (documented, not silently skipped).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def flatten_alpaca(*, execute: bool = False, alpaca: Optional[Any] = None) -> Dict[str, Any]:
    """Cancel ALL open orders + liquidate ALL positions on Alpaca. Broker-only; no DB/app deps.

    execute=False (default) -> DRY-RUN: report current positions/open-orders, change nothing.
    execute=True            -> `close_all_positions(cancel_orders=True)` (atomic cancel + liquidate).
    Returns a report dict; never raises (errors are collected) so an emergency caller always gets a
    structured result it can log/alert on.
    """
    report: Dict[str, Any] = {
        "venue": "ALPACA", "dry_run": not execute,
        "positions": [], "open_orders": None, "actions": [], "errors": [], "ok": False,
    }
    try:
        if alpaca is None:
            from app.integrations import get_alpaca_client
            alpaca = get_alpaca_client()
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"client init: {e}")
        return report

    # current positions (the wrapper returns dicts)
    try:
        positions = alpaca.get_positions() or []
        report["positions"] = [
            {"symbol": p.get("symbol"), "qty": p.get("qty"),
             "market_value": p.get("market_value")} for p in positions]
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"get_positions: {e}")

    # open orders (best-effort, for the report)
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        oo = alpaca.trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        report["open_orders"] = len(oo or [])
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"get_orders: {e}")

    if not execute:
        report["actions"].append(
            f"DRY-RUN: would cancel all open orders + liquidate {len(report['positions'])} "
            "position(s). Pass execute=True to act.")
        report["ok"] = True
        return report

    # ── EXECUTE: one atomic call cancels every open order AND liquidates every position ──
    try:
        resp = alpaca.trading_client.close_all_positions(cancel_orders=True) or []
        # Alpaca returns 207 MULTI-STATUS: each element carries a per-symbol HTTP status, so a symbol
        # can FAIL (halted/non-closable) while the call returns 200. In an emergency report we must
        # NOT claim ok unless EVERY position came back 2xx — surface the failures loudly instead.
        failed = []
        for r in resp:
            status = getattr(r, "status", None)
            sym = getattr(r, "symbol", None)
            if isinstance(r, dict):
                status = r.get("status", status)
                sym = r.get("symbol", sym)
            try:
                ok_item = 200 <= int(status) < 300
            except (TypeError, ValueError):
                ok_item = False           # unknown status -> treat as failure (conservative)
            if not ok_item:
                failed.append({"symbol": sym, "status": status})
        report["actions"].append(
            f"close_all_positions(cancel_orders=True) -> {len(resp)} response(s), {len(failed)} failed")
        if failed:
            report["errors"].append(f"positions NOT liquidated (per-symbol failure): {failed}")
            report["ok"] = False
            log.error("EMERGENCY FLATTEN partial: %d position(s) failed to liquidate: %s",
                      len(failed), failed)
        else:
            report["ok"] = True
            log.warning("EMERGENCY FLATTEN executed on Alpaca: %d position(s) liquidated, "
                        "orders cancelled", len(report["positions"]))
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"close_all_positions: {e}")
        log.error("EMERGENCY FLATTEN failed on Alpaca: %s", e)
    return report
