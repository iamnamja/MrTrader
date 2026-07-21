"""ibkr_readiness.py — read-only IBKR migration readiness probe (R1.2 tiny-live pre-flight).

Answers "how ready is the IBKR path?" at a glance so a silent gap — e.g. `ib_insync` not being
importable, which made the R1.1 shadow router skip for weeks and accrue ZERO comparison data — is
visible at daemon boot and on demand (`GET /api/ibkr/readiness`), instead of being discovered on a
Monday rebalance. This is the single pre-flight check before the owner-present R1.2 cutover.

Touches NOTHING live: no order is placed, no gateway session is opened (only a TCP-reachability
probe to host:port). NEVER raises — any error degrades a field, never the caller.

Stages (increasing readiness):
  BLOCKED            — ib_insync unavailable OR the shadow reconstruction self-test fails → R1.1
                       can't even run; fix the environment first.
  R1.1_SHADOW_READY  — shadow routing works (orders reconstruct + map clean), but the gateway is
                       unreachable → R1.0c-2b / R1.2 are blocked until IB Gateway/TWS is up.
  GATEWAY_UP         — gateway reachable but no account configured / Read-Only unknown.
  R1.0C2B_READY      — gateway reachable + account set → ready for the owner Read-Only-OFF smoke.
"""
from __future__ import annotations

import socket
from typing import Any, Dict, Optional

BLOCKED = "BLOCKED"
R1_1_SHADOW_READY = "R1.1_SHADOW_READY"
GATEWAY_UP = "GATEWAY_UP"
R1_0C2B_READY = "R1.0C2B_READY"


def _cfg(db, key: str) -> Optional[str]:
    """Read an agent-config value. Opens its own short session when db is None (get_agent_config
    needs a live session — passing None silently returns nothing, which would mis-report flags)."""
    try:
        from app.database.agent_config import get_agent_config
        if db is not None:
            return get_agent_config(db, key)
        from app.database.session import get_session
        with get_session() as _db:
            return get_agent_config(_db, key)
    except Exception:
        return None


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "on", "yes")


def _ib_insync_info() -> Dict[str, Any]:
    try:
        import ib_insync
        return {"available": True, "version": getattr(ib_insync, "__version__", None)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "version": None, "error": str(exc)}


def _gateway_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        s = socket.socket()
        s.settimeout(timeout)
        try:
            return s.connect_ex((host, int(port))) == 0
        finally:
            s.close()
    except Exception:  # noqa: BLE001
        return False


def _shadow_selftest(db) -> Dict[str, Any]:
    """Reconstruct one sample ETF order through the SAME path R1.1 uses (ib_insync +
    WritableIBKRAdapter mapping), placing nothing. Proves the shadow router will actually produce
    comparison data — the failure that silently skipped for weeks."""
    try:
        from app.live_trading import ibkr_shadow_router as isr
        rows = isr.route_shadow([{"symbol": "SPY", "side": "buy", "qty": 1}],
                                sleeve="readiness", db=db, enabled=True)
        ok = bool(rows) and all(r.get("match") for r in rows)
        return {"ok": ok, "n": len(rows)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "n": 0, "error": str(exc)}


def probe(db=None) -> Dict[str, Any]:
    """Read-only IBKR readiness snapshot. Never raises (any error → a minimal BLOCKED report)."""
    try:
        return _probe(db)
    except Exception as exc:  # noqa: BLE001 — the probe boundary must never raise into the caller
        return {"stage": BLOCKED, "ib_insync": {"available": False, "version": None},
                "shadow_route_selftest": {"ok": False, "n": 0}, "shadow_routing_flag": False,
                "enabled_flag": False, "read_only": True,
                "connection": {"host": None, "port": None, "account": None,
                               "gateway_reachable": False},
                "blockers": [f"readiness probe error: {exc}"]}


def _probe(db) -> Dict[str, Any]:
    ib = _ib_insync_info()
    host = str(_cfg(db, "ibkr.host") or "127.0.0.1")
    try:
        port = int(_cfg(db, "ibkr.port") or 7497)
    except (TypeError, ValueError):
        port = 7497
    account = _cfg(db, "ibkr.account")
    shadow_routing = _truthy(_cfg(db, "ibkr.shadow_routing"))
    enabled = _truthy(_cfg(db, "ibkr.enabled"))
    read_only_raw = _cfg(db, "ibkr.read_only")
    # Unset read_only defaults to ON (the safe, blocks-orders state) — matches the adapter default.
    read_only = True if read_only_raw is None else _truthy(read_only_raw)

    selftest = (_shadow_selftest(db) if ib["available"]
                else {"ok": False, "n": 0, "error": "ib_insync unavailable"})
    reachable = _gateway_reachable(host, port)

    blockers = []
    if not ib["available"]:
        blockers.append("ib_insync not importable in the daemon env — R1.1 shadow router skips "
                        "(zero comparison data); `pip install -r requirements.txt` + restart")
    elif not selftest["ok"]:
        blockers.append("shadow-route reconstruction self-test FAILED — mapping/adapter issue")
    if not reachable:
        blockers.append(f"IB Gateway/TWS not reachable at {host}:{port} — start it (paper) before "
                        "R1.0c-2b / R1.2")
    if reachable and not account:
        blockers.append("ibkr.account not configured")
    if reachable and read_only:
        blockers.append("gateway Read-Only API is ON (or unset) — the owner must turn it OFF for "
                        "the R1.0c-2b live smoke + real margin preview")

    if not ib["available"] or not selftest["ok"]:
        stage = BLOCKED
    elif not reachable:
        stage = R1_1_SHADOW_READY
    elif not account:
        stage = GATEWAY_UP
    else:
        stage = R1_0C2B_READY

    return {
        "stage": stage,
        "ib_insync": ib,
        "shadow_route_selftest": selftest,
        "shadow_routing_flag": shadow_routing,
        "enabled_flag": enabled,
        "read_only": read_only,
        "connection": {"host": host, "port": port, "account": account,
                       "gateway_reachable": reachable},
        "blockers": blockers,
    }


def format_line(rep: Dict[str, Any]) -> str:
    """One-line summary for the startup banner / logs."""
    ib = rep.get("ib_insync", {})
    conn = rep.get("connection", {})
    ibv = f"ib_insync {ib.get('version')}" if ib.get("available") else "ib_insync MISSING"
    gw = "gateway UP" if conn.get("gateway_reachable") else "gateway down"
    st = rep.get("shadow_route_selftest", {})
    shadow = "shadow OK" if st.get("ok") else "shadow FAIL"
    n_block = len(rep.get("blockers") or [])
    return (f"IBKR readiness: stage={rep.get('stage')} | {ibv} | {shadow} | {gw} "
            f"({conn.get('host')}:{conn.get('port')}) | read_only={rep.get('read_only')} | "
            f"{n_block} blocker(s)")
