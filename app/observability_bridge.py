"""
observability_bridge.py — R0.2 Phase 3: daemon→web WebSocket relay over Redis pub/sub.

PROBLEM (subprocess mode): the agents run in the standalone daemon, which has NO
WebSocket clients — the dashboard's WS clients connect to the FastAPI (web) process. So
agent-decision / trade / alert broadcasts generated in the daemon hit the daemon's empty
WebSocketManager and never reach the dashboard (the underlying data is still written to
Postgres — `agent_decisions` etc. — so REST/refresh views stay correct; only the LIVE
push is lost).

SOLUTION (mode-conditional, default-off):
  • daemon — `publish_ws_event` PUBLISHES each WS event to the Redis channel `ws_events`.
  • web — `ws_relay_loop` SUBSCRIBES and re-broadcasts to the local WS manager.
  • in_process (DEFAULT) — neither runs: `publish_ws_event` is a no-op (not the daemon)
    and the web never starts the relay. Local in-process broadcast is byte-identical to
    pre-Phase-3.

Pub/sub (not a Redis list) is deliberate: WS is live/ephemeral — if no web is subscribed,
events are dropped, never accumulated into an unbounded backlog. This module imports no
FastAPI (the WS manager is imported lazily, only inside the web-side relay).

Scope note: the news watchlist and the Trader mid-run cache (the ADR's other two P3
items) need NO bridge — both are mutated and read ONLY inside the brain process (the PM
writes the watchlist the co-located news_monitor reads; the Trader rebuilds
active_positions from the trades table on start), so they are already self-consistent in
whichever process runs the brain.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

log = logging.getLogger("mrtrader.observability_bridge")

WS_CHANNEL = "ws_events"


# ── Publisher (daemon only) ───────────────────────────────────────────────────

def publish_ws_event(message: dict) -> None:
    """Publish a WS broadcast onto Redis so the web can relay it to the dashboard. No-op
    unless this process is the daemon (the web/in_process already broadcasts locally to
    its own clients). Never raises — a dropped live update must never disrupt the brain."""
    from app.trading_runtime import is_daemon_process
    if not is_daemon_process():
        return
    try:
        from app.integrations import get_redis_queue
        # STRICT json.dumps (no default=) to mirror Starlette's ws.send_json exactly: the
        # only payloads relayed are those the local in-process broadcast could also
        # serialize — identical behavior in both modes (a non-JSON-native payload is
        # dropped here just as send_json would drop it locally), with no cross-mode drift.
        get_redis_queue().redis_client.publish(WS_CHANNEL, json.dumps(message))
    except Exception as exc:  # noqa: BLE001
        log.debug("publish_ws_event failed (dropped): %s", exc)


# ── Relay (web only, subprocess mode) ─────────────────────────────────────────

async def _relay_one(raw) -> None:
    """Decode one pub/sub payload and broadcast it to the local WS manager."""
    if raw is None:
        return
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    try:
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        log.debug("ws relay: dropping non-JSON message")
        return
    from app.api.websocket import manager
    await manager.broadcast(payload)


async def ws_relay_loop(*, poll_timeout: float = 1.0, max_iterations: Optional[int] = None) -> None:
    """Web-side loop: subscribe to `ws_events` and re-broadcast each message to the local
    WS manager. Blocking `get_message` runs in a thread so the event loop stays free.
    `max_iterations` bounds it for tests. Resilient: one bad message never stops the loop."""
    from app.integrations import get_redis_queue
    pubsub = get_redis_queue().redis_client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(WS_CHANNEL)
    log.info("ws relay subscribed to channel=%s", WS_CHANNEL)
    iterations = 0
    try:
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            try:
                msg = await asyncio.to_thread(pubsub.get_message, timeout=poll_timeout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("ws relay get_message error: %s", exc)
                await asyncio.sleep(1)
                continue
            if not msg or msg.get("type") != "message":
                continue
            try:
                await _relay_one(msg.get("data"))
            except Exception as exc:  # noqa: BLE001 — never die on one bad event
                log.error("ws relay broadcast error: %s", exc)
    except asyncio.CancelledError:
        log.info("ws relay stopping")
        try:
            pubsub.close()
        except Exception:  # noqa: BLE001
            pass
        raise
