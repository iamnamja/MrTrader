"""R0.2 Phase 3 — observability-bridge tests (app/observability_bridge.py).

Pins: (1) publish is daemon-only (the web/in_process never publishes → no double-delivery,
byte-identical in_process); (2) the web relay decodes + re-broadcasts, skips noise, and
survives a bad message.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import app.observability_bridge as ob

_REAL_SLEEP = asyncio.sleep


class _FakeRedisClient:
    def __init__(self):
        self.published = []
        self._pubsub = _FakePubSub()

    def publish(self, channel, data):
        self.published.append((channel, data))

    def pubsub(self, ignore_subscribe_messages=False):
        return self._pubsub


class _FakePubSub:
    def __init__(self):
        self.subscribed = []
        self._messages = []
        self.closed = False

    def subscribe(self, channel):
        self.subscribed.append(channel)

    def get_message(self, timeout=1.0):
        return self._messages.pop(0) if self._messages else None

    def close(self):
        self.closed = True


class _FakeQueue:
    def __init__(self):
        self.redis_client = _FakeRedisClient()


class _FakeManager:
    def __init__(self):
        self.broadcasts = []

    async def broadcast(self, message):
        self.broadcasts.append(message)


@pytest.fixture
def fake_queue(monkeypatch):
    q = _FakeQueue()
    monkeypatch.setattr("app.integrations.get_redis_queue", lambda: q)
    return q


@pytest.fixture
def fake_manager(monkeypatch):
    m = _FakeManager()
    monkeypatch.setattr("app.api.websocket.manager", m)
    return m


# ── publish_ws_event (daemon-only) ────────────────────────────────────────────
def test_publish_noop_when_not_daemon(monkeypatch, fake_queue):
    monkeypatch.setattr("app.trading_runtime._IS_DAEMON_PROCESS", False)
    ob.publish_ws_event({"type": "agent_decision", "data": {}})
    assert fake_queue.redis_client.published == []      # in_process/web never publishes


def test_publish_when_daemon(monkeypatch, fake_queue):
    monkeypatch.setattr("app.trading_runtime._IS_DAEMON_PROCESS", True)
    ob.publish_ws_event({"type": "trade_executed", "data": {"symbol": "SPY"}})
    assert len(fake_queue.redis_client.published) == 1
    channel, payload = fake_queue.redis_client.published[0]
    assert channel == ob.WS_CHANNEL
    assert json.loads(payload)["type"] == "trade_executed"


def test_publish_never_raises(monkeypatch):
    monkeypatch.setattr("app.trading_runtime._IS_DAEMON_PROCESS", True)

    def _boom():
        raise RuntimeError("redis down")
    monkeypatch.setattr("app.integrations.get_redis_queue", _boom)
    ob.publish_ws_event({"type": "x"})                  # must not raise


# ── _relay_one (decode + broadcast) ───────────────────────────────────────────
@pytest.mark.asyncio
async def test_relay_one_broadcasts_json_str(fake_manager):
    await ob._relay_one(json.dumps({"type": "alert", "data": {"m": "hi"}}))
    assert fake_manager.broadcasts == [{"type": "alert", "data": {"m": "hi"}}]


@pytest.mark.asyncio
async def test_relay_one_broadcasts_json_bytes(fake_manager):
    await ob._relay_one(json.dumps({"type": "x"}).encode("utf-8"))
    assert fake_manager.broadcasts == [{"type": "x"}]


@pytest.mark.asyncio
async def test_relay_one_drops_none_and_garbage(fake_manager):
    await ob._relay_one(None)
    await ob._relay_one("not json{{{")
    assert fake_manager.broadcasts == []                # nothing broadcast


# ── ws_relay_loop (bounded) ───────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_relay_loop_subscribes_and_relays(fake_queue, fake_manager):
    ps = fake_queue.redis_client._pubsub
    ps._messages = [
        {"type": "subscribe", "data": 1},                       # ignored (not a message)
        {"type": "message", "data": json.dumps({"type": "agent_decision", "data": {}})},
        None,                                                   # idle tick
    ]
    await ob.ws_relay_loop(poll_timeout=0, max_iterations=3)
    assert ps.subscribed == [ob.WS_CHANNEL]
    assert fake_manager.broadcasts == [{"type": "agent_decision", "data": {}}]


@pytest.mark.asyncio
async def test_relay_loop_survives_get_message_error(monkeypatch, fake_manager):
    class _BadPubSub(_FakePubSub):
        def get_message(self, timeout=1.0):
            raise RuntimeError("redis blip")
    q = _FakeQueue()
    q.redis_client._pubsub = _BadPubSub()
    monkeypatch.setattr("app.integrations.get_redis_queue", lambda: q)
    monkeypatch.setattr("asyncio.sleep", lambda *_a, **_k: _REAL_SLEEP(0))
    await ob.ws_relay_loop(poll_timeout=0, max_iterations=2)    # must not raise


@pytest.mark.asyncio
async def test_relay_loop_closes_pubsub_on_cancel(fake_queue, fake_manager):
    ps = fake_queue.redis_client._pubsub
    task = asyncio.create_task(ob.ws_relay_loop(poll_timeout=0))
    for _ in range(200):                                        # let it subscribe + idle
        if ps.subscribed:
            break
        await _REAL_SLEEP(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ps.closed is True                                    # cleaned up on shutdown


# ── send_update integration (broadcast + publish hook) ────────────────────────
@pytest.mark.asyncio
async def test_send_update_publishes_when_daemon(monkeypatch, fake_queue):
    from app.api.websocket import WebSocketManager
    monkeypatch.setattr("app.trading_runtime._IS_DAEMON_PROCESS", True)
    mgr = WebSocketManager()                            # no clients → local broadcast no-op
    await mgr.send_update("agent_decision", {"symbol": "SPY"})
    assert len(fake_queue.redis_client.published) == 1   # daemon also publishes for relay
    _, payload = fake_queue.redis_client.published[0]
    assert json.loads(payload)["type"] == "agent_decision"


@pytest.mark.asyncio
async def test_send_update_no_publish_when_not_daemon(monkeypatch, fake_queue):
    from app.api.websocket import WebSocketManager
    monkeypatch.setattr("app.trading_runtime._IS_DAEMON_PROCESS", False)
    mgr = WebSocketManager()
    await mgr.send_update("trade_executed", {"symbol": "SPY"})
    assert fake_queue.redis_client.published == []       # in_process/web: local broadcast only


# ── process role ──────────────────────────────────────────────────────────────
def test_mark_as_daemon_toggles(monkeypatch):
    import app.trading_runtime as tr
    monkeypatch.setattr(tr, "_IS_DAEMON_PROCESS", False)
    assert tr.is_daemon_process() is False
    tr.mark_as_daemon()
    assert tr.is_daemon_process() is True
