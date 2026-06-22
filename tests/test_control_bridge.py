"""R0.2 Phase 2 — web↔daemon control-bridge tests (app/control_bridge.py + route guards).

Pins the two invariants:
  1. DEFAULT (in_process): the control routes run their direct path — the bridge returns
     None and NOTHING is enqueued (byte-identical to pre-Phase-2).
  2. subprocess: the routes EMIT a command and DO NOT touch the web's (empty) orchestrator;
     the daemon's consumer/state-sync apply commands to the real brain and never die on a
     bad command.
"""
from __future__ import annotations

import asyncio

import pytest

import app.control_bridge as cb

_REAL_SLEEP = asyncio.sleep


async def _instant_sleep(*_a, **_k):
    # non-recursive replacement for asyncio.sleep in bounded-loop tests
    await _REAL_SLEEP(0)


# ── fakes ─────────────────────────────────────────────────────────────────────
class _FakeQueue:
    def __init__(self):
        self.pushed = []
        self._to_pop = []

    def push(self, queue, msg):
        self.pushed.append((queue, msg))
        return True

    def pop(self, queue, timeout=1):
        return self._to_pop.pop(0) if self._to_pop else None


class _FakePM:
    def __init__(self):
        self.calls = []
        self._analyzed_today = True
        self._selected_today = True

    async def select_instruments(self):
        self.calls.append("select_instruments")

    async def _analyze_swing_premarket(self):
        self.calls.append("analyze_swing")

    async def _send_swing_proposals(self):
        self.calls.append("send_swing")

    async def _retrain(self):
        self.calls.append("retrain")

    async def select_intraday_instruments(self, window=None, use_morning_candidates=True):
        self.calls.append(("intraday", window, use_morning_candidates))


class _FakeOrch:
    def __init__(self, pm=None):
        self.agents = {"portfolio_manager": pm} if pm else {}
        self.calls = []

    def pause_trading(self):
        self.calls.append("pause")

    def resume_trading(self):
        self.calls.append("resume")


@pytest.fixture
def fake_queue(monkeypatch):
    q = _FakeQueue()
    monkeypatch.setattr("app.integrations.get_redis_queue", lambda: q)
    return q


# ── emit_control_command ──────────────────────────────────────────────────────
def test_emit_known_command_pushes(fake_queue):
    assert cb.emit_control_command(cb.CMD_PAUSE) is True
    assert fake_queue.pushed == [(cb.COMMAND_QUEUE, {"cmd": "pause", "payload": {}})]


def test_emit_unknown_command_refused(fake_queue):
    assert cb.emit_control_command("nope") is False
    assert fake_queue.pushed == []          # never enqueued


def test_emit_redis_failure_returns_false(monkeypatch):
    def _boom():
        raise RuntimeError("redis down")
    monkeypatch.setattr("app.integrations.get_redis_queue", _boom)
    assert cb.emit_control_command(cb.CMD_PAUSE) is False   # no raise, just False


# ── bridge_or_none (mode-conditional route helper) ────────────────────────────
def test_bridge_or_none_in_process_returns_none(monkeypatch, fake_queue):
    monkeypatch.setattr("app.trading_runtime.web_boots_brain", lambda: True)
    assert cb.bridge_or_none(cb.CMD_PAUSE) is None   # route runs its direct path
    assert fake_queue.pushed == []                   # nothing enqueued in in_process


def test_bridge_or_none_subprocess_emits(monkeypatch, fake_queue):
    monkeypatch.setattr("app.trading_runtime.web_boots_brain", lambda: False)
    out = cb.bridge_or_none(cb.CMD_PAUSE)
    assert out and out["status"] == "queued" and out["command"] == "pause" and out["queued"] is True
    assert fake_queue.pushed == [(cb.COMMAND_QUEUE, {"cmd": "pause", "payload": {}})]


# ── dispatch_command ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_dispatch_pause_resume(monkeypatch):
    orch = _FakeOrch()
    monkeypatch.setattr("app.orchestrator.orchestrator", orch)
    bg = set()
    await cb.dispatch_command(cb.CMD_PAUSE, {}, background=bg)
    await cb.dispatch_command(cb.CMD_RESUME, {}, background=bg)
    assert orch.calls == ["pause", "resume"]


@pytest.mark.asyncio
async def test_dispatch_trigger_runs_pm_in_background(monkeypatch):
    pm = _FakePM()
    monkeypatch.setattr("app.orchestrator.orchestrator", _FakeOrch(pm))
    bg = set()
    await cb.dispatch_command(cb.CMD_TRIGGER_CYCLE, {}, background=bg)
    assert bg, "trigger should spawn a tracked background task"
    await asyncio.gather(*bg)               # let it finish
    assert pm.calls == ["select_instruments"]


@pytest.mark.asyncio
async def test_dispatch_trigger_swing_resets_flags(monkeypatch):
    pm = _FakePM()
    monkeypatch.setattr("app.orchestrator.orchestrator", _FakeOrch(pm))
    bg = set()
    await cb.dispatch_command(cb.CMD_TRIGGER_SWING, {}, background=bg)
    await asyncio.gather(*bg)
    assert pm._analyzed_today is False and pm._selected_today is False
    assert pm.calls == ["analyze_swing", "send_swing"]


@pytest.mark.asyncio
async def test_dispatch_job_pause_uses_payload(monkeypatch):
    calls = []
    monkeypatch.setattr("app.scheduler.scheduler.pause_job", lambda jid: calls.append(jid))
    bg = set()
    await cb.dispatch_command(cb.CMD_JOB_PAUSE, {"job_id": "trend_rebalance_trigger"}, background=bg)
    assert calls == ["trend_rebalance_trigger"]


@pytest.mark.asyncio
async def test_dispatch_unknown_command_no_raise():
    bg = set()
    await cb.dispatch_command("garbage", {}, background=bg)   # must not raise
    assert bg == set()


@pytest.mark.asyncio
async def test_dispatch_handler_error_is_swallowed(monkeypatch):
    # a control command whose handler raises must not propagate (consumer stays alive)
    class _Boom(_FakeOrch):
        def pause_trading(self):
            raise RuntimeError("boom")
    monkeypatch.setattr("app.orchestrator.orchestrator", _Boom())
    bg = set()
    await cb.dispatch_command(cb.CMD_PAUSE, {}, background=bg)   # no raise


# ── consume_control_commands (bounded) ────────────────────────────────────────
@pytest.mark.asyncio
async def test_consumer_drains_and_dispatches(monkeypatch, fake_queue):
    orch = _FakeOrch()
    monkeypatch.setattr("app.orchestrator.orchestrator", orch)
    fake_queue._to_pop = [{"cmd": "pause", "payload": {}}, None, {"cmd": "resume", "payload": {}}]
    await cb.consume_control_commands(poll_timeout=0, max_iterations=3)
    assert orch.calls == ["pause", "resume"]


@pytest.mark.asyncio
async def test_consumer_cancellation_cleans_up_inflight_trigger(monkeypatch):
    # MINOR-4 guard: a long-running manual trigger must be cancelled when the consumer
    # is cancelled on daemon shutdown (no orphaned task, no hang).
    inflight = {}

    class _SlowPM(_FakePM):
        async def select_instruments(self):
            inflight["task"] = asyncio.current_task()
            await _REAL_SLEEP(3600)        # never finishes on its own
    monkeypatch.setattr("app.orchestrator.orchestrator", _FakeOrch(_SlowPM()))
    q = _FakeQueue()
    q._to_pop = [{"cmd": "trigger_cycle", "payload": {}}]
    monkeypatch.setattr("app.integrations.get_redis_queue", lambda: q)

    task = asyncio.create_task(cb.consume_control_commands(poll_timeout=0))
    for _ in range(500):                   # wait until the trigger is in-flight
        if "task" in inflight:
            break
        await _REAL_SLEEP(0)
    assert "task" in inflight
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert inflight["task"].cancelled()    # the consumer cancelled the orphan


@pytest.mark.asyncio
async def test_consumer_survives_pop_error(monkeypatch):
    class _BadQueue(_FakeQueue):
        def pop(self, queue, timeout=1):
            raise RuntimeError("redis blip")
    monkeypatch.setattr("app.integrations.get_redis_queue", lambda: _BadQueue())
    monkeypatch.setattr("asyncio.sleep", _instant_sleep)
    await cb.consume_control_commands(poll_timeout=0, max_iterations=2)   # must not raise


# ── state sync ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_sync_detects_change(monkeypatch):
    class _KS:
        def __init__(self):
            self._active = False
            self.is_active = False

        def load_state(self):
            self.is_active = True       # simulate a web-triggered activation in Postgres
    monkeypatch.setattr("app.live_trading.kill_switch.kill_switch", _KS())
    assert await cb.sync_persisted_state() is True


@pytest.mark.asyncio
async def test_sync_no_change(monkeypatch):
    class _KS:
        is_active = False

        def load_state(self):
            pass
    monkeypatch.setattr("app.live_trading.kill_switch.kill_switch", _KS())
    assert await cb.sync_persisted_state() is False


@pytest.mark.asyncio
async def test_sync_error_no_raise(monkeypatch):
    class _KS:
        is_active = False

        def load_state(self):
            raise RuntimeError("db down")
    monkeypatch.setattr("app.live_trading.kill_switch.kill_switch", _KS())
    assert await cb.sync_persisted_state() is False   # swallowed


@pytest.mark.asyncio
async def test_state_sync_loop_bounded(monkeypatch):
    calls = {"n": 0}

    async def _fake_sync(_lg=None):
        calls["n"] += 1
        return False
    monkeypatch.setattr(cb, "sync_persisted_state", _fake_sync)
    monkeypatch.setattr("asyncio.sleep", _instant_sleep)
    await cb.state_sync_loop(interval=0, max_iterations=3)
    assert calls["n"] == 3


# ── route-level mode-conditional behavior (call the async route fns directly) ──
@pytest.mark.asyncio
async def test_route_pause_in_process_calls_orchestrator(monkeypatch, fake_queue):
    import app.api.orchestrator_routes as r
    monkeypatch.setattr("app.trading_runtime.web_boots_brain", lambda: True)
    orch = _FakeOrch()
    monkeypatch.setattr(r, "orchestrator", orch)
    out = await r.pause_trading()
    assert out == {"status": "trading_paused"}
    assert orch.calls == ["pause"]            # direct path
    assert fake_queue.pushed == []            # nothing enqueued


@pytest.mark.asyncio
async def test_route_pause_subprocess_emits_and_skips_orchestrator(monkeypatch, fake_queue):
    import app.api.orchestrator_routes as r
    monkeypatch.setattr("app.trading_runtime.web_boots_brain", lambda: False)
    orch = _FakeOrch()
    monkeypatch.setattr(r, "orchestrator", orch)
    out = await r.pause_trading()
    assert out["status"] == "queued"
    assert orch.calls == []                   # the web NEVER touches its empty orchestrator
    assert fake_queue.pushed == [(cb.COMMAND_QUEUE, {"cmd": "pause", "payload": {}})]


def test_kill_reload_notify_in_process_is_noop(monkeypatch, fake_queue):
    import app.api.routes as routes
    monkeypatch.setattr("app.trading_runtime.web_boots_brain", lambda: True)
    routes._notify_daemon_state_reload()
    assert fake_queue.pushed == []            # in_process: web owns the kill_switch → no command


def test_kill_reload_notify_subprocess_emits_reload(monkeypatch, fake_queue):
    import app.api.routes as routes
    monkeypatch.setattr("app.trading_runtime.web_boots_brain", lambda: False)
    routes._notify_daemon_state_reload()
    assert fake_queue.pushed == [(cb.COMMAND_QUEUE, {"cmd": "reload_state", "payload": {}})]


@pytest.mark.asyncio
async def test_route_increase_capital_subprocess_emits(monkeypatch, fake_queue):
    import app.api.routes as routes
    monkeypatch.setattr("app.trading_runtime.web_boots_brain", lambda: False)
    out = await routes.request_capital_increase()
    assert out["status"] == "queued" and out["command"] == "capital_advance"
    assert fake_queue.pushed == [(cb.COMMAND_QUEUE, {"cmd": "capital_advance", "payload": {}})]
