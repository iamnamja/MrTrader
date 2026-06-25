"""Alpha-v10 audit Wave 5i — BLOCKER: a standing pause must survive an agent crash-and-restart.

The orchestrator restarts a crashed agent by re-invoking run(). Previously run() hard-set
status='running', so a crash during a manual/auto pause silently RESUMED the agent (it self-un-paused
while the control plane still reported paused → the trader could place live orders the operator
believed were halted). Now run() preserves an existing 'paused' status, and _run_agent re-applies a
standing pause before the restart.
"""
from __future__ import annotations

import asyncio


# ── run() preserves a standing pause (BaseAgent._activate_status) ─────────────────
def test_activate_status_preserves_paused():
    from app.agents.base import BaseAgent

    class _A(BaseAgent):
        async def run(self):
            pass

    a = _A("t")
    a.status = "paused"
    a._activate_status()
    assert a.status == "paused"          # standing pause preserved across restart

    a.status = "initialized"
    a._activate_status()
    assert a.status == "running"         # normal first start activates

    a.status = "error"                   # crashed-while-running -> resume running
    a._activate_status()
    assert a.status == "running"


# ── _run_agent re-applies a standing pause after a crash restart ──────────────────
def test_run_agent_reapplies_pause_after_crash(monkeypatch):
    from app.orchestrator import AgentOrchestrator
    o = AgentOrchestrator.__new__(AgentOrchestrator)
    o._running = True
    o._manual_paused = True             # operator paused
    o._auto_paused = False
    o.agent_status = {}

    async def _log_error(*a, **k):
        return None
    o._log_error = _log_error

    # avoid the real 30s restart sleep
    async def _fast_sleep(_s):
        return None
    monkeypatch.setattr("app.orchestrator.asyncio.sleep", _fast_sleep)

    from app.agents.base import BaseAgent
    entry_status = []

    class _FakeAgent(BaseAgent):
        def __init__(self):
            super().__init__("trader")
            self.status = "paused"      # was paused by pause_trading()
            self.calls = 0

        async def run(self):
            self._activate_status()     # the REAL guard: preserves a standing pause
            self.calls += 1
            entry_status.append(self.status)
            if self.calls == 1:
                raise RuntimeError("boom")    # crash once
            o._running = False                # stop the restart loop on the 2nd run

    agent = _FakeAgent()
    asyncio.run(o._run_agent("trader", agent))
    # combined: _run_agent re-applies the standing pause, and run()'s _activate_status preserves it
    assert agent.status == "paused"
    assert entry_status[1] == "paused"        # 2nd run() entered paused (not silently resumed)


def test_run_agent_no_reapply_when_not_paused(monkeypatch):
    from app.orchestrator import AgentOrchestrator
    o = AgentOrchestrator.__new__(AgentOrchestrator)
    o._running = True
    o._manual_paused = False
    o._auto_paused = False
    o.agent_status = {}

    async def _log_error(*a, **k):
        return None
    o._log_error = _log_error

    async def _fast_sleep(_s):
        return None
    monkeypatch.setattr("app.orchestrator.asyncio.sleep", _fast_sleep)

    class _FakeAgent:
        def __init__(self):
            self.status = "running"
            self.calls = 0

        async def run(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            o._running = False

    agent = _FakeAgent()
    asyncio.run(o._run_agent("trader", agent))
    # no standing pause -> not forced to paused (agent free to resume running)
    assert agent.status != "paused"
