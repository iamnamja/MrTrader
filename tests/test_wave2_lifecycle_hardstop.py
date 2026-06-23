"""Alpha-v10 audit Wave 2 — pause/resume lifecycle (BLOCKER) + intraday daily-loss hard stop.

Pins: pause is reversible (auto-pause on an Alpaca blip auto-resumes on recovery; a manual pause is
NOT auto-resumed); the agent loops idle-while-paused (cooperative, stay alive); and the daily-loss
gate uses LIVE intraday equity (equity - last_equity) so the 2% hard stop binds during the session.
"""
from __future__ import annotations

import asyncio
import types
from unittest.mock import patch

from app.orchestrator import AgentOrchestrator


def _orch(agents):
    o = AgentOrchestrator()
    for name, a in agents.items():
        o.agents[name] = a
        o.agent_status[name] = "idle"
    return o


def _agent(status="running"):
    return types.SimpleNamespace(status=status)


# ── pause/resume reversibility (the BLOCKER) ─────────────────────────────────────
def test_manual_pause_then_resume(monkeypatch):
    from app import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.scheduler, "pause_job", lambda *a, **k: None)
    monkeypatch.setattr(orch_mod.scheduler, "resume_job", lambda *a, **k: None)
    a = _agent()
    o = _orch({"trader": a})
    o.pause_trading()
    assert a.status == "paused" and o._auto_paused is False     # manual pause not auto-flagged
    o.resume_trading()
    assert a.status == "running"


def test_pause_skips_stopped_agents(monkeypatch):
    from app import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.scheduler, "pause_job", lambda *a, **k: None)
    stopped = _agent(status="stopped")
    o = _orch({"trader": stopped})
    o.pause_trading()
    assert stopped.status == "stopped"                          # a stopped agent is not re-paused


def test_auto_pause_auto_resumes_on_alpaca_recovery(monkeypatch):
    from app import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.scheduler, "pause_job", lambda *a, **k: None)
    monkeypatch.setattr(orch_mod.scheduler, "resume_job", lambda *a, **k: None)
    a = _agent()
    o = _orch({"trader": a})

    monkeypatch.setattr("app.database.check_db_connection", lambda: True)
    monkeypatch.setattr("app.integrations.get_redis_queue",
                        lambda: types.SimpleNamespace(health_check=lambda: True))

    alpaca_up = {"v": False}
    monkeypatch.setattr("app.integrations.get_alpaca_client",
                        lambda: types.SimpleNamespace(health_check=lambda: alpaca_up["v"]))

    asyncio.run(o._health_check())                              # Alpaca DOWN -> auto-pause
    assert a.status == "paused" and o._auto_paused is True
    alpaca_up["v"] = True
    asyncio.run(o._health_check())                              # Alpaca UP -> auto-resume
    assert a.status == "running" and o._auto_paused is False


def test_manual_pause_is_not_auto_resumed(monkeypatch):
    from app import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.scheduler, "pause_job", lambda *a, **k: None)
    monkeypatch.setattr(orch_mod.scheduler, "resume_job", lambda *a, **k: None)
    a = _agent()
    o = _orch({"trader": a})
    o.pause_trading()                                           # MANUAL pause
    monkeypatch.setattr("app.database.check_db_connection", lambda: True)
    monkeypatch.setattr("app.integrations.get_redis_queue",
                        lambda: types.SimpleNamespace(health_check=lambda: True))
    monkeypatch.setattr("app.integrations.get_alpaca_client",
                        lambda: types.SimpleNamespace(health_check=lambda: True))
    asyncio.run(o._health_check())                              # healthy, but manual pause stands
    assert a.status == "paused"                                 # NOT auto-resumed


# ── intraday daily-loss hard stop ────────────────────────────────────────────────
def test_daily_pnl_uses_live_intraday_equity():
    from app.agents.risk_manager import RiskManager
    rm = RiskManager.__new__(RiskManager)
    rm.logger = __import__("logging").getLogger("t")
    # equity - last_equity = -3000 intraday (DB row would be 0.0 mid-session)
    acct = {"equity": 97_000.0, "last_equity": 100_000.0}
    assert rm._get_daily_pnl(acct) == -3000.0


def test_intraday_force_close_runs_even_when_paused(monkeypatch):
    # A pause (manual or auto) must NEVER leave intraday positions open overnight: the 3:45pm
    # force-close runs via the shared helper in both the running and paused branches.
    import datetime as _dt
    from app.agents.trader import Trader
    t = Trader.__new__(Trader)
    t.logger = __import__("logging").getLogger("t")
    t.status = "paused"
    t._force_closed_today = False
    t._fc_armed_date = None
    calls = {"n": 0}

    async def _fake_fc():
        calls["n"] += 1
    monkeypatch.setattr(t, "_force_close_intraday", _fake_fc)

    due = _dt.datetime(2026, 6, 22, 15, 46)        # Monday 15:46 ET — past the 15:45 trigger
    asyncio.run(t._maybe_force_close_intraday(due))
    assert calls["n"] == 1 and t._force_closed_today is True
    asyncio.run(t._maybe_force_close_intraday(due))  # already done today -> no repeat
    assert calls["n"] == 1
    # before the trigger -> not called
    t2 = Trader.__new__(Trader)
    t2.logger = t.logger
    t2._force_closed_today = False
    t2._fc_armed_date = None
    monkeypatch.setattr(t2, "_force_close_intraday", _fake_fc)
    asyncio.run(t2._maybe_force_close_intraday(_dt.datetime(2026, 6, 22, 10, 0)))
    assert calls["n"] == 1                            # still 1 — not due at 10:00


def test_daily_pnl_falls_back_to_db_when_account_unavailable(monkeypatch):
    from app.agents.risk_manager import RiskManager
    rm = RiskManager.__new__(RiskManager)
    rm.logger = __import__("logging").getLogger("t")
    monkeypatch.setattr(rm, "_fetch_account_state", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    class _Q:
        def filter_by(self, **k):
            return self

        def first(self):
            return types.SimpleNamespace(daily_pnl=-1234.0)

    class _DB:
        def query(self, *a):
            return _Q()

        def close(self):
            pass
    monkeypatch.setattr("app.agents.risk_manager.get_session", lambda: _DB())
    assert rm._get_daily_pnl() == -1234.0                       # live read failed -> DB realized
