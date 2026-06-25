"""Alpha-v10 audit Wave 5h — control-integrity fixes.

Pins:
  Bug A: an OPERATOR (manual) pause is never converted to an auto-pause by a health-check Alpaca
         blip, and is NEVER auto-resumed when Alpaca recovers (only an explicit resume lifts it).
  Bug B: the /health endpoint reports the kill-switch as a LATCH of (in-memory OR DB-persisted)
         active — so subprocess-mode staleness can't show 'trading' while the daemon has halted.
"""
from __future__ import annotations

import asyncio
import types


# ── Bug A: manual pause survives an Alpaca blip + recovery ────────────────────────
def _orch():
    from app.orchestrator import AgentOrchestrator
    o = AgentOrchestrator.__new__(AgentOrchestrator)
    o._auto_paused = False
    o._manual_paused = False
    o.agents = {"trader": types.SimpleNamespace(status="running")}
    return o


def _patch_scheduler(monkeypatch):
    import app.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.scheduler, "pause_job", lambda *a, **k: None)
    monkeypatch.setattr(orch_mod.scheduler, "resume_job", lambda *a, **k: None)


def test_manual_pause_not_auto_resumed(monkeypatch):
    _patch_scheduler(monkeypatch)
    o = _orch()
    o.pause_trading(auto=False)                 # operator pause
    assert o._manual_paused is True and o._auto_paused is False
    assert o.agents["trader"].status == "paused"

    # Alpaca goes down then recovers while manually paused
    async def _run():
        from app.database import check_db_connection  # noqa: F401
        # down: must NOT convert the manual pause into an auto-pause
        monkeypatch.setattr("app.database.check_db_connection", lambda: True)
        monkeypatch.setattr("app.integrations.get_redis_queue",
                            lambda: types.SimpleNamespace(health_check=lambda: True))
        monkeypatch.setattr("app.integrations.get_alpaca_client",
                            lambda: types.SimpleNamespace(health_check=lambda: False))
        await o._health_check()
        assert o._auto_paused is False           # NOT converted to auto
        assert o._manual_paused is True
        # recover: must NOT auto-resume the operator's halt
        monkeypatch.setattr("app.integrations.get_alpaca_client",
                            lambda: types.SimpleNamespace(health_check=lambda: True))
        await o._health_check()
        assert o.agents["trader"].status == "paused"   # still paused
        assert o._manual_paused is True
    asyncio.run(_run())

    # only an explicit resume lifts it
    o.resume_trading()
    assert o._manual_paused is False and o._auto_paused is False
    assert o.agents["trader"].status == "running"


def test_auto_pause_does_auto_resume(monkeypatch):
    _patch_scheduler(monkeypatch)
    o = _orch()

    async def _run():
        monkeypatch.setattr("app.database.check_db_connection", lambda: True)
        monkeypatch.setattr("app.integrations.get_redis_queue",
                            lambda: types.SimpleNamespace(health_check=lambda: True))
        # Alpaca down (from running) -> auto-pause
        monkeypatch.setattr("app.integrations.get_alpaca_client",
                            lambda: types.SimpleNamespace(health_check=lambda: False))
        await o._health_check()
        assert o._auto_paused is True and o._manual_paused is False
        assert o.agents["trader"].status == "paused"
        # Alpaca recovers -> auto-resume (no operator pause in force)
        monkeypatch.setattr("app.integrations.get_alpaca_client",
                            lambda: types.SimpleNamespace(health_check=lambda: True))
        await o._health_check()
        assert o._auto_paused is False
        assert o.agents["trader"].status == "running"
    asyncio.run(_run())


# ── Bug B: /health latches kill-switch active from in-memory OR the DB ─────────────
def test_health_latches_killswitch_from_db(monkeypatch):
    from app.api import routes

    monkeypatch.setattr(routes, "check_db_connection", lambda: True)
    monkeypatch.setattr(routes, "_redis",
                        lambda: types.SimpleNamespace(health_check=lambda: True))
    monkeypatch.setattr(routes, "_alpaca",
                        lambda: types.SimpleNamespace(health_check=lambda: True))
    monkeypatch.setattr(routes, "get_session",
                        lambda: types.SimpleNamespace(close=lambda: None))

    # in-memory singleton says INACTIVE (stale web process), but the DB says ACTIVE (daemon halted)
    from app.live_trading import kill_switch as ks_mod
    monkeypatch.setattr(ks_mod.kill_switch, "_active", False, raising=False)
    monkeypatch.setattr("app.database.config_store.get_config",
                        lambda db, key: True if key == ks_mod._CFG_KS_ACTIVE else None)

    out = asyncio.run(routes.get_health_alias.__wrapped__())
    assert out["kill_switch_active"] is True       # latched from DB despite stale singleton
    assert out["status"] == "halted"


def test_health_killswitch_inactive_when_both_clear(monkeypatch):
    from app.api import routes
    monkeypatch.setattr(routes, "check_db_connection", lambda: True)
    monkeypatch.setattr(routes, "_redis",
                        lambda: types.SimpleNamespace(health_check=lambda: True))
    monkeypatch.setattr(routes, "_alpaca",
                        lambda: types.SimpleNamespace(health_check=lambda: True))
    monkeypatch.setattr(routes, "get_session",
                        lambda: types.SimpleNamespace(close=lambda: None))
    from app.live_trading import kill_switch as ks_mod
    monkeypatch.setattr(ks_mod.kill_switch, "_active", False, raising=False)
    monkeypatch.setattr("app.database.config_store.get_config", lambda db, key: False)

    out = asyncio.run(routes.get_health_alias.__wrapped__())
    assert out["kill_switch_active"] is False
    assert out["status"] == "healthy"
