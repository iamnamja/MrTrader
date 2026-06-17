"""P1-1 — tests for the set_cash_config applier (the cash-sleeve enable switch).

Verifies the two independent switches (--enable / --arm) resolve to the right flag
values and that the baseline is the safe dormant+shadow state, without touching a real DB.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture()
def scc(monkeypatch):
    mod = importlib.import_module("scripts.set_cash_config")
    importlib.reload(mod)
    store: dict = {}

    # Seed schema defaults so reads before writes return the documented baseline.
    defaults = {
        "pm.cash_enabled": "false", "pm.cash_shadow": "true",
        "pm.cash_buffer_pct": 0.02, "pm.cash_universe": "SGOV,BIL",
        "pm.cash_rebalance_weekday": 0,
    }

    def _get(db, k):
        return store.get(k, defaults.get(k))

    def _set(db, k, v):
        store[k] = v

    monkeypatch.setattr(mod, "get_session", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(mod, "get_agent_config", _get)
    monkeypatch.setattr(mod, "set_agent_config", _set)
    mod._store = store  # expose for assertions
    return mod


def _run(mod, monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["set_cash_config"] + argv)
    mod.main()
    return mod._store


def test_baseline_is_dormant_and_shadow(scc, monkeypatch):
    store = _run(scc, monkeypatch, [])
    assert store["pm.cash_enabled"] == "false"
    assert store["pm.cash_shadow"] == "true"
    # buffer / universe / weekday are written explicitly (auditable live row)
    assert store["pm.cash_buffer_pct"] == 0.02
    assert store["pm.cash_universe"] == "SGOV,BIL"
    assert store["pm.cash_rebalance_weekday"] == 0


def test_enable_runs_in_shadow(scc, monkeypatch):
    store = _run(scc, monkeypatch, ["--enable"])
    assert store["pm.cash_enabled"] == "true"
    assert store["pm.cash_shadow"] == "true"   # still shadow until --arm


def test_enable_arm_goes_live(scc, monkeypatch):
    store = _run(scc, monkeypatch, ["--enable", "--arm"])
    assert store["pm.cash_enabled"] == "true"
    assert store["pm.cash_shadow"] == "false"


def test_arm_without_enable_stays_dormant(scc, monkeypatch):
    # --arm alone disarms shadow but the master flag keeps it dormant (no orders).
    store = _run(scc, monkeypatch, ["--arm"])
    assert store["pm.cash_enabled"] == "false"
    assert store["pm.cash_shadow"] == "false"


def test_show_writes_nothing(scc, monkeypatch):
    store = _run(scc, monkeypatch, ["--show"])
    assert store == {}   # --show is read-only


def test_dry_run_forces_shadow_and_restores_flags(scc, monkeypatch):
    # Pre-set a LIVE config; the dry-run must force shadow for the run then restore exactly.
    scc._store["pm.cash_enabled"] = "true"
    scc._store["pm.cash_shadow"] = "false"

    seen = {}

    def _fake_rebalance(db, force=False):
        seen["force"] = force
        seen["enabled_during"] = scc._store["pm.cash_enabled"]
        seen["shadow_during"] = scc._store["pm.cash_shadow"]
        return {"status": "ok", "mode": "shadow", "action": "deploy", "nav": 100_000,
                "cash_on_hand": 50_000, "buffer": 2000, "deployable": 48_000,
                "approved": [{"symbol": "SGOV", "side": "buy", "qty": 480}]}

    monkeypatch.setattr("app.live_trading.cash_sleeve.run_cash_rebalance", _fake_rebalance)
    _run(scc, monkeypatch, ["--dry-run"])

    assert seen["force"] is True
    assert seen["enabled_during"] == "true"    # forced enabled for the run
    assert seen["shadow_during"] == "true"     # forced shadow -> no orders
    # flags restored to the operator's pre-existing LIVE intent
    assert scc._store["pm.cash_enabled"] == "true"
    assert scc._store["pm.cash_shadow"] == "false"
