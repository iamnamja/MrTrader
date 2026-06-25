"""Phase H — H2: kill-switch state-machine wiring (shadow-first).

The R0.4 state machine is now wired: a module singleton (`kill_switch_sm`), a consult helper
(`evaluate_new_risk`, fail-safe), the recon-FAIL_CLOSED auto-trigger fed ONLY when reconciliation is
in ENFORCE (so shadow false-breaks can't latch it), and a dead-man heartbeat. Default mode is
'shadow' → the entry gates log but never block (no live change).
"""
from __future__ import annotations

import logging

import pytest

from app.live_trading import kill_switch_state as ksm
from app.live_trading.kill_switch_state import (
    evaluate_new_risk, kill_switch_sm, NORMAL, HALT_NEW_RISK,
)


@pytest.fixture(autouse=True)
def _reset_sm():
    """The singleton is module-level — reset to NORMAL + fresh heartbeat around each test."""
    kill_switch_sm.set_state(NORMAL, reason="test-reset", actor="test", manual=True)
    kill_switch_sm.heartbeat()
    yield
    kill_switch_sm.set_state(NORMAL, reason="test-reset", actor="test", manual=True)
    kill_switch_sm.heartbeat()


# ── evaluate_new_risk: the consult contract ──────────────────────────────────────
def test_off_mode_never_blocks():
    kill_switch_sm.set_state(HALT_NEW_RISK, reason="x", actor="t", manual=True)
    r = evaluate_new_risk("off", label="t")
    assert r["allow"] is True and r["would_block"] is False


def test_normal_state_allows_in_all_modes():
    for m in ("shadow", "enforce", "off"):
        r = evaluate_new_risk(m, label="t")
        assert r["allow"] is True and r["would_block"] is False and r["state"] == NORMAL


def test_shadow_logs_but_allows_when_halted():
    kill_switch_sm.set_state(HALT_NEW_RISK, reason="x", actor="t", manual=True)
    r = evaluate_new_risk("shadow", label="t", logger=logging.getLogger("t"))
    assert r["allow"] is True            # shadow never blocks
    assert r["would_block"] is True      # but records that it WOULD
    assert r["state"] == HALT_NEW_RISK


def test_enforce_blocks_when_halted():
    kill_switch_sm.set_state(HALT_NEW_RISK, reason="x", actor="t", manual=True)
    r = evaluate_new_risk("enforce", label="t")
    assert r["allow"] is False and r["would_block"] is True


def test_fail_safe_allows_on_error(monkeypatch):
    # a state-machine bug must never halt live trading -> allow
    monkeypatch.setattr(ksm.kill_switch_sm, "can_increase_risk",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    r = evaluate_new_risk("enforce", label="t")
    assert r["allow"] is True


def test_dead_man_escalates_then_enforce_blocks(monkeypatch):
    # stale heartbeat -> dead_man_check inside evaluate_new_risk escalates to HALT -> enforce blocks
    kill_switch_sm._last_heartbeat = 0.0   # ancient
    r = evaluate_new_risk("enforce", label="t")
    assert kill_switch_sm.state == HALT_NEW_RISK
    assert r["allow"] is False
    # auto can never exceed CANCEL_ONLY (never flatten)
    from app.live_trading.kill_switch_state import severity, CANCEL_ONLY
    assert severity(kill_switch_sm.state) <= severity(CANCEL_ONLY)


# ── recon-fail auto-trigger: fed ONLY in enforce ─────────────────────────────────
def _phantom_db():
    from tests.test_reconciliation_before_trade import FakeDB, _trade
    return FakeDB([_trade("SPY", "BUY", 150)])   # DB holds SPY; broker (below) shows nothing -> break


def test_recon_fail_escalates_sm_in_enforce():
    from app.live_trading import reconciliation as rec
    r = rec.shadow_reconcile_before_trade(_phantom_db(), [], nav=1e5, mode=rec.ENFORCE, label="trend")
    assert r.status == rec.FAIL_CLOSED
    assert kill_switch_sm.state == HALT_NEW_RISK      # enforce break latched the machine


def test_recon_fail_does_not_escalate_sm_in_shadow():
    from app.live_trading import reconciliation as rec
    r = rec.shadow_reconcile_before_trade(_phantom_db(), [], nav=1e5, mode=rec.SHADOW, label="trend")
    assert r.status == rec.FAIL_CLOSED
    assert kill_switch_sm.state == NORMAL             # shadow break must NOT latch the machine
