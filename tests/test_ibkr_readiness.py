"""Tests for the IBKR migration readiness probe (R1.2 tiny-live pre-flight).

Pins the stage logic + the blocker list so a silent gap (e.g. ib_insync unimportable -> the R1.1
shadow router skipping) surfaces at boot instead of on a Monday rebalance. The probe is read-only:
it places no order and opens no gateway session (only a TCP reachability check)."""
from __future__ import annotations

from app.live_trading import ibkr_readiness as r


def _patch(monkeypatch, *, ib_ok=True, shadow_ok=True, reachable=False, account=None,
           read_only=None, shadow_routing="true"):
    monkeypatch.setattr(r, "_ib_insync_info",
                        lambda: {"available": ib_ok, "version": "0.9.86" if ib_ok else None})
    monkeypatch.setattr(r, "_shadow_selftest", lambda db: {"ok": shadow_ok, "n": 1})
    monkeypatch.setattr(r, "_gateway_reachable", lambda host, port, timeout=1.5: reachable)
    cfg = {"ibkr.host": "127.0.0.1", "ibkr.port": "7497", "ibkr.account": account,
           "ibkr.read_only": read_only, "ibkr.shadow_routing": shadow_routing,
           "ibkr.enabled": "false"}
    monkeypatch.setattr(r, "_cfg", lambda db, key: cfg.get(key))


def test_blocked_when_ib_insync_missing(monkeypatch):
    _patch(monkeypatch, ib_ok=False)
    rep = r.probe()
    assert rep["stage"] == r.BLOCKED
    assert any("ib_insync" in b for b in rep["blockers"])


def test_blocked_when_shadow_selftest_fails(monkeypatch):
    _patch(monkeypatch, ib_ok=True, shadow_ok=False)
    rep = r.probe()
    assert rep["stage"] == r.BLOCKED
    assert any("reconstruction" in b for b in rep["blockers"])


def test_shadow_ready_when_gateway_down(monkeypatch):
    # the current real state: ib_insync + shadow OK, gateway not up yet
    _patch(monkeypatch, reachable=False)
    rep = r.probe()
    assert rep["stage"] == r.R1_1_SHADOW_READY
    assert rep["shadow_routing_flag"] is True
    assert any("not reachable" in b for b in rep["blockers"])


def test_gateway_up_but_no_account(monkeypatch):
    _patch(monkeypatch, reachable=True, account=None)
    rep = r.probe()
    assert rep["stage"] == r.GATEWAY_UP
    assert any("account" in b for b in rep["blockers"])


def test_ready_for_read_only_off_smoke(monkeypatch):
    # gateway up + account set -> ready for the owner R1.0c-2b step; read_only ON is flagged
    _patch(monkeypatch, reachable=True, account="DU123456", read_only="true")
    rep = r.probe()
    assert rep["stage"] == r.R1_0C2B_READY
    assert any("Read-Only" in b for b in rep["blockers"])


def test_read_only_defaults_on_when_unset(monkeypatch):
    _patch(monkeypatch, reachable=True, account="DU1", read_only=None)
    rep = r.probe()
    assert rep["read_only"] is True                    # unset -> the safe blocks-orders default


def test_format_line_is_a_one_liner(monkeypatch):
    _patch(monkeypatch, reachable=False)
    line = r.format_line(r.probe())
    assert "IBKR readiness" in line and "stage=" in line and "ib_insync 0.9.86" in line


def test_probe_never_raises(monkeypatch):
    monkeypatch.setattr(r, "_cfg", lambda db, key: (_ for _ in ()).throw(RuntimeError("boom")))
    # even with a broken config read, the probe must return a dict, never raise
    rep = r.probe()
    assert "stage" in rep
