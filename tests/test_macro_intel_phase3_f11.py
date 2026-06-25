"""Macro Intel Phase 3 F11 — idempotent skip-audit on repeated manual PM triggers.

A repeated trigger producing the SAME (symbol, strategy, reason) on the same ET day must NOT pile
up duplicate pm_skip rows. The dedup keys on the FULL reason (exact match), so distinct intraday
windows (kill_switch:09:45 vs :10:45) each still record, kill_switch never aliases kill_switch_late,
and it fails OPEN so a check error can never suppress a genuinely new abstention.
"""
from __future__ import annotations

import logging


# ── _write_skip_audit honours skip_audit_exists ─────────────────────────────────
def _pm():
    from app.agents.portfolio_manager import PortfolioManager
    pm = PortfolioManager.__new__(PortfolioManager)
    pm.logger = logging.getLogger("t")
    return pm


def _capture_write(monkeypatch):
    calls = []
    monkeypatch.setattr("app.database.decision_audit.write_decision",
                        lambda **k: calls.append(k))
    return calls


def test_blanket_duplicate_suppressed(monkeypatch):
    pm = _pm()
    calls = _capture_write(monkeypatch)
    # category already recorded today → no duplicate '*' row
    monkeypatch.setattr("app.database.decision_audit.skip_audit_exists", lambda *a: True)
    pm._write_skip_audit("swing", "kill_switch", [{"symbol": "AAPL", "confidence": 0.7}])
    assert calls == []


def test_blanket_first_writes(monkeypatch):
    pm = _pm()
    calls = _capture_write(monkeypatch)
    monkeypatch.setattr("app.database.decision_audit.skip_audit_exists", lambda *a: False)
    pm._write_skip_audit("swing", "kill_switch", [{"symbol": "AAPL", "confidence": 0.7}])
    assert len(calls) == 1 and calls[0]["symbol"] == "*"


def test_per_symbol_partial_dedup(monkeypatch):
    pm = _pm()
    calls = _capture_write(monkeypatch)
    # AAPL already recorded today, MSFT not → only MSFT writes
    monkeypatch.setattr("app.database.decision_audit.skip_audit_exists",
                        lambda sym, strat, base: sym == "AAPL")
    proposals = [{"symbol": "AAPL", "confidence": 0.6, "entry_price": 1.0},
                 {"symbol": "MSFT", "confidence": 0.6, "entry_price": 1.0}]
    pm._write_skip_audit("swing", "entry_gate", proposals)
    assert [c["symbol"] for c in calls] == ["MSFT"]


# ── skip_audit_exists base-match semantics ──────────────────────────────────────
class _FakeSession:
    def __init__(self, reasons):
        self._reasons = reasons

    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def all(self):
        return [(r,) for r in self._reasons]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_exists_exact_full_reason_match(monkeypatch):
    # exact full-reason match: same reason → True; a different reason → False
    from app.database import decision_audit as da
    monkeypatch.setattr("app.database.session.get_session",
                        lambda: _FakeSession(["pm_skip: opportunity_score_low:0.3"]))
    assert da.skip_audit_exists("*", "swing", "opportunity_score_low:0.3") is True
    assert da.skip_audit_exists("*", "swing", "opportunity_score_low:0.4") is False
    assert da.skip_audit_exists("*", "swing", "kill_switch") is False


def test_exists_distinct_windows_not_deduped(monkeypatch):
    # the HIGH fix: a 09:45 kill_switch must NOT suppress the 10:45 one (different window suffix)
    from app.database import decision_audit as da
    monkeypatch.setattr("app.database.session.get_session",
                        lambda: _FakeSession(["pm_skip: kill_switch:09:45"]))
    assert da.skip_audit_exists("*", "intraday", "kill_switch:09:45") is True   # repeat → deduped
    assert da.skip_audit_exists("*", "intraday", "kill_switch:10:45") is False  # new window → records


def test_exists_no_alias(monkeypatch):
    # kill_switch must NOT alias kill_switch_late (exact match)
    from app.database import decision_audit as da
    monkeypatch.setattr("app.database.session.get_session",
                        lambda: _FakeSession(["pm_skip: kill_switch_late:win"]))
    assert da.skip_audit_exists("*", "intraday", "kill_switch:win") is False
    assert da.skip_audit_exists("*", "intraday", "kill_switch_late:win") is True


def test_exists_fail_open(monkeypatch):
    # any error → False (never suppress a real abstention)
    from app.database import decision_audit as da

    def _boom():
        raise RuntimeError("db down")
    monkeypatch.setattr("app.database.session.get_session", _boom)
    assert da.skip_audit_exists("*", "swing", "kill_switch") is False
