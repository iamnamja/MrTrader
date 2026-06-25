"""Macro Intel Phase 0 — Decision-Linkage correctness (F1 date-scope + F4 skip-row collapse).

F4: blanket strategy-level abstentions (kill_switch, opportunity_score_low, macro_event_window, …)
    write ONE symbol='*' row instead of one-per-proposal (the "phantom AAPL swing" spam). Per-symbol
    gates still write one row each.
F1: /api/decision-audit/recent date-scopes to the last N ET days (default today) so a stale skip
    batch from a prior session / kill-switch window no longer surfaces forever.
"""
from __future__ import annotations

import logging


# ── F4: blanket skip reasons collapse to one '*' row ─────────────────────────────
def _pm():
    from app.agents.portfolio_manager import PortfolioManager
    pm = PortfolioManager.__new__(PortfolioManager)
    pm.logger = logging.getLogger("t")
    return pm


def _capture_write(monkeypatch):
    calls = []
    monkeypatch.setattr("app.database.decision_audit.write_decision",
                        lambda **k: calls.append(k))
    # F11 added an idempotency guard that reads the DB; these tests assert the COLLAPSE logic,
    # so disable dedup (return False) to keep them isolated from any rows already in the dev DB.
    monkeypatch.setattr("app.database.decision_audit.skip_audit_exists", lambda *a: False)
    return calls


def test_blanket_kill_switch_collapses_to_one_row(monkeypatch):
    pm = _pm()
    calls = _capture_write(monkeypatch)
    proposals = [{"symbol": s, "confidence": 0.7} for s in ("AAPL", "MSFT", "NVDA")]
    pm._write_skip_audit("swing", "kill_switch", proposals)
    assert len(calls) == 1                       # one summary row, not 3
    assert calls[0]["symbol"] == "*"
    assert calls[0]["block_reason"] == "pm_skip: kill_switch"


def test_blanket_with_suffix_still_collapses(monkeypatch):
    pm = _pm()
    calls = _capture_write(monkeypatch)
    proposals = [{"symbol": s, "confidence": 0.7} for s in ("AAPL", "MSFT")]
    # reason carries a ':' suffix (time window / score) — prefix still matches the blanket set
    pm._write_skip_audit("swing", "opportunity_score_low:0.41", proposals)
    assert len(calls) == 1 and calls[0]["symbol"] == "*"
    pm._write_skip_audit("swing", "macro_event_window:FOMC,CPI", proposals)
    assert len(calls) == 2 and calls[1]["symbol"] == "*"


def test_per_symbol_gate_still_writes_each(monkeypatch):
    pm = _pm()
    calls = _capture_write(monkeypatch)
    proposals = [{"symbol": s, "confidence": 0.7, "entry_price": 100.0} for s in ("AAPL", "MSFT", "NVDA")]
    # a NON-blanket (genuinely per-symbol) reason keeps one row per symbol
    pm._write_skip_audit("swing", "entry_gate", proposals)
    assert len(calls) == 3
    assert {c["symbol"] for c in calls} == {"AAPL", "MSFT", "NVDA"}


def test_no_proposals_writes_single_star(monkeypatch):
    pm = _pm()
    calls = _capture_write(monkeypatch)
    pm._write_skip_audit("intraday", "model_not_trained")
    assert len(calls) == 1 and calls[0]["symbol"] == "*"


# ── F1: /recent date-scopes to today (ET) by default ─────────────────────────────
class _FakeQuery:
    def __init__(self, sink):
        self.sink = sink

    def order_by(self, *a, **k):
        return self

    def filter(self, *a, **k):
        self.sink.append("filter")
        return self

    def limit(self, *a, **k):
        return self

    def all(self):
        return []


def _patch_session(monkeypatch, sink):
    from app.api import nis_routes

    class _DB:
        def query(self, *a, **k):
            return _FakeQuery(sink)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(nis_routes, "get_session", lambda: _DB())


def test_recent_applies_date_filter_by_default(monkeypatch):
    # NB: called directly (not via FastAPI), so pass resolved values not Query() defaults
    from app.api import nis_routes
    sink = []
    _patch_session(monkeypatch, sink)
    nis_routes.get_recent_decisions(limit=50, strategy=None, final_decision=None, days=1)
    assert "filter" in sink                              # a decided_at date filter was applied


def test_recent_days_zero_disables_date_filter(monkeypatch):
    from app.api import nis_routes
    sink = []
    _patch_session(monkeypatch, sink)
    nis_routes.get_recent_decisions(limit=50, strategy=None, final_decision=None, days=0)
    assert "filter" not in sink                          # no date/strategy/decision filter applied
