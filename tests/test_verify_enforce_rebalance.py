"""Tests for the enforce-rebalance verification (scripts/verify_enforce_rebalance.py).

Pins the report logic: enforce config OK, the un-backfillable CH0b scorecard-capture detection,
and the spurious-enforce-HOLD detection."""
from __future__ import annotations

import json
from datetime import date

import pytest

from scripts import verify_enforce_rebalance as ver


class _Q:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _DB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        return _Q(self._rows)

    def close(self):
        pass


class _Session:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return _DB(self._rows)

    def __exit__(self, *a):
        return False


class _Row:
    def __init__(self, final_decision, block_reason=None):
        self.strategy, self.final_decision, self.block_reason = "trend", final_decision, block_reason


def _patch(monkeypatch, *, config, scorecard_rows, decision_rows):
    monkeypatch.setattr("app.database.agent_config.get_agent_config",
                        lambda db, k: config.get(k))
    monkeypatch.setattr("app.database.session.get_session", lambda: _Session(decision_rows))
    monkeypatch.setattr("app.live_trading.back_validation.read_daily",
                        lambda since=None: scorecard_rows)


_ENFORCE = {"pm.whole_book_gate_mode": "enforce", "pm.reconciliation_mode": "enforce",
            "pm.per_name_gate_mode": "shadow", "pm.trend_enabled": "true", "pm.trend_shadow": "false"}


def _good_row():
    return {"trade_date": date.today().isoformat(),
            "intended_weights": json.dumps({"SPY": 0.2, "QQQ": 0.2}),
            "ungoverned_weights": json.dumps({"SPY": 0.4, "QQQ": 0.4}),
            "crash_mult": 1.0, "credit_mult": 1.0, "ladder_mult": 1.0, "overlay_mult": 1.0,
            "n_blocked": 0}


def test_clean_enforce_rebalance_is_ok(monkeypatch):
    _patch(monkeypatch, config=_ENFORCE, scorecard_rows=[_good_row()],
           decision_rows=[_Row("enter"), _Row("enter")])
    rep = ver.check()
    assert rep["status"] == "OK" and not rep["attention"]
    assert rep["scorecard"]["n_ungoverned"] == 2 and rep["scorecard"]["crash_mult"] == 1.0


def test_missing_ungoverned_weights_flags_unbackfillable(monkeypatch):
    row = _good_row()
    row["ungoverned_weights"] = None                    # CH0b counterfactual NOT captured
    _patch(monkeypatch, config=_ENFORCE, scorecard_rows=[row], decision_rows=[_Row("enter")])
    rep = ver.check()
    assert rep["status"] == "ATTENTION"
    assert any("ungoverned_weights MISSING" in a for a in rep["attention"])


def test_no_scorecard_row_flags(monkeypatch):
    _patch(monkeypatch, config=_ENFORCE, scorecard_rows=[], decision_rows=[])
    rep = ver.check()
    assert rep["status"] == "ATTENTION"
    assert any("no scorecard row" in a for a in rep["attention"])


def test_enforce_hold_is_flagged(monkeypatch):
    _patch(monkeypatch, config=_ENFORCE, scorecard_rows=[],
           decision_rows=[_Row("block", "whole_book_gate")])
    rep = ver.check()
    assert any("ENFORCE HOLD" in a and "whole_book_gate" in a for a in rep["attention"])


def test_config_not_enforce_is_flagged(monkeypatch):
    cfg = dict(_ENFORCE, **{"pm.whole_book_gate_mode": "shadow"})   # reverted / never flipped
    _patch(monkeypatch, config=cfg, scorecard_rows=[_good_row()], decision_rows=[_Row("enter")])
    rep = ver.check()
    assert any("whole_book_gate_mode" in a for a in rep["attention"])


def test_verification_event_is_registered_so_email_actually_sends():
    # the email is the job's whole purpose — an unregistered event_type is SILENTLY dropped by
    # notifier.enqueue (returns None, no raise). Pin that it's registered + renders a real subject.
    from app.notifications import notifier
    assert "enforce_rebalance_verification" in notifier.VALID_EVENTS
    subj, body = notifier.render("enforce_rebalance_verification", {
        "date": "2026-07-13", "status": "ATTENTION", "config": {"pm.whole_book_gate_mode": "enforce"},
        "scorecard": {"present": True, "n_intended": 5, "n_ungoverned": 5, "crash_mult": 1.0,
                      "overlay_mult": 1.0}, "decisions": {"n_total": 5}, "attention": ["x missing"]})
    assert "Enforce-rebalance verify" in subj and body
