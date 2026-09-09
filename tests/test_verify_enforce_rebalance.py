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
            "pm.per_name_gate_mode": "enforce",   # CH1 flipped 2026-09-07
            "pm.trend_enabled": "true", "pm.trend_shadow": "false"}


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


# ── 2026-09-09: the config check must be one-directional ──────────────────────
#
# EXPECT is a hand-maintained literal tracking a LIVE-TUNABLE DB config, and it drifted the
# moment CH1 flipped the per-name gate shadow -> enforce: the gate got STRONGER, the literal
# still said "shadow", and the weekly enforce-health email reported ATTENTION on a healthy
# book (observed 2026-09-08). A health check that cries wolf weekly trains the owner to
# ignore it. These pin the fix: flag WEAKER-than-intended, never merely different.

def _cfg(**over):
    c = dict(_ENFORCE)
    c.update(over)
    return c


def test_a_gate_weaker_than_intended_is_flagged(monkeypatch):
    from scripts.verify_enforce_rebalance import check

    _patch(monkeypatch, config=_cfg(**{"pm.per_name_gate_mode": "shadow"}),
           scorecard_rows=[_good_row()], decision_rows=[_Row("enter")])
    rep = check()
    assert any("WEAKER" in a and "per_name_gate_mode" in a for a in rep["attention"])


def test_a_gate_turned_off_is_flagged(monkeypatch):
    from scripts.verify_enforce_rebalance import check

    _patch(monkeypatch, config=_cfg(**{"pm.whole_book_gate_mode": "off"}),
           scorecard_rows=[_good_row()], decision_rows=[_Row("enter")])
    rep = check()
    assert any("WEAKER" in a and "whole_book_gate_mode" in a for a in rep["attention"])


def test_a_gate_stronger_than_recorded_is_reported_not_flagged(monkeypatch):
    """THE REGRESSION. Pretend the literal still lags a future flip: the gate is stronger
    than recorded, which is not a health problem and must not raise ATTENTION."""
    import scripts.verify_enforce_rebalance as v

    monkeypatch.setitem(v.EXPECT_MIN_MODE, "pm.per_name_gate_mode", "shadow")
    _patch(monkeypatch, config=_cfg(**{"pm.per_name_gate_mode": "enforce"}),
           scorecard_rows=[_good_row()], decision_rows=[_Row("enter")])
    rep = v.check()
    assert not any("per_name_gate_mode" in a for a in rep["attention"])
    assert any("per_name_gate_mode" in s for s in rep.get("stronger_than_expected", []))


def test_an_unrecognised_mode_is_flagged(monkeypatch):
    from scripts.verify_enforce_rebalance import check

    _patch(monkeypatch, config=_cfg(**{"pm.reconciliation_mode": "banana"}),
           scorecard_rows=[_good_row()], decision_rows=[_Row("enter")])
    rep = check()
    assert any("not a recognised mode" in a for a in rep["attention"])


def test_boolean_flags_are_still_compared_exactly(monkeypatch):
    """trend_enabled / trend_shadow are booleans, not a ladder — no ranking for them."""
    from scripts.verify_enforce_rebalance import check

    _patch(monkeypatch, config=_cfg(**{"pm.trend_shadow": "true"}),
           scorecard_rows=[_good_row()], decision_rows=[_Row("enter")])
    rep = check()
    assert any("trend_shadow" in a for a in rep["attention"])


def test_the_live_posture_from_2026_09_08_is_clean(monkeypatch):
    """The exact config that produced a false ATTENTION on the first CH1 enforce Monday."""
    from scripts.verify_enforce_rebalance import check

    _patch(monkeypatch, config=_ENFORCE, scorecard_rows=[_good_row()],
           decision_rows=[_Row("enter")])
    rep = check()
    assert rep["attention"] == []


def test_expect_min_mode_covers_every_gate_the_hold_reasons_name():
    """A gate that can HOLD a rebalance but has no expected posture is unmonitored."""
    from scripts.verify_enforce_rebalance import EXPECT_MIN_MODE, HOLD_REASONS

    monitored = {k.removeprefix("pm.").removesuffix("_mode") for k in EXPECT_MIN_MODE}
    for reason in ("whole_book_gate", "reconciliation", "per_name_gate"):
        assert reason in HOLD_REASONS
        assert reason in monitored or reason.replace("_gate", "") in monitored
