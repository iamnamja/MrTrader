"""Macro-banner staleness fix — /api/nis/macro must flag whether the served context is
actually TODAY's (is_today) so the dashboard never labels a prior-day snapshot "today".

Bug: before the 09:00 ET premarket run the endpoint falls back to the most-recent DB
snapshot (possibly days old) and the UI labelled its events "today", while the Macro Intel
tab showed the current (empty) state — the two disagreed.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from app.api import nis_routes


class _FakePremarket:
    def __init__(self, ctx):
        self.macro_context = ctx


def _live_ctx(as_of):
    return SimpleNamespace(
        as_of=as_of, overall_risk="HIGH", block_new_entries=True,
        global_sizing_factor=0.5, rationale="test", events_today=[])


def _snap(snapshot_date):
    return SimpleNamespace(
        as_of=datetime(2026, 6, 17, 9, 0), overall_risk="HIGH", block_new_entries=True,
        global_sizing_factor=0.5, rationale="test", snapshot_date=snapshot_date,
        events_json=[{"event_type": "FOMC"}])


@contextmanager
def _fake_session(snap):
    class _Q:
        def order_by(self, *_a):
            return self

        def first(self):
            return snap

    class _DB:
        def query(self, *_a):
            return _Q()
    yield _DB()


def test_live_context_is_today():
    with patch.object(nis_routes, "get_session"):
        with patch("app.agents.premarket.premarket_intel", _FakePremarket(_live_ctx(datetime.now()))):
            out = nis_routes.get_macro_context()
    assert out["source"] == "live"
    assert out["is_today"] is True
    assert "snapshot_date" in out


def test_db_snapshot_today_is_today_true():
    with patch("app.agents.premarket.premarket_intel", _FakePremarket(None)):
        with patch.object(nis_routes, "get_session", lambda: _fake_session(_snap(date.today()))):
            out = nis_routes.get_macro_context()
    assert out["source"] == "db_snapshot"
    assert out["is_today"] is True


def test_db_snapshot_stale_is_today_false():
    stale = date.today() - timedelta(days=5)
    with patch("app.agents.premarket.premarket_intel", _FakePremarket(None)):
        with patch.object(nis_routes, "get_session", lambda: _fake_session(_snap(stale))):
            out = nis_routes.get_macro_context()
    assert out["source"] == "db_snapshot"
    assert out["is_today"] is False           # the fix: a prior-day snapshot is flagged stale
    assert out["snapshot_date"] == stale.isoformat()
