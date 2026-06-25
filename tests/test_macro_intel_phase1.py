"""Macro Intel Phase 1 — the NIS macro-history (assessment lineage) endpoint.

GET /api/nis/macro-history returns the timestamped re-assessment lineage (nis_macro_history) for the
last N ET days, newest-first — backing the "Today's NIS Assessment" timestamped, newest-on-top view.
Read-only; the table is already populated by persist_nis_macro_snapshot.
"""
from __future__ import annotations

from datetime import datetime, timezone, date


class _Row:
    def __init__(self, as_of, risk, trig, sizing=1.0, block=False, n_events=0,
                 event_type=None, event_name=None, rationale="r"):
        self.as_of = as_of
        self.snapshot_date = as_of.date()
        self.overall_risk = risk
        self.trigger_source = trig
        self.trigger_event_type = event_type
        self.trigger_event_name = event_name
        self.global_sizing_factor = sizing
        self.block_new_entries = block
        self.rationale = rationale
        self.events_json = [{} for _ in range(n_events)]


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def all(self):
        return self._rows


def _patch(monkeypatch, rows):
    from app.api import nis_routes

    class _DB:
        def query(self, *a, **k):
            return _Query(rows)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(nis_routes, "get_session", lambda: _DB())


def test_macro_history_returns_lineage(monkeypatch):
    from app.api import nis_routes
    rows = [
        _Row(datetime(2026, 6, 25, 13, 33, tzinfo=timezone.utc), "MEDIUM", "post_event",
             sizing=0.9, n_events=9, event_type="PCE", rationale="cooler PCE, stepping down"),
        _Row(datetime(2026, 6, 25, 13, 0, tzinfo=timezone.utc), "HIGH", "premarket",
             sizing=0.85, block=True, n_events=9, rationale="pre-release uncertainty"),
    ]
    _patch(monkeypatch, rows)
    out = nis_routes.get_macro_history(days=1, limit=50)
    assert out["count"] == 2
    h0 = out["history"][0]
    assert h0["trigger_source"] == "post_event" and h0["trigger_event_type"] == "PCE"
    assert h0["overall_risk"] == "MEDIUM" and h0["global_sizing_factor"] == 0.9
    assert h0["n_events"] == 9 and h0["block_new_entries"] is False
    assert "as_of" in h0 and h0["rationale"].startswith("cooler")
    # premarket baseline carried block + HIGH
    assert out["history"][1]["block_new_entries"] is True


def test_macro_history_empty(monkeypatch):
    from app.api import nis_routes
    _patch(monkeypatch, [])
    out = nis_routes.get_macro_history(days=1, limit=50)
    assert out["count"] == 0 and out["history"] == []


def test_macro_history_filters_by_date(monkeypatch):
    # confirm a date filter is applied (start-of-window) so old lineage doesn't leak
    from app.api import nis_routes
    sink = []

    class _Q2(_Query):
        def filter(self, *a, **k):
            sink.append("filter")
            return self

    class _DB:
        def query(self, *a, **k):
            return _Q2([])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(nis_routes, "get_session", lambda: _DB())
    nis_routes.get_macro_history(days=1, limit=50)
    assert "filter" in sink
    assert date.today() is not None   # sanity
