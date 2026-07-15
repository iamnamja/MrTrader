"""Macro-Intel actuals display back-fill (fix for the '08:30 event stuck Pending' gap).

The macro snapshot is built once ~09:00 ET, so an event released later that day stays actual=None
-> 'Pending'. The display back-fills the actual on read (include_past_today fetch + _enrich_event),
without touching the cached trading decision."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.api import nis_routes as nr
from app.news.sources import fmp_source as fmp


# ── _enrich_event (pure display back-fill) ────────────────────────────────────
def test_enrich_fills_pending_actual_and_reclassifies():
    actuals = {"JOBLESS_CLAIMS": {"actual": 219.0, "estimate": 218.0, "prior": 215.0}}
    ev = {"event_type": "JOBLESS_CLAIMS", "actual": None, "estimate": 218.0, "prior": 215.0}
    out = nr._enrich_event(ev, actuals)
    assert out["actual"] == 219.0
    assert out["market_outcome"] != "pending"          # now classified, not Pending


def test_enrich_leaves_pending_when_no_actual_available():
    ev = {"event_type": "FOMC", "actual": None, "estimate": None, "prior": None}
    out = nr._enrich_event(ev, {})                      # nothing fetched -> stays Pending
    assert out["actual"] is None and out["market_outcome"] == "pending"


def test_enrich_does_not_override_an_existing_actual():
    actuals = {"JOBLESS_CLAIMS": {"actual": 219.0, "estimate": 218.0, "prior": 215.0}}
    ev = {"event_type": "JOBLESS_CLAIMS", "actual": 217.0, "estimate": 218.0, "prior": 215.0}
    out = nr._enrich_event(ev, actuals)
    assert out["actual"] == 217.0                       # keep the already-recorded value


def test_enrich_fills_missing_event_time_utc():
    # a date-less snapshot event gets its full ISO timestamp re-derived from the live calendar
    meta = {"PPI": {"event_time_utc": "2026-07-15T12:30:00+00:00"}}
    ev = {"event_type": "PPI", "event_time": "12:30 UTC", "event_time_utc": "", "actual": None}
    out = nr._enrich_event(ev, meta)
    assert out["event_time_utc"] == "2026-07-15T12:30:00+00:00"


def test_enrich_does_not_override_existing_event_time_utc():
    meta = {"PPI": {"event_time_utc": "2026-07-15T12:30:00+00:00"}}
    ev = {"event_type": "PPI", "event_time_utc": "2026-07-14T12:30:00+00:00", "actual": None}
    out = nr._enrich_event(ev, meta)
    assert out["event_time_utc"] == "2026-07-14T12:30:00+00:00"   # keep what the snapshot recorded


def test_todays_event_actuals_captures_time_for_unreleased(monkeypatch):
    # an unreleased (actual=None) look-ahead event still contributes its full timestamp
    from datetime import datetime, timezone
    monkeypatch.setattr(nr, "_actuals_cache", {})
    monkeypatch.setattr(
        "app.news.sources.fmp_source.fetch_economic_calendar",
        lambda **k: [{"event_type": "PPI", "event_name": "PPI MoM (Jun)", "actual": None,
                      "estimate": 0.0, "prior": 1.1,
                      "event_time": datetime(2026, 7, 15, 12, 30, tzinfo=timezone.utc)}])
    out = nr._todays_event_actuals()
    assert out["PPI"]["event_time_utc"] == "2026-07-15T12:30:00+00:00"
    assert "actual" not in out["PPI"]                    # unreleased -> no actual recorded


def test_todays_event_actuals_never_raises(monkeypatch):
    # a fetch failure must degrade to {} (display simply stays Pending), never 500 the route
    monkeypatch.setattr(nr, "_actuals_cache", {})
    monkeypatch.setattr("app.news.sources.fmp_source.fetch_economic_calendar",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("fmp down")))
    assert nr._todays_event_actuals() == {}


# ── fmp_source include_past_today opt-in ──────────────────────────────────────
def test_include_past_today_keeps_released_event_default_drops(monkeypatch):
    now = datetime.now(timezone.utc)
    if (now - now.replace(hour=0, minute=0, second=0, microsecond=0)) < timedelta(hours=3):
        pytest.skip("too early in the UTC day for a stable 2h-ago-still-today event")
    past = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")   # earlier TODAY, >1h ago

    class _Resp:
        status_code = 200

        def json(self):
            return [{"date": past, "event": "Initial Jobless Claims", "impact": "High",
                     "country": "US", "estimate": 218, "previous": 215, "actual": 219}]

    monkeypatch.setattr(fmp, "_key", lambda: "k")
    monkeypatch.setattr(fmp, "_ECON_CAL_DISABLED", False, raising=False)
    monkeypatch.setattr(fmp.requests, "get", lambda *a, **k: _Resp())

    # default: a >1h-past event is dropped (trading-gate view unchanged)
    assert fmp.fetch_economic_calendar(days_ahead=0, min_impact="low") == []
    # include_past_today: kept, with its now-available actual
    got = fmp.fetch_economic_calendar(days_ahead=0, min_impact="low", include_past_today=True)
    assert got and got[0]["actual"] == 219
