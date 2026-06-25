"""Alpha-v10 Wave 6e — macro calendar auto-fetches FOMC/CPI/NFP from a feed (hardcoded floor stays).

The FOMC/CPI/NFP dates were hardcoded for one reference year. Now a live econ-calendar feed (FMP) is
merged once per day to EXTEND forward coverage, UNIONed with the hardcoded floor so a feed outage can
never weaken the macro event-window gate. Feed supplies DATES; window TIMES are canonical per type.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.calendars.macro import MacroCalendar, FOMC_2026


def _utc(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_hardcoded_floor_present_without_feed(monkeypatch):
    # feed returns nothing -> the gate still has the full hardcoded floor (fail-safe)
    monkeypatch.setattr(MacroCalendar, "_fetch_feed_events", staticmethod(lambda days_ahead=120: []))
    cal = MacroCalendar()
    cal._refresh_events("2026-06-25")
    fomc_dates = {e.date_str for e in cal._events if e.event_type == "FOMC"}
    assert FOMC_2026.issubset(fomc_dates)        # floor intact


def test_feed_extends_forward_coverage(monkeypatch):
    # a 2027 FOMC date from the feed (beyond the hardcoded floor) is merged in
    from app.calendars import macro as macro_mod

    def _fake_feed(days_ahead=120):
        return [macro_mod.MacroEvent("FOMC", "2027-01-28", "14:00")]
    monkeypatch.setattr(MacroCalendar, "_fetch_feed_events", staticmethod(_fake_feed))
    cal = MacroCalendar()
    cal._refresh_events("2026-12-30")
    dates = {e.date_str for e in cal._events if e.event_type == "FOMC"}
    assert "2027-01-28" in dates                 # feed added forward coverage
    assert FOMC_2026.issubset(dates)             # floor still present (union)


def test_feed_does_not_duplicate_floor(monkeypatch):
    # a feed event that coincides with a hardcoded date must not duplicate it
    from app.calendars import macro as macro_mod
    dup = next(iter(FOMC_2026))

    def _fake_feed(days_ahead=120):
        return [macro_mod.MacroEvent("FOMC", dup, "14:00")]
    monkeypatch.setattr(MacroCalendar, "_fetch_feed_events", staticmethod(_fake_feed))
    cal = MacroCalendar()
    cal._refresh_events("2026-06-25")
    n = sum(1 for e in cal._events if e.event_type == "FOMC" and e.date_str == dup)
    assert n == 1                                # deduped


def test_fetch_feed_events_classifies_and_uses_canonical_times(monkeypatch):
    # the feed parser maps event_type->canonical window time and converts UTC date->ET
    import app.news.sources.economic_calendar as ec

    def _fake(days_ahead=1, min_impact="medium"):
        return [
            {"event_type": "FOMC", "event_time": _utc(2027, 3, 17, 18, 0)},   # 14:00 ET
            {"event_type": "CPI", "event_time": _utc(2027, 3, 10, 12, 30)},   # 08:30 ET
            {"event_type": "NFP", "event_time": _utc(2027, 3, 5, 13, 30)},
            {"event_type": "OTHER_HIGH", "event_time": _utc(2027, 3, 9, 12, 0)},  # ignored
        ]
    monkeypatch.setattr(ec, "fetch_economic_calendar", _fake)
    out = MacroCalendar._fetch_feed_events()
    by = {e.event_type: e for e in out}
    assert set(by) == {"FOMC", "CPI", "NFP"}     # OTHER_HIGH filtered out
    assert by["FOMC"].time_str == "14:00"
    assert by["CPI"].time_str == "08:30" and by["NFP"].time_str == "08:30"
    assert by["FOMC"].date_str == "2027-03-17"   # UTC->ET date


def test_fetch_feed_events_date_only_no_offbyone(monkeypatch):
    # a date-only feed value parses to midnight UTC; astimezone(ET) would roll it back a day.
    # The intended calendar date is the UTC date -> must NOT shift.
    import app.news.sources.economic_calendar as ec
    monkeypatch.setattr(ec, "fetch_economic_calendar",
                        lambda **k: [{"event_type": "FOMC", "event_time": _utc(2027, 9, 17, 0, 0)}])
    out = MacroCalendar._fetch_feed_events()
    assert len(out) == 1
    assert out[0].date_str == "2027-09-17"       # NOT 2027-09-16 (no off-by-one)
    assert out[0].time_str == "14:00"            # canonical window time still applied


def test_fetch_feed_events_failsafe_on_error(monkeypatch):
    # feed raising / returning None must yield [] (never propagate -> floor still gates)
    import app.news.sources.economic_calendar as ec
    monkeypatch.setattr(ec, "fetch_economic_calendar",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("feed down")))
    assert MacroCalendar._fetch_feed_events() == []
    monkeypatch.setattr(ec, "fetch_economic_calendar", lambda **k: None)
    assert MacroCalendar._fetch_feed_events() == []
