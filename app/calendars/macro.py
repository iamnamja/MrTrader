"""
Phase 59 — Macro Calendar Awareness.

Tracks high-impact scheduled macro events (FOMC, CPI, NFP, GDP) and provides:
  - Day-level awareness: is there a high-impact event today?
  - Window-level awareness: are we within ±N minutes of an event?
  - Sizing recommendation: reduce position size on high-impact days with elevated VIX

Data sources:
  - HARDCODED FOMC/CPI/NFP reference-year schedule = the fail-safe FLOOR (always present).
  - LIVE economic-calendar feed (FMP, via the shared NIS econ-calendar source) merged once per day
    to EXTEND/refresh forward coverage (so the gate keeps working past the hardcoded year without a
    code change). The feed supplies the DATE; the window TIME is the canonical fixed release time per
    type. UNION semantics: the gate is never weaker than the floor; a feed outage falls back to it.

All times are in US/Eastern. FOMC announcement 14:00 ET (±60 min); CPI/NFP 08:30 ET (±15 min).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ── Event window parameters ────────────────────────────────────────────────────
FOMC_WINDOW_MINUTES = 60      # block ±60 min around FOMC announcement (2 PM ET)
CPI_NFP_WINDOW_MINUTES = 15   # block ±15 min around CPI/NFP (8:30 AM ET)
HIGH_VIX_SIZE_REDUCTION = 0.15  # reduce position size by 15% when high_impact_today + VIX > 20

# ── Hardcoded 2026 FOMC meeting dates (announcement day, not start day) ────────
# Source: federalreserve.gov — press conferences at 2:30 PM ET, rate decision at 2:00 PM ET
FOMC_2026 = {
    "2026-01-29",
    "2026-03-19",
    "2026-05-07",
    "2026-06-18",
    "2026-07-30",
    "2026-09-17",
    "2026-10-29",
    "2026-12-10",
}

# Key CPI release times (8:30 AM ET) — approximate 2026 schedule
CPI_2026 = {
    "2026-01-15": "08:30",
    "2026-02-12": "08:30",
    "2026-03-12": "08:30",
    "2026-04-10": "08:30",
    "2026-05-13": "08:30",
    "2026-06-11": "08:30",
    "2026-07-15": "08:30",
    "2026-08-13": "08:30",
    "2026-09-11": "08:30",
    "2026-10-15": "08:30",
    "2026-11-12": "08:30",
    "2026-12-11": "08:30",
}

# NFP (Non-Farm Payrolls) release times (8:30 AM ET first Friday of each month)
NFP_2026 = {
    "2026-01-09": "08:30",
    "2026-02-06": "08:30",
    "2026-03-06": "08:30",
    "2026-04-03": "08:30",
    "2026-05-01": "08:30",
    "2026-06-05": "08:30",
    "2026-07-10": "08:30",
    "2026-08-07": "08:30",
    "2026-09-04": "08:30",
    "2026-10-02": "08:30",
    "2026-11-06": "08:30",
    "2026-12-04": "08:30",
}


@dataclass
class MacroEvent:
    event_type: str          # 'FOMC', 'CPI', 'NFP', 'GDP'
    date_str: str            # YYYY-MM-DD
    time_str: str            # HH:MM ET (24h)
    importance: str = "high"


@dataclass
class MacroContext:
    high_impact_today: bool = False
    within_event_window: bool = False
    next_event: Optional[str] = None         # human-readable e.g. "FOMC in 2d"
    events_today: List[MacroEvent] = field(default_factory=list)
    sizing_factor: float = 1.0               # 1.0 = normal, 0.85 = reduced
    block_new_entries: bool = False          # True only within event window


class MacroCalendar:
    """
    Singleton that builds a macro event schedule from hardcoded data and
    provides context queries used by PM and RM.
    """

    def __init__(self):
        # Build the HARDCODED floor only at import (no import-time network). The live feed is merged
        # lazily on the first get_context() of each day.
        self._events: List[MacroEvent] = self._build_hardcoded_events()
        self._events_built_date: Optional[str] = None   # date the feed was last merged (None = floor only)
        self._cache: Optional[MacroContext] = None
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 60.0  # refresh every 60s (window awareness needs fresh time)

    def get_context(self, vix: Optional[float] = None) -> MacroContext:
        # Merge the live econ-calendar feed once per day (extends/refreshes the hardcoded floor so the
        # gate keeps working past the hardcoded year without a code change). Fail-safe: a feed outage
        # leaves the floor intact — the gate is NEVER weaker than the hardcoded dates.
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        if self._events_built_date != today_str:
            self._refresh_events(today_str)
        now_mono = time.monotonic()
        if self._cache and now_mono - self._cache_ts < self._cache_ttl:
            return self._cache
        ctx = self._compute_context(vix)
        self._cache = ctx
        self._cache_ts = now_mono
        return ctx

    def is_entry_blocked(self) -> bool:
        return self.get_context().block_new_entries

    def high_impact_today(self) -> bool:
        return self.get_context().high_impact_today

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_context(self, vix: Optional[float]) -> MacroContext:
        now_et = datetime.now(ET)
        today_str = now_et.strftime("%Y-%m-%d")

        events_today = [e for e in self._events if e.date_str == today_str]
        high_impact_today = len(events_today) > 0
        within_window = False

        for event in events_today:
            window = FOMC_WINDOW_MINUTES if event.event_type == "FOMC" else CPI_NFP_WINDOW_MINUTES
            try:
                h, m = map(int, event.time_str.split(":"))
                event_dt = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
                delta_min = abs((now_et - event_dt).total_seconds() / 60)
                if delta_min <= window:
                    within_window = True
                    break
            except Exception:
                pass

        # Find next upcoming event
        next_event_str: Optional[str] = None
        for event in sorted(self._events, key=lambda e: e.date_str):
            if event.date_str >= today_str:
                days = (
                    datetime.strptime(event.date_str, "%Y-%m-%d").date()
                    - now_et.date()
                ).days
                if days == 0:
                    next_event_str = f"{event.event_type} today @ {event.time_str} ET"
                else:
                    next_event_str = f"{event.event_type} in {days}d ({event.date_str})"
                break

        sizing_factor = 1.0
        if high_impact_today and vix is not None and vix > 20:
            sizing_factor = round(1.0 - HIGH_VIX_SIZE_REDUCTION, 2)

        return MacroContext(
            high_impact_today=high_impact_today,
            within_event_window=within_window,
            next_event=next_event_str,
            events_today=events_today,
            sizing_factor=sizing_factor,
            block_new_entries=within_window,
        )

    @staticmethod
    def _build_hardcoded_events() -> List[MacroEvent]:
        """The fail-safe FLOOR: hardcoded FOMC/CPI/NFP for the reference year. Always present so a
        feed outage can never leave the gate without coverage."""
        events: List[MacroEvent] = []
        for date_str in FOMC_2026:
            events.append(MacroEvent("FOMC", date_str, "14:00"))
        for date_str, time_str in CPI_2026.items():
            events.append(MacroEvent("CPI", date_str, time_str))
        for date_str, time_str in NFP_2026.items():
            events.append(MacroEvent("NFP", date_str, time_str))
        return events

    @staticmethod
    def _fetch_feed_events(days_ahead: int = 120) -> List[MacroEvent]:
        """Fetch forward FOMC/CPI/NFP DATES from the economic-calendar feed (FMP, via the shared
        NIS source). The feed supplies the DATE (which changes year to year); the event WINDOW time
        is the canonical fixed release time per type (FOMC 14:00 ET, CPI/NFP 08:30 ET) — more
        reliable than a feed-reported time. FAIL-SAFE: returns [] on any error/unavailability so the
        hardcoded floor still gates (we never trade unprotected on a feed hiccup)."""
        out: List[MacroEvent] = []
        try:
            from app.news.sources.economic_calendar import fetch_economic_calendar
            events = fetch_economic_calendar(days_ahead=days_ahead, min_impact="high")
        except Exception as exc:
            logger.debug("macro calendar feed fetch failed (using hardcoded floor): %s", exc)
            return out
        if not events:   # None (unavailable) or [] (no high-impact events) -> floor only
            return out
        _canon = {"FOMC": "14:00", "CPI": "08:30", "NFP": "08:30"}
        for e in events:
            et = e.get("event_type")
            if et not in _canon:
                continue
            evt_time = e.get("event_time")
            try:
                if evt_time is None:
                    d = None
                elif evt_time.hour == 0 and evt_time.minute == 0:
                    # date-only feed value (parsed to midnight UTC): the intended calendar date IS the
                    # UTC date; astimezone(ET) would roll it back a day and misplace the window. Use
                    # the date as-is. (Proper-time values keep the UTC→ET conversion below.)
                    d = evt_time.strftime("%Y-%m-%d")
                else:
                    d = evt_time.astimezone(ET).strftime("%Y-%m-%d")
            except Exception:
                d = None
            if d:
                out.append(MacroEvent(et, d, _canon[et]))
        return out

    def _refresh_events(self, today_str: str) -> None:
        """Rebuild the event list = hardcoded floor UNION live feed (dedup by (type, date)). The
        union guarantees the gate is never weaker than the floor; the feed only ADDS coverage."""
        events = self._build_hardcoded_events()
        seen = {(e.event_type, e.date_str) for e in events}
        n_feed = 0
        for fe in self._fetch_feed_events():
            key = (fe.event_type, fe.date_str)
            if key not in seen:
                events.append(fe)
                seen.add(key)
                n_feed += 1
        self._events = events
        self._events_built_date = today_str
        if n_feed:
            logger.info("Macro calendar: merged %d feed event(s) beyond the hardcoded floor", n_feed)


# Module-level singleton
macro_calendar = MacroCalendar()
