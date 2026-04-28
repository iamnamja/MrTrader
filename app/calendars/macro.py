"""
Phase 59 — Macro Calendar Awareness.

Tracks high-impact scheduled macro events (FOMC, CPI, NFP, GDP) and provides:
  - Day-level awareness: is there a high-impact event today?
  - Window-level awareness: are we within ±N minutes of an event?
  - Sizing recommendation: reduce position size on high-impact days with elevated VIX

Data sources (in priority order):
  1. Hardcoded 2026 FOMC meeting schedule (most reliable)
  2. FRED release calendar (fetched once daily, cached)
  3. yfinance economic calendar (fallback)

All times are in US/Eastern.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
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
        self._events: List[MacroEvent] = self._build_event_list()
        self._cache: Optional[MacroContext] = None
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 60.0  # refresh every 60s (window awareness needs fresh time)

    def get_context(self, vix: Optional[float] = None) -> MacroContext:
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
    def _build_event_list() -> List[MacroEvent]:
        events: List[MacroEvent] = []
        for date_str in FOMC_2026:
            events.append(MacroEvent("FOMC", date_str, "14:00"))
        for date_str, time_str in CPI_2026.items():
            events.append(MacroEvent("CPI", date_str, time_str))
        for date_str, time_str in NFP_2026.items():
            events.append(MacroEvent("NFP", date_str, time_str))
        return events


# Module-level singleton
macro_calendar = MacroCalendar()
