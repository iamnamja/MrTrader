"""
macro_calendar.py — Fetch and cache macro event dates for walk-forward simulation.

Returns a set of dates on which high-impact macro events occurred (FOMC, NFP, CPI, GDP).
Used by WF-5a to block fold entries on macro event days — matching live PM behaviour.

Usage:
    from scripts.walkforward.macro_calendar import load_macro_blocked_dates
    blocked = load_macro_blocked_dates(start_date, end_date)
    is_blocked = some_date in blocked
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Set

logger = logging.getLogger(__name__)

# Known high-impact event keywords (matched case-insensitively)
_HIGH_IMPACT_KEYWORDS = [
    "fomc", "federal open market", "fed rate",
    "nonfarm payroll", "nfp",
    "consumer price index", "cpi",
    "gdp", "gross domestic product",
    "unemployment rate",
    "pce",
    "jackson hole",
]


def load_macro_blocked_dates(
    start: date,
    end: date,
    finnhub_token: str | None = None,
) -> Set[date]:
    """Return the set of dates with a high-impact macro event in [start, end].

    Fetches from Finnhub economic calendar if a token is available.
    Falls back to a hard-coded FOMC-only list for the most common events.
    Never raises — returns empty set on failure so the WF run is unblocked.
    """
    token = finnhub_token or os.environ.get("FINNHUB_API_KEY", "")
    if token:
        try:
            return _fetch_finnhub(start, end, token)
        except Exception as exc:
            logger.warning("Finnhub macro calendar fetch failed: %s — using fallback", exc)

    return _fallback_fomc_dates(start, end)


def _fetch_finnhub(start: date, end: date, token: str) -> Set[date]:
    import requests

    blocked: Set[date] = set()
    # Finnhub supports date range on the economic calendar endpoint
    url = (
        f"https://finnhub.io/api/v1/calendar/economic"
        f"?from={start.isoformat()}&to={end.isoformat()}&token={token}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    events = data.get("economicCalendar", [])

    for e in events:
        impact = (e.get("impact") or "").lower()
        if impact not in ("high",):
            continue
        evt_name = (e.get("event") or "").lower()
        if not any(kw in evt_name for kw in _HIGH_IMPACT_KEYWORDS):
            continue
        raw_time = e.get("time", "")
        if not raw_time:
            continue
        try:
            d = date.fromisoformat(raw_time[:10])
        except ValueError:
            continue
        if start <= d <= end:
            blocked.add(d)

    logger.info("Macro calendar: %d blocked dates (%s → %s) from Finnhub", len(blocked), start, end)
    return blocked


def _fallback_fomc_dates(start: date, end: date) -> Set[date]:
    """Hard-coded FOMC meeting dates 2020-2026 (announcement day only)."""
    fomc_dates = {
        # 2020
        date(2020, 1, 29), date(2020, 3, 3), date(2020, 3, 15), date(2020, 4, 29),
        date(2020, 6, 10), date(2020, 7, 29), date(2020, 9, 16), date(2020, 11, 5),
        date(2020, 12, 16),
        # 2021
        date(2021, 1, 27), date(2021, 3, 17), date(2021, 4, 28), date(2021, 6, 16),
        date(2021, 7, 28), date(2021, 9, 22), date(2021, 11, 3), date(2021, 12, 15),
        # 2022
        date(2022, 1, 26), date(2022, 3, 16), date(2022, 5, 4), date(2022, 6, 15),
        date(2022, 7, 27), date(2022, 9, 21), date(2022, 11, 2), date(2022, 12, 14),
        # 2023
        date(2023, 2, 1), date(2023, 3, 22), date(2023, 5, 3), date(2023, 6, 14),
        date(2023, 7, 26), date(2023, 9, 20), date(2023, 11, 1), date(2023, 12, 13),
        # 2024
        date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
        date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
        # 2025
        date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
        date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
        # 2026
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
        date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
    }
    result = {d for d in fomc_dates if start <= d <= end}
    logger.info("Macro calendar (fallback FOMC): %d blocked dates (%s → %s)", len(result), start, end)
    return result
