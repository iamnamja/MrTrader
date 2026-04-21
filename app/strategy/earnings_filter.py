"""
Earnings calendar blackout filter.

Blocks new position entries within BLACKOUT_DAYS calendar days before a
scheduled earnings release.  Uses yfinance for earnings dates and caches
results per symbol for CACHE_TTL_SECONDS to avoid hammering the API.

Usage:
    if is_earnings_blackout(symbol):
        return  # skip entry
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

BLACKOUT_DAYS = 2          # days before earnings to block new entries
CACHE_TTL_SECONDS = 3600   # re-fetch at most once per hour per symbol


class EarningsFilter:
    def __init__(self):
        # symbol → (next_earnings_date_or_None, fetched_at_monotonic)
        self._cache: Dict[str, Tuple[Optional[date], float]] = {}

    def is_blackout(self, symbol: str) -> bool:
        """Return True if today is within BLACKOUT_DAYS of the next earnings date."""
        next_date = self._get_next_earnings(symbol)
        if next_date is None:
            return False
        days_until = (next_date - date.today()).days
        if 0 <= days_until <= BLACKOUT_DAYS:
            logger.info(
                "%s: earnings blackout active — earnings in %d day(s) on %s",
                symbol, days_until, next_date,
            )
            return True
        return False

    def next_earnings_date(self, symbol: str) -> Optional[date]:
        """Return the next earnings date for display purposes (may be None)."""
        return self._get_next_earnings(symbol)

    def _get_next_earnings(self, symbol: str) -> Optional[date]:
        cached, fetched_at = self._cache.get(symbol, (None, -CACHE_TTL_SECONDS - 1))
        if time.monotonic() - fetched_at < CACHE_TTL_SECONDS:
            return cached
        result = self._fetch(symbol)
        self._cache[symbol] = (result, time.monotonic())
        return result

    @staticmethod
    def _fetch(symbol: str) -> Optional[date]:
        """Fetch next earnings date from yfinance. Returns None on any error."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is None:
                return None
            # calendar is a dict with key "Earnings Date" → list of Timestamps
            earnings_dates = cal.get("Earnings Date", [])
            if not earnings_dates:
                return None
            today = date.today()
            future = [
                d.date() if hasattr(d, "date") else d
                for d in earnings_dates
                if (d.date() if hasattr(d, "date") else d) >= today
            ]
            return min(future) if future else None
        except Exception as exc:
            logger.debug("Could not fetch earnings for %s: %s", symbol, exc)
            return None


# Module-level singleton
earnings_filter = EarningsFilter()


def is_earnings_blackout(symbol: str) -> bool:
    """Convenience wrapper used by generate_signal()."""
    return earnings_filter.is_blackout(symbol)


def days_until_earnings(symbol: str) -> Optional[int]:
    """
    Return number of calendar days until next earnings, or None if unknown.
    Used by PM position review to pro-actively exit before earnings.
    """
    from datetime import date as _date
    next_date = earnings_filter.next_earnings_date(symbol)
    if next_date is None:
        return None
    return (next_date - _date.today()).days
