"""
Phase 58 — Earnings Calendar Gate.

Blocks new entries within SWING_BLACKOUT_DAYS of an earnings print and
flags existing swing positions that would be held through earnings.

Uses yfinance for earnings dates (free, no API key).  Results are cached
per symbol for CACHE_TTL_SECONDS to avoid hammering the API.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

SWING_BLACKOUT_DAYS = 2      # block new swing entry if earnings within N trading days
INTRADAY_BLACKOUT_DAYS = 1   # block intraday entry if earnings today or pre-market tomorrow
EXIT_REVIEW_DAYS = 3         # flag existing positions for exit review this many days before earnings
CACHE_TTL_SECONDS = 3600     # re-fetch at most once per hour per symbol


@dataclass
class EarningsRisk:
    symbol: str
    next_earnings: Optional[date]
    days_until: Optional[int]          # None = unknown
    block_swing: bool = False
    block_intraday: bool = False
    exit_review: bool = False          # existing position should be reviewed for early exit
    reason: str = ""


class EarningsCalendar:
    """Earnings date lookup with in-process cache."""

    def __init__(self):
        # symbol → (next_earnings_date_or_None, fetched_at_monotonic)
        self._cache: Dict[str, Tuple[Optional[date], float]] = {}

    def get_earnings_risk(self, symbol: str, trade_type: str = "swing") -> EarningsRisk:
        """
        Return an EarningsRisk dataclass for the given symbol.
        trade_type: 'swing' or 'intraday'
        """
        next_date = self._get_next_earnings(symbol)
        days_until: Optional[int] = None
        if next_date is not None:
            days_until = (next_date - date.today()).days

        block_swing = False
        block_intraday = False
        exit_review = False

        if days_until is not None:
            block_swing = 0 <= days_until <= SWING_BLACKOUT_DAYS
            block_intraday = 0 <= days_until <= INTRADAY_BLACKOUT_DAYS
            exit_review = 0 <= days_until <= EXIT_REVIEW_DAYS

        reason = ""
        if block_swing or block_intraday:
            reason = f"earnings_in_{days_until}d" if days_until is not None else "earnings_unknown"

        return EarningsRisk(
            symbol=symbol,
            next_earnings=next_date,
            days_until=days_until,
            block_swing=block_swing,
            block_intraday=block_intraday,
            exit_review=exit_review,
            reason=reason,
        )

    def is_blocked(self, symbol: str, trade_type: str = "swing") -> bool:
        risk = self.get_earnings_risk(symbol, trade_type)
        if trade_type == "intraday":
            return risk.block_intraday
        return risk.block_swing

    def next_earnings_date(self, symbol: str) -> Optional[date]:
        return self._get_next_earnings(symbol)

    def days_until_earnings(self, symbol: str) -> Optional[int]:
        d = self._get_next_earnings(symbol)
        if d is None:
            return None
        return (d - date.today()).days

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_next_earnings(self, symbol: str) -> Optional[date]:
        cached_val, fetched_at = self._cache.get(symbol, (None, -CACHE_TTL_SECONDS - 1))
        if time.monotonic() - fetched_at < CACHE_TTL_SECONDS:
            return cached_val
        result = self._fetch(symbol)
        self._cache[symbol] = (result, time.monotonic())
        return result

    @staticmethod
    def _fetch(symbol: str) -> Optional[date]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is None:
                return None
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
earnings_calendar = EarningsCalendar()
