"""
Phase 81 — Earnings Calendar Gate (Finnhub primary, FMP fallback, fail-closed).

Blocks new entries within SWING_BLACKOUT_DAYS of an earnings print and
flags existing swing positions that would be held through earnings.

Data sources (in priority order):
  1. Finnhub earnings calendar (reuses finnhub_source.py — no extra API key)
  2. FMP earnings calendar (fallback — `/earning_calendar` endpoint)
  3. FAIL-CLOSED for swing (block_swing=True, reason="earnings_data_unavailable")
     FAIL-OPEN for intraday (short hold, acceptable risk)

Results cached per symbol for CACHE_TTL_SECONDS.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SWING_BLACKOUT_DAYS = 2      # block new swing entry if earnings within N trading days
INTRADAY_BLACKOUT_DAYS = 1   # block intraday entry if earnings today or pre-market tomorrow
EXIT_REVIEW_DAYS = 3         # flag existing positions for exit review this many days before earnings
CACHE_TTL_SECONDS = 3600     # re-fetch at most once per hour per symbol

_UNAVAILABLE = "earnings_data_unavailable"


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
        # symbol → (next_earnings_date_or_None, fetched_at_monotonic, data_ok)
        # data_ok=False means both sources failed → fail-closed
        self._cache: Dict[str, Tuple[Optional[date], float, bool]] = {}

    def get_earnings_risk(self, symbol: str, trade_type: str = "swing") -> EarningsRisk:
        next_date, data_ok = self._get_next_earnings(symbol)
        days_until: Optional[int] = None
        if next_date is not None:
            days_until = (next_date - date.today()).days

        # Data unavailable → fail-closed for swing, fail-open for intraday
        if not data_ok:
            return EarningsRisk(
                symbol=symbol,
                next_earnings=None,
                days_until=None,
                block_swing=True,
                block_intraday=False,
                exit_review=False,
                reason=_UNAVAILABLE,
            )

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
        d, _ = self._get_next_earnings(symbol)
        return d

    def days_until_earnings(self, symbol: str) -> Optional[int]:
        d, _ = self._get_next_earnings(symbol)
        if d is None:
            return None
        return (d - date.today()).days

    def prefetch(self, symbols: List[str]) -> None:
        """Warm cache for a batch of symbols — call at 06:00 ET pre-market."""
        logger.info("Prefetching earnings calendar for %d symbols", len(symbols))
        ok = fail = 0
        for sym in symbols:
            _, data_ok = self._get_next_earnings(sym)
            if data_ok:
                ok += 1
            else:
                fail += 1
        logger.info(
            "Earnings prefetch complete: %d ok, %d data-unavailable (fail-closed for swing)",
            ok, fail,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_next_earnings(self, symbol: str) -> Tuple[Optional[date], bool]:
        """Return (next_earnings_date_or_None, data_ok). data_ok=False means fail-closed."""
        cached = self._cache.get(symbol)
        if cached is not None:
            cached_date, fetched_at, data_ok = cached
            if time.monotonic() - fetched_at < CACHE_TTL_SECONDS:
                return cached_date, data_ok

        result, data_ok = self._fetch(symbol)
        self._cache[symbol] = (result, time.monotonic(), data_ok)
        return result, data_ok

    @staticmethod
    def _fetch(symbol: str) -> Tuple[Optional[date], bool]:
        """Try Finnhub → FMP → fail-closed. Returns (date_or_None, data_ok)."""
        # ── Source 1: Finnhub ────────────────────────────────────────────────
        try:
            from app.news.sources.finnhub_source import fetch_earnings_calendar
            today = date.today()
            window = today + timedelta(days=30)
            result = fetch_earnings_calendar([symbol], from_date=today, to_date=window)
            if result is not None:  # None means API call failed; {} means no earnings
                entry = result.get(symbol.upper())
                if entry and entry.get("date"):
                    d = entry["date"]
                    if isinstance(d, str):
                        d = date.fromisoformat(d)
                    logger.debug("Earnings %s via Finnhub: %s", symbol, d)
                    return d, True
                # Finnhub returned cleanly but no earnings in next 30 days → safe
                return None, True
        except Exception as exc:
            logger.warning("Finnhub earnings fetch failed for %s: %s — trying FMP", symbol, exc)

        # ── Source 2: FMP ────────────────────────────────────────────────────
        try:
            d = _fetch_fmp_next_earnings(symbol)
            logger.debug("Earnings %s via FMP: %s", symbol, d)
            return d, True
        except Exception as exc:
            logger.warning("FMP earnings fetch failed for %s: %s — fail-closed for swing", symbol, exc)

        # ── Both failed → fail-closed ────────────────────────────────────────
        return None, False


def _fetch_fmp_next_earnings(symbol: str) -> Optional[date]:
    """
    Fetch next upcoming earnings date from FMP `/earning_calendar` endpoint.
    Raises on any network/parse error so the caller can handle gracefully.
    """
    import requests
    from app.data.fmp_provider import _api_key  # reuse existing key helper
    today = date.today()
    to_date = today + timedelta(days=30)
    url = "https://financialmodelingprep.com/stable/earning_calendar"
    params = {
        "symbol": symbol,
        "from": today.isoformat(),
        "to": to_date.isoformat(),
        "apikey": _api_key(),
    }
    resp = requests.get(url, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or not data:
        return None
    dates: list[date] = []
    for entry in data:
        raw = entry.get("date")
        if raw:
            try:
                d = date.fromisoformat(str(raw)[:10])
                if d >= today:
                    dates.append(d)
            except ValueError:
                pass
    return min(dates) if dates else None


# Module-level singleton
earnings_calendar = EarningsCalendar()
