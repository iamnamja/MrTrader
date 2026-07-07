"""
exchange_calendar.py — Alpha-v10 R1.3: US futures-exchange trading calendar (holiday-aware).

The CME/CBOT/COMEX/NYMEX/CFE full-day closes follow the standard US equity holiday schedule (plus Good
Friday, which is NOT a federal holiday). The futures roll math (`futures_roll_policy`, `futures_roll_monitor`)
was weekday-only — a month-end holiday could compress the FND / last-trade safety margin (the Opus MINOR).
This computes the 10 full-close holidays ALGORITHMICALLY (any year, with the exchange weekend-observance
rule) so the business-day math is holiday-aware.

Scope: FULL-day closes only (the roll deadline is a full close). Early-close / half-days are NOT modelled —
they don't affect a roll deadline. Pre-2021 Juneteenth is included harmlessly (irrelevant to near-term rolls).
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import FrozenSet


def _observed(d: date) -> date:
    """Exchange weekend-observance for a FIXED-date holiday: Sat → prior Fri, Sun → following Mon."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th `weekday` (0=Mon..6=Sun) of the month (n=1..5)."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The LAST `weekday` of the month."""
    nxt = date(year + month // 12, month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _good_friday(year: int) -> date:
    """Good Friday = 2 days before Easter Sunday (Gregorian computus, Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day) - timedelta(days=2)


@lru_cache(maxsize=None)
def holidays(year: int) -> FrozenSet[date]:
    """The 10 US futures-exchange FULL-close holidays for `year` (observed dates)."""
    return frozenset({
        _observed(date(year, 1, 1)),          # New Year's Day
        _nth_weekday(year, 1, 0, 3),           # MLK — 3rd Monday of January
        _nth_weekday(year, 2, 0, 3),           # Washington's Birthday — 3rd Monday of February
        _good_friday(year),                    # Good Friday
        _last_weekday(year, 5, 0),             # Memorial Day — last Monday of May
        _observed(date(year, 6, 19)),          # Juneteenth
        _observed(date(year, 7, 4)),           # Independence Day
        _nth_weekday(year, 9, 0, 1),           # Labor Day — 1st Monday of September
        _nth_weekday(year, 11, 3, 4),          # Thanksgiving — 4th Thursday of November
        _observed(date(year, 12, 25)),         # Christmas Day
    })


def is_holiday(d: date) -> bool:
    # Check neighbouring years too: a New Year observed on Dec 31 belongs to the next year's set.
    return d in holidays(d.year) or d in holidays(d.year - 1) or d in holidays(d.year + 1)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and not is_holiday(d)


def minus_trading_days(d: date, n: int) -> date:
    """`d` shifted back `n` TRADING days (skips weekends AND exchange holidays)."""
    out = d
    while n > 0:
        out -= timedelta(days=1)
        if is_trading_day(out):
            n -= 1
    return out


def last_trading_day_of_month(year: int, month: int) -> date:
    """The last TRADING day of the month (skips a trailing weekend/holiday)."""
    nxt = date(year + month // 12, month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days_between(a: date, b: date) -> int:
    """Signed count of TRADING days from `a` to `b` (positive if b is after a)."""
    lo, hi = (a, b) if b >= a else (b, a)
    n, d = 0, lo
    while d < hi:
        d += timedelta(days=1)
        if is_trading_day(d):
            n += 1
    return n if b >= a else -n
