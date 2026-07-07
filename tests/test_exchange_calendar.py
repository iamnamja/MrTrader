"""R1.3 — exchange_calendar: US futures-exchange holidays + holiday-aware business-day math."""
from datetime import date

from app.live_trading import exchange_calendar as ec


def test_known_2026_holidays():
    h = ec.holidays(2026)
    assert date(2026, 1, 1) in h                 # New Year (Thu)
    assert date(2026, 1, 19) in h                # MLK — 3rd Mon Jan
    assert date(2026, 2, 16) in h                # Washington — 3rd Mon Feb
    assert date(2026, 4, 3) in h                 # Good Friday
    assert date(2026, 5, 25) in h                # Memorial — last Mon May
    assert date(2026, 7, 3) in h                 # Independence: Jul 4 is Sat → observed Fri Jul 3
    assert date(2026, 9, 7) in h                 # Labor — 1st Mon Sep
    assert date(2026, 11, 26) in h               # Thanksgiving — 4th Thu Nov
    assert date(2026, 12, 25) in h               # Christmas (Fri)


def test_weekend_observance_rules():
    assert ec._observed(date(2026, 7, 4)) == date(2026, 7, 3)     # Sat → prior Fri
    assert ec._observed(date(2027, 12, 25)) == date(2027, 12, 24)  # Sat → prior Fri
    assert ec._observed(date(2022, 12, 25)) == date(2022, 12, 26)  # Sun → following Mon


def test_is_trading_day():
    assert ec.is_trading_day(date(2026, 7, 2)) is True            # Thu
    assert ec.is_trading_day(date(2026, 7, 3)) is False           # Independence (observed)
    assert ec.is_trading_day(date(2026, 7, 4)) is False           # Sat
    assert ec.is_trading_day(date(2026, 12, 25)) is False         # Christmas


def test_minus_trading_days_skips_holiday():
    # 3 trading days before Mon 2026-07-06: skip Sun-5/Sat-4/holiday-Fri-3 → Thu-2 (1), Wed-1 (2), Tue Jun-30 (3).
    assert ec.minus_trading_days(date(2026, 7, 6), 3) == date(2026, 6, 30)


def test_last_trading_day_of_month_skips_trailing_holiday():
    # May 2026 ends Sun 31 → Sat 30 → Fri 29 (Memorial is the 25th, not trailing) → 2026-05-29.
    assert ec.last_trading_day_of_month(2026, 5) == date(2026, 5, 29)
    # Dec 2026 ends Thu 31 (trading) → 2026-12-31.
    assert ec.last_trading_day_of_month(2026, 12) == date(2026, 12, 31)


def test_new_year_observed_on_dec_31_boundary():
    # Jan 1 2028 is a Saturday → New Year observed Fri Dec 31 2027. The Dec-31 observance belongs to
    # the NEXT year's holiday set; is_holiday must still catch it (±1-year union), and the last trading
    # day of Dec 2027 must be Dec 30, NOT Dec 31.
    assert date(2027, 12, 31) in ec.holidays(2028)
    assert date(2028, 1, 1) not in ec.holidays(2028)         # no double-count
    assert ec.is_holiday(date(2027, 12, 31)) is True
    assert ec.is_trading_day(date(2027, 12, 31)) is False
    assert ec.last_trading_day_of_month(2027, 12) == date(2027, 12, 30)


def test_trading_days_between_signed_and_holiday_aware():
    # Wed Jul-1 → Mon Jul-6: Jul-2 (Thu), Jul-3 holiday, Jul-4/5 weekend, Jul-6 → 2 trading days.
    assert ec.trading_days_between(date(2026, 7, 1), date(2026, 7, 6)) == 2
    assert ec.trading_days_between(date(2026, 7, 6), date(2026, 7, 1)) == -2
    assert ec.trading_days_between(date(2026, 7, 6), date(2026, 7, 6)) == 0
