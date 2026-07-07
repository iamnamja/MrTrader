"""R1.3 — futures roll policy (shadow): candidate roll dates from IBKR contract fields, FND floor for
physically-delivered contracts, earliest-binding recommendation. Pure/deterministic; rolls nothing."""
from datetime import date

from app.live_trading import futures_roll_policy as rp


def test_settlement_taxonomy_and_failsafe_default():
    assert rp.settlement("ES") == rp.CASH and rp.settlement("VX") == rp.CASH
    assert rp.settlement("6E") == rp.FX
    assert rp.settlement("ZS") == rp.PHYSICAL and rp.settlement("CL") == rp.PHYSICAL
    assert rp.settlement("WHATEVER") == rp.PHYSICAL       # unknown → assume delivery risk (fail-safe)


def test_last_business_day_skips_weekend():
    # 2026-05-31 is a Sunday → last business day of May 2026 is Fri 2026-05-29.
    assert rp._last_business_day_of_month(2026, 5) == date(2026, 5, 29)


def test_first_notice_day_estimate_grain():
    # July-delivery grain: FND ≈ last business day of June = Tue 2026-06-30.
    assert rp.first_notice_day_estimate(2026, 7, "ZS") == date(2026, 6, 30)
    assert rp.first_notice_day_estimate(2026, 7, "ES") is None      # cash → no FND


def test_first_notice_day_wraps_january():
    # Jan-delivery → FND in the PRIOR December.
    fnd = rp.first_notice_day_estimate(2026, 1, "GC")
    assert fnd is not None and fnd.year == 2025 and fnd.month == 12


def test_cash_market_uses_fixed_calendar_rule_only():
    rd = rp.compute_roll_dates("ES", contract_month="202609", last_trade="20260918")
    assert rd.scheduled_expiry == date(2026, 9, 15)
    assert rd.fixed_roll == date(2026, 9, 10)             # scheduled - 5 days
    assert rd.fnd_floor is None                           # cash → no FND floor
    assert rd.recommended == date(2026, 9, 10)            # fixed rule binds
    assert rp.should_roll(rd, date(2026, 7, 7)) is False  # not due in July


def test_grain_fnd_floor_binds_earlier_than_fixed():
    # THE key finding: the July soybean's FND floor (2026-06-30) binds well before the fixed rule
    # (2026-07-10) and the last-trade (2026-07-14) — and it is ALREADY PAST today (2026-07-07),
    # so a live book would be sitting in the delivery window. The policy flags roll_due=True.
    rd = rp.compute_roll_dates("ZS", contract_month="202607", last_trade="20260714")
    assert rd.fixed_roll == date(2026, 7, 10)
    assert rd.fnd_floor == date(2026, 6, 30)
    assert rd.recommended == date(2026, 6, 30)            # earliest binding = FND floor
    assert rp.should_roll(rd, date(2026, 7, 7)) is True   # overdue — delivery-risk avoided


def test_energy_last_trade_cap_binds_before_the_late_fnd_estimate():
    # THE energy hazard (Opus MAJOR-1/-2): CL Aug last-trade ≈ 2026-07-21, but the prior-month-end FND
    # estimate is 2026-07-31 (AFTER the contract stops trading) and fixed is 2026-08-10. The last_trade
    # cap must bind so we NEVER recommend rolling crude after it has gone to delivery.
    rd = rp.compute_roll_dates("CL", contract_month="202608", last_trade="20260721")
    assert rd.fnd_floor == date(2026, 7, 31) and rd.fixed_roll == date(2026, 8, 10)
    assert rd.last_trade_cap == rp._minus_business_days(date(2026, 7, 21), 3)   # 3 BD before last-trade
    assert rd.recommended == rd.last_trade_cap and rd.recommended < rd.last_trade


def test_recommended_never_after_last_trade_for_any_physical_market():
    # Invariant: for every physically-delivered market the recommendation is strictly before last-trade.
    cases = {"CL": ("202608", "20260721"), "NG": ("202608", "20260729"),
             "GC": ("202608", "20260827"), "ZS": ("202607", "20260714"),
             "ZN": ("202609", "20260921")}
    for root, (cm, lt) in cases.items():
        rd = rp.compute_roll_dates(root, contract_month=cm, last_trade=lt)
        assert rd.recommended is not None and rd.recommended < rd.last_trade, root


def test_malformed_contract_month_degrades_to_none_not_crash():
    for bad in ("202613", "202600", "20260", "oops", None):
        rd = rp.compute_roll_dates("ZS", contract_month=bad, last_trade="20260714")
        assert rd.fixed_roll is None and rd.fnd_floor is None      # no crash; degrades
        assert rd.recommended == rd.last_trade_cap                 # still capped by last-trade


def test_roll_report_snapshot_and_logs():
    snap = rp.roll_report("ZS", date(2026, 7, 7), contract_month="202607", last_trade="20260714")
    assert snap["root"] == "ZS" and snap["settlement"] == "physical"
    assert snap["recommended"] == "2026-06-30" and snap["roll_due"] is True
    assert snap["fixed_roll"] == "2026-07-10" and snap["fnd_floor"] == "2026-06-30"
    assert snap["liquidity_roll"] is None                 # not wired yet (instrumentation slot)


def test_liquidity_roll_is_recorded_but_not_yet_binding():
    # A supplied vol/OI crossover date is logged but does NOT change the recommendation (R1.3 measures
    # it first; making it binding requires re-validating the signal under the new roll).
    liq = date(2026, 6, 20)
    rd = rp.compute_roll_dates("ZS", contract_month="202607", last_trade="20260714", liquidity_roll=liq)
    assert rd.liquidity_roll == liq
    assert rd.recommended == date(2026, 6, 30)            # still the FND floor, not the earlier liq date
