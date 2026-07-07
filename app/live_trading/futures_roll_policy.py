"""
futures_roll_policy.py — Alpha-v10 R1.3: WHEN to roll a futures position (shadow instrumentation).

Recon (2026-07-07, live paper gateway) established the design:
  - IBKR `reqContractDetails` exposes NO First Notice Day — only `contractMonth` (delivery month),
    `lastTradeDateOrContractMonth`, `realExpirationDate`, `timeZoneId`. So the FND floor must be
    DERIVED per-market from the delivery month, not read from the broker.
  - The backtested carry/xSMOM edge rolls on a FIXED rule: front = nearest contract whose SCHEDULED
    (day-15-of-delivery-month) expiry is > ROLL_BUFFER_DAYS (=5) out (`app/research/futures_roll.py`).
  - A fixed "N days before last-trade" rule is DANGEROUS for physically-delivered contracts: e.g. the
    July soybean's First Notice Day (~last business day of June) is ~2 weeks before last-trade, so a
    late roll sits in the delivery window (assignment risk) and in an illiquid, rolled-away contract.

So the roll date is the EARLIEST (most conservative) of the applicable candidates:
  1. `fixed`      — 5 calendar days before the SCHEDULED expiry (matches the backtest; every market).
  2. `fnd`        — First-Notice-Day floor, PHYSICAL-delivery markets only (never hold into delivery).
  3. `last_trade` cap — always: N business days before the IBKR-supplied last-trade date. This is the
     hard backstop and the ONLY thing that protects ENERGY (CL/NG): their true expiry falls BEFORE the
     prior-month-end FND estimate, so the FND floor alone would land after the contract stopped trading.
  4. `liquidity` (INSTRUMENTED, not yet binding) — the vol/OI crossover to the next contract.

R1.3 is SHADOW: `roll_report()` computes + logs all candidate dates so we measure how far apart they
are per product on real data BEFORE choosing fixed-vs-dynamic (and any live roll that diverges from the
backtested `fixed` rule must be re-validated against the signal — a roll change silently alters returns).
Nothing here places or rolls anything.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Optional

from app.live_trading import exchange_calendar as ec

log = logging.getLogger(__name__)

ROLL_BUFFER_DAYS = 5          # mirror app/research/futures_carry.ROLL_BUFFER_DAYS (backtest convention)
LAST_TRADE_BUFFER_BDAYS = 3   # hard backstop: never recommend rolling later than this many trading
#                               days before the IBKR last-trade date (protects energy — see module doc)

# Settlement taxonomy for our 16-market universe (roots = instrument_master.FUT.<root>).
#   "cash"     — cash-settled, no delivery: roll on the calendar rule only.
#   "physical" — physically delivered with a delivery month + First Notice Day: an FND floor applies.
#   "fx"       — deliverable FX, but settle-on-expiry (no early delivery-month/FND risk): calendar rule.
CASH = "cash"
PHYSICAL = "physical"
FX = "fx"
SETTLEMENT: Dict[str, str] = {
    "ES": CASH, "NQ": CASH, "RTY": CASH, "VX": CASH,          # index + VIX (cash-settled)
    "6E": FX, "6J": FX,                                       # FX (settle on expiry)
    "CL": PHYSICAL, "NG": PHYSICAL,                           # energy
    "GC": PHYSICAL, "SI": PHYSICAL, "HG": PHYSICAL,           # metals
    "ZC": PHYSICAL, "ZS": PHYSICAL,                           # grains
    "ZN": PHYSICAL, "ZB": PHYSICAL, "ZF": PHYSICAL,           # Treasury (deliverable)
}


def settlement(root: str) -> str:
    """Settlement class for a futures root; defaults to PHYSICAL (fail-SAFE — assume delivery risk)."""
    return SETTLEMENT.get(str(root).upper(), PHYSICAL)


def _minus_business_days(d: date, n: int) -> date:
    """`d` shifted back `n` TRADING days — HOLIDAY-AWARE (exchange_calendar), so a month-end holiday
    can't silently compress the FND / last-trade safety margin (was weekday-only; Opus MINOR)."""
    return ec.minus_trading_days(d, n)


def _last_business_day_of_month(year: int, month: int) -> date:
    """Last TRADING day of the month — HOLIDAY-AWARE. The FND estimate keys off this, so a month-end
    holiday now correctly pulls the estimate ~1 day earlier (the safe direction) instead of being late."""
    return ec.last_trading_day_of_month(year, month)


def first_notice_day_estimate(delivery_year: int, delivery_month: int, root: str) -> Optional[date]:
    """ESTIMATE of First Notice Day for a physical-delivery contract, from the delivery month.

    Rule of thumb used for the SHADOW floor: FND ≈ last business day of the month BEFORE the delivery
    month — correct/standard for CBOT grains (ZC/ZS) and COMEX metals (GC/SI/HG); safe-EARLY for
    Treasuries (ZN/ZB/ZF), whose delivery is driven by delivery-month position days (this estimate sits
    ~1 BD before First Delivery Day, so it's conservative — not a literal FND). **NOT safe for energy
    (CL/NG): their true expiry/last-trade falls BEFORE this prior-month-end estimate**, so the FND floor
    would land LATE — energy is protected only by the `last_trade` cap in `RollDates.recommended`, not
    by this value. Returns None for cash/FX. An ESTIMATE — verified per-market before any live roll."""
    if settlement(root) != PHYSICAL:
        return None
    m0 = delivery_month - 1
    y0 = delivery_year
    if m0 == 0:
        m0, y0 = 12, delivery_year - 1
    return _last_business_day_of_month(y0, m0)


@dataclass(frozen=True)
class RollDates:
    """All candidate roll dates for one held contract (any may be None if inputs are missing)."""
    root: str
    settlement: str
    delivery_month: Optional[str]        # 'YYYYMM' from IBKR contractMonth
    last_trade: Optional[date]           # realized last-trade / expiry
    scheduled_expiry: Optional[date]     # day-15-of-delivery-month proxy (backtest basis)
    fixed_roll: Optional[date]           # scheduled_expiry - ROLL_BUFFER_DAYS (the backtest rule)
    fnd_floor: Optional[date]            # First-Notice-Day estimate (physical only)
    liquidity_roll: Optional[date]       # vol/OI crossover (instrumentation; None until wired)

    @property
    def last_trade_cap(self) -> Optional[date]:
        """Hard backstop: LAST_TRADE_BUFFER_BDAYS trading days before the IBKR last-trade date. The one
        per-contract truth from the broker — this is what protects energy (whose expiry precedes the
        prior-month-end FND estimate) from being recommended a roll AFTER it has stopped trading."""
        return _minus_business_days(self.last_trade, LAST_TRADE_BUFFER_BDAYS) if self.last_trade else None

    @property
    def recommended(self) -> Optional[date]:
        """The roll date we'd ACT on: earliest (most conservative) of ALL binding candidates — the
        backtested `fixed` rule, the `fnd` floor (physical only), AND the `last_trade` cap (always).
        `liquidity_roll` is logged, not yet binding."""
        cands = [d for d in (self.fixed_roll, self.fnd_floor, self.last_trade_cap) if d is not None]
        return min(cands) if cands else None


def _parse_yyyymm(s: Optional[str]):
    try:
        s = str(s)
        y, m = int(s[:4]), int(s[4:6])
    except (TypeError, ValueError):
        return None
    return (y, m) if 1 <= m <= 12 and 1900 <= y <= 2100 else None   # reject malformed IBKR months


def _parse_yyyymmdd(s: Optional[str]) -> Optional[date]:
    try:
        s = str(s)
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (TypeError, ValueError):
        return None


def compute_roll_dates(root: str, *, contract_month: Optional[str] = None,
                       last_trade: Optional[str] = None,
                       liquidity_roll: Optional[date] = None) -> RollDates:
    """Compute all candidate roll dates for a held contract from the fields IBKR actually returns
    (`contractMonth`='YYYYMM', `lastTradeDateOrContractMonth`='YYYYMMDD'). Pure/deterministic."""
    root = str(root).upper()
    stl = settlement(root)
    ym = _parse_yyyymm(contract_month)
    lt = _parse_yyyymmdd(last_trade)
    sched = fixed = fnd = None
    if ym is not None:
        y, m = ym
        sched = date(y, m, 15)                                   # day-15 scheduled-expiry proxy
        fixed = sched - timedelta(days=ROLL_BUFFER_DAYS)
        fnd = first_notice_day_estimate(y, m, root)
    return RollDates(root=root, settlement=stl, delivery_month=contract_month, last_trade=lt,
                     scheduled_expiry=sched, fixed_roll=fixed, fnd_floor=fnd,
                     liquidity_roll=liquidity_roll)


def should_roll(rd: RollDates, today: date) -> bool:
    """SHADOW decision: is `today` at/after the recommended (earliest-binding) roll date?"""
    rec = rd.recommended
    return rec is not None and today >= rec


def roll_report(root: str, today: date, *, contract_month: Optional[str] = None,
                last_trade: Optional[str] = None,
                liquidity_roll: Optional[date] = None) -> Dict[str, object]:
    """Instrumentation: compute the candidate roll dates + the shadow decision, LOG them, and return
    a snapshot dict. This is what accrues the fixed-vs-FND-vs-liquidity gap data during the R1.3 soak.
    Places/rolls NOTHING."""
    rd = compute_roll_dates(root, contract_month=contract_month, last_trade=last_trade,
                            liquidity_roll=liquidity_roll)
    due = should_roll(rd, today)
    snap = {
        "root": rd.root, "settlement": rd.settlement, "delivery_month": rd.delivery_month,
        "today": today.isoformat(),
        "last_trade": rd.last_trade.isoformat() if rd.last_trade else None,
        "fixed_roll": rd.fixed_roll.isoformat() if rd.fixed_roll else None,
        "fnd_floor": rd.fnd_floor.isoformat() if rd.fnd_floor else None,
        "last_trade_cap": rd.last_trade_cap.isoformat() if rd.last_trade_cap else None,
        "liquidity_roll": rd.liquidity_roll.isoformat() if rd.liquidity_roll else None,
        "recommended": rd.recommended.isoformat() if rd.recommended else None,
        "roll_due": due,
    }
    log.info("ibkr ROLL-SHADOW [%s/%s]: due=%s recommended=%s (fixed=%s fnd=%s liq=%s last_trade=%s)",
             rd.root, rd.settlement, due, snap["recommended"], snap["fixed_roll"], snap["fnd_floor"],
             snap["liquidity_roll"], snap["last_trade"])
    return snap
