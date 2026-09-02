"""Standing accounting + control-liveness invariants (2026-09-02).

WHY THIS EXISTS
---------------
Every serious defect found in the 2026-08/09 review was caught the same way: by checking arithmetic
against the broker instead of trusting numbers that looked plausible. But those checks were run BY
HAND, reactively, because someone happened to look. Nothing ran them on a schedule, and the cost of
that shows in how long each defect survived:

    regime scorer pinned to v9 while v40 existed      ~7 weeks
    crash governor unable to compute (VIX3M gone)     ~5 weeks
    order-status enum leak disabling three features   months
    DBC double-buy, 11.5% of equity                   until someone looked

Not one of those raised an error. Every one would have tripped an invariant below.

DESIGN
------
* READ-ONLY. Nothing here mutates state, places orders, or escalates the kill switch. The position
  check deliberately calls the pure `reconcile()` rather than `reconcile_and_alert(mode=enforce)`,
  which would latch HALT_NEW_RISK — a monitor must never change what it monitors.
* EVERY CHECK INDEPENDENTLY GUARDED. One failing check degrades the report; it never suppresses the
  others. A monitor its own errors can silence is worse than no monitor.
* A check that cannot EVALUATE is a breach, not a pass. "Could not determine" is the state the
  crash governor sat in for five weeks while reporting nothing.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Tolerances. Generous against fees + intraday mark timing; a real attribution bug is far larger
# (the errors this replaces were $52, $199, $1,051).
ACCOUNT_IDENTITY_TOL = 25.0
SLEEVE_NAV_TOL = 25.0
# Trading days the P&L scorecard may lag before we call it stalled — long enough for a weekend
# plus a holiday, short enough that three months of NULLs could not recur unnoticed.
SCORECARD_MAX_STALE_DAYS = 4


@dataclass
class Check:
    name: str
    ok: Optional[bool]          # None = could not evaluate (treated as a breach)
    detail: str = ""
    value: Optional[float] = None

    @property
    def breached(self) -> bool:
        return self.ok is not True


@dataclass
class InvariantReport:
    checks: List[Check] = field(default_factory=list)

    @property
    def breaches(self) -> List[Check]:
        return [c for c in self.checks if c.breached]

    @property
    def ok(self) -> bool:
        return not self.breaches

    def summary(self) -> str:
        if self.ok:
            return f"all {len(self.checks)} invariants OK"
        return "; ".join(f"{c.name}: {c.detail}" for c in self.breaches)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "n_checks": len(self.checks),
                "breaches": [asdict(c) for c in self.breaches],
                "checks": [asdict(c) for c in self.checks]}


def _deposits(alpaca) -> float:
    """Net external cash in (JNLC/CSD/CSW), read from the broker rather than hardcoded."""
    from app.config import settings
    import httpx
    base = (getattr(settings, "alpaca_base_url", "") or "").rstrip("/")
    hdr = {"APCA-API-KEY-ID": settings.alpaca_api_key,
           "APCA-API-SECRET-KEY": settings.alpaca_secret_key}
    total = 0.0
    for kind in ("JNLC", "CSD", "CSW"):
        r = httpx.get(f"{base}/account/activities/{kind}", timeout=20,
                      params={"page_size": 100}, headers=hdr)
        if r.status_code == 200 and isinstance(r.json(), list):
            total += sum(float(x.get("net_amount") or 0) for x in r.json())
    return total


def _fees(alpaca) -> float:
    from app.config import settings
    import httpx
    base = (getattr(settings, "alpaca_base_url", "") or "").rstrip("/")
    r = httpx.get(f"{base}/account/activities/FEE", timeout=20, params={"page_size": 200},
                  headers={"APCA-API-KEY-ID": settings.alpaca_api_key,
                           "APCA-API-SECRET-KEY": settings.alpaca_secret_key})
    if r.status_code != 200 or not isinstance(r.json(), list):
        return 0.0
    return sum(float(x.get("net_amount") or 0) for x in r.json())


def check_account_identity(alpaca=None) -> Check:
    """equity == deposits + FIFO_realized + unrealized + fees.

    The master arithmetic check. A break means fills are missing, positions are phantom, or the
    P&L engine disagrees with the broker — the class of problem that produced a double-buy.
    """
    from app.analytics.execution_pnl import compute_realized_pnl
    if alpaca is None:
        from app.integrations.alpaca import AlpacaClient
        alpaca = AlpacaClient()
    fills = alpaca.get_all_orders()
    pos = alpaca.get_positions()
    acct = alpaca.get_account()
    pm = {p["symbol"]: float(p["qty"]) for p in pos}
    realized = sum(v["realized_pnl"] for v in compute_realized_pnl(fills, pm).values()
                   if v["realized_pnl"] is not None)
    unreal = sum(float(p["unrealized_pl"]) for p in pos)
    equity = float(acct["equity"])
    predicted = _deposits(alpaca) + realized + unreal + _fees(alpaca)
    resid = equity - predicted
    ok = abs(resid) <= ACCOUNT_IDENTITY_TOL
    return Check("account_identity", ok,
                 f"residual ${resid:+,.2f} (tol ${ACCOUNT_IDENTITY_TOL:.0f})", resid)


def check_sleeve_pnl_ties_to_nav() -> Check:
    """Σ per-sleeve cumulative P&L == the account's NAV move over the same window."""
    from app.integrations.alpaca import AlpacaClient
    from app.live_trading.sleeve_pnl import (
        compute_daily_pnl, daily_pnl_series, load_snapshots,
    )
    snaps = load_snapshots()
    if len(snaps) < 2:
        return Check("sleeve_pnl_vs_nav", None, "fewer than 2 snapshots")
    fills = AlpacaClient().get_all_orders()
    total = 0.0
    for sleeve in ("trend", "cash"):
        rows = daily_pnl_series(compute_daily_pnl(fills, snaps, sleeve=sleeve))
        marked = [r for r in rows if r.get("unrealized") is not None]
        if marked:
            total += float(marked[-1]["cumulative"])
    navs = [s["nav"] for s in snaps if s.get("nav")]
    if len(navs) < 2:
        return Check("sleeve_pnl_vs_nav", None, "no NAV series")
    resid = total - (navs[-1] - navs[0])
    ok = abs(resid) <= SLEEVE_NAV_TOL
    return Check("sleeve_pnl_vs_nav", ok,
                 f"residual ${resid:+,.2f} (tol ${SLEEVE_NAV_TOL:.0f})", resid)


def check_position_reconcile() -> Check:
    """DB intent == broker reality. READ-ONLY: the pure reconcile(), never the enforcing wrapper."""
    import app.live_trading.reconciliation as R
    from app.database import SessionLocal
    from app.integrations.alpaca import AlpacaClient
    db = SessionLocal()
    try:
        exp = R.db_expected_positions(db, R.im.ALPACA)
        pend = R.db_pending_positions(db, R.im.ALPACA)
        act = R.alpaca_actual_positions(AlpacaClient().get_positions(), R.im.ALPACA)
        res = R.reconcile(exp, act, pending_qty=pend)
    finally:
        db.close()
    if res.position_breaks:
        detail = ", ".join(f"{b.instrument_id} exp {b.expected_qty:g} act {b.actual_qty:g}"
                           for b in res.position_breaks)
        return Check("position_reconcile", False, f"{res.status}: {detail}")
    return Check("position_reconcile", True, str(res.status))


def check_crash_governor_live() -> Check:
    """Can the crash governor actually COMPUTE?

    It is fail-safe, so when its inputs vanish it returns 1.0 and looks identical to a healthy
    'no de-risk needed'. That is exactly how it sat inert for five weeks while VIX3M was missing.
    Liveness here means: enabled, and holding enough paired VIX/VIX3M history to evaluate.
    """
    import pandas as pd
    from app.database import SessionLocal
    from app.database.agent_config import get_agent_config
    from app.data.macro_history import load_macro_history
    db = SessionLocal()
    try:
        enabled = str(get_agent_config(db, "pm.crash_governor_enabled")).lower() in ("true", "1", "yes")
    finally:
        db.close()
    if not enabled:
        return Check("crash_governor_live", True, "disabled by config (not a breach)")
    mh = load_macro_history()
    if mh is None or mh.empty or "vix" not in mh.columns or "vix3m" not in mh.columns:
        return Check("crash_governor_live", False, "macro history missing vix/vix3m")
    tail = mh.tail(6)
    usable = int((tail["vix"].notna() & tail["vix3m"].notna()).sum())
    last_pair = mh[mh["vix"].notna() & mh["vix3m"].notna()]
    age_days = None
    if not last_pair.empty:
        age_days = (pd.Timestamp.today().normalize()
                    - pd.Timestamp(str(last_pair["date"].iloc[-1])[:10])).days
    ok = usable >= 2 and (age_days is not None and age_days <= 7)
    return Check("crash_governor_live", ok,
                 f"{usable}/6 recent paired obs, newest pair {age_days}d old", float(usable))


def check_scorecard_recording() -> Check:
    """Is the P&L scorecard still accruing? Self-referential on purpose — three months of NULL
    columns went unnoticed precisely because nothing asserted that recording was happening."""
    import sqlite3
    from datetime import date
    from app.live_trading import cash_tracker, trend_tracker
    stale: List[str] = []
    for name, tracker, table in (("trend", trend_tracker, "trend_daily"),
                                 ("cash", cash_tracker, "cash_daily")):
        try:
            c = sqlite3.connect(str(tracker.DB_PATH))
            row = c.execute(
                f"SELECT MAX(trade_date) FROM {table} WHERE cumulative_pnl IS NOT NULL").fetchone()
            last = row[0] if row else None
        except Exception as exc:  # noqa: BLE001
            stale.append(f"{name}: unreadable ({type(exc).__name__})")
            continue
        if not last:
            stale.append(f"{name}: no P&L rows at all")
            continue
        lag = (date.today() - date.fromisoformat(str(last)[:10])).days
        if lag > SCORECARD_MAX_STALE_DAYS:
            stale.append(f"{name}: last row {last} ({lag}d ago)")
    if stale:
        return Check("scorecard_recording", False, "; ".join(stale))
    return Check("scorecard_recording", True, "both sleeves current")


_CHECKS = (
    check_account_identity,
    check_sleeve_pnl_ties_to_nav,
    check_position_reconcile,
    check_crash_governor_live,
    check_scorecard_recording,
)


def run_all() -> InvariantReport:
    """Run every invariant. Never raises; a check that blows up is reported as a breach, because
    'could not evaluate' is the state these exist to surface."""
    report = InvariantReport()
    for fn in _CHECKS:
        name = fn.__name__.replace("check_", "")
        try:
            report.checks.append(fn())
        except Exception as exc:  # noqa: BLE001
            log.exception("invariant %s failed to evaluate", name)
            report.checks.append(Check(name, None, f"evaluation error: {type(exc).__name__}: {exc}"))
    if report.breaches:
        log.warning("INVARIANT BREACH: %s", report.summary())
    else:
        log.info("invariants: %s", report.summary())
    return report
