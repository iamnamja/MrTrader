"""P1-4 — live-vs-sim back-validation for the trend sleeve (the live edge).

A strategy can't graduate on research alone — it must show **live ≈ sim**. The existing
trend_tracker compares realized Sharpe to a CONSTANT (+0.71 backtest), which can't isolate
execution friction. This module adds the missing instrument: a date-matched daily
INTENDED-vs-ACTUAL tracking-error series with a graduation verdict.

**Why intended-vs-actual (not an independently reconstructed backtest):** an independent sim
book rebalances on a different calendar (the deep-history modular grid, not live Mondays) and
uses split/div-ADJUSTED yfinance closes vs Alpaca's raw marks — both inject divergence that
has NOTHING to do with execution. Instead we capture the sleeve's OWN intended weights at each
rebalance (`run_trend_rebalance` -> summary['intended_weights'], post inverse-vol + alloc +
governor) and replay them on the SAME Alpaca price panel and SAME calendar as the actual held
book. The only thing that can differ is execution friction: whole-share rounding, the 80%
gross cap, per-name caps, PEAD crowding, partial/failed fills, timing. That is exactly what
"live ≈ sim" must measure.

Layers:
  1. record_daily_snapshot() — EOD (trading days only): full-universe Alpaca closes + the
     trend-tagged held qty + NAV.
  2. record_rebalance_intent() — on the rebalance day: the intended weights + blocked count +
     governor multiplier (diagnostics). Carried forward until the next rebalance.
  3. compute_report() / weekly_report() — per consecutive trading-day pair, build
     intended_return = Σ intended_w · r_sym and actual_return = Σ actual_w · r_sym (same
     prices), then report correlation / annualized tracking error / drift / drag and a
     PASS / WATCH / FAIL / BUILDING verdict.

Report-only. Touches no order path. Mirrors trend_tracker.py (append-only sqlite, never
raises, env-overridable DB for pytest-xdist isolation).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("MRTRADER_BACKVAL_DB", str(_ROOT / "data" / "back_validation.db")))

# Graduation thresholds (trailing window). Intended-vs-actual tracking is TIGHT by
# construction (same book, same prices) — friction should be small. Report-only.
MIN_DAYS_FOR_VERDICT = 15           # below this -> BUILDING
PASS_MIN_CORR = 0.90                # daily intended-vs-actual correlation
PASS_MAX_ABS_DRIFT_ANN = 0.015      # |annualized actual-minus-intended drift| <= 1.5%/yr
PASS_MAX_TE_ANN = 0.02              # annualized tracking error <= 2%/yr
WATCH_MIN_CORR = 0.75               # below this (with valid corr) -> FAIL
ANN = 252

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trend_backval_daily (
    trade_date        TEXT PRIMARY KEY,
    nav               REAL,
    prices            TEXT,    -- JSON {symbol: close} for the full trend universe
    positions         TEXT,    -- JSON {symbol: qty} trend-tagged held
    intended_weights  TEXT,    -- JSON {symbol: fraction_of_nav} (rebalance days; carried fwd)
    n_positions       INTEGER,
    n_blocked         INTEGER, -- RM/quality blocks at the rebalance (missed-trade proxy)
    overlay_mult      REAL,    -- governor multiplier applied at rebalance (diagnostic)
    created_at        REAL
);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=10)
    c.execute("PRAGMA journal_mode=WAL;")
    c.executescript(_SCHEMA)
    return c


# ──────────────────────────────────────────────────────────────────────────────────
# Layer 1 — EOD snapshot (trading days only)
# ──────────────────────────────────────────────────────────────────────────────────
def record_daily_snapshot(db=None, *, asof: _date | str | None = None) -> bool:
    """Capture the trend sleeve's EOD state: full-universe Alpaca closes + trend-held qty +
    NAV. Trading days only (skips holidays/weekends via the latest-bar-date guard). Never
    raises. The daily INTENDED/ACTUAL returns are derived later in compute_report.
    """
    try:
        from app.integrations import get_alpaca_client
        from app.database.db import SessionLocal
        from app.live_trading import trend_sleeve as _ts
        from app.database.agent_config import get_agent_config

        today = asof or _date.today()
        today = today if isinstance(today, _date) else _date.fromisoformat(str(today))

        _own_db = db is None
        if _own_db:
            db = SessionLocal()
        try:
            universe = [s.strip().upper()
                        for s in str(get_agent_config(db, "pm.trend_universe")).split(",")
                        if s.strip()]
            alpaca = get_alpaca_client()
            held = _ts._current_trend_positions(db, alpaca)   # {sym: int qty}, trend-tagged
        finally:
            if _own_db:
                db.close()

        prices_df = _ts._fetch_prices(alpaca, universe)
        if prices_df is None or prices_df.empty:
            log.warning("back_validation: price panel unavailable — snapshot skipped")
            return False
        # Trading-day guard (M1): only snapshot if the latest bar is for `today`. On a
        # holiday the 16:15 cron still fires but the panel's last bar is stale -> skip, so
        # we never record a flat duplicate or mis-date a multi-day move.
        last_bar = prices_df.index[-1]
        last_bar = last_bar.date() if hasattr(last_bar, "date") else last_bar
        if str(last_bar) != today.isoformat():
            log.info("back_validation: latest bar %s != today %s (non-trading day) — skip",
                     last_bar, today.isoformat())
            return False

        prices = {s: float(v) for s, v in prices_df.iloc[-1].to_dict().items()
                  if v is not None and float(v) == float(v)}  # drop NaN
        acct = alpaca.get_account()
        nav = float(acct.get("portfolio_value") or acct.get("equity") or 0.0)
        positions = {s: float(q) for s, q in held.items() if q}

        with _conn() as c:
            c.execute(
                "INSERT INTO trend_backval_daily(trade_date, nav, prices, positions, "
                "n_positions, created_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(trade_date) DO UPDATE SET nav=excluded.nav, "
                "prices=excluded.prices, positions=excluded.positions, "
                "n_positions=excluded.n_positions",
                (today.isoformat(), nav, json.dumps(prices), json.dumps(positions),
                 len(positions), time.time()),
            )
        log.info("back_validation snapshot %s: nav=%.2f held=%d prices=%d",
                 today.isoformat(), nav, len(positions), len(prices))
        return True
    except Exception:
        log.exception("back_validation.record_daily_snapshot failed (swallowed)")
        return False


def record_rebalance_intent(summary: dict, *, asof: _date | str | None = None) -> bool:
    """Persist the sleeve's INTENDED book + diagnostics from a run_trend_rebalance summary.
    Upserts onto the rebalance day's row (COALESCE-merges with the daily snapshot). Never
    raises.

    Records ONLY for genuine LIVE rebalances. A SHADOW run still produces intended_weights and
    status='ok', but places no orders -> the actual held book is empty -> recording its intent
    would make the verdict scream FAIL (intended full book vs ~0 actual) exactly while the
    system behaves as designed. So shadow/dormant/failed runs are skipped (no intent recorded
    -> compute_report stays BUILDING, the correct state for a non-trading book)."""
    try:
        iw = summary.get("intended_weights")
        if (not iw or summary.get("status") not in ("ok", None)
                or summary.get("mode") != "live"):
            return False
        today = asof or _date.today()
        td = today.isoformat() if isinstance(today, _date) else str(today)
        n_blocked = len(summary.get("blocked", []) or [])
        overlay = summary.get("overlay_mult")
        with _conn() as c:
            c.execute(
                "INSERT INTO trend_backval_daily(trade_date, intended_weights, n_blocked, "
                "overlay_mult, created_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(trade_date) DO UPDATE SET "
                "intended_weights=excluded.intended_weights, n_blocked=excluded.n_blocked, "
                "overlay_mult=excluded.overlay_mult",
                (td, json.dumps(iw), n_blocked,
                 (float(overlay) if overlay is not None else None), time.time()),
            )
        log.info("back_validation intent %s: %d names, blocked=%d", td, len(iw), n_blocked)
        return True
    except Exception:
        log.exception("back_validation.record_rebalance_intent failed (swallowed)")
        return False


def read_daily(since: _date | str | None = None) -> list[dict[str, Any]]:
    """Return snapshot rows (optionally since a date) as dicts. Never raises."""
    try:
        with _conn() as c:
            c.row_factory = sqlite3.Row
            if since is not None:
                sd = since.isoformat() if isinstance(since, _date) else str(since)
                rows = c.execute(
                    "SELECT * FROM trend_backval_daily WHERE trade_date >= ? ORDER BY trade_date",
                    (sd,)).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM trend_backval_daily ORDER BY trade_date").fetchall()
            return [dict(r) for r in rows]
    except Exception:
        log.exception("back_validation.read_daily failed (swallowed)")
        return []


# ──────────────────────────────────────────────────────────────────────────────────
# Layer 2 — report (pure computation; no network)
# ──────────────────────────────────────────────────────────────────────────────────
def _json(s: Optional[str]) -> dict:
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def daily_pairs(snapshots: list[dict]) -> tuple[list[tuple[float, float]], dict]:
    """Build (actual_return, intended_return) NAV-fraction pairs over consecutive trading
    days, replaying the carried-forward intended weights on the SAME Alpaca prices as the
    actual held book. Pure. Returns (pairs, diag) where diag counts dropped days.

      actual_return_t   = Σ_sym  (qty_{t-1}·price_{sym,t-1}/nav_{t-1}) · (price_{sym,t}/price_{sym,t-1} - 1)
      intended_return_t = Σ_sym  intended_w_sym(≤t-1)                   · (price_{sym,t}/price_{sym,t-1} - 1)

    Only symbols priced on BOTH days contribute (no look-ahead; entries/exits handled by the
    full-universe price panel). Days before the first rebalance (no intended book) are dropped.
    """
    pairs: list[tuple[float, float]] = []
    diag = {"no_intent_days": 0, "bad_days": 0}
    prev = None
    carried_intent: Optional[dict] = None
    for row in snapshots:
        iw = _json(row.get("intended_weights"))
        # carry-forward intent updates AFTER using prev as the t-1 anchor
        if prev is not None:
            try:
                p0 = _json(prev.get("prices"))
                p1 = _json(row.get("prices"))
                pos0 = _json(prev.get("positions"))
                nav0 = float(prev.get("nav") or 0.0)
                if carried_intent is None:
                    diag["no_intent_days"] += 1
                elif nav0 > 0:
                    a_ret = 0.0
                    i_ret = 0.0
                    for sym, px1 in p1.items():
                        px0 = p0.get(sym)
                        if not px0 or not px1:
                            continue
                        r = float(px1) / float(px0) - 1.0
                        qty = float(pos0.get(sym, 0.0))
                        a_ret += (qty * float(px0) / nav0) * r
                        i_ret += float(carried_intent.get(sym, 0.0)) * r
                    pairs.append((a_ret, i_ret))
                else:
                    diag["bad_days"] += 1
            except Exception:
                diag["bad_days"] += 1
                log.exception("daily_pairs: bad row %s", row.get("trade_date"))
        if iw:
            carried_intent = iw
        prev = row
    return pairs, diag


def _tracking_metrics(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Pure tracking metrics from aligned (actual, intended) daily-return pairs."""
    import numpy as np
    n = len(pairs)
    if n < 2:
        return {"n_days": n, "corr": None, "te_ann": None, "drift_ann": None,
                "actual_cum": None, "intended_cum": None, "drag_bps_day": None,
                "actual_sharpe": None, "intended_sharpe": None}
    actual = np.array([p[0] for p in pairs], dtype=float)
    intended = np.array([p[1] for p in pairs], dtype=float)
    diff = actual - intended

    def _sharpe(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(ANN)) if sd > 0 else None

    corr = None
    if actual.std(ddof=1) > 0 and intended.std(ddof=1) > 0:
        corr = float(np.corrcoef(actual, intended)[0, 1])
    return {
        "n_days": n, "corr": corr,
        "te_ann": float(diff.std(ddof=1) * np.sqrt(ANN)),
        "drift_ann": float(diff.mean() * ANN),
        "actual_cum": float(np.prod(1.0 + actual) - 1.0),
        "intended_cum": float(np.prod(1.0 + intended) - 1.0),
        "drag_bps_day": float(diff.mean() * 1e4),
        "actual_sharpe": _sharpe(actual), "intended_sharpe": _sharpe(intended),
    }


def _verdict(m: dict) -> str:
    """Lead with tracking error + drift (robust when the sleeve is legitimately flat);
    correlation is a secondary gate. BUILDING below the min-days floor."""
    if m["n_days"] < MIN_DAYS_FOR_VERDICT or m["te_ann"] is None:
        return "BUILDING"
    te, drift, corr = m["te_ann"], abs(m["drift_ann"]), m["corr"]
    # FAIL on a big gap or a clearly-broken correlation (only when corr is well-defined).
    if te > 2 * PASS_MAX_TE_ANN or drift > 2 * PASS_MAX_ABS_DRIFT_ANN or \
            (corr is not None and corr < WATCH_MIN_CORR):
        return "FAIL"
    # PASS when the gap is tight. A flat-but-tracking window (corr undefined, te tiny) PASSES.
    if te <= PASS_MAX_TE_ANN and drift <= PASS_MAX_ABS_DRIFT_ANN and \
            (corr is None or corr >= PASS_MIN_CORR):
        return "PASS"
    return "WATCH"


@dataclass
class BackValReport:
    verdict: str
    n_days: int
    corr: Optional[float]
    tracking_error_ann: Optional[float]
    drift_ann: Optional[float]
    slippage_drag_bps_day: Optional[float]
    actual_cum_return: Optional[float]
    intended_cum_return: Optional[float]
    actual_sharpe_navcontrib: Optional[float]
    intended_sharpe_navcontrib: Optional[float]
    governor_days: int
    total_blocked: int
    window_start: Optional[str]
    window_end: Optional[str]
    note: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def compute_report(start: _date | str | None = None,
                   end: _date | str | None = None) -> BackValReport:
    """Build the intended-vs-actual tracking report over the snapshot window. Never raises.

    Distinguishes ERROR (something broke) from BUILDING (just not enough history yet)."""
    try:
        snaps = read_daily(since=start)
        if end is not None:
            ed = end.isoformat() if isinstance(end, _date) else str(end)
            snaps = [s for s in snaps if s["trade_date"] <= ed]
        gov_days = sum(1 for s in snaps if (s.get("overlay_mult") or 1.0) < 1.0)
        blocked = sum(int(s.get("n_blocked") or 0) for s in snaps)
        ws = snaps[0]["trade_date"] if snaps else None
        we = snaps[-1]["trade_date"] if snaps else None

        pairs, diag = daily_pairs(snaps)
        m = _tracking_metrics(pairs)
        notes = []
        if diag["no_intent_days"]:
            notes.append(f"{diag['no_intent_days']} day(s) had no intended book yet")
        if diag["bad_days"]:
            notes.append(f"{diag['bad_days']} day(s) dropped (missing nav/snapshot)")
        note = "; ".join(notes) or None
        # Detect a stuck/empty instrument: snapshots exist but no usable pairs.
        if len(snaps) >= MIN_DAYS_FOR_VERDICT and m["n_days"] == 0:
            log.warning("back_validation: %d snapshots but 0 usable pairs (%s)", len(snaps), diag)
        return BackValReport(
            verdict=_verdict(m), n_days=m["n_days"], corr=m["corr"],
            tracking_error_ann=m["te_ann"], drift_ann=m["drift_ann"],
            slippage_drag_bps_day=m["drag_bps_day"],
            actual_cum_return=m["actual_cum"], intended_cum_return=m["intended_cum"],
            actual_sharpe_navcontrib=m["actual_sharpe"],
            intended_sharpe_navcontrib=m["intended_sharpe"],
            governor_days=gov_days, total_blocked=blocked,
            window_start=ws, window_end=we, note=note)
    except Exception:
        log.exception("back_validation.compute_report failed (swallowed)")
        return BackValReport("ERROR", 0, None, None, None, None, None, None, None, None,
                             0, 0, None, None, note="exception in compute_report")


def weekly_report(send: bool = True, lookback_days: int = 120) -> dict:
    """Trailing-window intended-vs-actual report, optionally emailed. Never raises.

    Emails on a real verdict (PASS/WATCH/FAIL) and on ERROR (so a broken/stuck instrument is
    surfaced, not silently swallowed). BUILDING is logged but not emailed."""
    start = _date.today() - timedelta(days=lookback_days)
    rep = compute_report(start=start)
    payload = rep.as_dict()
    if rep.verdict == "BUILDING":
        payload["skipped"] = f"building history ({rep.n_days}/{MIN_DAYS_FOR_VERDICT} days)"
        log.info("back_validation.weekly_report: %s", payload["skipped"])
        return payload
    if send:
        try:
            from app.notifications import notifier
            notifier.enqueue("trend_backval_weekly", payload,
                             dedup_key=f"trend_backval_{rep.window_end}")
        except Exception:
            log.exception("back_validation.weekly_report notify failed (swallowed)")
    return payload
