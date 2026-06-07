"""
Trend (TSMOM) sleeve live-vs-backtest tracking artifact.

Mirrors pead_tracker.py. The live trend sleeve will diverge from the clean
+0.71 standalone backtest for two structural reasons worth attributing rather
than swallowing: (1) the backtest used yfinance auto-adjusted closes while live
uses Alpaca daily bars (different split/dividend adjustment), and (2) a wall-clock
Monday rebalance vs the backtest's modular 5-day grid. This module surfaces the
realized trend-book P&L so that divergence is measurable.

Two artifacts:
  1. Daily row (record_daily): per rebalance/trading day — n positions, gross
     deployed, turnover, daily/cumulative P&L.
  2. Weekly rollup (weekly_rollup): realized Sharpe of the trend-tagged book vs the
     +0.71 backtest expectation, emailed via notifier.enqueue("trend_weekly").

Standalone sqlite (mirrors notifier.py / pead_tracker.py) — append-only, never
raises, safe to call from the live scheduled job. Scoped entirely to the trend sleeve.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
# Env-overridable so pytest-xdist workers each use an isolated file (no lock) —
# same convention as pead_tracker.py (per PR #400 per-worker SQLite isolation).
DB_PATH = Path(os.environ.get("MRTRADER_TREND_TRACKING_DB", str(_ROOT / "data" / "trend_tracking.db")))

BACKTEST_SHARPE = 0.71  # validated standalone TSMOM CPCV/backtest expectation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trend_daily (
    trade_date          TEXT PRIMARY KEY,
    n_positions         INTEGER,
    gross_deployed      REAL,
    turnover            REAL,
    realized_pnl        REAL,
    unrealized_pnl      REAL,
    daily_pnl           REAL,
    cumulative_pnl      REAL,
    extra               TEXT,
    created_at          REAL
);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=10)
    c.execute("PRAGMA journal_mode=WAL;")
    c.executescript(_SCHEMA)
    return c


def record_daily(
    trade_date: _date | str | None = None,
    *,
    n_positions: int | None = None,
    gross_deployed: float | None = None,
    turnover: float | None = None,
    realized_pnl: float | None = None,
    unrealized_pnl: float | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Upsert today's trend tracking row. Never raises.

    PARTIAL UPSERT semantics: every field defaults to None meaning "don't touch"
    (COALESCE(excluded, existing) on conflict). daily_pnl/cumulative_pnl are derived
    only when P&L inputs are supplied this call. Returns True on success.
    """
    if trade_date is None:
        trade_date = _date.today()
    td = trade_date.isoformat() if isinstance(trade_date, _date) else str(trade_date)

    daily_pnl = None
    pnl_supplied = realized_pnl is not None or unrealized_pnl is not None
    if pnl_supplied:
        daily_pnl = float(realized_pnl or 0.0) + float(unrealized_pnl or 0.0)

    def _i(x):
        return int(x) if x is not None else None

    def _f(x):
        return float(x) if x is not None else None

    try:
        with _conn() as c:
            cumulative_pnl = None
            if pnl_supplied:
                prior = c.execute(
                    "SELECT cumulative_pnl FROM trend_daily "
                    "WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1",
                    (td,),
                ).fetchone()
                prior_cum = float(prior[0]) if prior and prior[0] is not None else 0.0
                cumulative_pnl = prior_cum + daily_pnl

            c.execute(
                "INSERT INTO trend_daily(trade_date, n_positions, gross_deployed, "
                "turnover, realized_pnl, unrealized_pnl, daily_pnl, cumulative_pnl, "
                "extra, created_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(trade_date) DO UPDATE SET "
                "n_positions=COALESCE(excluded.n_positions, n_positions), "
                "gross_deployed=COALESCE(excluded.gross_deployed, gross_deployed), "
                "turnover=COALESCE(excluded.turnover, turnover), "
                "realized_pnl=COALESCE(excluded.realized_pnl, realized_pnl), "
                "unrealized_pnl=COALESCE(excluded.unrealized_pnl, unrealized_pnl), "
                "daily_pnl=COALESCE(excluded.daily_pnl, daily_pnl), "
                "cumulative_pnl=COALESCE(excluded.cumulative_pnl, cumulative_pnl), "
                "extra=COALESCE(excluded.extra, extra)",
                (
                    td, _i(n_positions), _f(gross_deployed), _f(turnover),
                    _f(realized_pnl), _f(unrealized_pnl), daily_pnl, cumulative_pnl,
                    (json.dumps(extra, default=str) if extra is not None else None),
                    time.time(),
                ),
            )
        return True
    except Exception:
        log.exception("trend_tracker.record_daily failed (swallowed)")
        return False


def read_daily(since: _date | str | None = None) -> list[dict[str, Any]]:
    """Return daily rows (optionally since a date) as a list of dicts. Never raises."""
    try:
        with _conn() as c:
            c.row_factory = sqlite3.Row
            if since is not None:
                sd = since.isoformat() if isinstance(since, _date) else str(since)
                rows = c.execute(
                    "SELECT * FROM trend_daily WHERE trade_date >= ? ORDER BY trade_date",
                    (sd,),
                ).fetchall()
            else:
                rows = c.execute("SELECT * FROM trend_daily ORDER BY trade_date").fetchall()
            return [dict(r) for r in rows]
    except Exception:
        log.exception("trend_tracker.read_daily failed (swallowed)")
        return []


def _realized_sharpe(daily_pnls: list[float], gross: list[float]) -> float | None:
    """Annualised Sharpe of daily P&L returns (pnl / gross deployed). None if insufficient."""
    import numpy as np
    rets = [pnl / g for pnl, g in zip(daily_pnls, gross) if g and g > 0]
    if len(rets) < 2:
        return None
    arr = np.array(rets, dtype=float)
    sd = arr.std(ddof=1)
    if sd == 0:
        return None
    return float(arr.mean() / sd * np.sqrt(252))


def weekly_rollup(
    week_ending: _date | str | None = None,
    send: bool = True,
    min_days: int = 3,
) -> dict[str, Any]:
    """Trailing-7-day trend realized Sharpe vs the +0.71 backtest, optionally emailed
    via notifier.enqueue("trend_weekly", ...). Never raises.

    Vacuous-email guard: if fewer than ``min_days`` rows have gross_deployed > 0 the
    email is SKIPPED. Returns the payload dict (also when send=False, for testing).
    """
    if week_ending is None:
        week_ending = _date.today()
    we = week_ending if isinstance(week_ending, _date) else _date.fromisoformat(str(week_ending))
    since = we - timedelta(days=6)
    rows = read_daily(since=since)

    def _g(r, k, default=0):
        v = r.get(k)
        return default if v is None else v

    daily_pnls = [_g(r, "daily_pnl", 0.0) for r in rows]
    gross = [_g(r, "gross_deployed", 0.0) for r in rows]
    sharpe = _realized_sharpe(daily_pnls, gross)
    n_deployed_days = sum(1 for g in gross if g and g > 0)
    cum_pnl = _g(rows[-1], "cumulative_pnl", 0.0) if rows else 0.0
    avg_positions = (round(sum(_g(r, "n_positions") for r in rows) / len(rows), 1)
                     if rows else 0)
    avg_gross = (round(sum(gross) / max(1, n_deployed_days), 2) if n_deployed_days else 0.0)

    payload: dict[str, Any] = {
        "week_ending": we.isoformat(),
        "realized_sharpe": (round(sharpe, 3) if sharpe is not None else "n/a"),
        "backtest_sharpe": f"+{BACKTEST_SHARPE:.3f}",
        "n_days": len(rows),
        "avg_positions": avg_positions,
        "avg_gross_deployed": avg_gross,
        "cumulative_pnl": round(float(cum_pnl), 2),
    }

    if n_deployed_days < min_days:
        payload["skipped"] = "insufficient data"
        log.info(
            "trend_tracker.weekly_rollup: only %d deployed day(s) (< min_days=%d) — "
            "skipping email for week_ending=%s",
            n_deployed_days, min_days, we.isoformat(),
        )
        return payload

    if send:
        try:
            from app.notifications import notifier
            notifier.enqueue("trend_weekly", payload,
                             dedup_key=f"trend_weekly_{we.isoformat()}")
        except Exception:
            log.exception("trend_tracker.weekly_rollup notify failed (swallowed)")

    return payload
