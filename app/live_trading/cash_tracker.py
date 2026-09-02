"""Cash / T-bill sleeve daily tracking artifact (P1-1). Mirrors trend_tracker.py.

Records the idle capital parked in T-bills each rebalance so the cash sleeve's contribution
(the risk-free yield the book now earns instead of zero) is a measurable return stream, and a
weekly rollup surfaces idle-capital utilization. Standalone sqlite, append-only, never raises.
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
DB_PATH = Path(os.environ.get("MRTRADER_CASH_TRACKING_DB",
                              str(_ROOT / "data" / "cash_tracking.db")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cash_daily (
    trade_date      TEXT PRIMARY KEY,
    n_positions     INTEGER,
    tbill_deployed  REAL,    -- $ idle capital parked in T-bills
    cash_buffer     REAL,    -- settled cash left as buffer
    extra           TEXT,
    created_at      REAL
);
"""


# P&L columns added 2026-09-01. The cash sleeve carried NO P&L fields at all, yet its SGOV income
# is comparable in magnitude to the trend sleeve's own P&L — a trend-only scorecard misrepresents
# the book. Same idempotent ALTER pattern as back_validation (SQLite has no ADD COLUMN IF NOT
# EXISTS). Semantics match trend_tracker: `unrealized_pnl` is a LEVEL, `daily_pnl` is
# realized + Δ(level).
_ADDED_COLUMNS = {
    "realized_pnl": "REAL",
    "unrealized_pnl": "REAL",
    "daily_pnl": "REAL",
    "cumulative_pnl": "REAL",
    # Dividend accrual added 2026-09-02 — see trend_tracker/_ADDED_COLUMNS and
    # app/live_trading/dividends.py. Kept separate from daily_pnl so the series still reconciles.
    "dividend_accrual": "REAL",
    "cumulative_dividend": "REAL",
    "cumulative_economic": "REAL",
}


def _ensure_columns(c: sqlite3.Connection) -> None:
    have = {r[1] for r in c.execute("PRAGMA table_info(cash_daily)").fetchall()}
    for col, typ in _ADDED_COLUMNS.items():
        if col not in have:
            c.execute(f"ALTER TABLE cash_daily ADD COLUMN {col} {typ}")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=10)
    c.execute("PRAGMA journal_mode=WAL;")
    c.executescript(_SCHEMA)
    _ensure_columns(c)
    return c


def record_daily(trade_date: _date | str | None = None, *, n_positions: int | None = None,
                 tbill_deployed: float | None = None, cash_buffer: float | None = None,
                 realized_pnl: float | None = None, unrealized_pnl: float | None = None,
                 daily_pnl_override: float | None = None,
                 cumulative_pnl_override: float | None = None,
                 dividend_accrual: float | None = None,
                 cumulative_dividend: float | None = None,
                 cumulative_economic: float | None = None,
                 extra: dict[str, Any] | None = None) -> bool:
    """Upsert today's cash-sleeve row. Partial upsert (None = don't touch). Never raises.

    `unrealized_pnl` is the LEVEL of open-position gain (mark minus cost basis), NOT the daily
    change — `daily_pnl` is derived as realized + Δ(level), so passing the level twice would
    re-count the same open gain every day and drift `cumulative_pnl`.
    """
    td = trade_date or _date.today()
    td = td.isoformat() if isinstance(td, _date) else str(td)

    def _f(x):
        return float(x) if x is not None else None

    pnl_supplied = realized_pnl is not None or unrealized_pnl is not None
    try:
        with _conn() as c:
            daily_pnl = cumulative_pnl = None
            if pnl_supplied:
                prior = c.execute(
                    "SELECT cumulative_pnl, unrealized_pnl FROM cash_daily "
                    "WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1", (td,),
                ).fetchone()
                prior_cum = float(prior[0]) if prior and prior[0] is not None else 0.0
                prior_unreal = float(prior[1]) if prior and prior[1] is not None else 0.0
                daily_pnl = float(realized_pnl or 0.0) + (float(unrealized_pnl or 0.0) - prior_unreal)
                cumulative_pnl = prior_cum + daily_pnl
            # See trend_tracker.record_daily: overrides let a caller with an already-correct
            # series write it verbatim rather than have it re-derived from a missing anchor.
            if daily_pnl_override is not None:
                daily_pnl = float(daily_pnl_override)
            if cumulative_pnl_override is not None:
                cumulative_pnl = float(cumulative_pnl_override)

            c.execute(
                "INSERT INTO cash_daily(trade_date, n_positions, tbill_deployed, cash_buffer, "
                "realized_pnl, unrealized_pnl, daily_pnl, cumulative_pnl, extra, created_at, "
                "dividend_accrual, cumulative_dividend, cumulative_economic) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(trade_date) DO UPDATE SET "
                "n_positions=COALESCE(excluded.n_positions, n_positions), "
                "tbill_deployed=COALESCE(excluded.tbill_deployed, tbill_deployed), "
                "cash_buffer=COALESCE(excluded.cash_buffer, cash_buffer), "
                "realized_pnl=COALESCE(excluded.realized_pnl, realized_pnl), "
                "unrealized_pnl=COALESCE(excluded.unrealized_pnl, unrealized_pnl), "
                "daily_pnl=COALESCE(excluded.daily_pnl, daily_pnl), "
                "cumulative_pnl=COALESCE(excluded.cumulative_pnl, cumulative_pnl), "
                "extra=COALESCE(excluded.extra, extra), "
                "dividend_accrual=COALESCE(excluded.dividend_accrual, dividend_accrual), "
                "cumulative_dividend=COALESCE(excluded.cumulative_dividend, cumulative_dividend), "
                "cumulative_economic=COALESCE(excluded.cumulative_economic, cumulative_economic)",
                (td, (int(n_positions) if n_positions is not None else None),
                 _f(tbill_deployed), _f(cash_buffer),
                 _f(realized_pnl), _f(unrealized_pnl), daily_pnl, cumulative_pnl,
                 (json.dumps(extra, default=str) if extra is not None else None), time.time(),
                 _f(dividend_accrual), _f(cumulative_dividend), _f(cumulative_economic)))
        return True
    except Exception:
        log.exception("cash_tracker.record_daily failed (swallowed)")
        return False


def read_daily(since: _date | str | None = None) -> list[dict[str, Any]]:
    try:
        with _conn() as c:
            c.row_factory = sqlite3.Row
            if since is not None:
                sd = since.isoformat() if isinstance(since, _date) else str(since)
                rows = c.execute("SELECT * FROM cash_daily WHERE trade_date >= ? ORDER BY trade_date",
                                 (sd,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM cash_daily ORDER BY trade_date").fetchall()
            return [dict(r) for r in rows]
    except Exception:
        log.exception("cash_tracker.read_daily failed (swallowed)")
        return []


def weekly_rollup(week_ending: _date | str | None = None, send: bool = True,
                  min_days: int = 1) -> dict[str, Any]:
    """Trailing-7-day idle-capital utilization, optionally emailed via notifier. Never raises."""
    we = week_ending or _date.today()
    we = we if isinstance(we, _date) else _date.fromisoformat(str(we))
    rows = read_daily(since=we - timedelta(days=6))

    def _g(r, k):
        v = r.get(k)
        return 0.0 if v is None else float(v)

    deployed = [_g(r, "tbill_deployed") for r in rows]
    avg_deployed = round(sum(deployed) / len(deployed), 2) if deployed else 0.0
    last_deployed = round(deployed[-1], 2) if deployed else 0.0
    payload: dict[str, Any] = {
        "week_ending": we.isoformat(),
        "n_days": len(rows),
        "avg_tbill_deployed": avg_deployed,
        "latest_tbill_deployed": last_deployed,
        "latest_cash_buffer": round(_g(rows[-1], "cash_buffer"), 2) if rows else 0.0,
    }
    if len(rows) < min_days:
        payload["skipped"] = "insufficient data"
        return payload
    if send:
        try:
            from app.notifications import notifier
            notifier.enqueue("cash_weekly", payload, dedup_key=f"cash_weekly_{we.isoformat()}")
        except Exception:
            log.exception("cash_tracker.weekly_rollup notify failed (swallowed)")
    return payload
