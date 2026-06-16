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


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=10)
    c.execute("PRAGMA journal_mode=WAL;")
    c.executescript(_SCHEMA)
    return c


def record_daily(trade_date: _date | str | None = None, *, n_positions: int | None = None,
                 tbill_deployed: float | None = None, cash_buffer: float | None = None,
                 extra: dict[str, Any] | None = None) -> bool:
    """Upsert today's cash-sleeve row. Partial upsert (None = don't touch). Never raises."""
    td = trade_date or _date.today()
    td = td.isoformat() if isinstance(td, _date) else str(td)

    def _f(x):
        return float(x) if x is not None else None

    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO cash_daily(trade_date, n_positions, tbill_deployed, cash_buffer, "
                "extra, created_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(trade_date) DO UPDATE SET "
                "n_positions=COALESCE(excluded.n_positions, n_positions), "
                "tbill_deployed=COALESCE(excluded.tbill_deployed, tbill_deployed), "
                "cash_buffer=COALESCE(excluded.cash_buffer, cash_buffer), "
                "extra=COALESCE(excluded.extra, extra)",
                (td, (int(n_positions) if n_positions is not None else None),
                 _f(tbill_deployed), _f(cash_buffer),
                 (json.dumps(extra, default=str) if extra is not None else None), time.time()))
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
