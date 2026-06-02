"""
PEAD live-vs-backtest tracking artifact.

The only real validation of the wired PEAD selector is live-vs-backtest:
the live "risk-managed" variant keeps overlays (regime multiplier, NIS sizing,
opportunity-score gate, macro-calendar block, RM 10-rule chain) on top of the
validated +0.546 long-only PEAD config, so it WILL diverge from the clean
backtest. This module surfaces that divergence so the tracking error can be
attributed to the kept overlays rather than silently swallowed.

Two artifacts:
  1. Daily row (record_daily): per trading day — signals, entered, filled,
     fill rate, gross deployed, daily/cumulative P&L, VIX, vix_block_fired,
     and per-overlay suppression counts (opportunity / macro / RM).
  2. Weekly rollup (weekly_rollup): realized Sharpe of the PEAD-tagged book vs
     the +0.546 backtest expectation, emailed via notifier.enqueue("pead_weekly").

Standalone sqlite (mirrors notifier.py) — append-only, never raises, safe to
call from the live agent loop. Scoped entirely to the PEAD selector.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import date as _date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _ROOT / "data" / "pead_tracking.db"

BACKTEST_SHARPE = 0.546  # validated CPCV long-only expectation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pead_daily (
    trade_date          TEXT PRIMARY KEY,
    n_signals           INTEGER,
    n_entered           INTEGER,
    n_filled            INTEGER,
    fill_rate           REAL,
    gross_deployed      REAL,
    realized_pnl        REAL,
    unrealized_pnl      REAL,
    daily_pnl           REAL,
    cumulative_pnl      REAL,
    vix_level           REAL,
    vix_block_fired     INTEGER,
    suppressed_opportunity INTEGER,
    suppressed_macro    INTEGER,
    suppressed_rm       INTEGER,
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
    n_signals: int = 0,
    n_entered: int = 0,
    n_filled: int = 0,
    gross_deployed: float = 0.0,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    vix_level: float | None = None,
    vix_block_fired: bool = False,
    suppressed_opportunity: int = 0,
    suppressed_macro: int = 0,
    suppressed_rm: int = 0,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Upsert today's PEAD tracking row. Never raises.

    cumulative_pnl is computed as prior-day cumulative + this day's daily P&L
    (realized + unrealized). Returns True on success.
    """
    if trade_date is None:
        trade_date = _date.today()
    td = trade_date.isoformat() if isinstance(trade_date, _date) else str(trade_date)

    fill_rate = (n_filled / n_entered) if n_entered else 0.0
    daily_pnl = float(realized_pnl) + float(unrealized_pnl)

    try:
        with _conn() as c:
            prior = c.execute(
                "SELECT cumulative_pnl FROM pead_daily "
                "WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1",
                (td,),
            ).fetchone()
            prior_cum = float(prior[0]) if prior and prior[0] is not None else 0.0
            cumulative_pnl = prior_cum + daily_pnl

            c.execute(
                "INSERT INTO pead_daily(trade_date, n_signals, n_entered, n_filled, "
                "fill_rate, gross_deployed, realized_pnl, unrealized_pnl, daily_pnl, "
                "cumulative_pnl, vix_level, vix_block_fired, suppressed_opportunity, "
                "suppressed_macro, suppressed_rm, extra, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(trade_date) DO UPDATE SET "
                "n_signals=excluded.n_signals, n_entered=excluded.n_entered, "
                "n_filled=excluded.n_filled, fill_rate=excluded.fill_rate, "
                "gross_deployed=excluded.gross_deployed, realized_pnl=excluded.realized_pnl, "
                "unrealized_pnl=excluded.unrealized_pnl, daily_pnl=excluded.daily_pnl, "
                "cumulative_pnl=excluded.cumulative_pnl, vix_level=excluded.vix_level, "
                "vix_block_fired=excluded.vix_block_fired, "
                "suppressed_opportunity=excluded.suppressed_opportunity, "
                "suppressed_macro=excluded.suppressed_macro, "
                "suppressed_rm=excluded.suppressed_rm, extra=excluded.extra",
                (
                    td, int(n_signals), int(n_entered), int(n_filled),
                    round(fill_rate, 4), float(gross_deployed), float(realized_pnl),
                    float(unrealized_pnl), float(daily_pnl), float(cumulative_pnl),
                    (float(vix_level) if vix_level is not None else None),
                    1 if vix_block_fired else 0,
                    int(suppressed_opportunity), int(suppressed_macro), int(suppressed_rm),
                    json.dumps(extra or {}, default=str), time.time(),
                ),
            )
        return True
    except Exception:
        log.exception("pead_tracker.record_daily failed (swallowed)")
        return False


def read_daily(since: _date | str | None = None) -> list[dict[str, Any]]:
    """Return daily rows (optionally since a date) as a list of dicts. Never raises."""
    try:
        with _conn() as c:
            c.row_factory = sqlite3.Row
            if since is not None:
                sd = since.isoformat() if isinstance(since, _date) else str(since)
                rows = c.execute(
                    "SELECT * FROM pead_daily WHERE trade_date >= ? ORDER BY trade_date",
                    (sd,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM pead_daily ORDER BY trade_date"
                ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        log.exception("pead_tracker.read_daily failed (swallowed)")
        return []


def _realized_sharpe(daily_pnls: list[float], gross: list[float]) -> float | None:
    """Annualised Sharpe of daily P&L returns (pnl / gross deployed). None if insufficient."""
    import numpy as np
    rets = []
    for pnl, g in zip(daily_pnls, gross):
        if g and g > 0:
            rets.append(pnl / g)
    if len(rets) < 2:
        return None
    arr = np.array(rets, dtype=float)
    sd = arr.std(ddof=1)
    if sd == 0:
        return None
    return float(arr.mean() / sd * np.sqrt(252))


def weekly_rollup(week_ending: _date | str | None = None, send: bool = True) -> dict[str, Any]:
    """Compute the trailing-7-day PEAD realized Sharpe vs the +0.546 backtest and
    (optionally) email it via notifier.enqueue("pead_weekly", ...). Never raises.

    Returns the payload dict (also when send=False, for testing).
    """
    if week_ending is None:
        week_ending = _date.today()
    we = week_ending if isinstance(week_ending, _date) else _date.fromisoformat(str(week_ending))
    since = we - timedelta(days=6)

    rows = read_daily(since=since)
    daily_pnls = [r["daily_pnl"] for r in rows]
    gross = [r["gross_deployed"] for r in rows]
    sharpe = _realized_sharpe(daily_pnls, gross)

    n_signals = sum(r["n_signals"] for r in rows)
    n_entered = sum(r["n_entered"] for r in rows)
    n_filled = sum(r["n_filled"] for r in rows)
    cum_pnl = rows[-1]["cumulative_pnl"] if rows else 0.0
    fill_rates = [r["fill_rate"] for r in rows if r["n_entered"]]
    avg_fill = round(sum(fill_rates) / len(fill_rates), 4) if fill_rates else None
    vix_blocks = sum(r["vix_block_fired"] for r in rows)
    sup_opp = sum(r["suppressed_opportunity"] for r in rows)
    sup_macro = sum(r["suppressed_macro"] for r in rows)
    sup_rm = sum(r["suppressed_rm"] for r in rows)

    payload: dict[str, Any] = {
        "week_ending": we.isoformat(),
        "realized_sharpe": (round(sharpe, 3) if sharpe is not None else "n/a"),
        "backtest_sharpe": f"+{BACKTEST_SHARPE:.3f}",
        "n_days": len(rows),
        "signals_entered_filled": f"{n_signals} / {n_entered} / {n_filled}",
        "cumulative_pnl": round(float(cum_pnl), 2),
        "avg_fill_rate": avg_fill,
        "vix_blocks_fired": vix_blocks,
        "suppressed_breakdown": f"opp={sup_opp} macro={sup_macro} rm={sup_rm}",
    }

    if send:
        try:
            from app.notifications import notifier
            notifier.enqueue("pead_weekly", payload,
                             dedup_key=f"pead_weekly_{we.isoformat()}")
        except Exception:
            log.exception("pead_tracker.weekly_rollup notify failed (swallowed)")

    return payload
