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
from datetime import date as _date, timedelta
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
    n_signals: int | None = None,
    n_entered: int | None = None,
    n_filled: int | None = None,
    gross_deployed: float | None = None,
    realized_pnl: float | None = None,
    unrealized_pnl: float | None = None,
    vix_level: float | None = None,
    vix_block_fired: bool | None = None,
    suppressed_opportunity: int | None = None,
    suppressed_macro: int | None = None,
    suppressed_rm: int | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Upsert today's PEAD tracking row. Never raises.

    PARTIAL UPSERT semantics: every field defaults to None meaning "don't touch".
    On INSERT a None field is stored as its column default (0 / NULL / False); on
    CONFLICT a None field is left UNCHANGED (COALESCE(excluded, existing)). This lets
    the signals-stage writer set n_signals/vix while the EOD writer sets
    gross/realized/unrealized — each preserves the other's fields.

    fill_rate, daily_pnl, cumulative_pnl are DERIVED. They are recomputed only when
    their inputs are supplied this call (n_entered/n_filled for fill_rate;
    realized_pnl/unrealized_pnl for the P&L fields); otherwise the existing values
    are preserved. Returns True on success.
    """
    if trade_date is None:
        trade_date = _date.today()
    td = trade_date.isoformat() if isinstance(trade_date, _date) else str(trade_date)

    # Derived fields are recomputed only when their inputs are supplied this call.
    fill_rate = None
    if n_entered is not None or n_filled is not None:
        _ne = int(n_entered or 0)
        _nf = int(n_filled or 0)
        fill_rate = round((_nf / _ne) if _ne else 0.0, 4)

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
            # cumulative_pnl = prior-day cumulative + this day's daily P&L. Only
            # recomputed when P&L was supplied this call.
            cumulative_pnl = None
            if pnl_supplied:
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
                "n_signals=COALESCE(excluded.n_signals, n_signals), "
                "n_entered=COALESCE(excluded.n_entered, n_entered), "
                "n_filled=COALESCE(excluded.n_filled, n_filled), "
                "fill_rate=COALESCE(excluded.fill_rate, fill_rate), "
                "gross_deployed=COALESCE(excluded.gross_deployed, gross_deployed), "
                "realized_pnl=COALESCE(excluded.realized_pnl, realized_pnl), "
                "unrealized_pnl=COALESCE(excluded.unrealized_pnl, unrealized_pnl), "
                "daily_pnl=COALESCE(excluded.daily_pnl, daily_pnl), "
                "cumulative_pnl=COALESCE(excluded.cumulative_pnl, cumulative_pnl), "
                "vix_level=COALESCE(excluded.vix_level, vix_level), "
                "vix_block_fired=COALESCE(excluded.vix_block_fired, vix_block_fired), "
                "suppressed_opportunity=COALESCE(excluded.suppressed_opportunity, suppressed_opportunity), "
                "suppressed_macro=COALESCE(excluded.suppressed_macro, suppressed_macro), "
                "suppressed_rm=COALESCE(excluded.suppressed_rm, suppressed_rm), "
                "extra=COALESCE(excluded.extra, extra)",
                (
                    td, _i(n_signals), _i(n_entered), _i(n_filled),
                    fill_rate, _f(gross_deployed), _f(realized_pnl),
                    _f(unrealized_pnl), daily_pnl, cumulative_pnl,
                    _f(vix_level),
                    (None if vix_block_fired is None else (1 if vix_block_fired else 0)),
                    _i(suppressed_opportunity), _i(suppressed_macro), _i(suppressed_rm),
                    (json.dumps(extra, default=str) if extra is not None else None),
                    time.time(),
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


def weekly_rollup(
    week_ending: _date | str | None = None,
    send: bool = True,
    min_days: int = 3,
) -> dict[str, Any]:
    """Compute the trailing-7-day PEAD realized Sharpe vs the +0.546 backtest and
    (optionally) email it via notifier.enqueue("pead_weekly", ...). Never raises.

    Vacuous-email guard: if fewer than ``min_days`` daily rows in the window have
    gross_deployed > 0 (i.e. the PEAD book wasn't actually deployed), the email is
    SKIPPED — the payload is returned with skipped="insufficient data" and not sent.

    Returns the payload dict (also when send=False, for testing).
    """
    if week_ending is None:
        week_ending = _date.today()
    we = week_ending if isinstance(week_ending, _date) else _date.fromisoformat(str(week_ending))
    since = we - timedelta(days=6)

    rows = read_daily(since=since)

    # Partial-upsert rows may carry NULLs for fields a given writer didn't set; treat
    # NULL as 0 for aggregation.
    def _g(r, k, default=0):
        v = r.get(k)
        return default if v is None else v

    daily_pnls = [_g(r, "daily_pnl", 0.0) for r in rows]
    gross = [_g(r, "gross_deployed", 0.0) for r in rows]
    sharpe = _realized_sharpe(daily_pnls, gross)
    n_deployed_days = sum(1 for g in gross if g and g > 0)

    n_signals = sum(_g(r, "n_signals") for r in rows)
    n_entered = sum(_g(r, "n_entered") for r in rows)
    n_filled = sum(_g(r, "n_filled") for r in rows)
    cum_pnl = _g(rows[-1], "cumulative_pnl", 0.0) if rows else 0.0
    fill_rates = [_g(r, "fill_rate", 0.0) for r in rows if _g(r, "n_entered")]
    avg_fill = round(sum(fill_rates) / len(fill_rates), 4) if fill_rates else None
    vix_blocks = sum(_g(r, "vix_block_fired") for r in rows)
    sup_opp = sum(_g(r, "suppressed_opportunity") for r in rows)
    sup_macro = sum(_g(r, "suppressed_macro") for r in rows)
    sup_rm = sum(_g(r, "suppressed_rm") for r in rows)

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

    # Vacuous-email guard: don't send a "Sharpe: n/a" weekly when the PEAD book was
    # barely (or never) deployed. The dedup_key is still honoured when we DO send.
    if n_deployed_days < min_days:
        payload["skipped"] = "insufficient data"
        log.info(
            "pead_tracker.weekly_rollup: only %d deployed day(s) (< min_days=%d) — "
            "skipping email for week_ending=%s",
            n_deployed_days, min_days, we.isoformat(),
        )
        return payload

    if send:
        try:
            from app.notifications import notifier
            notifier.enqueue("pead_weekly", payload,
                             dedup_key=f"pead_weekly_{we.isoformat()}")
        except Exception:
            log.exception("pead_tracker.weekly_rollup notify failed (swallowed)")

    return payload
