"""
Sleeve-allocator recompute audit trail — Alpha-v4 P3.

One row per weekly allocator recompute: the scheme used, the resulting per-sleeve
weights, the live regime label, and whether the result came from the allocator or a
fixed-weight fallback (disabled / warmup / stale). Lets us see, after the fact, what
the allocator decided and why — without trusting the live agent_config snapshot alone.

Standalone sqlite (mirrors trend_tracker.py) — append-only, never raises, safe to call
from the scheduled job.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import date as _date
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
# Env-overridable so pytest-xdist workers each use an isolated file (no lock).
DB_PATH = Path(os.environ.get("MRTRADER_ALLOCATOR_TRACKING_DB",
                              str(_ROOT / "data" / "allocator_tracking.db")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS allocator_daily (
    compute_date     TEXT PRIMARY KEY,
    scheme           TEXT,
    enabled          INTEGER,
    source           TEXT,          -- 'allocator' | 'fallback_disabled' | 'fallback_warmup' | 'fallback_error'
    trend_weight     REAL,
    pead_weight      REAL,
    regime           TEXT,
    n_days           INTEGER,
    extra            TEXT,
    created_at       REAL
);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=10)
    c.execute("PRAGMA journal_mode=WAL;")
    c.executescript(_SCHEMA)
    return c


def record(
    compute_date: _date | str | None = None,
    *,
    scheme: str | None = None,
    enabled: bool | None = None,
    source: str | None = None,
    trend_weight: float | None = None,
    pead_weight: float | None = None,
    regime: str | None = None,
    n_days: int | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Upsert one allocator-recompute row (one per day). Never raises."""
    if compute_date is None:
        compute_date = _date.today()
    cd = compute_date.isoformat() if isinstance(compute_date, _date) else str(compute_date)
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO allocator_daily(compute_date, scheme, enabled, source, "
                "trend_weight, pead_weight, regime, n_days, extra, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(compute_date) DO UPDATE SET "
                "scheme=excluded.scheme, enabled=excluded.enabled, source=excluded.source, "
                "trend_weight=excluded.trend_weight, pead_weight=excluded.pead_weight, "
                "regime=excluded.regime, n_days=excluded.n_days, extra=excluded.extra",
                (
                    cd, scheme,
                    (None if enabled is None else (1 if enabled else 0)),
                    source,
                    (float(trend_weight) if trend_weight is not None else None),
                    (float(pead_weight) if pead_weight is not None else None),
                    regime,
                    (int(n_days) if n_days is not None else None),
                    (json.dumps(extra, default=str) if extra is not None else None),
                    time.time(),
                ),
            )
        return True
    except Exception:
        log.exception("allocator_tracker.record failed (swallowed)")
        return False


def read(since: _date | str | None = None) -> list[dict[str, Any]]:
    """Return allocator rows (optionally since a date) as dicts. Never raises."""
    try:
        with _conn() as c:
            c.row_factory = sqlite3.Row
            if since is not None:
                sd = since.isoformat() if isinstance(since, _date) else str(since)
                rows = c.execute(
                    "SELECT * FROM allocator_daily WHERE compute_date >= ? ORDER BY compute_date",
                    (sd,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM allocator_daily ORDER BY compute_date"
                ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        log.exception("allocator_tracker.read failed (swallowed)")
        return []
