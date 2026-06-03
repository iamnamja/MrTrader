"""proposal_log.selector migration — adds PM selector-source attribution column.

Mirrors Trade.selector onto proposal_log so PEAD (and other directional selectors:
quality_short, factor_portfolio, ml_model) are distinguishable from baseline
swing/intraday proposals in the UI proposal log.

Idempotent: checks information_schema (PostgreSQL) / PRAGMA (SQLite) before ADD COLUMN.
Adding a nullable column is a metadata-only operation on PostgreSQL — safe to run
against the live DB while the server is up. Run once before deploying the code that
writes/reads proposal_log.selector.

Usage:
    python scripts/migrations/2026_06_proposal_log_selector.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import re

import sqlalchemy as sa
from app.database.session import get_session, init_db

TABLE = "proposal_log"
NEW_COLUMNS = [
    ("selector", "VARCHAR(32)"),
]
INDEX_NAME = "ix_proposal_log_selector"  # matches SQLAlchemy's auto index name

# Directional proposals embed their selector in batch_id as: dir_{selector}_{YYYYMMDD}_{HHMMSS}
# (selector may itself contain underscores, e.g. quality_short). This recovers it.
_BATCH_SELECTOR_RE = re.compile(r"^dir_(.+)_\d{8}_\d{6}$")


def _existing_columns(db, table: str) -> set:
    """Return existing column names for ``table`` (PostgreSQL + SQLite compatible)."""
    try:
        # PostgreSQL
        result = db.execute(sa.text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
        ), {"t": table})
        cols = {row[0] for row in result.fetchall()}
        if cols:
            return cols
    except Exception:
        pass
    try:
        # SQLite fallback
        result = db.execute(sa.text(f"PRAGMA table_info({table})"))
        return {row[1] for row in result.fetchall()}
    except Exception:
        return set()


def _backfill_directional(db) -> int:
    """Recover selector for historical directional proposals from their batch_id.

    DB-agnostic (parses in Python, not SQL regex) so it is safe on SQLite test DBs.
    Idempotent: only touches rows whose selector is still NULL/empty.
    Returns the number of rows updated.
    """
    rows = db.execute(sa.text(
        f"SELECT DISTINCT batch_id FROM {TABLE} "
        "WHERE batch_id LIKE 'dir%' AND (selector IS NULL OR selector = '')"
    )).fetchall()

    batch_to_sel: dict[str, str] = {}
    for (batch_id,) in rows:
        m = _BATCH_SELECTOR_RE.match(batch_id or "")
        if m:
            batch_to_sel[batch_id] = m.group(1)

    updated = 0
    for batch_id, sel in batch_to_sel.items():
        res = db.execute(sa.text(
            f"UPDATE {TABLE} SET selector = :s "
            "WHERE batch_id = :b AND (selector IS NULL OR selector = '')"
        ), {"s": sel, "b": batch_id})
        updated += res.rowcount or 0
    return updated


def run():
    init_db()
    with get_session() as db:
        existing_cols = _existing_columns(db, TABLE)

        added = []
        for col_name, col_type in NEW_COLUMNS:
            if col_name not in existing_cols:
                db.execute(sa.text(
                    f"ALTER TABLE {TABLE} ADD COLUMN {col_name} {col_type} NULL"
                ))
                added.append(col_name)
        db.commit()

        # Index (CREATE INDEX IF NOT EXISTS is supported on both PostgreSQL and SQLite)
        try:
            db.execute(sa.text(f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON {TABLE} (selector)"))
            db.commit()
        except Exception as exc:
            print(f"Index creation skipped ({exc})")

        # Backfill historical directional (PEAD / quality_short / …) proposals
        backfilled = _backfill_directional(db)
        db.commit()

    if added:
        print(f"Added {len(added)} column(s) to {TABLE}: {', '.join(added)}")
    else:
        print(f"Column already exists on {TABLE} - no schema change.")
    print(f"Backfilled selector on {backfilled} historical directional row(s).")


if __name__ == "__main__":
    run()
