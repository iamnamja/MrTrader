"""Regime V2 schema migration — adds new feature/probability columns to regime_snapshots.

Idempotent: checks information_schema before each ADD COLUMN (PostgreSQL-compatible).
Run once before deploying regime v2 code.

Usage:
    python scripts/migrations/2026_05_regime_v2_schema.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import sqlalchemy as sa
from app.database.session import get_session, init_db

NEW_COLUMNS = [
    # VIX term structure & trend
    ("vix_5d_change",             "FLOAT"),
    ("vix_term_ratio",            "FLOAT"),
    # Extended SPY trend
    ("spy_50d_return",            "FLOAT"),
    ("spy_above_ma50",            "FLOAT"),
    ("spy_above_ma200",           "FLOAT"),
    # Breadth
    ("breadth_rsp_spy_ratio_20d", "FLOAT"),
    # Credit
    ("credit_hyg_ief_5d",         "FLOAT"),
    ("credit_hyg_ief_20d",        "FLOAT"),
    # Sector dispersion
    ("sector_dispersion_20d",     "FLOAT"),
    ("sector_leader_lag_20d",     "FLOAT"),
    # Class probabilities from multinomial model
    ("prob_risk_off",             "FLOAT"),
    ("prob_risk_caution",         "FLOAT"),
    ("prob_risk_on",              "FLOAT"),
    # Rule-based training label
    ("regime_label_rule",         "VARCHAR(15)"),
]


def _existing_columns(db) -> set:
    """Return set of existing column names in regime_snapshots (PostgreSQL + SQLite compat)."""
    url = str(db.bind.engine.url) if hasattr(db, "bind") else ""
    try:
        # PostgreSQL
        result = db.execute(sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'regime_snapshots'"
        ))
        return {row[0] for row in result.fetchall()}
    except Exception:
        pass
    try:
        # SQLite fallback
        result = db.execute(sa.text("PRAGMA table_info(regime_snapshots)"))
        return {row[1] for row in result.fetchall()}
    except Exception:
        return set()


def run():
    init_db()
    with get_session() as db:
        existing_cols = _existing_columns(db)

        added = []
        for col_name, col_type in NEW_COLUMNS:
            if col_name not in existing_cols:
                db.execute(sa.text(
                    f"ALTER TABLE regime_snapshots ADD COLUMN {col_name} {col_type} NULL"
                ))
                added.append(col_name)
        db.commit()

    if added:
        print(f"Added {len(added)} columns: {', '.join(added)}")
    else:
        print("All columns already exist — no changes made.")


if __name__ == "__main__":
    run()
