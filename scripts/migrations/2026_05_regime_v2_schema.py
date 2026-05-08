"""Regime V2 schema migration — adds new feature/probability columns to regime_snapshots.

Idempotent: checks PRAGMA table_info before each ADD COLUMN.
Run once before deploying regime v2 code.

Usage:
    python scripts/migrations/2026_05_regime_v2_schema.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

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
    # Rule-based training label (0=RISK_OFF, 1=RISK_CAUTION, 2=RISK_ON)
    ("regime_label_rule",         "VARCHAR(15)"),
]


def run():
    init_db()
    with get_session() as db:
        result = db.execute(__import__("sqlalchemy").text(
            "PRAGMA table_info(regime_snapshots)"
        ))
        existing_cols = {row[1] for row in result.fetchall()}

    added = []
    with get_session() as db:
        for col_name, col_type in NEW_COLUMNS:
            if col_name not in existing_cols:
                db.execute(__import__("sqlalchemy").text(
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
