"""
Migration: gate calibration schema additions.

Adds columns to decision_audit and creates nis_macro_snapshots + scan_abstentions tables.
Safe to run multiple times (all statements use IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).

Usage:
    python scripts/migrate_gate_calibration.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database.session import get_session
from app.database.models import Base
from sqlalchemy import text


ALTER_STATEMENTS = [
    # Extend decision_audit with gate calibration columns
    "ALTER TABLE decision_audit ADD COLUMN IF NOT EXISTS gate_category VARCHAR(20)",
    "ALTER TABLE decision_audit ADD COLUMN IF NOT EXISTS price_at_decision FLOAT",
    "ALTER TABLE decision_audit ADD COLUMN IF NOT EXISTS direction VARCHAR(5)",
    "ALTER TABLE decision_audit ADD COLUMN IF NOT EXISTS outcome_fetched_at TIMESTAMP",
    # Index gate_category for calibration queries
    "CREATE INDEX IF NOT EXISTS ix_decision_audit_gate_category ON decision_audit (gate_category)",
]


def run():
    from app.database.session import engine

    print("Running gate calibration migration...")

    with engine.connect() as conn:
        # 1. ALTER existing table
        for stmt in ALTER_STATEMENTS:
            print(f"  {stmt[:80]}...")
            conn.execute(text(stmt))

        # 2. Create new tables via SQLAlchemy (NisMacroSnapshot, ScanAbstention)
        #    create_all with checkfirst=True is safe — skips existing tables
        Base.metadata.create_all(bind=engine, checkfirst=True)
        print("  Created new tables (nis_macro_snapshots, scan_abstentions) if not exist")

        conn.commit()

    print("Migration complete.")


if __name__ == "__main__":
    run()
