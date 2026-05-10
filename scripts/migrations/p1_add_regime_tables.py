"""
P1 migration: add daily_regime_scores and regime_gate_events tables.

Safe to run multiple times (CREATE IF NOT EXISTS semantics via SQLAlchemy).
Also seeds daily_regime_scores from macro_history.parquet so historical
regime data is queryable immediately after migration.

Usage:
    python scripts/migrations/p1_add_regime_tables.py
    python scripts/migrations/p1_add_regime_tables.py --dry-run
    python scripts/migrations/p1_add_regime_tables.py --no-seed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("p1_migration")


def run(dry_run: bool = False, seed: bool = True) -> None:
    from app.database.models import DailyRegimeScore, RegimeGateEvent
    from app.database.session import engine, SessionLocal

    logger.info("Creating tables: daily_regime_scores, regime_gate_events")
    if dry_run:
        logger.info("DRY RUN — no changes made")
        return

    # Create only the new tables (existing tables untouched)
    DailyRegimeScore.__table__.create(engine, checkfirst=True)
    RegimeGateEvent.__table__.create(engine, checkfirst=True)
    logger.info("Tables created (or already existed)")

    if not seed:
        logger.info("Skipping seed (--no-seed)")
        return

    # Seed daily_regime_scores from macro_history.parquet
    macro_path = Path("data/macro/macro_history.parquet")
    if not macro_path.exists():
        logger.warning("macro_history.parquet not found — skipping seed")
        return

    import pandas as pd
    from app.ml.regime_score_pit import compute_pit_regime_series

    logger.info("Seeding daily_regime_scores from %s", macro_path)
    df = pd.read_parquet(macro_path)
    series = compute_pit_regime_series(df)
    series = series.dropna(subset=["composite_score"])

    db = SessionLocal()
    try:
        inserted = 0
        skipped = 0
        for ts, row in series.iterrows():
            d = ts.date()
            existing = db.query(DailyRegimeScore).filter_by(date=d).first()
            if existing:
                skipped += 1
                continue
            db.add(DailyRegimeScore(
                date=d,
                spy_above_ma50=float(row["spy_above_ma50"]),
                spy_above_ma200=float(row["spy_above_ma200"]),
                vix_term_ratio=float(row["vix_term_ratio"]),
                breadth_20d_change=float(row["breadth_20d_change"]),
                credit_20d_change=float(row["credit_20d_change"]),
                composite_score=float(row["composite_score"]),
            ))
            inserted += 1

            if inserted % 200 == 0:
                db.commit()
                logger.info("  %d rows inserted so far...", inserted)

        db.commit()
        logger.info("Seed complete: %d inserted, %d already existed", inserted, skipped)
    except Exception as exc:
        db.rollback()
        logger.error("Seed failed: %s", exc)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-seed", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run, seed=not args.no_seed)
