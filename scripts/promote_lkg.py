"""
P1 — promote_lkg.py: mark the current ACTIVE model as Last-Known-Good (LKG).

Usage:
    python scripts/promote_lkg.py --model swing
    python scripts/promote_lkg.py --model intraday
    python scripts/promote_lkg.py --model swing --version 182   # override

When PM loads a model and the CPCV gate fails, it falls back to the LKG version.
This script persists the LKG pointer to the Configuration table so it survives restarts.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Promote current ACTIVE model to LKG")
    parser.add_argument("--model", choices=["swing", "intraday"], required=True)
    parser.add_argument("--version", type=int, default=None,
                        help="Explicit version to promote (default: query ACTIVE from DB)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print what would be done without writing to DB")
    args = parser.parse_args()

    from app.database.session import SessionLocal
    from app.database.models import ModelVersion

    db = SessionLocal()
    try:
        version = args.version
        if version is None:
            row = (
                db.query(ModelVersion)
                .filter_by(model_type=args.model, status="ACTIVE")
                .order_by(ModelVersion.version.desc())
                .first()
            )
            if row is None:
                print(f"[ERROR] No ACTIVE {args.model} model found in DB. "
                      f"Use --version to specify explicitly.")
                sys.exit(1)
            version = row.version

        print(f"Promoting {args.model} v{version} as LKG...")
        if args.dry_run:
            print(f"[DRY RUN] Would write {args.model}.last_known_good_version = {version}")
            return

        from app.strategy.benign_gate import set_lkg_version
        set_lkg_version(args.model, version)
        print(f"[OK] {args.model}.last_known_good_version = {version}")

    finally:
        db.close()


def restore_lkg(model_name: str) -> bool:
    """
    Activate the LKG version of model_name if the current ACTIVE version is RETIRED.
    Returns True if a restore was performed, False if not needed or LKG not available.

    Called by PM._weekly_retrain() when the new model fails the walk-forward gate.
    """
    import logging
    logger = logging.getLogger(__name__)

    from app.strategy.benign_gate import get_lkg_version
    lkg_ver = get_lkg_version(model_name)

    if lkg_ver is None:
        logger.warning("LKG restore: no LKG version found for %s", model_name)
        return False

    from app.database.session import SessionLocal
    from app.database.models import ModelVersion

    db = SessionLocal()
    try:
        lkg_row = db.query(ModelVersion).filter_by(
            model_type=model_name, version=lkg_ver
        ).first()
        if lkg_row is None:
            logger.error("LKG restore: v%d not found in DB for %s", lkg_ver, model_name)
            return False

        if lkg_row.status == "ACTIVE":
            logger.info("LKG restore: %s v%d already ACTIVE — no action needed", model_name, lkg_ver)
            return False

        # Demote any current ACTIVE to RETIRED, then promote LKG to ACTIVE
        current_active = (
            db.query(ModelVersion)
            .filter_by(model_type=model_name, status="ACTIVE")
            .all()
        )
        for row in current_active:
            row.status = "RETIRED"
            logger.warning("LKG restore: demoted %s v%d → RETIRED", model_name, row.version)

        lkg_row.status = "ACTIVE"
        db.commit()
        logger.warning(
            "LKG restore: %s v%d promoted back to ACTIVE (LKG rollback)", model_name, lkg_ver
        )
        return True

    except Exception as exc:
        logger.error("LKG restore failed: %s", exc)
        db.rollback()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    main()
