"""
Weekly model retraining script — Phase B4 (MLOps).

Designed to be called every Friday after market close (16:30 ET).
Can be run via cron, Windows Task Scheduler, or GitHub Actions.

Usage:
    python scripts/retrain_cron.py [--model-type lambdarank] [--years 5] [--dry-run]

What it does:
  1. Retrains the swing model on SP_500_TICKERS with the latest N years of data
  2. Logs SHAP feature importance to the ModelVersion DB record
  3. Alerts (WARNING log) if new OOS AUC < 0.65 (concept drift)
  4. Archives the previous ACTIVE model version
  5. Keeps at most 3 saved model files; deletes older ones

Exit codes:
  0  Success
  1  Training failed
  2  AUC drift detected (new AUC < 0.65) — model saved but flag raised
"""
import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [retrain] %(message)s",
)
logger = logging.getLogger(__name__)

AUC_DRIFT_THRESHOLD = 0.65
KEEP_MODEL_VERSIONS = 3


def archive_old_versions(db, current_version: int) -> None:
    """Set older ACTIVE model versions to ARCHIVED."""
    from app.database.models import ModelVersion
    old = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.model_name == "swing",
            ModelVersion.status == "ACTIVE",
            ModelVersion.version != current_version,
        )
        .all()
    )
    for mv in old:
        mv.status = "ARCHIVED"
        logger.info("Archived swing model v%d", mv.version)
    db.commit()


def prune_old_model_files(model_dir: Path, keep: int = KEEP_MODEL_VERSIONS) -> None:
    """Delete oldest saved .pkl files, keeping the N most recent."""
    files = sorted(model_dir.glob("swing_v*.pkl"), key=lambda p: p.stat().st_mtime)
    for f in files[:-keep]:
        try:
            f.unlink()
            logger.info("Pruned old model file: %s", f.name)
        except Exception as exc:
            logger.warning("Could not prune %s: %s", f.name, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly swing model retraining")
    parser.add_argument("--model-type", default="lambdarank",
                        choices=["lambdarank", "xgboost", "lightgbm"])
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--model-dir", default="app/ml/models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run feature engineering only, skip training + save")
    args = parser.parse_args()

    from app.ml.training import ModelTrainer
    from app.utils.constants import SP_500_TICKERS
    from app.database.session import get_session

    logger.info(
        "Weekly retrain starting — model_type=%s years=%d symbols=%d dry_run=%s",
        args.model_type, args.years, len(SP_500_TICKERS), args.dry_run,
    )

    if args.dry_run:
        logger.info("DRY RUN — skipping training and save")
        return 0

    try:
        trainer = ModelTrainer(
            model_type=args.model_type,
            label_scheme="lambdarank",
            use_feature_store=True,
            model_dir=args.model_dir,
        )
        version = trainer.train(symbols=SP_500_TICKERS, years=args.years)
        logger.info("Retrain complete — new model version: v%d", version)
    except Exception as exc:
        logger.error("Retrain failed: %s", exc, exc_info=True)
        return 1

    # Archive old versions and prune files
    db = get_session()
    try:
        archive_old_versions(db, version)
    finally:
        db.close()

    prune_old_model_files(Path(args.model_dir), keep=KEEP_MODEL_VERSIONS)

    # Check for AUC drift
    db = get_session()
    try:
        from app.database.models import ModelVersion
        mv = (
            db.query(ModelVersion)
            .filter_by(model_name="swing", version=version)
            .first()
        )
        if mv and mv.performance:
            auc = mv.performance.get("auc")
            if auc is not None and auc < AUC_DRIFT_THRESHOLD:
                logger.warning(
                    "AUC DRIFT: v%d AUC=%.4f < %.2f — model may be stale, "
                    "review training data and feature distribution",
                    version, auc, AUC_DRIFT_THRESHOLD,
                )
                return 2
    finally:
        db.close()

    logger.info("Weekly retrain complete — v%d active", version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
