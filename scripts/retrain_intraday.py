"""
Intraday model retraining script.

Usage:
    python scripts/retrain_intraday.py [--days 730] [--dry-run]

Exit codes:
  0  Success
  1  Training failed
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [retrain_intraday] %(message)s",
)
logger = logging.getLogger(__name__)

AUC_DRIFT_THRESHOLD = 0.58
KEEP_MODEL_VERSIONS = 3


def prune_old_model_files(model_dir: Path, keep: int = KEEP_MODEL_VERSIONS) -> None:
    files = sorted(model_dir.glob("intraday_v*.pkl"), key=lambda p: p.stat().st_mtime)
    for f in files[:-keep]:
        try:
            f.unlink()
            logger.info("Pruned old model file: %s", f.name)
        except Exception as exc:
            logger.warning("Could not prune %s: %s", f.name, exc)


def main() -> int:
    import os as _os
    parser = argparse.ArgumentParser(description="Intraday model retraining")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--model-dir", default="app/ml/models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN — skipping training")
        return 0

    logger.info("Intraday retrain starting — days=%d", args.days)

    try:
        from app.ml.intraday_training import IntradayModelTrainer
        trainer = IntradayModelTrainer(model_dir=args.model_dir)
        version = trainer.train_model(days=args.days)
        logger.info("Retrain complete — new intraday model version: v%d", version)
    except Exception as exc:
        logger.error("Retrain failed: %s", exc, exc_info=True)
        return 1

    prune_old_model_files(Path(args.model_dir), keep=KEEP_MODEL_VERSIONS)

    from app.database.session import get_session
    from app.database.models import ModelVersion
    db = get_session()
    try:
        mv = db.query(ModelVersion).filter_by(model_name="intraday", version=version).first()
        if mv and mv.performance:
            auc = mv.performance.get("auc")
            logger.info("Intraday v%d — AUC=%.4f", version, auc or 0)
            if auc is not None and auc < AUC_DRIFT_THRESHOLD:
                logger.warning(
                    "AUC DRIFT: intraday v%d AUC=%.4f < %.2f",
                    version, auc, AUC_DRIFT_THRESHOLD,
                )
    finally:
        db.close()

    logger.info("Intraday retrain complete — v%d active", version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
