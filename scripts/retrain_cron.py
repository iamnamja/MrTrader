"""
Weekly model retraining script — gate-enforced, aligned with retrain_config.py.

Runs swing + intraday training, then validates each through the walk-forward gate.
If a model fails the gate, the previous ACTIVE version is restored automatically.

Usage (from project root):
    python scripts/retrain_cron.py [--swing-only] [--intraday-only] [--dry-run]

Exit codes:
  0  Both models passed (or were not run)
  1  Training error
  2  One or both models failed the walk-forward gate (previous champion kept)
"""
import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "24")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "24")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [retrain] %(message)s",
)
logger = logging.getLogger(__name__)


def prune_old_model_files(model_dir: Path, keep: int = 3) -> None:
    """Delete oldest swing_v*.pkl files, keeping the N most recent."""
    files = sorted(model_dir.glob("swing_v*.pkl"), key=lambda p: p.stat().st_mtime)
    for f in files[:-keep]:
        try:
            f.unlink()
            logger.info("Pruned old model file: %s", f.name)
        except Exception as exc:
            logger.warning("Could not prune %s: %s", f.name, exc)


def _previous_active(db, strategy: str):
    from app.database.models import ModelVersion
    row = (
        db.query(ModelVersion)
        .filter_by(model_name=strategy, status="ACTIVE")
        .order_by(ModelVersion.version.desc())
        .first()
    )
    return row.version if row else None


def _restore_previous(strategy: str, prev_version, new_version):
    from app.database.session import get_session
    from app.database.models import ModelVersion
    if prev_version is None:
        return
    db = get_session()
    try:
        new = db.query(ModelVersion).filter_by(model_name=strategy, version=new_version).first()
        if new:
            new.status = "RETIRED"
        prev = db.query(ModelVersion).filter_by(model_name=strategy, version=prev_version).first()
        if prev:
            prev.status = "ACTIVE"
        db.commit()
        logger.info("%s v%d gate failed — restored v%d as ACTIVE", strategy, new_version, prev_version)
    finally:
        db.close()


def run_swing(dry_run: bool) -> bool:
    """Train swing model + walk-forward gate. Returns True if promoted."""
    from app.ml.retrain_config import SWING_RETRAIN, SWING_GATE
    from app.ml.training import ModelTrainer
    from app.database.session import get_session

    db = get_session()
    try:
        prev = _previous_active(db, "swing")
    finally:
        db.close()

    logger.info("Swing retrain — model_type=%s hpo=%d wf_folds=%d prev_active=v%s",
                SWING_RETRAIN["model_type"], SWING_RETRAIN["hpo_trials"],
                SWING_RETRAIN["walk_forward_folds"], prev)

    if dry_run:
        logger.info("DRY RUN — skipping swing training")
        return True

    trainer = ModelTrainer(
        model_type=SWING_RETRAIN["model_type"],
        hpo_trials=SWING_RETRAIN["hpo_trials"],
        n_workers=SWING_RETRAIN["n_workers"],
    )
    try:
        version = trainer.train_model(
            fetch_fundamentals=SWING_RETRAIN["fetch_fundamentals"]
        )
        logger.info("Swing v%d trained — running walk-forward gate...", version)
    except Exception as e:
        logger.error("Swing training failed: %s", e, exc_info=True)
        return False

    try:
        from scripts.walkforward_tier3 import run_swing_walkforward
        wf = run_swing_walkforward(
            n_folds=SWING_RETRAIN["walk_forward_folds"],
            model_version=version,
        )
        avg_sh = wf.avg_sharpe
        min_sh = wf.min_sharpe
        gate_ok = (avg_sh >= SWING_GATE["min_avg_sharpe"] and
                   min_sh >= SWING_GATE["min_fold_sharpe"])

        ModelTrainer.record_tier3_result(version, avg_sh, [f.sharpe for f in wf.folds], gate_ok)

        if gate_ok:
            logger.info("Swing v%d GATE PASSED (avg_sharpe=%.3f min_sharpe=%.3f) — now ACTIVE",
                        version, avg_sh, min_sh)
            return True
        else:
            logger.warning("Swing v%d GATE FAILED (avg_sharpe=%.3f min_sharpe=%.3f) — restoring v%d",
                           version, avg_sh, min_sh, prev)
            _restore_previous("swing", prev, version)
            return False
    except Exception as e:
        logger.error("Swing walk-forward gate failed: %s", e, exc_info=True)
        _restore_previous("swing", prev, version)
        return False


def run_intraday(dry_run: bool) -> bool:
    """Train intraday model + walk-forward gate. Returns True if promoted."""
    from app.ml.retrain_config import INTRADAY_RETRAIN, INTRADAY_GATE
    from app.ml.intraday_training import IntradayModelTrainer
    from app.database.session import get_session

    db = get_session()
    try:
        prev = _previous_active(db, "intraday")
    finally:
        db.close()

    logger.info("Intraday retrain — days=%d fetch_spy=%s prev_active=v%s",
                INTRADAY_RETRAIN["days"], INTRADAY_RETRAIN["fetch_spy"], prev)

    if dry_run:
        logger.info("DRY RUN — skipping intraday training")
        return True

    trainer = IntradayModelTrainer()
    try:
        version = trainer.train_model(**INTRADAY_RETRAIN)
        logger.info("Intraday v%d trained — running walk-forward gate...", version)
    except Exception as e:
        logger.error("Intraday training failed: %s", e, exc_info=True)
        return False

    try:
        from scripts.walkforward_tier3 import run_intraday_walkforward
        wf = run_intraday_walkforward(n_folds=3, model_version=version)
        avg_sh = wf.avg_sharpe
        min_sh = wf.min_sharpe
        gate_ok = (avg_sh >= INTRADAY_GATE["min_avg_sharpe"] and
                   min_sh >= INTRADAY_GATE["min_fold_sharpe"])

        trainer.record_tier3_result(version, avg_sh, [f.sharpe for f in wf.folds], gate_ok)

        if gate_ok:
            logger.info("Intraday v%d GATE PASSED (avg_sharpe=%.3f min_sharpe=%.3f) — now ACTIVE",
                        version, avg_sh, min_sh)
            return True
        else:
            logger.warning("Intraday v%d GATE FAILED (avg_sharpe=%.3f min_sharpe=%.3f) — restoring v%d",
                           version, avg_sh, min_sh, prev)
            _restore_previous("intraday", prev, version)
            return False
    except Exception as e:
        logger.error("Intraday walk-forward gate failed: %s", e, exc_info=True)
        _restore_previous("intraday", prev, version)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate-enforced weekly model retrain")
    parser.add_argument("--swing-only", action="store_true", help="Retrain swing model only")
    parser.add_argument("--intraday-only", action="store_true", help="Retrain intraday model only")
    parser.add_argument("--dry-run", action="store_true", help="Skip training, just log what would run")
    args = parser.parse_args()

    run_both = not args.swing_only and not args.intraday_only

    results = {}

    if run_both or args.swing_only:
        results["swing"] = run_swing(args.dry_run)

    if run_both or args.intraday_only:
        results["intraday"] = run_intraday(args.dry_run)

    # Reap loky workers
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=False, kill_workers=True)
    except Exception:
        pass

    logger.info("Retrain complete — %s", results)

    if not all(results.values()):
        logger.warning("One or more models failed the gate — previous champions retained")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
