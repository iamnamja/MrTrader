"""
Phase C+ swing_v201: LambdaRank + 14 IC features + 3 interaction terms.
Run after v200/v200b comparison is done.

Decision tree:
- If v200 (LambdaRank) > v200b (XGBoost binary) by >0.05 Sharpe: keep LambdaRank
- If v200b wins or ties: switch to xgboost with triple_barrier label
- Either way: add 3 interaction features and run v201

Usage:
    python scripts/run_v201_lambdarank_plus.py [--use-xgboost]
"""
import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml.retrain_config import MAX_THREADS as _max_threads
os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_max_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_max_threads))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [v201] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-xgboost", action="store_true",
                        help="Use XGBoost binary instead of LambdaRank")
    args = parser.parse_args()

    from app.ml.retrain_config import (
        PHASE_C_PLUS_FEATURE_KEEP_LIST, PHASE_C_V2_FEATURE_KEEP_LIST,
        SWING_GATE, MAX_WORKERS,
    )
    from app.ml.training import ModelTrainer
    from app.database.session import get_session
    from scripts.retrain_cron import _previous_active, _restore_previous

    model_type = "xgboost" if args.use_xgboost else "lambdarank"
    label_scheme = "triple_barrier" if args.use_xgboost else "lambdarank"

    # v209a: revert to 17-feature clean baseline (drop sector-neutral features
    # that hurt v208 — Opus analysis shows they landed HPO in a worse basin).
    # Switch back to PHASE_C_PLUS_FEATURE_KEEP_LIST for this run.
    feature_list = PHASE_C_PLUS_FEATURE_KEEP_LIST
    n_feats = len(feature_list)

    db = get_session()
    try:
        prev = _previous_active(db, "swing")
    finally:
        db.close()

    logger.info("v209a: %s, %d features (17-feat clean baseline), prev_active=v%s",
                model_type, n_feats, prev)
    logger.info("Features: %s", feature_list)

    trainer = ModelTrainer(
        model_type=model_type,
        label_scheme=label_scheme,
        hpo_trials=50,
        n_workers=MAX_WORKERS,
        feature_keep_list=feature_list,
    )

    try:
        version = trainer.train_model(
            fetch_fundamentals=False,
            exclude_risk_off_days=True,
            use_union_label=False,
        )
        logger.info("v201 trained as v%d — running walk-forward gate...", version)
    except Exception as e:
        logger.error("v201 training failed: %s", e, exc_info=True)
        return 1

    try:
        from scripts.walkforward_tier3 import run_swing_walkforward

        wf = run_swing_walkforward(
            n_folds=5,
            total_years=6,
            model_version=version,
            use_opportunity_score=True,
            no_prefilters=True,
        )
        avg_sh = wf.avg_sharpe
        min_sh = wf.min_sharpe
        gate_ok = (avg_sh >= SWING_GATE["min_avg_sharpe"] and
                   min_sh >= SWING_GATE["min_fold_sharpe"])

        ModelTrainer.record_tier3_result(version, avg_sh, [f.sharpe for f in wf.folds], gate_ok)
        verdict = "GATE PASSED" if gate_ok else "GATE FAILED"
        logger.info("v201 v%d %s (avg_sharpe=%.3f min=%.3f folds=%s)",
                    version, verdict, avg_sh, min_sh,
                    [round(f.sharpe, 3) for f in wf.folds])

        try:
            from app.notifications import notifier as _notifier
            _notifier.enqueue("training_complete", {
                "model": f"swing ({model_type}, 17 feat)",
                "version": version,
                "sharpe": round(avg_sh, 3),
                "gate_result": verdict,
                "log_path": f"logs/retrain_v201_2026.log",
                "fold_table_html": "<pre>" + "\n".join(
                    f"Fold {f.fold}: Sharpe={f.sharpe:.3f}" for f in wf.folds
                ) + "</pre>",
            })
        except Exception:
            pass

        if not gate_ok:
            _restore_previous("swing", prev, version)

        return 0 if gate_ok else 2

    except Exception as e:
        logger.error("v201 walk-forward failed: %s", e, exc_info=True)
        _restore_previous("swing", prev, version)
        return 1


if __name__ == "__main__":
    sys.exit(main())
