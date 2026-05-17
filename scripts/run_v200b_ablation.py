"""
Phase C ablation: swing_v200b — XGBoost binary classifier on 14 IC-validated features.
Critical comparison against v200a (LambdaRank) to isolate whether LambdaRank
or just feature pruning is responsible for any improvement.

Usage:
    python scripts/run_v200b_ablation.py
"""
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
    format="%(asctime)s %(levelname)s [v200b] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    from app.ml.retrain_config import PHASE_C_FEATURE_KEEP_LIST, SWING_GATE, MAX_WORKERS
    from app.ml.training import ModelTrainer
    from app.database.session import get_session
    from scripts.retrain_cron import _previous_active, _restore_previous

    db = get_session()
    try:
        prev = _previous_active(db, "swing")
    finally:
        db.close()

    logger.info("v200b ablation: XGBoost binary, 14 features, prev_active=v%s", prev)

    trainer = ModelTrainer(
        model_type="xgboost",
        label_scheme="triple_barrier",   # standard binary classifier
        hpo_trials=20,                   # low-capacity: 20 trials, Opus recommendation
        n_workers=MAX_WORKERS,
        feature_keep_list=PHASE_C_FEATURE_KEEP_LIST,
    )

    try:
        version = trainer.train_model(
            fetch_fundamentals=False,
            exclude_risk_off_days=True,
            use_union_label=False,
        )
        logger.info("v200b trained as v%d — running walk-forward gate...", version)
    except Exception as e:
        logger.error("v200b training failed: %s", e, exc_info=True)
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
        logger.info("v200b v%d %s (avg_sharpe=%.3f min_sharpe=%.3f)",
                    version, verdict, avg_sh, min_sh)

        try:
            from app.notifications import notifier as _notifier
            _notifier.enqueue("diag_complete", {
                "script": f"run_v200b_ablation.py (XGBoost binary, 14 feat, v{version})",
                "duration": "see log",
                "outcome": f"{verdict}: avg_sharpe={avg_sh:.3f}, min_sharpe={min_sh:.3f}",
                "artifacts": [],
                "summary_html": f"<pre>v200b v{version}: {verdict}\nAvg Sharpe: {avg_sh:.3f}\nMin Sharpe: {min_sh:.3f}\nFold Sharpes: {[round(f.sharpe,3) for f in wf.folds]}</pre>",
            })
        except Exception:
            pass

        if not gate_ok:
            _restore_previous("swing", prev, version)

        return 0 if gate_ok else 2

    except Exception as e:
        logger.error("v200b walk-forward failed: %s", e, exc_info=True)
        _restore_previous("swing", prev, version)
        return 1


if __name__ == "__main__":
    sys.exit(main())
