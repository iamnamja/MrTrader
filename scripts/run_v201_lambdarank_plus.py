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

    from app.ml.retrain_config import PHASE_C_PLUS_FEATURE_KEEP_LIST, SWING_GATE, MAX_WORKERS
    from app.ml.training import ModelTrainer
    from app.database.session import get_session
    from scripts.retrain_cron import _previous_active, _restore_previous

    model_type = "xgboost" if args.use_xgboost else "lambdarank"
    label_scheme = "triple_barrier" if args.use_xgboost else "lambdarank"

    db = get_session()
    try:
        prev = _previous_active(db, "swing")
    finally:
        db.close()

    logger.info("v201: %s, 17 features (14+3 interactions), prev_active=v%s",
                model_type, prev)
    logger.info("Features: %s", PHASE_C_PLUS_FEATURE_KEEP_LIST)

    trainer = ModelTrainer(
        model_type=model_type,
        label_scheme=label_scheme,
        hpo_trials=50,
        n_workers=MAX_WORKERS,
        feature_keep_list=PHASE_C_PLUS_FEATURE_KEEP_LIST,
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

        # v206: regime-gated inference — build benign_blocked_dates from PIT regime scores.
        # Training already uses exclude_risk_off_days=True; gating test entries too fixes
        # the train/test distribution mismatch (training = risk-on only, test = all days).
        benign_blocked_dates = None
        try:
            from app.ml.regime_score_pit import build_regime_score_map
            from app.ml.retrain_config import BENIGN_REGIME_THRESHOLD
            _score_map = build_regime_score_map()
            benign_blocked_dates = {d for d, s in _score_map.items() if s < BENIGN_REGIME_THRESHOLD}
            logger.info("BenignGate: %d adverse-regime dates blocked (threshold=%.2f)",
                        len(benign_blocked_dates), BENIGN_REGIME_THRESHOLD)
        except Exception as _be:
            logger.warning("BenignGate setup failed — running without regime gate: %s", _be)

        wf = run_swing_walkforward(
            n_folds=5,
            total_years=6,
            model_version=version,
            use_opportunity_score=True,
            no_prefilters=True,
            benign_blocked_dates=benign_blocked_dates,
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
