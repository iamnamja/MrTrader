"""
LambdaRank training + rebalance-mode WF gate for v217+ swing models.

Usage:
    python scripts/run_v201_lambdarank_plus.py               # train + gate
    python scripts/run_v201_lambdarank_plus.py --gate-only 218  # re-gate existing model
    python scripts/run_v201_lambdarank_plus.py --use-xgboost    # XGBoost binary (not recommended)

WF gate runs in REBALANCE mode (top-30 portfolio, 20-day rebalance, regime gate + inv-vol).
LambdaRank models must NOT be gated in scan mode — the ranker outputs relative ranks, not
calibrated trade-success probabilities. Scan-mode gate was the root cause of v216/v218 misdiagnosis.
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


def _run_gate(version: int, model_type: str, prev_version: int) -> int:
    """Run rebalance-mode WF gate on an already-trained model version."""
    from app.ml.retrain_config import SWING_GATE
    from app.ml.training import ModelTrainer
    from scripts.retrain_cron import _restore_previous
    from scripts.walkforward_tier3 import run_swing_walkforward

    logger.info(
        "Running rebalance-mode WF gate on v%d (model_type=%s, prev_active=v%s)",
        version, model_type, prev_version,
    )

    # Gate MUST use rebalance mode for LambdaRank models.
    # Scan mode is invalid: ranker scores are relative ranks, not calibrated signals.
    use_rebalance = (model_type == "lambdarank")
    if not use_rebalance:
        logger.warning(
            "model_type=%s — running scan-mode gate (not rebalance). "
            "Only use lambdarank for production models.",
            model_type,
        )

    wf = run_swing_walkforward(
        n_folds=5,
        total_years=6,
        model_version=version,
        use_opportunity_score=False,    # ranker scores are not probabilities
        no_prefilters=True,
        # Rebalance-mode params (production architecture)
        rebalance_mode=use_rebalance,
        rebalance_days=20,
        rebalance_target_n=30,
        rebalance_add_threshold=15,
        rebalance_drop_threshold=30,
        rebalance_sector_cap=0.30,
        rebalance_min_adv=20_000_000.0,
        rebalance_regime_gate=use_rebalance,
        rebalance_regime_spy_ma_days=200,
        rebalance_inv_vol=use_rebalance,
        rebalance_inv_vol_lookback=20,
        rebalance_inv_vol_min_mult=0.5,
        rebalance_inv_vol_max_mult=2.0,
    )

    avg_sh = wf.avg_sharpe
    min_sh = wf.min_sharpe
    gate_ok = (avg_sh >= SWING_GATE["min_avg_sharpe"] and
               min_sh >= SWING_GATE["min_fold_sharpe"])

    ModelTrainer.record_tier3_result(version, avg_sh, [f.sharpe for f in wf.folds], gate_ok)
    verdict = "GATE PASSED" if gate_ok else "GATE FAILED"
    logger.info(
        "v%d %s (avg_sharpe=%.3f min=%.3f folds=%s)",
        version, verdict, avg_sh, min_sh,
        [round(f.sharpe, 3) for f in wf.folds],
    )

    try:
        from app.notifications import notifier as _notifier
        _notifier.enqueue("training_complete", {
            "model": f"swing ({model_type}, rebalance-mode gate)",
            "version": version,
            "sharpe": round(avg_sh, 3),
            "gate_result": verdict,
            "fold_table_html": "<pre>" + "\n".join(
                f"Fold {f.fold}: Sharpe={f.sharpe:.3f}" for f in wf.folds
            ) + "</pre>",
        })
    except Exception:
        pass

    if not gate_ok:
        _restore_previous("swing", prev_version, version)

    return 0 if gate_ok else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-xgboost", action="store_true",
                        help="Use XGBoost binary instead of LambdaRank (not recommended)")
    parser.add_argument("--gate-only", type=int, metavar="VERSION",
                        help="Skip training; re-run WF gate on an existing model version")
    args = parser.parse_args()

    from app.ml.retrain_config import (
        BENIGN_SWING_FEATURES,
        SWING_GATE, MAX_WORKERS,
    )
    from app.ml.training import ModelTrainer
    from app.database.session import get_session
    from scripts.retrain_cron import _previous_active, _restore_previous

    model_type = "xgboost" if args.use_xgboost else "lambdarank"

    db = get_session()
    try:
        prev = _previous_active(db, "swing")
    finally:
        db.close()

    if args.gate_only is not None:
        return _run_gate(args.gate_only, model_type, prev)

    label_scheme = "triple_barrier" if args.use_xgboost else "lambdarank"

    # v217: IC-audited 17-feature cross-regime set (2026-05-24 audit).
    # Drops bull-tape-only factors (gross_margin, revenue_growth, near_52w_high,
    # trend_consistency_63d); adds counter-trend (reversal_5d_vol_weighted,
    # downtrend) and WQ alphas (wq_alpha35/40/43) with positive 2022 bear-year IC.
    feature_list = BENIGN_SWING_FEATURES
    n_feats = len(feature_list)

    logger.info("v217: %s, %d features, NDCG@3 HPO seed=42, num_leaves<=31, prev_active=v%s",
                model_type, n_feats, prev)
    logger.info("Features: %s", feature_list)

    trainer = ModelTrainer(
        model_type=model_type,
        label_scheme=label_scheme,
        hpo_trials=50,
        hpo_seed=42,
        hpo_ndcg_k=3,
        n_workers=MAX_WORKERS,
        feature_keep_list=feature_list,
    )

    try:
        version = trainer.train_model(
            fetch_fundamentals=False,
            exclude_risk_off_days=True,
            use_union_label=False,
        )
        logger.info("v217 trained as v%d — running rebalance-mode WF gate...", version)
    except Exception as e:
        logger.error("v217 training failed: %s", e, exc_info=True)
        return 1

    try:
        return _run_gate(version, model_type, prev)
    except Exception as e:
        logger.error("v217 walk-forward gate failed: %s", e, exc_info=True)
        _restore_previous("swing", prev, version)
        return 1


if __name__ == "__main__":
    sys.exit(main())
