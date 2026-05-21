"""Temp script: factor WF v9 — top_n=5, NO momentum filter (ablation: does momentum help?)."""
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from app.ml.retrain_config import MAX_THREADS as _max_threads
os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [factor_wf_v9] %(message)s")
logger = logging.getLogger(__name__)

GATE = {"min_avg_sharpe": 0.80, "min_fold_sharpe": -0.30}
SCORER_CONFIG = {
    "top_n": 5,
    "long_short": False,
    "vix_threshold": 30.0,
    "spy_ma_window": 200,
    # no momentum filter — ablation to see if 20d filter adds value vs pure top-5
}


def main() -> int:
    from app.ml.factor_scorer import FactorPortfolioScorer
    from scripts.walkforward_tier3 import run_swing_walkforward

    scorer = FactorPortfolioScorer(**SCORER_CONFIG)
    logger.info("Factor WF v9: top_n=5, NO momentum filter, VIX≤30, SPY-200DMA")

    wf = run_swing_walkforward(
        n_folds=5, total_years=6,
        use_opportunity_score=False, no_prefilters=True,
        feature_cache_disable=True, scorer_instance=scorer,
    )

    avg_sh, min_sh = wf.avg_sharpe, wf.min_sharpe
    gate_ok = avg_sh >= GATE["min_avg_sharpe"] and min_sh >= GATE["min_fold_sharpe"]
    verdict = "GATE PASSED" if gate_ok else "GATE FAILED"
    logger.info("v9 %s — avg=%.3f min=%.3f folds=%s", verdict, avg_sh, min_sh,
                [round(f.sharpe, 3) for f in wf.folds])

    try:
        from app.notifications.notifier import _smtp_send
        fold_rows = "\n".join(f"  Fold {f.fold}: Sharpe={f.sharpe:.3f}  trades={f.trades}" for f in wf.folds)
        _smtp_send(
            subject=f"MrTrader Factor WF v9: {verdict} (avg={avg_sh:.3f})",
            html_body=f"<h2>Factor WF v9 (top_n=5, no momentum)</h2><p><b>{verdict}</b> avg={avg_sh:.3f} min={min_sh:.3f}</p><pre>{fold_rows}</pre>",
        )
    except Exception as e:
        logger.warning("Email failed: %s", e)

    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
