"""
G-Pre — Model Walk-Forward (ML model.predict, not FactorPortfolioScorer).

This runs the standard 5-fold, 6-year walk-forward using the TRAINED LambdaRank
model (swing_vN) instead of the hand-crafted FactorPortfolioScorer.

IMPORTANT: All prior WF runs (factor portfolio iterations) used scorer_instance
to bypass model.predict entirely. Score reconciliation audit showed Spearman=0.035
between scorer and model rankings — they are statistically independent.
This script provides the FIRST honest WF evaluation of the trained ML model.

LOOK-AHEAD DISCLOSURE: swing_v214 was trained on all available data including
the WF test windows. Results should be treated as an OOS test, not true WF.
If results show signal, schedule per-fold retraining for a proper WF.

Usage:
    python scripts/run_model_walkforward.py [--model swing_v214]
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [model_wf] %(message)s",
)
logger = logging.getLogger(__name__)

GATE = {"min_avg_sharpe": 0.80, "min_fold_sharpe": -0.30}
# Pre-registered kill criterion (see MASTER_BACKLOG.md):
# If avg Sharpe < 0.40 AND >= 2 folds negative -> KILL single-name picker
KILL_THRESHOLD = 0.40


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="swing_v214", help="Model version name (no .pkl)")
    args = parser.parse_args()

    from scripts.walkforward_tier3 import run_swing_walkforward

    logger.info(
        "Model walk-forward: 5 folds, 6yr, model=%s (NO scorer override)",
        args.model,
    )
    logger.warning(
        "DISCLOSURE: %s was trained on all available data — results are OOS test, "
        "not true WF. If signal found, run per-fold retraining for proper WF.",
        args.model,
    )

    wf = run_swing_walkforward(
        n_folds=5,
        total_years=6,
        use_opportunity_score=False,
        no_prefilters=True,
        feature_cache_disable=False,  # use cache for speed
        scorer_instance=None,         # KEY: no scorer override -> uses model.predict
        use_factor_portfolio=False,   # KEY: pure model path
        model_version=None,           # load latest
    )

    avg_sh = wf.avg_sharpe
    min_sh = wf.min_sharpe
    n_negative = sum(1 for f in wf.folds if f.sharpe < 0)

    gate_ok = avg_sh >= GATE["min_avg_sharpe"] and min_sh >= GATE["min_fold_sharpe"]
    kill_triggered = avg_sh < KILL_THRESHOLD and n_negative >= 2

    if gate_ok:
        verdict = "GATE PASSED"
    elif kill_triggered:
        verdict = "KILL CRITERION TRIGGERED"
    else:
        verdict = "GATE FAILED (continue with label/feature work)"

    logger.info(
        "Model WF %s — avg_sharpe=%.3f  min_fold=%.3f  n_negative_folds=%d",
        verdict, avg_sh, min_sh, n_negative,
    )
    logger.info("Folds: %s", [round(f.sharpe, 3) for f in wf.folds])

    if kill_triggered:
        logger.critical(
            "KILL CRITERION MET: avg Sharpe %.3f < %.1f with %d negative folds. "
            "See MASTER_BACKLOG.md kill criterion — pivot to ETF strategy.",
            avg_sh, KILL_THRESHOLD, n_negative,
        )

    try:
        from dotenv import load_dotenv
        load_dotenv()
        from app.notifications.notifier import _smtp_send
        fold_rows = "\n".join(
            f"  Fold {f.fold}: Sharpe={f.sharpe:.3f}  trades={f.trades}  "
            f"winrate={f.win_rate:.1%}  maxDD={f.max_drawdown:.1%}"
            for f in wf.folds
        )
        color = "green" if gate_ok else ("red" if kill_triggered else "orange")
        _smtp_send(
            subject=f"MrTrader Model WF: {verdict} (avg={avg_sh:.3f})",
            html_body=f"""
<h2>Model Walk-Forward Result (swing_v214, ML model.predict)</h2>
<p style="color:{color}"><b>{verdict}</b></p>
<ul>
  <li>Avg Sharpe: {avg_sh:.3f} (gate: >=0.80, kill: &lt;0.40)</li>
  <li>Min fold: {min_sh:.3f} (gate: >=-0.30)</li>
  <li>Negative folds: {n_negative}/5</li>
</ul>
<pre>{fold_rows}</pre>
<p><b>Context:</b> First WF run using model.predict instead of FactorPortfolioScorer.
Prior score reconciliation audit showed scorer-model Spearman=0.035 (independent systems).
All prior WF Sharpe numbers described the scorer, not this model.</p>
<p><small>Disclosure: swing_v214 trained on all data (look-ahead risk). If signal found,
schedule per-fold retraining for proper WF.</small></p>
""",
        )
    except Exception as e:
        logger.warning("Email failed: %s", e)

    if gate_ok:
        return 0
    elif kill_triggered:
        return 3  # distinct exit code for kill criterion
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
