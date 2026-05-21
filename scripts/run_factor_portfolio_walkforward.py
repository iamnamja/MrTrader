"""
Phase D: Walk-forward validation of the factor portfolio scorer.

Runs the standard 5-fold, 6-year walk-forward using FactorPortfolioScorer
instead of an ML model. No training needed — pure scoring.

Usage:
    python scripts/run_factor_portfolio_walkforward.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [factor_wf] %(message)s",
)
logger = logging.getLogger(__name__)

GATE = {"min_avg_sharpe": 0.80, "min_fold_sharpe": -0.30}

# v4: top_n=10 + require stock beats SPY over 20 days (market-relative momentum filter).
# Hypothesis: In 2024 Mag7 concentration era, most stocks underperformed SPY even while
# individually up. Requiring outperformance vs SPY naturally selects the Mag7 leaders
# and other market-beating names — exactly the stocks that drove 2024 returns.
# This is cross-sectional momentum (the strongest documented factor).
SCORER_CONFIG = {
    "top_n": 10,
    "long_short": False,
    "vix_threshold": 30.0,
    "spy_ma_window": 200,
    "require_spy_outperform_days": 20,  # only buy stocks that beat SPY over last 20 days
}


def main() -> int:
    from app.ml.factor_scorer import FactorPortfolioScorer
    from scripts.walkforward_tier3 import run_swing_walkforward

    scorer = FactorPortfolioScorer(**SCORER_CONFIG)

    logger.info(
        "Factor portfolio walk-forward: 5 folds, 6yr, top-%d, VIX≤%.0f, SPY-200DMA, SPY-rel-mom%dd",
        SCORER_CONFIG["top_n"], SCORER_CONFIG["vix_threshold"],
        SCORER_CONFIG.get("require_spy_outperform_days", 0),
    )

    wf = run_swing_walkforward(
        n_folds=5,
        total_years=6,
        use_opportunity_score=False,
        no_prefilters=True,
        feature_cache_disable=True,
        scorer_instance=scorer,  # triggers VIX download + wires vix_history to scorer
    )

    avg_sh = wf.avg_sharpe
    min_sh = wf.min_sharpe
    gate_ok = avg_sh >= GATE["min_avg_sharpe"] and min_sh >= GATE["min_fold_sharpe"]

    verdict = "GATE PASSED" if gate_ok else "GATE FAILED"
    logger.info(
        "Factor portfolio WF %s — avg_sharpe=%.3f  min_fold=%.3f  folds=%s",
        verdict, avg_sh, min_sh,
        [round(f.sharpe, 3) for f in wf.folds],
    )

    try:
        from dotenv import load_dotenv
        load_dotenv()
        from app.notifications.notifier import _smtp_send
        fold_rows = "\n".join(
            f"  Fold {f.fold}: Sharpe={f.sharpe:.3f}  trades={f.trades}" for f in wf.folds
        )
        config_str = ", ".join(f"{k}={v}" for k, v in SCORER_CONFIG.items())
        _smtp_send(
            subject=f"MrTrader Factor Portfolio WF: {verdict} (avg={avg_sh:.3f})",
            html_body=f"""
<h2>Factor Portfolio Walk-Forward Result</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Avg Sharpe: {avg_sh:.3f} (gate: ≥0.80)</li>
  <li>Min fold: {min_sh:.3f} (gate: ≥-0.30)</li>
</ul>
<pre>{fold_rows}</pre>
<p><small>Config: {config_str}</small></p>
""",
        )
    except Exception as _e:
        logger.warning("Email notification failed: %s", _e)

    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
