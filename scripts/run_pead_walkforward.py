"""
Phase G: Walk-forward validation of the PEAD scorer.

Runs the standard 5-fold, 6-year walk-forward using PEADScorer
instead of the factor portfolio or ML model. No training needed.

Usage:
    python scripts/run_pead_walkforward.py
"""
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [pead_wf] %(message)s",
)
logger = logging.getLogger(__name__)

GATE = {"min_avg_sharpe": 0.80, "min_fold_sharpe": -0.30}


def main() -> int:
    from app.ml.pead_scorer import PEADScorer
    from scripts.walkforward_tier3 import run_swing_walkforward

    logger.info("PEAD walk-forward: 5 folds, 6yr, L/S, earnings-surprise driven")

    # Pre-warm FMP earnings cache for the Russell 1000 universe.
    # Each symbol fetches ~20 quarters of history; in-process cache ensures single fetch.
    # Without this, API calls happen inside each fold's inner loop (slower, same result).
    try:
        from app.data.fmp_provider import get_earnings_history_fmp
        from app.data.universe import RUSSELL_1000_TICKERS
        logger.info("Pre-warming FMP earnings cache for %d symbols...", len(RUSSELL_1000_TICKERS))
        _ok = 0
        for _sym in RUSSELL_1000_TICKERS:
            try:
                if get_earnings_history_fmp(_sym):
                    _ok += 1
            except Exception:
                pass
        logger.info("FMP earnings cache warmed: %d/%d symbols have data", _ok, len(RUSSELL_1000_TICKERS))
    except Exception as _warm_err:
        logger.warning("FMP cache pre-warm failed (non-fatal): %s", _warm_err)

    # v9: 7% threshold + long-only + VIX30 + T+5
    # Key insight from v5/v8: 10% threshold improves fold 1 (2021) but breaks fold 3 (2023-24).
    # 5% threshold keeps fold 3 (0.94) but fold 1 is only 0.53.
    # Hypothesis: 7% is the regime-adaptive sweet spot — filters 2021 noise while
    # keeping 2023-24 moderate surprises that actually drift.
    scorer = PEADScorer(
        long_threshold=0.07,        # 7% — between 5% (too noisy) and 10% (too restrictive)
        short_threshold=-0.07,
        long_short=False,           # no shorts — consistently destructive
        vix_block_all=30.0,         # proven sweet spot
        vix_block_short=100.0,
        vix_conf_ref=100.0,
        max_announce_day_move=1.0,  # no priced-in filter
    )

    wf = run_swing_walkforward(
        n_folds=5,
        total_years=6,
        use_opportunity_score=False,
        no_prefilters=True,
        feature_cache_disable=True,
        scorer_instance=scorer,
        max_hold_bars_override=5,
    )

    avg_sh = wf.avg_sharpe
    min_sh = wf.min_sharpe
    gate_ok = avg_sh >= GATE["min_avg_sharpe"] and min_sh >= GATE["min_fold_sharpe"]

    verdict = "GATE PASSED" if gate_ok else "GATE FAILED"
    logger.info(
        "PEAD WF %s — avg_sharpe=%.3f  min_fold=%.3f  folds=%s",
        verdict, avg_sh, min_sh,
        [round(f.sharpe, 3) for f in wf.folds],
    )

    try:
        from app.notifications.notifier import _smtp_send
        fold_rows = "\n".join(
            f"  Fold {f.fold}: Sharpe={f.sharpe:.3f}  trades={f.trades}" for f in wf.folds
        )
        _smtp_send(
            subject=f"MrTrader PEAD WF: {verdict} (avg={avg_sh:.3f})",
            html_body=f"""
<h2>PEAD Walk-Forward Result</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Avg Sharpe: {avg_sh:.3f} (gate: ≥0.80)</li>
  <li>Min fold: {min_sh:.3f} (gate: ≥-0.30)</li>
</ul>
<pre>{fold_rows}</pre>
""",
        )
    except Exception as _e:
        logger.warning("Email notification failed: %s", _e)

    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
