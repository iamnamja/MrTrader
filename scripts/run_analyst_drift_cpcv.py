"""
Alpha-v3 A1: CPCV validation of the analyst up/downgrade drift edge.

Runs the AnalystRevisionScorer through the reusable EventEdgeStrategy harness
(scripts/walkforward/event_edge.py) — the SAME trusted CPCV + gate path as PEAD.

Env levers:
  ANALYST_LONG_SHORT=1   enable the short leg (long upgrades + short downgrades)
  ANALYST_MAX_HOLD_BARS  drift hold window in bars (default 20 ~= 4 weeks)
  ANALYST_MAX_DAYS_AFTER act within N days of the rating change (default 5)
  ANALYST_MIN_NET        net-momentum confirmation threshold (default 1)
  CPCV_SMOKE=1           fast smoke: k=3, 2yr, env ANALYST_SMOKE_SYMBOLS subset

Usage:
    python scripts/run_analyst_drift_cpcv.py [--hypothesis-id HYP-ID | --exploratory]
    CPCV_SMOKE=1 python scripts/run_analyst_drift_cpcv.py
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from app.ml.retrain_config import MAX_THREADS as _max_threads
os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))

from scripts.walkforward.event_edge import EventEdgeStrategy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [analyst_cpcv] %(message)s")
logger = logging.getLogger(__name__)

_SMOKE = os.environ.get("CPCV_SMOKE") == "1"
CPCV_K = 3 if _SMOKE else 8
CPCV_PATHS = 2
TOTAL_YEARS = 2 if _SMOKE else 6


def build_scorer():
    from app.ml.analyst_revision_scorer import AnalystRevisionScorer
    return AnalystRevisionScorer(
        lookback_days=30,
        max_days_after=int(os.environ.get("ANALYST_MAX_DAYS_AFTER", "5")),
        min_net_momentum=float(os.environ.get("ANALYST_MIN_NET", "1")),
        long_short=os.environ.get("ANALYST_LONG_SHORT") == "1",
        vix_block_all=30.0,
    )


def main() -> int:
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.registry_enforcement import add_arguments, begin_run

    ap = argparse.ArgumentParser(description="Analyst up/downgrade drift CPCV")
    add_arguments(ap)  # --hypothesis-id / --exploratory (all-optional)
    args = ap.parse_args()
    # Registry enforcement FAILS FAST — before the multi-hour fetch/run.
    run = begin_run(args.hypothesis_id, exploratory=args.exploratory)

    if _SMOKE and os.environ.get("ANALYST_SMOKE_SYMBOLS"):
        symbols = os.environ["ANALYST_SMOKE_SYMBOLS"].split(",")
    else:
        symbols = list(RUSSELL_1000_TICKERS)

    hold = int(os.environ.get("ANALYST_MAX_HOLD_BARS", "20"))
    strategy = EventEdgeStrategy(
        scorer=build_scorer(),
        symbols=symbols,
        model_type="analyst_drift",
        transaction_cost_pct=0.0005,
        max_hold_bars_override=hold,
    )

    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strategy.fetch_data(start_all, end_all)

    logger.info("Running ANALYST-DRIFT CPCV: k=%d paths=%d years=%d hold=%d ls=%s smoke=%s",
                CPCV_K, CPCV_PATHS, TOTAL_YEARS, hold,
                os.environ.get("ANALYST_LONG_SHORT") == "1", _SMOKE)

    result = run_cpcv(
        strategy=strategy,
        purge_days=10,
        embargo_days=10,
        n_folds=CPCV_K,
        n_paths=CPCV_PATHS,
        total_years=TOTAL_YEARS,
    )
    result.print()

    gate_detail = result.gate_detail()
    gate_ok = all(v for _, v in gate_detail.values())
    verdict = "CPCV GATE PASSED" if gate_ok else "CPCV GATE FAILED"
    logger.info("ANALYST-DRIFT CPCV %s - mean_sharpe=%.3f p5=%.3f p95=%.3f",
                verdict, result.mean_sharpe, result.p5_sharpe, result.p95_sharpe)

    # Best-effort registry recording (never crashes the run). decision=None:
    # promotion is owner-gated, never auto-decided from one run.
    if run is not None:
        run.record({
            "mean_sharpe": result.mean_sharpe,
            "p5_sharpe": result.p5_sharpe,
            "p95_sharpe": result.p95_sharpe,
            "tstat": result.path_sharpe_tstat,
            "n_paths": len(result.path_sharpes),
            "gate_detail": gate_detail,
            "gate_ok": gate_ok,
        }, decision=None)

    if not _SMOKE:
        try:
            from app.notifications.notifier import _smtp_send
            _smtp_send(
                subject=f"MrTrader A1 analyst-drift CPCV: {verdict} (mean={result.mean_sharpe:.3f})",
                html_body=f"""
<h2>Alpha-v3 A1 — Analyst up/downgrade drift CPCV</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Mean Sharpe: {result.mean_sharpe:.3f} (gate ≥0.80)</li>
  <li>P5 Sharpe: {result.p5_sharpe:.3f} · P95: {result.p95_sharpe:.3f}</li>
  <li>Gate detail: {gate_detail}</li>
  <li>N paths: {len(result.path_sharpes)}</li>
</ul>
<p>vs PEAD +0.546. If null, analyst-drift joins the closed list; if alive, candidate edge #2.</p>
""",
            )
        except Exception as _e:
            logger.warning("Email notification failed: %s", _e)

    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
