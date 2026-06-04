"""
Alpha-v3 A2: CPCV validation of the dollar-neutral short-interest factor.

Long lowest-days-to-cover / short highest-days-to-cover (the Boehmer/Asquith
short-interest anomaly), dollar-neutral by construction, through the reusable
EventEdgeStrategy harness — same CPCV + gate path as PEAD/A1. Because the book is
neutral by construction, this CPCV result IS the beta-isolated test (no separate
L/S re-run needed); we still CAPM-cross-check with scripts/analyst_beta_check.py
patterns if it looks alive.

Env levers:
  SI_N_PER_LEG       names per side (default 25)
  SI_MIN_DTC_SHORT   min days-to-cover to short (default 2.0)
  SI_MAX_HOLD_BARS   hold window (default 20 ~= 4wk; SI anomaly horizon)
  SI_SHORT_ONLY=1    short-only (no long leg)
  CPCV_SMOKE=1       fast smoke: k=3, 2yr, SI_SMOKE_SYMBOLS subset

Usage:
    python scripts/run_si_factor_cpcv.py
    CPCV_SMOKE=1 python scripts/run_si_factor_cpcv.py
"""
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
                    format="%(asctime)s %(levelname)s [si_factor_cpcv] %(message)s")
logger = logging.getLogger(__name__)

_SMOKE = os.environ.get("CPCV_SMOKE") == "1"
CPCV_K = 3 if _SMOKE else 8
CPCV_PATHS = 2
TOTAL_YEARS = 2 if _SMOKE else 6


def build_scorer():
    from app.ml.short_interest_factor_scorer import ShortInterestFactorScorer
    return ShortInterestFactorScorer(
        n_per_leg=int(os.environ.get("SI_N_PER_LEG", "25")),
        min_dtc_short=float(os.environ.get("SI_MIN_DTC_SHORT", "2.0")),
        long_short=os.environ.get("SI_SHORT_ONLY") != "1",
        vix_block_all=30.0,
    )


def main() -> int:
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.walkforward.cpcv import run_cpcv
    from app.data.short_interest_provider import load_short_interest

    si = load_short_interest(refresh=True)
    if si is None or si.empty:
        logger.error("short_interest.parquet empty — run scripts/backfill_short_interest.py first")
        return 1
    logger.info("SI store: %d rows, %d tickers", len(si), si["ticker"].nunique())

    if _SMOKE and os.environ.get("SI_SMOKE_SYMBOLS"):
        symbols = os.environ["SI_SMOKE_SYMBOLS"].split(",")
    else:
        symbols = list(RUSSELL_1000_TICKERS)

    hold = int(os.environ.get("SI_MAX_HOLD_BARS", "20"))
    strategy = EventEdgeStrategy(
        scorer=build_scorer(),
        symbols=symbols,
        model_type="si_factor",
        transaction_cost_pct=0.0005,
        max_hold_bars_override=hold,
    )

    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strategy.fetch_data(start_all, end_all)

    logger.info("Running SI-FACTOR CPCV: k=%d paths=%d years=%d hold=%d short_only=%s smoke=%s",
                CPCV_K, CPCV_PATHS, TOTAL_YEARS, hold,
                os.environ.get("SI_SHORT_ONLY") == "1", _SMOKE)

    result = run_cpcv(strategy=strategy, purge_days=10, embargo_days=10,
                      n_folds=CPCV_K, n_paths=CPCV_PATHS, total_years=TOTAL_YEARS)
    result.print()

    gate_detail = result.gate_detail()
    gate_ok = all(v for _, v in gate_detail.values())
    verdict = "CPCV GATE PASSED" if gate_ok else "CPCV GATE FAILED"
    logger.info("SI-FACTOR CPCV %s - mean_sharpe=%.3f p5=%.3f p95=%.3f",
                verdict, result.mean_sharpe, result.p5_sharpe, result.p95_sharpe)

    if not _SMOKE:
        try:
            from app.notifications.notifier import _smtp_send
            _smtp_send(
                subject=f"MrTrader A2 SI-factor CPCV: {verdict} (mean={result.mean_sharpe:.3f})",
                html_body=f"""
<h2>Alpha-v3 A2 — Dollar-neutral short-interest factor CPCV</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Mean Sharpe: {result.mean_sharpe:.3f} (gate &ge;0.80)</li>
  <li>P5 Sharpe: {result.p5_sharpe:.3f} &middot; P95: {result.p95_sharpe:.3f}</li>
  <li>Gate detail: {gate_detail}</li>
  <li>N paths: {len(result.path_sharpes)}</li>
</ul>
<p>Dollar-neutral by construction (long low-DTC / short high-DTC) -> this IS the
beta-isolated test. vs PEAD +0.546. If null, the SI anomaly joins the closed list;
if alive, candidate edge #2.</p>
""",
            )
        except Exception as _e:
            logger.warning("Email notification failed: %s", _e)

    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
