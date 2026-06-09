"""
OPT-3: CPCV validation of the earnings IV-crush options sleeve — the program's FIRST verdict.

Runs the OptionsStrategy adapter (earnings IV-crush iron condor) through the SAME trusted
path as every equity edge: scripts.walkforward.cpcv.run_cpcv + the significance gate + CAPM
residual-α. Per the KEEP/KILL protocol it sweeps the spread-cost stress (1× and 2×) — a KEEP
must survive 2×. A KILL here is a success of the harness (cf. reversal / carry).

Requires the options backfill (scripts/backfill_options.py) to have populated
data/options_bars.parquet for the universe + window.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/run_options_ivcrush_cpcv.py
    OPT_SPREAD_MULTS=1,2,3 python scripts/run_options_ivcrush_cpcv.py
"""
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [opt_ivcrush] %(message)s")
logger = logging.getLogger(__name__)

CPCV_K = 8
CPCV_PATHS = 2
TOTAL_YEARS = 4

# The broad backfilled universe (matches scripts/backfill_options.py defaults + extras).
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG",
    "TSLA", "AMD", "NFLX", "AVGO", "JPM", "BAC", "WFC", "GS", "MS", "XOM", "CVX", "XLE",
    "UNH", "JNJ", "PFE", "LLY", "HD", "WMT", "COST", "NKE", "MCD", "KO", "PEP", "DIS",
    "INTC", "CSCO", "ORCL", "CRM", "ADBE", "GLD", "SLV", "TLT", "BA", "CAT", "V", "MA",
]
# Index/commodity ETFs have no single-name earnings event -> exclude from the event universe.
NON_EARNINGS = {"SPY", "QQQ", "IWM", "DIA", "XLE", "GLD", "SLV", "TLT"}


def _run_one(spread_mult: float):
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.options_strategy import OptionsStrategy

    syms = [s for s in UNIVERSE if s not in NON_EARNINGS]
    strat = OptionsStrategy(symbols=syms, spread_mult=spread_mult)
    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strat.fetch_data(start_all, end_all)

    logger.info("=== IV-crush CPCV @ spread_mult=%.1fx (k=%d, paths=%d) ===",
                spread_mult, CPCV_K, CPCV_PATHS)
    result = run_cpcv(strategy=strat, purge_days=10, embargo_days=10,
                      n_folds=CPCV_K, n_paths=CPCV_PATHS, total_years=TOTAL_YEARS)
    result.print()
    try:
        detail = result.gate_detail()
        gate_ok = all(v for _, v in detail.values())
    except Exception:
        gate_ok = False
    logger.info("spread_mult=%.1fx -> gate_ok=%s", spread_mult, gate_ok)
    return spread_mult, result, gate_ok


def main() -> int:
    mults = [float(x) for x in os.environ.get("OPT_SPREAD_MULTS", "1,2").split(",")]
    outcomes = []
    for m in mults:
        outcomes.append(_run_one(m))

    print("\n" + "=" * 78)
    print("  OPT-3 EARNINGS IV-CRUSH — spread-stress sweep summary")
    print("=" * 78)
    for m, res, ok in outcomes:
        sharpe = getattr(res, "mean_sharpe", getattr(res, "avg_sharpe", float("nan")))
        t = getattr(res, "path_sharpe_tstat", float("nan"))
        ra_t = getattr(res, "residual_alpha_t_hac", float("nan"))
        print(f"  {m:.1f}x spread: mean_sharpe={sharpe:.3f}  path_t={t:.2f}  "
              f"residual_alpha_t_HAC={ra_t:.2f}  gate={'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    # KEEP requires the gate to pass at 2x (or the largest mult tested).
    stress = max(outcomes, key=lambda o: o[0])
    verdict = "KEEP" if stress[2] else "KILL"
    print(f"  VERDICT (must pass at {stress[0]:.1f}x): {verdict}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
