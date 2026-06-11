"""
OPT-4: CPCV validation of the systematic INDEX short-vol sleeve.

Short iron condors on SPY/QQQ/IWM opened on a fixed cadence (the index variance risk premium),
through the SAME trusted path as every edge: run_cpcv + significance gate + CAPM residual-α,
with the mandatory 1x/2x spread-stress sweep. Index option spreads are ~pennies and the VRP is
fatter + crisis-NEGATIVE (pairs with the trend sleeve) — the cost wall that killed single-name
earnings short-vol (OPT-3) is minimal here, so this is the better-founded short-vol bet.

Requires the options backfill (scripts/backfill_options.py) covering SPY/QQQ/IWM.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/run_index_shortvol_cpcv.py
    OPT_SPREAD_MULTS=1,2,3 python scripts/run_index_shortvol_cpcv.py
    python scripts/run_index_shortvol_cpcv.py [--hypothesis-id HYP-ID | --exploratory]
"""
import argparse
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
                    format="%(asctime)s %(levelname)s [idx_shortvol] %(message)s")
logger = logging.getLogger(__name__)

CPCV_K = 8
CPCV_PATHS = 2
TOTAL_YEARS = 4
UNIVERSE = ["SPY", "QQQ", "IWM"]
# Short-strike distance in REALIZED-vol SDs. Canonical short-vol sells ~16-delta ≈ 1 IMPLIED
# SD ≈ ~1.5 realized SD (implied > realized = the VRP). 1.0 sits inside the implied move and
# breaches ~32% -> structurally negative; ~1.5 targets the ~15% breach where the VRP is paid.
SD_MULT = float(os.environ.get("OPT_SD_MULT", "1.5"))


def _run_one(spread_mult: float):
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.options_strategy import IndexShortVolStrategy

    strat = IndexShortVolStrategy(symbols=UNIVERSE, spread_mult=spread_mult, sd_mult=SD_MULT)
    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strat.fetch_data(start_all, end_all)

    logger.info("=== Index short-vol CPCV @ spread_mult=%.1fx (k=%d, paths=%d) ===",
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
    from scripts.walkforward.registry_enforcement import add_arguments, begin_run

    ap = argparse.ArgumentParser(
        description="OPT-4 index short-vol CPCV (spread-stress sweep)")
    add_arguments(ap)  # --hypothesis-id / --exploratory (all-optional)
    args = ap.parse_args()
    # Registry enforcement FAILS FAST — before the multi-hour fetch/run.
    run = begin_run(args.hypothesis_id, exploratory=args.exploratory)

    mults = [float(x) for x in os.environ.get("OPT_SPREAD_MULTS", "1,2").split(",")]
    outcomes = [_run_one(m) for m in mults]

    print("\n" + "=" * 78)
    print("  OPT-4 INDEX SHORT-VOL — spread-stress sweep summary")
    print("=" * 78)
    for m, res, ok in outcomes:
        sharpe = getattr(res, "mean_sharpe", getattr(res, "avg_sharpe", float("nan")))
        t = getattr(res, "path_sharpe_tstat", float("nan"))
        ra_t = getattr(res, "residual_alpha_t_hac", float("nan"))
        pf = getattr(res, "avg_profit_factor", float("nan"))
        print(f"  {m:.1f}x spread: mean_sharpe={sharpe:.3f}  path_t={t:.2f}  "
              f"residual_alpha_t_HAC={ra_t:.2f}  PF={pf:.2f}  gate={'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    stress = max(outcomes, key=lambda o: o[0])
    verdict = "KEEP" if stress[2] else "KILL"
    print(f"  VERDICT (must pass at {stress[0]:.1f}x): {verdict}")
    print("=" * 78)

    # Best-effort registry recording (never crashes the run). decision=None:
    # promotion is owner-gated, never auto-decided from one run.
    if run is not None:
        run.record({
            "spread_mults": mults,
            "outcomes": [
                {"spread_mult": m,
                 "mean_sharpe": getattr(res, "mean_sharpe", None),
                 "tstat": getattr(res, "path_sharpe_tstat", None),
                 "gate_ok": ok}
                for m, res, ok in outcomes
            ],
            "stress_mult": stress[0],
            "stress_gate_ok": stress[2],
            "verdict": verdict,
        }, decision=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
