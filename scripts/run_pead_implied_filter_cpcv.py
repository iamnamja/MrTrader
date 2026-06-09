"""
OPT-5: does the options-implied "priced-in" filter improve PEAD?

Runs the PEAD CPCV (the host sleeve's OWN gate) twice on the SAME options-covered window:
  baseline  — committed PEAD scorer (no options filter)
  filtered  — same scorer + the implied-move filter (skip entries whose realized announce-day
              move was within/under the pre-earnings IMPLIED move: realized/implied < threshold)
and reports the delta. Judged on PEAD's existing gate — no options execution, no alpha-gate-vs-
risk-premium mismatch (the lens that paused the standalone options sleeves).

Window = TOTAL_YEARS (default 2) to match the 2y R1K options backfill, so both arms see the same
period and the filter can actually act on every signal. A KILL (no improvement) is an honest
result — the prior PRICE-based priced-in filter HURT PEAD, so this is genuinely uncertain.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/run_pead_implied_filter_cpcv.py
    OPT_IMPLIED_RATIO=1.25 python scripts/run_pead_implied_filter_cpcv.py
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
                    format="%(asctime)s %(levelname)s [pead_implied] %(message)s")
logger = logging.getLogger(__name__)

CPCV_K = 8
CPCV_PATHS = 2
TOTAL_YEARS = 2   # match the 2y R1K options backfill window
IMPLIED_RATIO = float(os.environ.get("OPT_IMPLIED_RATIO", "1.0"))  # skip if realized/implied < this


def _run(label: str, with_filter: bool):
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.run_pead_cpcv import PEADStrategy, build_pead_scorer

    scorer = build_pead_scorer()
    if with_filter:
        from app.data.options_signal import ImpliedMoveProvider
        scorer.implied_move_fn = ImpliedMoveProvider().implied_move
        scorer.min_move_vs_implied = IMPLIED_RATIO

    strat = PEADStrategy(scorer=scorer, symbols=list(RUSSELL_1000_TICKERS),
                         transaction_cost_pct=0.0005)
    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strat.fetch_data(start_all, end_all)
    logger.info("=== PEAD CPCV (%s) k=%d paths=%d years=%d ===",
                label, CPCV_K, CPCV_PATHS, TOTAL_YEARS)
    res = run_cpcv(strategy=strat, purge_days=10, embargo_days=10,
                   n_folds=CPCV_K, n_paths=CPCV_PATHS, total_years=TOTAL_YEARS)
    res.print()
    return res


def _m(res):
    def g(*names):
        for n in names:
            v = getattr(res, n, None)
            if v is not None:
                return v
        return float("nan")
    return {
        "sharpe": g("mean_sharpe", "avg_sharpe"),
        "t": g("path_sharpe_tstat"),
        "ra_t": g("residual_alpha_t_hac"),
        "ra_sharpe": g("residual_alpha_sharpe", "hedged_sharpe"),
        "pf": g("avg_profit_factor"),
        "pos": g("pct_positive"),
    }


def main() -> int:
    base = _m(_run("baseline", with_filter=False))
    filt = _m(_run(f"implied-filter@{IMPLIED_RATIO}", with_filter=True))
    print("\n" + "=" * 80)
    print(f"  OPT-5 PEAD options-implied priced-in filter (realized/implied < {IMPLIED_RATIO} -> skip)")
    print("=" * 80)
    print(f"  {'arm':<22}{'mean_sharpe':>12}{'path_t':>9}{'resid_a_t':>11}{'PF':>7}{'%pos':>7}")
    for name, m in [("baseline", base), ("implied-filter", filt)]:
        _pos = m["pos"] * 100 if isinstance(m["pos"], (int, float)) and m["pos"] == m["pos"] \
            else float("nan")
        print(f"  {name:<22}{m['sharpe']:>12.3f}{m['t']:>9.2f}{m['ra_t']:>11.2f}"
              f"{m['pf']:>7.2f}{_pos:>6.1f}%")
    d_sharpe = filt["sharpe"] - base["sharpe"]
    d_pf = filt["pf"] - base["pf"]
    d_rasharpe = filt["ra_sharpe"] - base["ra_sharpe"]
    print("-" * 80)
    print(f"  delta: mean_sharpe {d_sharpe:+.3f}, PF {d_pf:+.2f}, "
          f"beta-hedged Sharpe {d_rasharpe:+.3f} (resid-α t {base['ra_t']:.2f}->{filt['ra_t']:.2f})")
    # residual-α now available (EventEdge emits daily_returns_dated). The make-or-break check for
    # a PEAD enhancement is whether the lift is ALPHA (beta-hedged Sharpe up, β ~flat) vs just
    # more beta. Significance (resid-α t > 2) still needs power (more data / thresholds).
    improves = d_sharpe > 0.10 and d_pf > 0.0
    alpha_like = d_rasharpe > 0.05 and filt["ra_t"] >= base["ra_t"]
    if improves and alpha_like:
        verdict = ("FILTER IMPROVES PEAD and the lift is ALPHA-LIKE (beta-hedged Sharpe up, "
                   "β ~flat) — STRONGER lead. Still UNDERPOWERED (resid-α t<2, thin 2y sample, "
                   "single threshold, gate not cleared): confirm with threshold robustness + "
                   "more data before any deploy.")
    elif improves:
        verdict = ("FILTER improves headline Sharpe but the lift is NOT clearly alpha "
                   "(beta-hedged Sharpe flat) — likely beta; do not deploy.")
    else:
        verdict = "NO IMPROVEMENT (drop filter)"
    print(f"  VERDICT: {verdict}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
