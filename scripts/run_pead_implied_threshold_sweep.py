"""
OPT-5 threshold robustness: is the implied-move filter's PEAD lift robust across thresholds,
or an artifact of the single 1.0 threshold?

#423 found the filter (skip PEAD entries whose realized announce-day move was within the
pre-earnings IMPLIED move, i.e. realized/implied < threshold) improves PEAD with an ALPHA-like
lift — but at a SINGLE threshold (1.0) on a thin 2y/8-fold sample, so it could be a
multiplicity/overfit artifact. This sweeps the baseline (no filter) ONCE and the filter at
{0.75, 1.0, 1.25} on the SAME options-covered window, then judges:

  ROBUST  -> the lift (mean-Sharpe AND beta-hedged/alpha Sharpe) is positive and of similar
             magnitude across all three thresholds (a plateau) — consistent with a real effect.
  FRAGILE -> the lift exists only at 1.0 and vanishes/flips at 0.75 or 1.25 (a spike) —
             consistent with threshold overfit; do not pursue without more data.

Same gate/lens as #421/#423: judged on PEAD's OWN CPCV (no options execution). A null is an
honest result. NOT a deploy decision — robustness is necessary, not sufficient (significance
still needs power → Phase 4 more-data).

Usage:
    PYTHONIOENCODING=utf-8 python scripts/run_pead_implied_threshold_sweep.py
"""
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [pead_thr_sweep] %(message)s")
logger = logging.getLogger(__name__)

CPCV_K = 8
CPCV_PATHS = 2
TOTAL_YEARS = 2          # match the 2y R1K options backfill window
THRESHOLDS = [0.75, 1.0, 1.25]


def _run(label, with_filter, ratio):
    """One PEAD CPCV arm. ratio only used when with_filter=True."""
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.run_pead_cpcv import PEADStrategy, build_pead_scorer

    scorer = build_pead_scorer()
    if with_filter:
        from app.data.options_signal import ImpliedMoveProvider
        scorer.implied_move_fn = ImpliedMoveProvider().implied_move
        scorer.min_move_vs_implied = ratio

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
        # CPCVResult exposes the beta-hedged Sharpe as `residual_sharpe` (cpcv.py:84).
        # It is None when SPY align < 30 obs -> g() returns NaN, handled in the verdict.
        "ra_sharpe": g("residual_sharpe"),
        "pf": g("avg_profit_factor"),
        "pos": g("pct_positive"),
    }


def _fin(x):
    """True iff x is a finite, non-NaN number (rejects NaN AND +/-inf)."""
    return isinstance(x, (int, float)) and math.isfinite(x)


def classify_robustness(base, rows, thresholds):
    """Pure robustness classifier — no I/O, so it is unit-testable.

    base / rows[thr] are _m() metric dicts. Returns a dict with the verdict label, the
    human-readable text, the per-threshold deltas, the mean-Sharpe spread, and the
    availability flags.

    PRIMARY criterion = mean-Sharpe plateau: the lift must be present at ALL thresholds
    (Δ > 0.10) with a tight spread (<= 0.40). A lift only at 1.0 (with 0.75/1.25 flat or
    negative) is a single-threshold spike -> overfit-suspect. SECONDARY confirmation = the
    beta-hedged (alpha) Sharpe lift, WHEN computable; if SPY-align < 30 obs the hedged
    Sharpe is None -> NaN here, and we say so explicitly rather than silently failing.
    """
    d_sharpes = [rows[t]["sharpe"] - base["sharpe"] for t in thresholds]
    d_hedged = [rows[t]["ra_sharpe"] - base["ra_sharpe"] for t in thresholds]

    hedged_ok = _fin(base["ra_sharpe"]) and all(_fin(rows[t]["ra_sharpe"]) for t in thresholds)
    mean_ok = _fin(base["sharpe"]) and all(_fin(d) for d in d_sharpes)

    finite_ds = [d for d in d_sharpes if _fin(d)]
    spread = (max(finite_ds) - min(finite_ds)) if finite_ds else float("nan")
    mean_lifts = mean_ok and all(ds > 0.10 for ds in d_sharpes)
    mean_plateau = mean_lifts and spread <= 0.40
    hedged_lifts = hedged_ok and all(dh > 0.0 for dh in d_hedged)

    if not mean_ok:
        label = "INCONCLUSIVE"
        verdict = ("INCONCLUSIVE — mean-Sharpe deltas are non-finite (a CPCV metric failed to "
                   "populate). Do NOT read robustness into this; investigate the result object "
                   "before re-running.")
    elif mean_plateau and hedged_lifts:
        label = "ROBUST"
        verdict = ("ROBUST — the lift (mean-Sharpe + beta-hedged Sharpe) is positive at ALL "
                   f"thresholds with a tight spread ({spread:.3f}); consistent with a real "
                   "effect, not a single-threshold artifact. Still UNDERPOWERED (2y/8-fold) -> "
                   "Phase 4 more-data for significance before any deploy.")
    elif mean_plateau and not hedged_ok:
        label = "ROBUST_MEAN_ONLY"
        verdict = ("ROBUST ON MEAN-SHARPE; HEDGED UNAVAILABLE — mean-Sharpe lift is positive at "
                   f"ALL thresholds with a tight spread ({spread:.3f}), but the beta-hedged alpha "
                   "Sharpe could not be computed (SPY-align < 30 obs), so the lift may be partly "
                   "market beta. Directional lead only; needs Phase-4 data + a hedged re-check.")
    elif mean_lifts and (hedged_lifts or not hedged_ok):
        label = "DIRECTIONAL"
        note = "" if hedged_ok else " (hedged unavailable)"
        verdict = (f"DIRECTIONALLY ROBUST but spread-wide ({spread:.3f}){note} — mean-Sharpe "
                   "improves at all thresholds but magnitude varies; soft lead, prioritize data.")
    else:
        label = "FRAGILE"
        improving = [thr for thr, ds, dh in zip(thresholds, d_sharpes, d_hedged)
                     if _fin(ds) and ds > 0.10 and (not hedged_ok or (_fin(dh) and dh > 0.0))]
        note = "" if hedged_ok else " [hedged Sharpe unavailable — judged on mean-Sharpe only]"
        verdict = (f"FRAGILE — the lift is NOT robust across thresholds (only {improving} "
                   f"improve{note}). Consistent with threshold overfit; do NOT pursue the "
                   "filter without more data / a pre-registered threshold.")

    return {"label": label, "verdict": verdict, "spread": spread,
            "d_sharpes": d_sharpes, "d_hedged": d_hedged,
            "mean_ok": mean_ok, "hedged_ok": hedged_ok}


def main() -> int:
    base = _m(_run("baseline", with_filter=False, ratio=None))
    rows = {}
    for thr in THRESHOLDS:
        rows[thr] = _m(_run(f"filter@{thr}", with_filter=True, ratio=thr))

    cls = classify_robustness(base, rows, THRESHOLDS)
    d_sharpes, d_hedged = cls["d_sharpes"], cls["d_hedged"]

    def pct(x):
        return x * 100 if isinstance(x, (int, float)) and x == x else float("nan")

    print("\n" + "=" * 92)
    print("  OPT-5 THRESHOLD ROBUSTNESS — implied-move filter vs PEAD baseline (2y R1K, k=8/p=2)")
    print("=" * 92)
    hdr = (f"  {'arm':<16}{'mean_shrp':>11}{'path_t':>8}{'resid_a_t':>11}"
           f"{'hedged_shrp':>13}{'PF':>7}{'%pos':>7}")
    print(hdr)
    print(f"  {'baseline':<16}{base['sharpe']:>11.3f}{base['t']:>8.2f}{base['ra_t']:>11.2f}"
          f"{base['ra_sharpe']:>13.3f}{base['pf']:>7.2f}{pct(base['pos']):>6.1f}%")
    for thr in THRESHOLDS:
        m = rows[thr]
        print(f"  {('filter@'+str(thr)):<16}{m['sharpe']:>11.3f}{m['t']:>8.2f}{m['ra_t']:>11.2f}"
              f"{m['ra_sharpe']:>13.3f}{m['pf']:>7.2f}{pct(m['pos']):>6.1f}%")
    print("-" * 92)
    print(f"  {'Δ vs baseline':<16}{'Δmean_shrp':>11}{'':>8}{'':>11}{'Δhedged':>13}{'ΔPF':>7}")
    for thr, ds, dh in zip(THRESHOLDS, d_sharpes, d_hedged):
        print(f"  {('@'+str(thr)):<16}{ds:>+11.3f}{'':>8}{'':>11}{dh:>+13.3f}"
              f"{rows[thr]['pf']-base['pf']:>+7.2f}")
    print("-" * 92)
    print(f"  VERDICT: {cls['verdict']}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
