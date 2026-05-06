"""
bootstrap_sharpe.py — Phase 1d: Bootstrap walk-forward to quantify selection bias.

Resamples fold boundaries N times (perturbing split dates by ±jitter days) and
runs the full walk-forward on each resample. Builds a distribution of Sharpe
ratios to assess whether the reported champion Sharpe is in the lucky tail.

Also computes the Deflated Sharpe Ratio (DSR) correcting for the number of
model variants tried (Bailey & López de Prado 2014).

Usage:
    python scripts/bootstrap_sharpe.py --model swing --n-resamples 200
    python scripts/bootstrap_sharpe.py --model intraday --n-resamples 200
    python scripts/bootstrap_sharpe.py --model swing --n-resamples 50 --jitter 20

Exit codes:
    0 — original Sharpe is NOT in top 10% of bootstrap distribution (not lucky)
    1 — original Sharpe IS in top 10% (selection bias likely), or error
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import norm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute the Deflated Sharpe Ratio (DSR) from Bailey & López de Prado (2014).

    DSR answers: what is the probability that the observed Sharpe is above zero
    after correcting for the number of trials tested?

    Returns (DSR_value, p_value) where p_value = P(true Sharpe > 0 | selection bias).
    """
    if n_trials <= 1 or n_obs <= 1:
        return observed_sharpe, 0.5

    # Expected maximum Sharpe from n_trials independent trials (eq. 10 in paper)
    euler_mascheroni = 0.5772156649
    sr_star = (
        (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * math.e))
    )

    # Variance of Sharpe estimator (eq. 7)
    sr_var = (
        1
        + (observed_sharpe ** 2) * (1 - skewness * observed_sharpe)
        + ((excess_kurtosis - 1) / 4) * observed_sharpe ** 2
    ) / (n_obs - 1)

    if sr_var <= 0:
        return observed_sharpe, 0.5

    # DSR (eq. 11)
    dsr_z = (observed_sharpe - sr_star) / math.sqrt(sr_var)
    p_value = norm.cdf(dsr_z)

    return dsr_z, p_value


def _print_header(msg: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {msg}")
    print(f"{'=' * 62}")


def _print_sub(msg: str) -> None:
    print(f"\n  {msg}")


def run_bootstrap_swing(
    n_resamples: int = 200,
    jitter_days: int = 30,
    n_folds: int = 3,
    total_years: int = 5,
    swing_cost_bps: float = 5.0,
    swing_purge_days: int = 10,
    n_trials_tested: int = 15,
    model_version: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Run bootstrap walk-forward for swing model."""
    from scripts.walkforward_tier3 import run_swing_walkforward

    rng = random.Random(seed)

    _print_header(f"Bootstrap Walk-Forward — SWING (n={n_resamples} resamples, ±{jitter_days}d jitter)")

    # First: run the canonical walk-forward (no jitter) to get the baseline
    print("\n  Running canonical walk-forward (no jitter)...")
    t0 = time.time()
    canonical = run_swing_walkforward(
        n_folds=n_folds,
        total_years=total_years,
        transaction_cost_pct=swing_cost_bps / 10_000 / 2,
        purge_days=swing_purge_days,
        model_version=model_version,
    )
    canonical_sharpe = canonical.avg_sharpe
    canonical_elapsed = time.time() - t0
    print(f"  Canonical avg Sharpe: {canonical_sharpe:.3f}  ({canonical_elapsed:.0f}s)")

    # Bootstrap resamples: perturb the effective total_years slightly
    bootstrap_sharpes: List[float] = []

    def _one_resample(i: int) -> float:
        jitter = rng.randint(-jitter_days, jitter_days)
        effective_years = max(2, total_years) + jitter / 365.0
        try:
            result = run_swing_walkforward(
                n_folds=n_folds,
                total_years=effective_years,
                transaction_cost_pct=swing_cost_bps / 10_000 / 2,
                purge_days=swing_purge_days,
                model_version=model_version,
            )
            return result.avg_sharpe
        except Exception as exc:
            logger.warning("Resample %d failed: %s", i, exc)
            return float("nan")

    print(f"  Running {n_resamples} resamples (this will take a while)...")
    completed = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_one_resample, i): i for i in range(n_resamples)}
        for fut in as_completed(futures):
            sharpe = fut.result()
            if not math.isnan(sharpe):
                bootstrap_sharpes.append(sharpe)
            completed += 1
            if completed % 10 == 0:
                so_far = [s for s in bootstrap_sharpes if not math.isnan(s)]
                print(f"  Progress: {completed}/{n_resamples}  median so far: {np.median(so_far):.3f}")

    return _summarize_bootstrap(
        model="swing",
        canonical_sharpe=canonical_sharpe,
        bootstrap_sharpes=bootstrap_sharpes,
        n_trials_tested=n_trials_tested,
        n_obs=sum(f.trades for f in canonical.folds),
    )


def run_bootstrap_intraday(
    n_resamples: int = 200,
    jitter_days: int = 30,
    n_folds: int = 3,
    total_days: int = 730,
    intraday_cost_bps: float = 15.0,
    intraday_purge_days: int = 2,
    n_trials_tested: int = 15,
    model_version: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Run bootstrap walk-forward for intraday model."""
    from scripts.walkforward_tier3 import run_intraday_walkforward

    rng = random.Random(seed)

    _print_header(f"Bootstrap Walk-Forward — INTRADAY (n={n_resamples} resamples, ±{jitter_days}d jitter)")

    print("\n  Running canonical walk-forward (no jitter)...")
    t0 = time.time()
    canonical = run_intraday_walkforward(
        n_folds=n_folds,
        total_days=total_days,
        transaction_cost_pct=intraday_cost_bps / 10_000 / 2,
        purge_days=intraday_purge_days,
        model_version=model_version,
    )
    canonical_sharpe = canonical.avg_sharpe
    canonical_elapsed = time.time() - t0
    print(f"  Canonical avg Sharpe: {canonical_sharpe:.3f}  ({canonical_elapsed:.0f}s)")

    bootstrap_sharpes: List[float] = []

    def _one_resample(i: int) -> float:
        jitter = rng.randint(-jitter_days, jitter_days)
        effective_days = max(180, total_days + jitter)
        try:
            result = run_intraday_walkforward(
                n_folds=n_folds,
                total_days=effective_days,
                transaction_cost_pct=intraday_cost_bps / 10_000 / 2,
                purge_days=intraday_purge_days,
                model_version=model_version,
            )
            return result.avg_sharpe
        except Exception as exc:
            logger.warning("Resample %d failed: %s", i, exc)
            return float("nan")

    print(f"  Running {n_resamples} resamples (this will take a while)...")
    completed = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_one_resample, i): i for i in range(n_resamples)}
        for fut in as_completed(futures):
            sharpe = fut.result()
            if not math.isnan(sharpe):
                bootstrap_sharpes.append(sharpe)
            completed += 1
            if completed % 10 == 0:
                so_far = [s for s in bootstrap_sharpes if not math.isnan(s)]
                print(f"  Progress: {completed}/{n_resamples}  median so far: {np.median(so_far):.3f}")

    return _summarize_bootstrap(
        model="intraday",
        canonical_sharpe=canonical_sharpe,
        bootstrap_sharpes=bootstrap_sharpes,
        n_trials_tested=n_trials_tested,
        n_obs=sum(f.trades for f in canonical.folds),
    )


def _summarize_bootstrap(
    model: str,
    canonical_sharpe: float,
    bootstrap_sharpes: List[float],
    n_trials_tested: int,
    n_obs: int,
) -> dict:
    arr = np.array(bootstrap_sharpes)
    valid = arr[~np.isnan(arr)]

    if len(valid) == 0:
        print("  ERROR: No valid bootstrap resamples completed.")
        return {}

    median = float(np.median(valid))
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    p10 = float(np.percentile(valid, 10))
    p25 = float(np.percentile(valid, 25))
    p75 = float(np.percentile(valid, 75))
    p90 = float(np.percentile(valid, 90))

    # What percentile is the canonical result in?
    pct_rank = float(np.mean(valid <= canonical_sharpe)) * 100

    # DSR
    skewness = float(
        np.mean(((valid - mean) / std) ** 3) if std > 0 else 0.0
    )
    excess_kurtosis = float(
        np.mean(((valid - mean) / std) ** 4) - 3 if std > 0 else 0.0
    )
    dsr_z, dsr_p = _deflated_sharpe_ratio(
        observed_sharpe=canonical_sharpe,
        n_trials=n_trials_tested,
        n_obs=n_obs,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )

    selection_bias_flag = pct_rank >= 90.0

    _print_header(f"Bootstrap Results — {model.upper()}")
    print(f"\n  Canonical Sharpe:    {canonical_sharpe:+.3f}")
    print(f"  Bootstrap median:    {median:+.3f}")
    print(f"  Bootstrap mean:      {mean:+.3f}")
    print(f"  Bootstrap std:       {std:.3f}")
    print(f"  10th–90th pct:       [{p10:+.3f}, {p90:+.3f}]")
    print(f"  25th–75th pct:       [{p25:+.3f}, {p75:+.3f}]")
    print(f"  Canonical pct rank:  {pct_rank:.1f}th percentile")
    print(f"  N resamples valid:   {len(valid)}/{len(bootstrap_sharpes)}")
    print()
    print(f"  Deflated Sharpe Ratio (DSR):")
    print(f"    N trials tested: {n_trials_tested}")
    print(f"    DSR z-score:     {dsr_z:+.3f}")
    print(f"    P(SR > 0 | bias):{dsr_p:.3f}  {'✅ significant' if dsr_p > 0.95 else '❌ not significant'}")
    print()
    if selection_bias_flag:
        print(f"  ⚠️  SELECTION BIAS WARNING: canonical Sharpe is in top {100-pct_rank:.0f}% of bootstrap distribution.")
        print(f"     Reported result likely inflated by lucky fold boundaries.")
    else:
        print(f"  ✅ Canonical Sharpe is at the {pct_rank:.0f}th percentile — not in lucky tail.")
    print()
    print(f"  Histogram (rough):")
    bins = np.linspace(min(valid.min(), canonical_sharpe) - 0.1,
                       max(valid.max(), canonical_sharpe) + 0.1, 12)
    hist, edges = np.histogram(valid, bins=bins)
    max_bar = max(hist) or 1
    for count, left, right in zip(hist, edges[:-1], edges[1:]):
        bar = "█" * int(count / max_bar * 30)
        marker = " ← canonical" if left <= canonical_sharpe < right else ""
        print(f"    [{left:+.2f},{right:+.2f}] {bar}{marker}")

    return {
        "model": model,
        "canonical_sharpe": canonical_sharpe,
        "bootstrap_median": median,
        "bootstrap_mean": mean,
        "bootstrap_std": std,
        "p10": p10,
        "p90": p90,
        "canonical_pct_rank": pct_rank,
        "dsr_z": dsr_z,
        "dsr_p": dsr_p,
        "selection_bias_flag": selection_bias_flag,
        "n_valid": len(valid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1d: Bootstrap walk-forward + DSR")
    parser.add_argument("--model", choices=["swing", "intraday", "both"], default="both")
    parser.add_argument("--n-resamples", type=int, default=200,
                        help="Number of bootstrap resamples (default: 200)")
    parser.add_argument("--jitter", type=int, default=30,
                        help="Max fold boundary jitter in calendar days (default: 30)")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--years", type=int, default=5, help="Swing history years")
    parser.add_argument("--days", type=int, default=730, help="Intraday history days")
    parser.add_argument("--swing-cost-bps", type=float, default=5.0)
    parser.add_argument("--intraday-cost-bps", type=float, default=15.0)
    parser.add_argument("--swing-purge-days", type=int, default=10)
    parser.add_argument("--intraday-purge-days", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=15,
                        help="Number of model variants tested historically (for DSR; default: 15)")
    parser.add_argument("--swing-model-version", type=int, default=0)
    parser.add_argument("--intraday-model-version", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    swing_ver = args.swing_model_version or None
    intraday_ver = args.intraday_model_version or None

    results = {}
    selection_bias_found = False

    if args.model in ("swing", "both"):
        r = run_bootstrap_swing(
            n_resamples=args.n_resamples,
            jitter_days=args.jitter,
            n_folds=args.folds,
            total_years=args.years,
            swing_cost_bps=args.swing_cost_bps,
            swing_purge_days=args.swing_purge_days,
            n_trials_tested=args.n_trials,
            model_version=swing_ver,
            seed=args.seed,
        )
        results["swing"] = r
        if r.get("selection_bias_flag"):
            selection_bias_found = True

    if args.model in ("intraday", "both"):
        r = run_bootstrap_intraday(
            n_resamples=args.n_resamples,
            jitter_days=args.jitter,
            n_folds=args.folds,
            total_days=args.days,
            intraday_cost_bps=args.intraday_cost_bps,
            intraday_purge_days=args.intraday_purge_days,
            n_trials_tested=args.n_trials,
            model_version=intraday_ver,
            seed=args.seed,
        )
        results["intraday"] = r
        if r.get("selection_bias_flag"):
            selection_bias_found = True

    return 1 if selection_bias_found else 0


if __name__ == "__main__":
    sys.exit(main())
