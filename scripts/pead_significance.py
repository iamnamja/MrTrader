"""
ALPHA v2 §1.3 — PEAD honest significance harness.

THE LAST PHASE-1 DE-RISK GATE. PEAD's validated long-only edge is reported at CPCV
mean Sharpe +0.546 with a PATH t-stat = 2.26 (N_eff = n_folds = 8). For an EVENT
strategy that t-stat is OPTIMISTIC: earnings cluster into ~4 quarterly seasons/year,
~40-trading-day holds overlap heavily, and within-season cross-sectional returns are
correlated — so the true independent-information count is well below 8 and the real
significance is weaker than t=2.26. §1.3 produces significance estimates whose UNIT OF
INDEPENDENCE is the EARNINGS EVENT (clustered by calendar quarter), not the CPCV fold
or the trading day, to confirm (or temper) the edge before any capital decision. The
§1.1 cost analysis already concluded "statistical significance, not cost, is now PEAD's
binding constraint."

This harness MIRRORS scripts/pead_crisis_robustness.py exactly:
  * load PEAD data ONCE (re-fetch dominates runtime),
  * run the committed C(8,2) CPCV ONCE on BYTE-IDENTICAL data + folds, deterministic +
    anchored to retrain_as_of(), CAPTURING per-fold trades + daily equity curves via the
    PURE-ADDITIVE _last_trades / _last_equity_curve side-channels (no re-simulation),
  * a BASELINE self-validation that the captured realized full-sample daily Sharpe and
    the per-fold path reconstruction reconcile with the known +0.546 (logged LOUDLY on
    divergence),
  * ASCII-safe output, artifacts (JSON/CSV) written FIRST then a guarded table print.

────────────────────────────────────────────────────────────────────────────────────
WHAT IS BOOTSTRAPPED / HAC'd, AND HOW IT RECONCILES WITH +0.546 / t=2.26
────────────────────────────────────────────────────────────────────────────────────
There are THREE distinct lenses on the SAME edge; we do NOT conflate them:

  (i)   CPCV PATH construct (the existing anchor) — mean over C(8,2) path-Sharpes
        = +0.546, path t-stat = mean / (std / sqrt(N_eff=8)) = 2.26. This is a
        property of the CROSS-VALIDATION (path dispersion across fold combos), NOT a
        property of one realized return stream. We KEEP and REPORT it unchanged as the
        conservative anchor (decision (c), N_eff = n_folds floor).

  (ii)  Realized TRADE-return stream — every closed trade across all captured folds,
        each tagged with its earnings-proximate ENTRY date + symbol. The observed
        statistic is the per-trade Sharpe  mean(pnl_pct) / std(pnl_pct)  (a PER-EVENT
        statistic, NOT annualized — annualization needs a fixed period; trades do not
        have one). This is what the EVENT-CLUSTERED BLOCK BOOTSTRAP resamples
        (analysis (a)).

  (iii) Realized DAILY portfolio return stream — the union of the per-fold daily equity
        curves differenced into daily returns (folds are disjoint test windows in a CPCV
        path, so the daily series is a real out-of-sample portfolio return series). The
        observed statistic is the full-sample ANNUALIZED (252) daily Sharpe. This is
        what the NEWEY-WEST / HAC t-stat penalizes for the ~40-day overlapping-hold
        autocorrelation (analysis (b)).

Reconciliation: (ii) and (iii) are the REALIZED stream; (i) is the CPCV path construct.
They are different denominators on the same alpha and need NOT be numerically equal. The
self-validation we DO assert is that recomputing each CPCV path's Sharpe from the captured
per-fold daily curves and averaging reproduces run_cpcv's own mean Sharpe (~+0.546) to a
tight tolerance — i.e. the capture is faithful to the committed run. We report all three
side by side so the reader sees the honest-event significance (a)/(b) AGAINST the optimistic
path t (i) and can temper accordingly.

────────────────────────────────────────────────────────────────────────────────────
THE FOUR SUB-ANALYSES (all on the one shared capture)
────────────────────────────────────────────────────────────────────────────────────
(a) EVENT-CLUSTERED BLOCK BOOTSTRAP — cluster trades by EARNINGS SEASON = calendar
    quarter of the trade's entry date (the unit of independence: a quarter's worth of
    overlapping, cross-sectionally-correlated earnings names). Resample WHOLE clusters
    WITH replacement (a block/stationary bootstrap over event-clusters — resampling the
    whole cluster preserves within-quarter correlation AND the overlapping-hold
    structure). Recompute the trade-Sharpe per resample -> sampling distribution ->
    report a one-sided bootstrap p-value (H0: Sharpe <= 0) AND a percentile CI on the
    Sharpe. N resamples = 10000 (full) / 200 (smoke), fixed seed.

(b) NEWEY-WEST / HAC t-stat — on the daily return series, with a Bartlett-kernel lag
    L = 60 trading days (>= the ~40-day overlapping hold, rounded up to absorb the full
    autocorrelation tail; see HAC_LAG). statsmodels is NOT a dependency, so we hand-roll
    the Bartlett HAC (the same estimator OLS-on-a-constant uses) and the unit test pins
    it to a hand-checked reference. The HAC t < the naive OLS t on positively
    autocorrelated returns — that gap IS the overlap penalty.

(c) N_eff = n_folds FLOOR — keep/report the existing CPCV path t-stat (~2.26) as the
    conservative anchor for comparison.

(d) TREND-REGIME STRATIFICATION (§1.2 carry-over) — tag each trade by SPY >< its 200d
    SMA (PIT: using only SPY closes <= the trade's entry date) at entry. Report the
    trade-Sharpe + event-clustered bootstrap p WITHIN up-trend and down-trend separately,
    and the % of trades / % of total P&L from each. Goal: confirm the edge is not driven
    by a few correlated up-trend clusters, and characterize the down-trend behavior
    (§1.2 showed PEAD bleeds in down-trends).

Usage:
    # Full run (re-fetches ~1000 symbols / 6yr, runs the full CPCV once — LONG):
    python scripts/pead_significance.py

    # Fast smoke (tiny universe + short window + tiny CPCV + few resamples; for tests/CI):
    python scripts/pead_significance.py --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.ml.retrain_config import MAX_THREADS as _max_threads

os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))

logger = logging.getLogger("pead_significance")

# ── Determinism + analysis constants ────────────────────────────────────────────────
BOOTSTRAP_SEED = 1303          # fixed seed -> same resample draws every run
N_RESAMPLES = 10000            # full-run bootstrap resamples
SMOKE_N_RESAMPLES = 200        # smoke-run resamples (fast, still exercises the path)
CI_ALPHA = 0.05                # 95% percentile CI (2.5%..97.5%)

# (b) Newey-West / HAC lag. The PEAD hold is ~40 trading days (DEFAULT_MAX_HOLD_BARS=40);
# overlapping holds induce autocorrelation out to ~40 lags. We round up to 60 to absorb
# the full tail (a longer lag is conservative — it can only WIDEN the HAC std / SHRINK
# the t). Choosing exactly the hold length would under-penalize the tail.
HAC_LAG = 60

# Baseline self-validation: recomputing each CPCV path's Sharpe from the captured per-fold
# daily curves and averaging MUST reproduce run_cpcv's own mean Sharpe (~+0.546). A miss of
# more than this proves the capture diverged from the committed run (LOUD). Same floor logic
# as the §1.2 LOCO self-check: run_cpcv stores each fold sharpe round(sharpe,3) while we
# recompute unrounded from the same curves, so per-path means can differ by ~5e-4 from 3dp
# rounding alone; 1e-3 stays far below any real grouping divergence.
SELF_CHECK_TOL = 1e-3
BASELINE_EXPECTED_SHARPE = 0.546   # committed +0.546 path-mean (sanity log only)

LOG_DIR = ROOT / "logs"
ARTIFACT_DIR = ROOT / "logs"


def _setup_logging(log_path: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [pead_sig] %(message)s",
        handlers=handlers,
        force=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════════
# SEPARATELY-TESTABLE CORE FUNCTIONS (unit tests hit these with synthetic inputs)
# ═══════════════════════════════════════════════════════════════════════════════════

def trade_sharpe(returns: List[float]) -> float:
    """Per-trade (per-EVENT) Sharpe = mean / sample-std (ddof=1). NOT annualized.

    This is the cross-sectional-return statistic the event-clustered bootstrap resamples.
    Annualizing would require a fixed period that trade returns do not have; the bootstrap
    only needs a monotone-in-edge statistic, and mean/std is the natural one. Returns 0.0
    for < 2 returns or zero dispersion.
    """
    import numpy as np
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(arr.mean() / sd)


def cluster_key(d: date) -> str:
    """Earnings-SEASON cluster key = calendar quarter of the trade's (event) date.

    Calendar quarter is the defensible cluster unit: US large-caps report in ~4 quarterly
    seasons (roughly Jan-Feb, Apr-May, Jul-Aug, Oct-Nov), so all names that report in the
    same quarter overlap in time and share macro/sector co-movement -> they are NOT
    independent. The quarter cleanly buckets the ~40-day overlapping holds into one block.
    """
    d = d.date() if hasattr(d, "date") else d
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def event_clustered_bootstrap(
    trade_returns: List[float],
    trade_clusters: List[str],
    n_resamples: int,
    seed: int,
    ci_alpha: float = CI_ALPHA,
) -> Dict:
    """Block bootstrap over EVENT CLUSTERS (resample whole clusters WITH replacement).

    Mechanism (a stationary/block bootstrap whose blocks are the earnings-season clusters):
      1. group trade returns by cluster key,
      2. draw len(clusters) clusters WITH replacement,
      3. concatenate the drawn clusters' returns and recompute trade_sharpe,
      4. repeat n_resamples times -> the sampling distribution of the Sharpe.

    Resampling WHOLE clusters (not individual trades) preserves the within-cluster
    correlation and the overlapping-hold structure, so the resulting CI/p respect the
    event-level dependence the naive iid bootstrap ignores.

    Reported:
      observed   : trade_sharpe on the full sample.
      p_value    : one-sided bootstrap p for H0: Sharpe <= 0, by the TEXTBOOK RETURN-SHIFT
                   null. We impose H0 at the DATA level: subtract the observed MEAN RETURN
                   from every trade return so the population has exactly zero mean (Sharpe 0),
                   then resample WHOLE CLUSTERS with replacement FROM THAT shifted population
                   and recompute the Sharpe per resample -> the null sampling distribution of
                   the statistic under H0. p = fraction of null-Sharpes >= the observed
                   Sharpe. Under true noise the observed is itself ~0 and lands mid-null ->
                   p~=0.5; under a real edge the observed sits far in the null's right tail ->
                   p small. (This is the standard "shift to satisfy H0, resample the shifted
                   data" bootstrap test, equivalent in spirit to the earlier re-center-the-
                   Sharpe-distribution shortcut but textbook-correct: the null is built from a
                   genuine zero-mean population, not by translating the statistic.)
      ci_low/high: percentile CI at [ci_alpha/2, 1-ci_alpha/2] on the UN-shifted resample
                   distribution (the CI describes the sampling distribution of the ACTUAL
                   statistic, so it must use the real, un-shifted returns).
      n_clusters : number of distinct event clusters (the effective independence count).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    by_cluster: Dict[str, List[float]] = {}
    for r, c in zip(trade_returns, trade_clusters):
        by_cluster.setdefault(c, []).append(float(r))
    clusters = list(by_cluster.keys())
    n_clusters = len(clusters)
    observed = trade_sharpe(trade_returns)
    if n_clusters < 2 or len(trade_returns) < 2:
        return {
            "observed_sharpe": round(observed, 6),
            "p_value": 1.0,
            "ci_low": round(observed, 6),
            "ci_high": round(observed, 6),
            "ci_width": 0.0,
            "n_clusters": n_clusters,
            "n_trades": len(trade_returns),
            "n_resamples": 0,
        }
    # UN-shifted cluster arrays (for the CI) and H0-shifted arrays (for the null): the shift
    # subtracts the GLOBAL observed mean return so the whole shifted population has mean 0.
    grand_mean = float(np.asarray(trade_returns, dtype=float).mean())
    cluster_arrays = [np.asarray(by_cluster[c], dtype=float) for c in clusters]
    cluster_arrays_null = [a - grand_mean for a in cluster_arrays]
    dist = np.empty(n_resamples, dtype=float)       # un-shifted: CI
    null_dist = np.empty(n_resamples, dtype=float)  # H0-shifted: p-value
    idx_space = np.arange(n_clusters)
    for b in range(n_resamples):
        pick = rng.choice(idx_space, size=n_clusters, replace=True)
        dist[b] = trade_sharpe(np.concatenate([cluster_arrays[i] for i in pick]).tolist())
        null_dist[b] = trade_sharpe(
            np.concatenate([cluster_arrays_null[i] for i in pick]).tolist())
    # Textbook return-shift null: p = fraction of zero-mean-population resample Sharpes that
    # are at least as large as the observed Sharpe.
    p_value = float(np.mean(null_dist >= observed))
    ci_low = float(np.percentile(dist, 100 * (ci_alpha / 2)))
    ci_high = float(np.percentile(dist, 100 * (1 - ci_alpha / 2)))
    return {
        "observed_sharpe": round(observed, 6),
        "p_value": round(p_value, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "ci_width": round(ci_high - ci_low, 6),
        "n_clusters": n_clusters,
        "n_trades": len(trade_returns),
        "n_resamples": n_resamples,
    }


def iid_bootstrap_ci(
    trade_returns: List[float],
    n_resamples: int,
    seed: int,
    ci_alpha: float = CI_ALPHA,
) -> Dict:
    """Naive IID (per-trade) bootstrap CI — the OPTIMISTIC comparison baseline.

    Resamples individual trades with replacement (ignores clustering). On autocorrelated /
    clustered input this UNDERSTATES uncertainty, so its CI is NARROWER than the
    event-clustered CI. The harness reports both so the clustering penalty is explicit
    (the unit test asserts clustered_width > iid_width on clustered input).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(trade_returns, dtype=float)
    n = arr.size
    observed = trade_sharpe(trade_returns)
    if n < 2:
        return {"observed_sharpe": round(observed, 6), "ci_low": round(observed, 6),
                "ci_high": round(observed, 6), "ci_width": 0.0, "n_resamples": 0}
    dist = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        pick = rng.integers(0, n, size=n)
        dist[b] = trade_sharpe(arr[pick].tolist())
    ci_low = float(np.percentile(dist, 100 * (ci_alpha / 2)))
    ci_high = float(np.percentile(dist, 100 * (1 - ci_alpha / 2)))
    return {
        "observed_sharpe": round(observed, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "ci_width": round(ci_high - ci_low, 6),
        "n_resamples": n_resamples,
    }


def newey_west_tstat(returns: List[float], lag: int) -> Dict:
    """Newey-West / HAC t-stat for H0: mean daily return == 0, Bartlett kernel, `lag` lags.

    This is the HAC t-stat of the sample mean (equivalently OLS of the return series on a
    constant). statsmodels is not a dependency so we hand-roll the standard Bartlett-kernel
    HAC variance of the mean:

        gamma_0 = (1/T) * sum_t (r_t - rbar)^2
        gamma_k = (1/T) * sum_{t=k+1}^{T} (r_t - rbar)(r_{t-k} - rbar)
        S_hac   = gamma_0 + 2 * sum_{k=1}^{L} (1 - k/(L+1)) * gamma_k     (Bartlett weights)
        var(mean) = S_hac / T
        t_hac     = rbar / sqrt(var(mean))

    We also report the naive OLS t (lag=0, i.e. iid var(mean) = gamma_0_unbiased / T) and an
    ANNUALIZED Sharpe for context. On POSITIVELY autocorrelated returns S_hac > gamma_0, so
    t_hac < t_ols — that shrinkage is the overlapping-hold penalty. p_value is one-sided
    (H0: mean <= 0) from the t distribution (scipy) with T-1 dof; if scipy is missing we
    fall back to the standard-normal survival function.
    """
    import numpy as np
    arr = np.asarray(returns, dtype=float)
    T = arr.size
    if T < 3:
        return {"t_hac": 0.0, "t_ols": 0.0, "p_value_hac": 1.0, "lag": lag,
                "n_obs": int(T), "ann_sharpe": 0.0, "mean": 0.0}
    rbar = float(arr.mean())
    dev = arr - rbar
    gamma0 = float(np.dot(dev, dev) / T)
    if gamma0 <= 1e-18:
        return {"t_hac": 0.0, "t_ols": 0.0, "p_value_hac": 1.0, "lag": lag,
                "n_obs": int(T), "ann_sharpe": 0.0, "mean": rbar}
    L = max(0, min(int(lag), T - 1))
    s_hac = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gamma_k = float(np.dot(dev[k:], dev[:-k]) / T)
        s_hac += 2.0 * w * gamma_k
    # HAC variance of the mean can go non-positive on small/odd samples; floor it.
    s_hac = max(s_hac, 1e-18)
    var_mean_hac = s_hac / T
    t_hac = rbar / math.sqrt(var_mean_hac)
    # Naive OLS t of the mean: unbiased sample variance / T.
    var_mean_ols = float(arr.var(ddof=1)) / T
    t_ols = rbar / math.sqrt(var_mean_ols) if var_mean_ols > 0 else 0.0
    # Annualized daily Sharpe for context (mean/std * sqrt(252)).
    sd = float(arr.std(ddof=1))
    ann_sharpe = (rbar / sd * math.sqrt(252)) if sd > 0 else 0.0
    # One-sided p (H0: mean <= 0).
    try:
        from scipy import stats as _st
        p_hac = float(_st.t.sf(t_hac, df=max(T - 1, 1)))
    except Exception:  # pragma: no cover - scipy is present in this env
        from math import erfc
        p_hac = 0.5 * erfc(t_hac / math.sqrt(2.0))
    return {
        "t_hac": round(t_hac, 4),
        "t_ols": round(t_ols, 4),
        "p_value_hac": round(p_hac, 6),
        "lag": L,
        "n_obs": int(T),
        "ann_sharpe": round(ann_sharpe, 4),
        "mean": rbar,
    }


def spy_trend_at(entry_d: date, spy_closes, ma_window: int = 200) -> Optional[str]:
    """PIT trend regime at `entry_d`: 'up' if SPY close >= its `ma_window`d SMA, else 'down'.

    Uses ONLY SPY closes with date <= entry_d (a future SPY move cannot change a past
    trade's tag — the unit test pins this). Returns None when fewer than `ma_window`
    closes exist on/before entry_d (cannot form the MA -> untagged, excluded from the
    stratified buckets). `spy_closes` is a pandas Series indexed by datetime/date.
    """
    import pandas as pd
    if spy_closes is None or len(spy_closes) == 0:
        return None
    ed = entry_d.date() if hasattr(entry_d, "date") else entry_d
    ts = pd.Timestamp(ed)
    # Strictly PIT: keep only closes dated on/before the entry day.
    prior = spy_closes[spy_closes.index <= ts]
    if len(prior) < ma_window:
        return None
    ma = float(prior.iloc[-ma_window:].mean())
    last = float(prior.iloc[-1])
    return "up" if last >= ma else "down"


# ═══════════════════════════════════════════════════════════════════════════════════
# CAPTURE HARNESS (load once, run the committed CPCV once, capture trades + curves)
# ═══════════════════════════════════════════════════════════════════════════════════

def _make_scorer():
    """Build the committed long-only +0.546 PEAD scorer (VIX>30 block). Identical config
    to scripts/run_pead_cpcv.build_pead_scorer default and pead_crisis_robustness baseline."""
    from app.ml.pead_scorer import PEADScorer
    return PEADScorer(
        long_threshold=0.05,
        short_threshold=-0.05,
        long_short=False,
        vix_block_all=30.0,
        vix_block_short=100.0,
        vix_conf_ref=100.0,
        max_announce_day_move=1.0,
        require_positive_revision=False,
        min_analyst_momentum=0.0,
    )


def _capture_run(base, purge_days, embargo_days, k, paths, years):
    """Run the committed CPCV once, capturing per-fold (trades, equity_curve) keyed by the
    GLOBAL fold id run_cpcv assigns. Mirrors pead_crisis_robustness._loco_capture_run, but
    also captures the trade objects via the §1.3 _last_trades side-channel.

    Returns (CPCVResult, {fold_id: {"trades": [...], "equity": [...]}}).
    """
    from scripts.walkforward.cpcv import run_cpcv
    base.scorer = _make_scorer()
    captured: Dict[int, Dict] = {}
    inner_run_fold = base.run_fold

    def _wrapped(fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        fold = inner_run_fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end)
        trades = getattr(base, "_last_trades", None)
        ec = getattr(base, "_last_equity_curve", None)
        captured[fold_idx] = {
            "trades": list(trades) if trades is not None else [],
            "equity": list(ec) if ec is not None else [],
        }
        return fold

    base.run_fold = _wrapped  # type: ignore[assignment]
    try:
        result = run_cpcv(
            strategy=base, purge_days=purge_days, embargo_days=embargo_days,
            n_folds=k, n_paths=paths, total_years=years,
        )
    finally:
        base.run_fold = inner_run_fold  # type: ignore[assignment]
    return result, captured


def _path_sharpe_from_curves(result, fold_curves) -> Tuple[float, int]:
    """Recompute mean path Sharpe from captured per-fold daily equity curves, grouped by
    run_cpcv's REAL path_fold_members (the single source of truth). Returns (mean, n_paths).

    Used for the baseline self-validation (must reproduce result.mean_sharpe ~+0.546).
    """
    import numpy as np
    from scripts.pead_crisis_robustness import _sharpe_from_equity
    path_sharpes: List[float] = []
    for members in result.path_fold_members:
        fold_sharpes = []
        for fid in members:
            ec = fold_curves.get(fid, {}).get("equity") or []
            if not ec:
                continue
            s, _ = _sharpe_from_equity(ec, None)
            fold_sharpes.append(s)
        if fold_sharpes:
            path_sharpes.append(float(np.mean(fold_sharpes)))
    if not path_sharpes:
        return 0.0, 0
    return float(np.mean(path_sharpes)), len(path_sharpes)


def _pool_unique_trades(result, fold_curves) -> List:
    """Pool the realized trades across the CPCV PATH folds, de-duplicated by
    (symbol, entry_date, exit_date). A trading day / trade appears in multiple CPCV paths
    (combinatorial multiplicity), so we take the UNION of trades over the surviving global
    fold ids (each fold's test window is disjoint) — that union is one out-of-sample trade
    set, counted once. Mirrors the 'unique trading day' correction in CPCVResult.unique_obs.
    """
    seen = set()
    pooled = []
    member_ids = {fid for members in result.path_fold_members for fid in members}
    for fid in sorted(member_ids):
        for t in fold_curves.get(fid, {}).get("trades", []):
            key = (getattr(t, "symbol", None),
                   getattr(t, "entry_date", None),
                   getattr(t, "exit_date", None))
            if key in seen:
                continue
            seen.add(key)
            pooled.append(t)
    return pooled


def _pool_unique_daily_returns(result, fold_curves) -> List[float]:
    """Pool ONE coherent out-of-sample daily return series across the CPCV fold curves.

    CRITICAL DE-DUP BY CALENDAR WINDOW (not global fold id). run_cpcv assigns each fold a
    GLOBAL id  = combo_idx * len(all_boundaries) + ti + 1  that is DISTINCT per combo, but
    the calendar test window [te_start, te_end] depends ONLY on the within-combo index `ti`
    (the same `ti` recurs in many combos). For a rules-based expanding-window strategy like
    PEAD the captured curve for a given `ti` is BYTE-IDENTICAL across every combo it appears
    in. Concatenating by global fold id would therefore repeat each calendar window ~k-1
    times adjacently (e.g. C(8,2): 48 surviving ids -> only 7 distinct ti windows, each ~6-7x)
    -> a ~7x-too-long series with each day repeated -> massive ARTIFICIAL autocorrelation ->
    an INVALID Newey-West / OLS t. We therefore key each captured curve by its distinct
    calendar window (first_date, last_date) and keep ONE representative per window, mirroring
    what _pool_unique_trades does for the trade stream (dedup by content key).

    CPCV test folds ti=1..k are DISJOINT by construction, so the kept windows are disjoint in
    calendar time; concatenating them in chronological order yields one clean NON-overlapping
    OOS daily series (each calendar day exactly once). Each fold's returns are differenced
    from its OWN equity curve only (never across a window boundary).

    Returns the pooled daily return list. The caller (run_analysis) asserts coherence.
    """
    member_ids = {fid for members in result.path_fold_members for fid in members}
    # Dedup by distinct calendar window -> one representative curve per window.
    by_window: Dict[Tuple[date, date], List[Tuple[date, float]]] = {}
    for fid in member_ids:
        ec = fold_curves.get(fid, {}).get("equity") or []
        pts = [(d.date() if hasattr(d, "date") else d, float(v)) for d, v in ec]
        if len(pts) < 2:
            continue
        win = (pts[0][0], pts[-1][0])
        # Keep the first representative seen for each distinct window (they are byte-identical
        # across combos for a rules-based expanding-window strategy; any one is correct).
        by_window.setdefault(win, pts)
    # Chronological order of the disjoint windows by their start date.
    windows = sorted(by_window.keys(), key=lambda w: w[0])
    out: List[float] = []
    for win in windows:
        pts = by_window[win]
        rets = [(pts[i][1] - pts[i - 1][1]) / max(pts[i - 1][1], 1e-9)
                for i in range(1, len(pts))]
        out.extend(rets)
    return out


def _pool_unique_daily_returns_audit(result, fold_curves) -> Dict:
    """Same window de-dup as _pool_unique_daily_returns but returns the per-window structure
    for the loud self-validation in run_analysis: the distinct windows (chronological), the
    set of calendar dates each window contributes (the LATER date of each daily return), and
    the total series length. Used to assert no calendar day appears twice and the series
    length == sum of per-window return counts."""
    member_ids = {fid for members in result.path_fold_members for fid in members}
    by_window: Dict[Tuple[date, date], List[Tuple[date, float]]] = {}
    for fid in member_ids:
        ec = fold_curves.get(fid, {}).get("equity") or []
        pts = [(d.date() if hasattr(d, "date") else d, float(v)) for d, v in ec]
        if len(pts) < 2:
            continue
        win = (pts[0][0], pts[-1][0])
        by_window.setdefault(win, pts)
    windows = sorted(by_window.keys(), key=lambda w: w[0])
    win_dates: List[List[date]] = []
    series_len = 0
    for win in windows:
        pts = by_window[win]
        # Each daily return is dated by its LATER endpoint (the day the return is realized).
        dates = [pts[i][0] for i in range(1, len(pts))]
        win_dates.append(dates)
        series_len += len(dates)
    return {"windows": windows, "win_dates": win_dates, "series_len": series_len,
            "n_distinct_windows": len(windows)}


def validate_daily_series(daily_returns, result, fold_curves) -> Dict:
    """Self-validation for the pooled OOS daily return series (the HAC input).

    Asserts the series is ONE coherent non-overlapping stream:
      * len(daily_returns) == sum of per-distinct-window return counts (no inflation), and
      * no calendar day appears in more than one kept window (windows disjoint), and
      * the union of per-window calendar days has no duplicate (each day exactly once).

    Returns {"ok": bool, "reason": str, "series_len", "n_distinct_windows", "n_unique_days"}.
    This is the loud guard against the fold-overlap corruption: on the byte-identical
    per-ti duplication the old by-fold-id concat produced, series_len would be ~k x the
    distinct-day count and the same calendar days would recur -> ok=False.
    """
    audit = _pool_unique_daily_returns_audit(result, fold_curves)
    series_len = audit["series_len"]
    n_windows = audit["n_distinct_windows"]
    # 1. No length inflation: pooled length must equal the distinct-window return total.
    if len(daily_returns) != series_len:
        return {
            "ok": False,
            "reason": (f"series length {len(daily_returns)} != sum of distinct-window "
                       f"return counts {series_len} (length inflated -> overlap duplication)"),
            "series_len": series_len, "n_distinct_windows": n_windows, "n_unique_days": 0,
        }
    # 2. Every calendar day appears exactly once across the kept windows.
    all_days: List[date] = [d for dates in audit["win_dates"] for d in dates]
    unique_days = set(all_days)
    if len(all_days) != len(unique_days):
        dupes = len(all_days) - len(unique_days)
        return {
            "ok": False,
            "reason": (f"{dupes} calendar day(s) appear in more than one kept window "
                       f"(windows NOT disjoint -> artificial autocorrelation)"),
            "series_len": series_len, "n_distinct_windows": n_windows,
            "n_unique_days": len(unique_days),
        }
    # 3. Length must match the unique-calendar-day count exactly (each day once).
    if series_len != len(unique_days):
        return {
            "ok": False,
            "reason": (f"series length {series_len} != unique calendar-day count "
                       f"{len(unique_days)}"),
            "series_len": series_len, "n_distinct_windows": n_windows,
            "n_unique_days": len(unique_days),
        }
    return {
        "ok": True, "reason": "coherent non-overlapping daily series",
        "series_len": series_len, "n_distinct_windows": n_windows,
        "n_unique_days": len(unique_days),
    }


def _trade_records(trades, spy_closes, ma_window: int = 200) -> List[Dict]:
    """Turn captured Trade objects into significance records: return, event date (entry),
    symbol, cluster (calendar quarter of entry), PIT trend regime at entry."""
    recs: List[Dict] = []
    for t in trades:
        ed = getattr(t, "entry_date", None)
        pnl = getattr(t, "pnl_pct", None)
        if ed is None or pnl is None:
            continue
        ed_d = ed.date() if hasattr(ed, "date") else ed
        recs.append({
            "symbol": getattr(t, "symbol", None),
            "entry_date": ed_d,
            "pnl_pct": float(pnl),
            "pnl": float(getattr(t, "pnl", 0.0) or 0.0),
            "cluster": cluster_key(ed_d),
            "trend": spy_trend_at(ed_d, spy_closes, ma_window),
        })
    return recs


def run_analysis(
    smoke: bool = False,
    symbols: Optional[List[str]] = None,
    total_years: Optional[int] = None,
    cpcv_k: Optional[int] = None,
    cpcv_paths: Optional[int] = None,
    purge_days: int = 10,
    embargo_days: int = 10,
    n_resamples: Optional[int] = None,
    seed: int = BOOTSTRAP_SEED,
    hac_lag: int = HAC_LAG,
) -> Dict:
    """Run the §1.3 significance harness on a SINGLE shared data load + ONE captured CPCV."""
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.run_pead_cpcv import PEADStrategy, CPCV_K, CPCV_PATHS, TOTAL_YEARS

    _years = total_years if total_years is not None else (1 if smoke else TOTAL_YEARS)
    _k = cpcv_k if cpcv_k is not None else (4 if smoke else CPCV_K)
    _paths = cpcv_paths if cpcv_paths is not None else CPCV_PATHS
    _nres = n_resamples if n_resamples is not None else (SMOKE_N_RESAMPLES if smoke else N_RESAMPLES)
    if symbols is None:
        symbols = list(RUSSELL_1000_TICKERS)[:8] if smoke else list(RUSSELL_1000_TICKERS)

    logger.info(
        "PEAD significance: smoke=%s years=%d CPCV C(%d,%d) symbols=%d resamples=%d "
        "seed=%d hac_lag=%d",
        smoke, _years, _k, _paths, len(symbols), _nres, seed, hac_lag,
    )

    base = PEADStrategy(scorer=_make_scorer(), symbols=symbols, transaction_cost_pct=0.0005)
    t0 = time.time()
    try:
        from app.ml.retrain_config import retrain_as_of
        end_all = datetime.combine(retrain_as_of(), datetime.min.time())
    except Exception:
        end_all = datetime.now()
    start_all = end_all - timedelta(days=_years * 365 + 30)
    base.fetch_data(start_all, end_all)
    logger.info("Single shared data load complete in %.1fs", time.time() - t0)

    # ── Run the committed CPCV ONCE, capturing per-fold trades + daily curves ──────────
    result, fold_curves = _capture_run(base, purge_days, embargo_days, _k, _paths, _years)
    real_mean = float(result.mean_sharpe)
    path_tstat = float(result.path_sharpe_tstat)
    logger.info("Captured CPCV: mean Sharpe %+.4f  path t-stat %+.3f (N_eff=%d folds)  "
                "%d surviving paths", real_mean, path_tstat, result.n_folds,
                len(result.path_sharpes))

    spy_closes = getattr(base, "spy_prices", None)

    # ── BASELINE self-validation: path Sharpe from captured curves must == run_cpcv mean ─
    recon_mean, recon_paths = _path_sharpe_from_curves(result, fold_curves)
    self_check_ok = abs(recon_mean - real_mean) <= SELF_CHECK_TOL
    if not self_check_ok:
        logger.error(
            "BASELINE SELF-VALIDATION FAILED: path Sharpe recomputed from captured curves "
            "%+.6f does NOT reproduce run_cpcv mean Sharpe %+.6f (|diff|=%.6f > tol %.0e). "
            "The capture DIVERGES from the committed run; significance estimates may be "
            "computed on the WRONG stream. INVESTIGATE.",
            recon_mean, real_mean, abs(recon_mean - real_mean), SELF_CHECK_TOL,
        )
    else:
        logger.info(
            "BASELINE self-validation OK: recomputed path Sharpe %+.6f reproduces "
            "run_cpcv mean %+.6f (|diff|=%.2e <= tol %.0e); both near committed +%.3f",
            recon_mean, real_mean, abs(recon_mean - real_mean), SELF_CHECK_TOL,
            BASELINE_EXPECTED_SHARPE,
        )

    # ── Pool the realized streams (unique trades + daily returns) ─────────────────────
    pooled_trades = _pool_unique_trades(result, fold_curves)
    daily_returns = _pool_unique_daily_returns(result, fold_curves)

    # ── LOUD self-validation: the daily series must be ONE coherent non-overlapping OOS ──
    # stream — each calendar window kept once, windows disjoint, every calendar day appearing
    # EXACTLY ONCE. This guards against the fold-overlap corruption (see
    # _pool_unique_daily_returns docstring): if it ever returned per-fold-id concatenation
    # again, the length would balloon ~kx and calendar days would repeat -> we abort loudly.
    daily_audit = validate_daily_series(daily_returns, result, fold_curves)
    if not daily_audit["ok"]:
        logger.error(
            "DAILY-SERIES SELF-VALIDATION FAILED: %s. The Newey-West HAC series is CORRUPTED "
            "(likely CPCV fold-overlap duplication); HAC/OLS t-stats would be INVALID. ABORTING.",
            daily_audit["reason"],
        )
        raise AssertionError(
            "PEAD daily-series self-validation failed: " + daily_audit["reason"]
        )
    logger.info(
        "DAILY-SERIES self-validation OK: %d distinct calendar windows -> %d daily obs, each "
        "calendar day exactly once, windows disjoint and chronological.",
        daily_audit["n_distinct_windows"], daily_audit["series_len"],
    )

    recs = _trade_records(pooled_trades, spy_closes)
    trade_returns = [r["pnl_pct"] for r in recs]
    trade_clusters = [r["cluster"] for r in recs]
    logger.info("Realized stream: %d unique trades across %d clusters, %d daily-return obs",
                len(trade_returns), len(set(trade_clusters)), len(daily_returns))

    # ── (a) Event-clustered block bootstrap (+ iid comparison) ────────────────────────
    boot = event_clustered_bootstrap(trade_returns, trade_clusters, _nres, seed)
    iid = iid_bootstrap_ci(trade_returns, _nres, seed)
    boot["iid_ci_width"] = iid.get("ci_width", 0.0)
    boot["clustering_widens_ci"] = (boot.get("ci_width", 0.0) >= iid.get("ci_width", 0.0))

    # ── (b) Newey-West / HAC t-stat on the daily return series ────────────────────────
    nw = newey_west_tstat(daily_returns, hac_lag)

    # ── (d) Trend-regime stratification ───────────────────────────────────────────────
    total_pnl = sum(r["pnl"] for r in recs) or 0.0
    strat_rows: List[Dict] = []
    for regime in ("up", "down"):
        sub = [r for r in recs if r["trend"] == regime]
        sub_ret = [r["pnl_pct"] for r in sub]
        sub_clu = [r["cluster"] for r in sub]
        sub_pnl = sum(r["pnl"] for r in sub)
        sub_boot = event_clustered_bootstrap(sub_ret, sub_clu, _nres, seed)
        strat_rows.append({
            "regime": regime,
            "n_trades": len(sub),
            "pct_trades": round(len(sub) / max(len(recs), 1), 4),
            "pct_pnl": round(sub_pnl / total_pnl, 4) if abs(total_pnl) > 1e-9 else 0.0,
            "trade_sharpe": sub_boot["observed_sharpe"],
            "bootstrap_p": sub_boot["p_value"],
            "ci_low": sub_boot["ci_low"],
            "ci_high": sub_boot["ci_high"],
            "n_clusters": sub_boot["n_clusters"],
        })
    n_untagged = sum(1 for r in recs if r["trend"] is None)

    out: Dict = {
        "smoke": smoke,
        "config": {
            "total_years": _years, "cpcv_k": _k, "cpcv_paths": _paths,
            "purge_days": purge_days, "embargo_days": embargo_days,
            "n_symbols": len(symbols), "n_resamples": _nres, "seed": seed,
            "hac_lag": hac_lag, "cluster_def": "calendar_quarter",
        },
        "baseline_self_validation": {
            "real_cpcv_mean_sharpe": round(real_mean, 6),
            "recomputed_path_sharpe": round(recon_mean, 6),
            "n_recomputed_paths": recon_paths,
            "self_check_ok": self_check_ok,
            "self_check_tol": SELF_CHECK_TOL,
            "baseline_expected_sharpe": BASELINE_EXPECTED_SHARPE,
        },
        "anchor_cpcv_path_tstat": {  # (c) N_eff = n_folds floor
            "mean_path_sharpe": round(real_mean, 6),
            "path_tstat": round(path_tstat, 4),
            "n_eff_folds": result.n_folds,
            "n_paths": len(result.path_sharpes),
            "note": ("CONSERVATIVE ANCHOR: CPCV path t-stat with N_eff=n_folds (NOT n_paths). "
                     "This is the CROSS-VALIDATION construct, distinct from the realized-stream "
                     "bootstrap/HAC below — reported side by side, not conflated."),
        },
        "event_clustered_bootstrap": boot,         # (a)
        "iid_bootstrap": iid,                       # optimistic comparison
        "newey_west_hac": nw,                       # (b)
        "trend_stratification": {                   # (d)
            "rows": strat_rows,
            "n_untagged": n_untagged,
            "ma_window": 200,
            "note": ("PIT trend tag: SPY close >< its 200d SMA at trade entry (only SPY "
                     "closes <= entry). %pnl shows edge concentration; down-trend behavior "
                     "characterizes the §1.2 bleed."),
        },
        "interpretation": (
            "Event-clustered bootstrap p (H0: Sharpe<=0) and the Newey-West HAC t are the "
            "HONEST event-level / overlap-aware significance; compare them to the CPCV path "
            "t-stat anchor (N_eff=n_folds). If the clustered p stays small and the HAC t "
            "stays meaningfully positive, the edge survives honest dependence accounting; if "
            "they collapse toward p~0.5 / t~0 while the path t looked strong, the +0.546/2.26 "
            "was optimistic and the edge should be TEMPERED before capital."
        ),
    }
    return out


# ═══════════════════════════════════════════════════════════════════════════════════
# REPORTING + ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════════

def _print_report(out: Dict) -> None:
    smoke = "   [SMOKE]" if out.get("smoke") else ""
    print()
    print("=" * 100)
    print("PEAD HONEST SIGNIFICANCE HARNESS  (Alpha v2 1.3)" + smoke)
    print("=" * 100)

    bsv = out["baseline_self_validation"]
    print("\nBASELINE SELF-VALIDATION  (capture reproduces the committed +0.546 run)")
    print("-" * 100)
    print(f"  run_cpcv mean Sharpe:        {bsv['real_cpcv_mean_sharpe']:+.4f}")
    print(f"  recomputed from curves:      {bsv['recomputed_path_sharpe']:+.4f}  "
          f"({bsv['n_recomputed_paths']} paths)")
    print(f"  self-check: {'OK (reproduces)' if bsv['self_check_ok'] else 'DIVERGES - INVESTIGATE'} "
          f"(tol {bsv['self_check_tol']:.0e}; expected ~+{bsv['baseline_expected_sharpe']:.3f})")

    a = out["anchor_cpcv_path_tstat"]
    print("\n(c) CPCV PATH t-stat ANCHOR  (N_eff = n_folds; the conservative comparison)")
    print("-" * 100)
    print(f"  mean path Sharpe {a['mean_path_sharpe']:+.4f}   path t-stat {a['path_tstat']:+.3f}   "
          f"N_eff={a['n_eff_folds']} folds (NOT {a['n_paths']} paths)")

    b = out["event_clustered_bootstrap"]
    iid = out["iid_bootstrap"]
    print("\n(a) EVENT-CLUSTERED BLOCK BOOTSTRAP  (cluster = calendar quarter; resample clusters)")
    print("-" * 100)
    print(f"  observed trade-Sharpe:       {b['observed_sharpe']:+.4f}   "
          f"({b['n_trades']} trades, {b['n_clusters']} event clusters)")
    print(f"  bootstrap p (H0 Sharpe<=0):  {b['p_value']:.4f}   "
          f"resamples={b['n_resamples']}")
    print(f"  95% CI (clustered):          [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  "
          f"width={b.get('ci_width', 0.0):.4f}")
    print(f"  95% CI (iid, optimistic):    [{iid.get('ci_low', 0):+.4f}, {iid.get('ci_high', 0):+.4f}]  "
          f"width={iid.get('ci_width', 0.0):.4f}")
    print(f"  clustering widens CI:        {'YES (honest)' if b.get('clustering_widens_ci') else 'NO'}")

    nw = out["newey_west_hac"]
    print("\n(b) NEWEY-WEST / HAC t-stat on the daily return series")
    print("-" * 100)
    print(f"  daily obs {nw['n_obs']}   ann Sharpe {nw['ann_sharpe']:+.4f}   "
          f"Bartlett lag L={nw['lag']}")
    print(f"  naive OLS t:   {nw['t_ols']:+.3f}")
    print(f"  HAC  t:        {nw['t_hac']:+.3f}   one-sided p={nw['p_value_hac']:.4f}   "
          f"(HAC < OLS = overlap penalty)")

    st = out["trend_stratification"]
    print("\n(d) TREND-REGIME STRATIFICATION  (SPY >< 200d SMA at entry, PIT)")
    print("-" * 100)
    print(f"{'regime':>8} {'nTrades':>8} {'%trades':>8} {'%pnl':>8} {'tradeSharpe':>12} "
          f"{'bootP':>8} {'CI_low':>9} {'CI_high':>9} {'clusters':>9}")
    for r in st["rows"]:
        print(f"{r['regime']:>8} {r['n_trades']:>8d} {r['pct_trades']*100:>7.1f}% "
              f"{r['pct_pnl']*100:>7.1f}% {r['trade_sharpe']:>+12.4f} {r['bootstrap_p']:>8.4f} "
              f"{r['ci_low']:>+9.4f} {r['ci_high']:>+9.4f} {r['n_clusters']:>9d}")
    if st.get("n_untagged"):
        print(f"  ({st['n_untagged']} trades untagged: < 200 SPY closes before entry)")
    print()
    print(f"  {out['interpretation']}")
    print("=" * 100)
    print()


def _write_artifacts(out: Dict, stamp: str) -> Dict[str, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACT_DIR / f"pead_significance_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    paths = {"json": json_path}
    # Flat summary CSV (one row) for the scalar headline stats.
    b = out["event_clustered_bootstrap"]
    nw = out["newey_west_hac"]
    a = out["anchor_cpcv_path_tstat"]
    summary_path = ARTIFACT_DIR / f"pead_significance_summary_{stamp}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["cpcv_mean_path_sharpe", a["mean_path_sharpe"]])
        w.writerow(["cpcv_path_tstat_Neff_folds", a["path_tstat"]])
        w.writerow(["observed_trade_sharpe", b["observed_sharpe"]])
        w.writerow(["n_trades", b["n_trades"]])
        w.writerow(["n_event_clusters", b["n_clusters"]])
        w.writerow(["bootstrap_p_value", b["p_value"]])
        w.writerow(["bootstrap_ci_low", b["ci_low"]])
        w.writerow(["bootstrap_ci_high", b["ci_high"]])
        w.writerow(["clustered_ci_width", b.get("ci_width", 0.0)])
        w.writerow(["iid_ci_width", b.get("iid_ci_width", 0.0)])
        w.writerow(["hac_t_stat", nw["t_hac"]])
        w.writerow(["ols_t_stat", nw["t_ols"]])
        w.writerow(["hac_p_value", nw["p_value_hac"]])
        w.writerow(["hac_lag", nw["lag"]])
        w.writerow(["daily_ann_sharpe", nw["ann_sharpe"]])
    paths["summary"] = summary_path
    # Per-regime stratification CSV.
    strat_path = ARTIFACT_DIR / f"pead_significance_trend_{stamp}.csv"
    fields = ["regime", "n_trades", "pct_trades", "pct_pnl", "trade_sharpe",
              "bootstrap_p", "ci_low", "ci_high", "n_clusters"]
    with open(strat_path, "w", newline="", encoding="utf-8") as f:
        ww = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        ww.writeheader()
        for r in out["trend_stratification"]["rows"]:
            ww.writerow(r)
    paths["trend"] = strat_path
    logger.info("Wrote artifacts: %s", ", ".join(str(p) for p in paths.values()))
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PEAD honest significance harness (Alpha v2 1.3)")
    p.add_argument("--smoke", action="store_true",
                   help="fast smoke: tiny universe + short window + tiny CPCV + few resamples")
    args = p.parse_args(argv)

    try:
        if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_smoke" if args.smoke else "")
    log_path = LOG_DIR / f"pead_significance_{stamp}.log"
    _setup_logging(log_path)

    out = run_analysis(smoke=args.smoke)

    # Artifacts FIRST (durable), then a guarded table print (console-encoding safe).
    paths = _write_artifacts(out, stamp)
    out["artifacts"] = {k: str(v) for k, v in paths.items()}
    try:
        _print_report(out)
    except Exception as exc:  # pragma: no cover - defensive console guard
        logger.warning("Report print failed (%s: %s); artifacts already written to %s",
                       type(exc).__name__, exc, paths)

    logger.info("PEAD significance complete. Log: %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
