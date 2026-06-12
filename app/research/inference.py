"""
inference.py — the Ruler-v2 PURE inference core (Alpha-v7 Phase B, Phase 1).

The statistics that replace the broken CPCV path-Sharpe t-stat (which measured
cross-fold consistency, not significance — see docs/reference/RULER_V2_DESIGN.md).
Everything here is PURE: numpy/scipy in, frozen dataclasses out, no I/O, no config
reads. The Ruler-v2 orchestrator (app/research/ruler_v2.py, Phase 2) composes these
into the paper/capital verdicts.

  hac_sharpe              — autocorrelation-robust significance of SR>0 via the
                            Newey-West HAC t-test of the MEAN return (Lo 2002's idea
                            applied to the mean; testing SR>0 ≡ testing mean>0 since
                            vol>0). This is the GATING significance instrument. It is
                            NOT Lo's full Sharpe-estimator SE — it omits the
                            (1 + ½·SR²)/T term from estimating σ, which makes it
                            anti-conservative by ≤1% at daily SR ≤ 0.2 (negligible at
                            realistic daily Sharpe). `se_sr_ann_implied` is the SR SE
                            BACK-DERIVED from this t (= sr_ann/t), not an independent
                            Lo SE — do not combine it as if it were one.
  stationary_bootstrap_sr — Politis-Romano (1994) stationary-bootstrap P(SR>0) + CI
                            (the robustness twin; block length is data-driven).
  pbo_cscv                — Bailey/Lopez de Prado (2017) Probability of Backtest
                            Overfitting via CSCV (needs an N_configs x N_blocks
                            performance matrix; non-gating / undefined for one config).
  multifactor_alpha       — the canonical multi-factor Newey-West residual-alpha
                            (moved here from app/research/options_xs_ls.py, which now
                            re-exports it; generalizes attribution.capm_alpha).

CONTRACT (all functions): below MIN_INFERENCE_OBS observations, or on a degenerate
(zero-variance) series, return the point estimate where computable plus
`gating=False` and a `reason` — they NEVER raise and NEVER fabricate a t-stat
(mirrors capm_alpha's n<30 zero-fill discipline). `gating` says whether the result
is trustworthy enough to feed a CAPITAL-tier decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── frozen module defaults (the SSOT copies live in retrain_config at Phase 2) ──
ANN = 252                      # trading days / year
MIN_INFERENCE_OBS = 60         # below this, no gating verdict (point estimate only)
HAC_SR_LAG = 10                # Bartlett-kernel lag for the Sharpe HAC SE (OD-1)
BOOTSTRAP_REPS = 2000          # stationary-bootstrap resamples
EPS = 1e-12                    # zero-variance floor (mirrors book_gate ZERO_VARIANCE_STD)


# ═════════════════════════════════════════════════════════════════════════════
# 1. HAC Sharpe-ratio significance (Lo 2002)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HACSharpeResult:
    n: int
    sr_daily: float
    sr_ann: float
    # back-derived from t (= sr_ann/t); NOT an independent Lo SE (see module
    # docstring); inf if t==0.
    se_sr_ann_implied: float
    t_stat: float             # mean / NW-HAC SE(mean) — the SR>0 significance test
    p_one_sided: float        # P(SR <= 0); small = significant positive SR
    hac_lag: int              # the EFFECTIVE lag used (min(HAC_SR_LAG, n-1))
    gating: bool
    reason: str = ""


def _nw_longrun_var_of_mean(u: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett-kernel) long-run variance of the sample mean from the
    demeaned series `u` (length n): gamma_0 + 2*sum_{k=1..lag} (1-k/(lag+1))*gamma_k,
    gamma_k = (1/n) sum u_t u_{t-k}. Bartlett guarantees a non-negative estimate."""
    n = len(u)
    g0 = float(u @ u) / n
    lrv = g0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gk = float(u[k:] @ u[:-k]) / n
        lrv += 2.0 * w * gk
    return max(lrv, 0.0)


def hac_sharpe(returns: Sequence[float], *, hac_lag: int = HAC_SR_LAG,
               annualize: int = ANN) -> HACSharpeResult:
    """Autocorrelation-robust significance of SR>0 on a daily OOS return series.

    The point Sharpe is mean/std (annualized by sqrt(annualize)). The SIGNIFICANCE
    is the Newey-West HAC t-test of the mean return (mean / HAC-SE(mean)); since
    vol>0, "SR>0" is exactly "mean>0", so this is the right autocorrelation-robust
    test. This applies Lo (2002)'s autocorrelation idea to the mean — it is the
    gating instrument, but NOT Lo's full SR-estimator SE (it omits the (1+½SR²)/T
    σ-estimation term; anti-conservative by ≤1% at daily SR≤0.2 — negligible).
    Bartlett kernel, effective lag = min(hac_lag, n-1)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < MIN_INFERENCE_OBS:
        return HACSharpeResult(n, 0.0, 0.0, float("inf"), 0.0, 0.5, 0,
                               gating=False, reason="too few obs")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= EPS:
        return HACSharpeResult(n, 0.0, 0.0, float("inf"), 0.0, 0.5, 0,
                               gating=False, reason="degenerate vol")
    sr_daily = mu / sd
    sr_ann = sr_daily * np.sqrt(annualize)
    lag = int(min(hac_lag, n - 1))
    lrv = _nw_longrun_var_of_mean(r - mu, lag)
    se_mu = np.sqrt(lrv / n)
    t = mu / se_mu if se_mu > 0 else 0.0
    p = float(norm.sf(t))                       # one-sided: P(SR<=0)
    se_implied = abs(sr_ann / t) if t != 0 else float("inf")
    return HACSharpeResult(n, float(sr_daily), float(sr_ann), float(se_implied),
                           float(t), p, lag, gating=True)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Stationary bootstrap (Politis-Romano 1994)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BootstrapSRResult:
    n: int
    sr_ann_point: float
    p_sr_gt_0: float          # fraction of resampled annualized SR > 0
    ci_low_95: float
    ci_high_95: float
    mean_block_len: float
    n_reps: int
    gating: bool
    reason: str = ""


def _auto_block_len(r: np.ndarray) -> float:
    """Data-driven mean block length = the INTEGRATED AUTOCORRELATION TIME
    tau = 1 + 2*sum_k rho_k, summing consecutive autocorrelations until the first
    one inside the +-2/sqrt(n) white-noise band (Sokal's window). A near-IID series
    -> tau~1 (block ~1); a persistent series -> larger blocks that grow with the
    dependence (not just the first-crossing LAG, which under-estimates slow decay).
    Floored at 1, capped at n//4; ceil(n^(1/3)) fallback on a degenerate series."""
    n = len(r)
    u = r - r.mean()
    denom = float(u @ u)
    if denom <= EPS:
        return max(1.0, np.ceil(n ** (1.0 / 3.0)))
    band = 2.0 / np.sqrt(n)
    max_lag = min(n - 1, max(1, n // 4))
    tau = 1.0
    for k in range(1, max_lag + 1):
        rho = float(u[k:] @ u[:-k]) / denom
        if abs(rho) < band:
            break
        tau += 2.0 * rho
    blk = int(round(tau))
    return float(min(max(1, blk), max_lag))


def _stationary_bootstrap_indices(n: int, p: float, rng) -> np.ndarray:
    """One Politis-Romano stationary-bootstrap index path of length n: start at a
    random index; each step with prob p jump to a fresh random index, else advance
    circularly (geometric block lengths with mean 1/p)."""
    idx = np.empty(n, dtype=np.int64)
    cur = int(rng.integers(0, n))
    jumps = rng.random(n) < p
    for i in range(n):
        idx[i] = cur
        cur = int(rng.integers(0, n)) if jumps[i] else (cur + 1) % n
    return idx


def stationary_bootstrap_sr(returns: Sequence[float], *,
                            n_reps: int = BOOTSTRAP_REPS,
                            mean_block_len: Optional[float] = None,
                            annualize: int = ANN,
                            seed: int = 0) -> BootstrapSRResult:
    """One-sided P(SR>0) + 95% CI for the annualized Sharpe via the stationary
    bootstrap. Block length is data-driven (_auto_block_len) unless supplied.
    Seeded -> reproducible. Resamples whose draw is degenerate-vol contribute SR=0."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < MIN_INFERENCE_OBS:
        return BootstrapSRResult(n, 0.0, 0.5, 0.0, 0.0, 0.0, 0,
                                 gating=False, reason="too few obs")
    sd0 = float(r.std(ddof=1))
    if sd0 <= EPS:
        return BootstrapSRResult(n, 0.0, 0.5, 0.0, 0.0, 0.0, 0,
                                 gating=False, reason="degenerate vol")
    blk = float(mean_block_len) if mean_block_len is not None else _auto_block_len(r)
    blk = max(1.0, blk)
    p = 1.0 / blk
    rng = np.random.default_rng(seed)
    sqrt_ann = np.sqrt(annualize)
    sims = np.empty(n_reps, dtype=float)
    for b in range(n_reps):
        s = r[_stationary_bootstrap_indices(n, p, rng)]
        sd = s.std(ddof=1)
        sims[b] = (s.mean() / sd) * sqrt_ann if sd > EPS else 0.0
    sr_point = float(r.mean() / sd0 * sqrt_ann)
    p_gt0 = float(np.mean(sims > 0.0))
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return BootstrapSRResult(n, sr_point, p_gt0, float(lo), float(hi), blk,
                             int(n_reps), gating=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3. PBO — Probability of Backtest Overfitting (Bailey / Lopez de Prado 2017, CSCV)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PBOResult:
    pbo: float                 # P(IS-best config is below the OOS median); nan if M<2
    n_configs: int
    n_splits: int
    prob_oos_loss: float       # P(OOS performance < 0 | IS-selected)
    logits: List[float] = field(default_factory=list)
    gating: bool = False
    reason: str = ""


# Above this many C(S, S/2) partitions, randomly sample instead of enumerating.
_PBO_MAX_ENUM_SPLITS = 20000


def pbo_cscv(is_oos_perf, *, n_splits: Optional[int] = None,
             seed: int = 0) -> PBOResult:
    """CSCV Probability of Backtest Overfitting from an N_configs x N_blocks matrix
    of PER-BLOCK out-of-sample performance (e.g. each block's Sharpe or mean return).

    For every way of splitting the S time-blocks into equal IS/OOS halves: pick the
    IS-best config, take its OOS relative rank omega in (0,1), accumulate the logit
    log(omega/(1-omega)). PBO = fraction of splits where the IS-best lands below the
    OOS median (logit < 0); prob_oos_loss = fraction where its OOS performance < 0.

    ⚠️ LEAKAGE: this CANNOT detect a caller that produced `is_oos_perf` from
    configs fit on the FULL sample — the matrix MUST be genuine per-block OOS
    performance (each block scored by a model that did not see it). M<2 configs ->
    undefined (no selection); odd S -> the last block is dropped."""
    from itertools import combinations

    P = np.asarray(is_oos_perf, dtype=float)
    if P.ndim != 2:
        return PBOResult(float("nan"), 0, 0, 0.0, gating=False,
                         reason="is_oos_perf must be 2-D (configs x blocks)")
    M, S = P.shape
    if not np.isfinite(P).all():
        # A NaN/inf cell (e.g. a block with no trades) would silently corrupt the
        # mean/argmax/rank into a finite, confidently-wrong gating PBO. Fail closed.
        return PBOResult(float("nan"), M, 0, 0.0, gating=False,
                         reason="non-finite performance cell")
    if M < 2:
        return PBOResult(float("nan"), M, 0, 0.0, gating=False,
                         reason="single config — PBO not applicable (no selection)")
    if S % 2 == 1:             # CSCV needs an even number of symmetric blocks
        P = P[:, : S - 1]
        S -= 1
    if S < 2:
        return PBOResult(float("nan"), M, 0, 0.0, gating=False,
                         reason="too few time blocks")
    half = S // 2
    blocks = list(range(S))

    # Enumerate all symmetric splits, or sample if there are too many.
    from math import comb
    total = comb(S, half)
    if n_splits is not None or total > _PBO_MAX_ENUM_SPLITS:
        target = int(n_splits) if n_splits is not None else _PBO_MAX_ENUM_SPLITS
        rng = np.random.default_rng(seed)
        splits = (tuple(np.sort(rng.choice(S, half, replace=False)))
                  for _ in range(target))
    else:
        splits = combinations(blocks, half)

    logits: List[float] = []
    oos_losses: List[bool] = []
    eps = 1.0 / (M + 1.0)       # rank-based; keep omega strictly inside (0,1)
    for is_idx in splits:
        is_set = set(is_idx)
        oos_idx = [j for j in blocks if j not in is_set]
        is_perf = P[:, list(is_idx)].mean(axis=1)
        oos_perf = P[:, oos_idx].mean(axis=1)
        n_star = int(np.argmax(is_perf))
        # OOS relative rank of the IS-best config (1..M), as omega in (0,1). Use a
        # tie-fair MID-rank: strictly-worse count + (ties+1)/2. (A plain `<=` would
        # award the IS-best the most favorable rank among ties, biasing PBO DOWN —
        # the optimistic/unsafe direction for a promotion gate; equals `<=`-rank
        # when there are no ties.)
        better = int((oos_perf < oos_perf[n_star]).sum())
        ties = int((oos_perf == oos_perf[n_star]).sum())     # >=1 (incl. itself)
        rank = better + (ties + 1) / 2.0
        omega = min(max(rank / (M + 1.0), eps), 1.0 - eps)
        logits.append(float(np.log(omega / (1.0 - omega))))
        oos_losses.append(bool(oos_perf[n_star] < 0.0))
    arr = np.asarray(logits)
    pbo = float(np.mean(arr < 0.0))
    return PBOResult(pbo, int(M), int(len(logits)),
                     float(np.mean(oos_losses)), logits, gating=True)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Canonical multi-factor Newey-West residual-alpha
#    (moved here from options_xs_ls.py, which now re-exports it)
# ═════════════════════════════════════════════════════════════════════════════

def multifactor_alpha(r_book: pd.Series, factors: pd.DataFrame,
                      hac_lag: int = 5) -> dict:
    """OLS r_book = alpha + B·factors with Newey-West HAC alpha t-stat — the
    multi-factor generalization of attribution.capm_alpha (same Bartlett-kernel
    sandwich). The Ruler-v2 CANONICAL residual-alpha (a premia book needs
    multi-factor residualization, not SPY-only). Aligns book + factors on common
    dates, drops NaNs; zero-filled dict if < 30 aligned obs or a collinear design.

    Returns: n, alpha_ann, alpha_bps_d, t_alpha_ols, t_alpha_hac, betas (dict),
    resid_sharpe (factor-hedged), r2, raw_sharpe."""
    f = factors.dropna()
    common = r_book.dropna().index.intersection(f.index)
    y = r_book.reindex(common).to_numpy(dtype=float)
    F = f.reindex(common).to_numpy(dtype=float)
    names = list(f.columns)
    if F.ndim == 1:
        F = F[:, None]
    # finite-mask BOTH sides — pandas .dropna() does NOT drop ±inf, and an inf
    # factor makes np.linalg.lstsq raise (SVD non-convergence), which would break
    # the module's "never raise / fail to the zero-fill" contract.
    if y.size and F.size:
        keep = np.isfinite(y) & np.isfinite(F).all(axis=1)
        y, F = y[keep], F[keep]
    n = len(y)
    k = F.shape[1] if F.ndim == 2 else 0
    _zero = {"n": n, "alpha_ann": 0.0, "alpha_bps_d": 0.0, "t_alpha_ols": 0.0,
             "t_alpha_hac": 0.0, "betas": {}, "resid_sharpe": 0.0, "r2": 0.0,
             "raw_sharpe": 0.0}
    if n < 30 or k == 0:
        return _zero
    X = np.column_stack([np.ones(n), F])  # [1, f1..fk]
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        # Degenerate/collinear factor design — the regression is ill-posed.
        return _zero
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_d = float(coef[0])
    betas = {names[j]: float(coef[1 + j]) for j in range(k)}
    resid = y - X @ coef
    dof = n - (k + 1)
    s2 = float(resid @ resid) / dof if dof > 0 else 0.0
    se_ols = float(np.sqrt(max(s2 * XtX_inv[0, 0], 0.0)))
    t_ols = alpha_d / se_ols if se_ols > 0 else 0.0

    # Newey-West HAC sandwich on the full [1, F] design.
    Xr = X * resid[:, None]
    Smat = Xr.T @ Xr
    for L in range(1, hac_lag + 1):
        w = 1.0 - L / (hac_lag + 1.0)
        G = Xr[L:].T @ Xr[:-L]
        Smat += w * (G + G.T)
    cov_hac = XtX_inv @ Smat @ XtX_inv
    se_hac = float(np.sqrt(max(cov_hac[0, 0], 0.0)))
    t_hac = alpha_d / se_hac if se_hac > 0 else 0.0

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    raw_sharpe = float(y.mean() / y.std() * np.sqrt(ANN)) if y.std() > 0 else 0.0
    hedged = y - (X[:, 1:] @ coef[1:])  # alpha + resid = factor-hedged stream
    resid_sharpe = (float(hedged.mean() / hedged.std() * np.sqrt(ANN))
                    if hedged.std() > 0 else 0.0)
    return {"n": n, "alpha_ann": alpha_d * ANN, "alpha_bps_d": alpha_d * 1e4,
            "t_alpha_ols": t_ols, "t_alpha_hac": t_hac, "betas": betas,
            "resid_sharpe": resid_sharpe, "r2": r2, "raw_sharpe": raw_sharpe}
