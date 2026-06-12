"""
bayes_sr.py — the Ruler-v2 Bayesian posterior on the annualized Sharpe ratio
(Alpha-v7 Phase B). PURE: scalars in, a frozen dataclass out, no I/O, no config reads.

This is the multiplicity defense that REPLACES the saturated Deflated Sharpe Ratio.
The DSR collapsed because N_TRIALS_TESTED hit 300 and the haircut stopped moving;
here the registry's TRUE trial count tightens a mean-zero prior, and the backtest +
(optional) live-paper Sharpes enter as precision-weighted normal observations. The
closed-form posterior gives P(SR>0) directly.

Model (conjugate normal-normal on the annualized SR):
  prior    SR ~ N(0, τ²),  τ = prior_sd / √(1 + log(max(N_trials, 1)))
  backtest sr_bt ~ N(SR, σ_bt²)               (σ_bt = HAC SE of the annualized SR)
  live     sr_lp ~ N(SR, σ_lp²)               (optional; realized paper P&L)
  ⇒ posterior precision  Λ = 1/τ² + 1/σ_bt² [+ 1/σ_lp²]
    posterior mean        μ = (0/τ² + sr_bt/σ_bt² [+ sr_lp/σ_lp²]) / Λ
    posterior var         1/Λ
    P(SR>0)               = Φ(μ · √Λ)

An UNINFORMATIVE observation (σ = ∞ / nan / ≤0, e.g. the HAC t≈0 case where
se_sr_ann_implied is inf) contributes ZERO precision — it neither helps nor hurts,
and with only the prior the posterior is N(0, τ²) ⇒ P(SR>0)=0.5. Never raises.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm

EPS = 1e-12


@dataclass(frozen=True)
class PosteriorSRResult:
    posterior_mean: float       # E[SR | prior, backtest, live]
    posterior_sd: float         # sqrt(posterior variance)
    p_sr_gt_0: float            # P(SR > 0 | ·) — the CAPITAL gate reads this
    prior_sd: float             # τ, AFTER the trial-count shrinkage
    n_trials: int               # the registry trial count that set τ
    used_backtest: bool         # backtest observation contributed precision
    used_live: bool             # live-paper observation contributed precision
    gating: bool                # False when NO informative observation entered
    reason: str = ""


def _precision(se: Optional[float]) -> float:
    """1/σ² for a finite positive σ, else 0 (an uninformative observation)."""
    if se is None:
        return 0.0
    try:
        s = float(se)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(s) or s <= EPS:
        return 0.0
    return 1.0 / (s * s)


def posterior_sr(sr_backtest: float, se_backtest: Optional[float], *,
                 n_trials: int, prior_sd: float,
                 sr_live: Optional[float] = None,
                 se_live: Optional[float] = None) -> PosteriorSRResult:
    """Closed-form Bayesian posterior P(SR>0) on the annualized Sharpe.

    sr_backtest / se_backtest — the OOS point SR and its HAC SE (from
    inference.hac_sharpe: sr_ann + se_sr_ann_implied). se_backtest may be inf/None
    (HAC t≈0) ⇒ the backtest is uninformative and only the prior + any live obs set
    the posterior. n_trials is the registry's TRUE trial count (NOT the saturated
    DSR 300); it shrinks the prior toward zero. sr_live/se_live optionally fold in
    realized live-paper performance the same precision-weighted way.

    Fails to gating=False (P=0.5, posterior=prior) when NO observation is informative
    or prior_sd is degenerate — never raises, never fabricates significance.
    """
    tau0 = float(prior_sd)
    nt = max(int(n_trials), 1)
    if not math.isfinite(tau0) or tau0 <= EPS:
        return PosteriorSRResult(0.0, 0.0, 0.5, 0.0, nt, False, False,
                                 gating=False, reason="degenerate prior_sd")
    # Multiplicity shrinkage: more trials → tighter mean-zero prior.
    tau = tau0 / math.sqrt(1.0 + math.log(nt))
    prior_prec = 1.0 / (tau * tau)

    prec_bt = _precision(se_backtest)
    prec_lp = _precision(se_live)
    used_bt = prec_bt > 0.0 and sr_backtest is not None and math.isfinite(float(sr_backtest))
    used_lp = prec_lp > 0.0 and sr_live is not None and math.isfinite(float(sr_live))

    lam = prior_prec + (prec_bt if used_bt else 0.0) + (prec_lp if used_lp else 0.0)
    num = 0.0  # prior mean is 0 → contributes 0 to the numerator
    if used_bt:
        num += float(sr_backtest) * prec_bt
    if used_lp:
        num += float(sr_live) * prec_lp
    mu = num / lam
    var = 1.0 / lam
    sd = math.sqrt(var)
    p = float(norm.cdf(mu / sd)) if sd > 0 else (1.0 if mu > 0 else 0.0)

    gating = used_bt or used_lp   # the prior alone is not a verdict
    reason = "" if gating else "no informative observation (prior only)"
    return PosteriorSRResult(float(mu), float(sd), p, float(tau), nt,
                             used_bt, used_lp, gating=gating, reason=reason)
