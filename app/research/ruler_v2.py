"""
ruler_v2.py — the Ruler-v2 gate ORCHESTRATOR (Alpha-v7 Phase B, Phase 2).

Composes the pure inference core (app/research/inference.py) + the Bayesian
posterior (app/research/bayes_sr.py) + the Ruler-v2 config constants into the
two-tier PAPER/CAPITAL verdict, returned as a {key: (observed, ok)} detail dict in
the EXACT shape CPCVResult.significance_gate_detail uses — so the existing
`all(ok …)` reducer in CPCVResult.gate_passed works unchanged.

SHIPS DARK: reached only when GATE_MODE="ruler_v2" (not the default). The
significance / mean_sharpe branches never call into this module, so the legacy
test corpus + recorded kill ledger are byte-for-byte unaffected.

Tier philosophy (the deliberate INVERSION of the legacy gate):
  PAPER   = PLAUSIBILITY — no significance t. A pre-registered, economically-
            motivated sleeve with a believable (not implausibly high) point SR and a
            non-catastrophic worst regime earns a paper slot. PF/Calmar are DEMOTED
            to report-only (off the AND) — they were the legacy Type-II machine.
  CAPITAL = SIGNIFICANCE — REQUIRES a live-paper observation (the posterior is
            P(SR>0 | backtest AND live paper); a backtest alone can NEVER reach
            capital — structural, not threshold-dependent) AND Bayesian posterior
            P(SR>0) ≥ threshold (the multiplicity defense that replaces the saturated
            DSR) AND multi-factor residual-α t_HAC ≥ floor (primary) AND stationary-
            bootstrap P(SR>0) ≥ threshold AND (PBO ≤ max, when >1 configs) AND a hard
            POWER FLOOR (n_obs, n_folds) that fails closed regardless of the posterior.
            Through the CPCVResult.gate_detail/gate_passed dispatch no live_paper is
            supplied, so the CPCV entry point evaluates CAPITAL as backtest-only and
            therefore always FAILS CLOSED there — live-paper-informed CAPITAL is an
            explicit call into evaluate()/gate_passed() with live_paper=… (Phase D).

Inputs the orchestrator reads off the result (duck-typed CPCVResult): the dated OOS
book return series (`oos_returns_dated`, the Phase-2 additive field), `mean_sharpe`,
`path_sharpes`, `n_folds`, `worst_regime_sharpe`, `regime_insufficient_obs`, and the
already-computed `residual_alpha_t_hac` diagnostic. NOTHING here mutates the result
except the informational `requires_human_review_flag` (mirrors the legacy waiver).

NOTE (dark-phase honesty): two design criteria are REPORT-ONLY here because their
inputs aren't on a single CPCVResult — the 2× cost-stress survival (needs a re-run)
and PBO (needs an N_configs×N_blocks matrix). They appear in the detail as
informational keys and are wired to gate in a later phase; they are NOT silently
passed inside the boolean AND.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.research import inference as inf
from app.research import bayes_sr

# Detail keys that are REPORT-ONLY — excluded from the gate's boolean AND (they show
# up in the reporter as context / flags, never silently flip a pass to a fail).
INFORMATIONAL_KEYS = frozenset({
    "requires_human_review",
    "pf_report", "calmar_report",
    "hac_t_report", "point_sr_report", "bootstrap_psr_report",
    "cost_stress_report",   # needs a 2× cost re-run — gates in a later phase
    "pbo_report",           # needs an N_configs×N_blocks matrix — non-gating for M=1
})


def capital_factor_set(component_type: str, available_factors) -> List[str]:
    """OD-9: the factors the CAPITAL residual-α should residualize against for a sleeve
    of `component_type`, given the `available_factors` columns — every base factor
    EXCEPT the premium this sleeve is paid to harvest. PRINCIPLE: a trend sleeve must
    NOT be residualized against the trend (TSMOM) factor, or you hedge out the very
    edge you're certifying; a declared diversifier / pure-alpha sleeve residualizes
    against ALL available base factors. The harvested-premium map lives in
    retrain_config.RULERV2_HARVESTED_FACTOR (owner-ratified)."""
    from app.ml.retrain_config import RULERV2_HARVESTED_FACTOR
    harvested = RULERV2_HARVESTED_FACTOR.get((component_type or "").lower(), None)
    avail = list(available_factors)
    return [f for f in avail if f != harvested]


def residual_alpha_t(book_returns: pd.Series, factor_panel: pd.DataFrame, *,
                     component_type: str, hac_lag: int = 5) -> float:
    """OD-9 CAPITAL residual-α t_HAC: multi-factor NW-HAC alpha of `book_returns` on
    the harvested-premium-EXCLUDED factor set (per capital_factor_set). The premia-book
    generalization of the SPY-only `capm_alpha` diagnostic — wire this into run_cpcv's
    residual-alpha computation to populate CPCVResult.residual_alpha_t_hac under a
    premia book. Returns 0.0 (the fail-closed value) when no usable factor remains."""
    cols = [c for c in capital_factor_set(component_type, list(factor_panel.columns))
            if c in factor_panel.columns]
    if not cols:
        return 0.0
    return float(inf.multifactor_alpha(book_returns, factor_panel[cols],
                                       hac_lag=hac_lag)["t_alpha_hac"])


def _oos_return_array(result) -> np.ndarray:
    """The concatenated OOS book daily returns as a date-ordered, date-deduped 1-D
    array, from the pure-additive `oos_returns_dated` field (list of (date, ret)).
    Empty array when the field is absent/empty (legacy results) → every inference
    instrument fails closed (gating=False), so the gate fails closed too."""
    pairs: List[Tuple] = list(getattr(result, "oos_returns_dated", None) or [])
    if not pairs:
        return np.empty(0, dtype=float)
    # Sort by date, dedupe keeping first (mirror run_cpcv's residual-alpha assembly).
    # Sort/dedupe on str(date): for ISO 'YYYY-MM-DD' (what run_cpcv emits) lexical ==
    # chronological, and str() also keeps datetime/Timestamp inputs chronological —
    # so a stray non-string date can't silently misorder the series (the HAC/bootstrap
    # stats are autocorrelation-sensitive, so order correctness is load-bearing).
    seen = set()
    ordered: List[float] = []
    for d, r in sorted(pairs, key=lambda p: str(p[0])):
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        try:
            ordered.append(float(r))
        except (TypeError, ValueError):
            continue
    return np.asarray(ordered, dtype=float)


def _regime_backstop(result, *, floor: float,
                     regime_waiver_approved: bool) -> Tuple[float, bool, bool]:
    """Worst-regime "not catastrophic" backstop with the event-sparsity waiver.
    Returns (observed, ok, waived). Mirrors the significance backstop: None worst-
    regime due to genuine event-sparsity (regime_insufficient_obs) is WAIVED on
    paper / capital-with-approval (ok=True, waived=True, → human-review flag); None
    for any other reason fails CLOSED."""
    wrs = getattr(result, "worst_regime_sharpe", None)
    if wrs is not None:
        return float(wrs), float(wrs) >= floor, False
    insufficient = bool(getattr(result, "regime_insufficient_obs", False))
    if insufficient and regime_waiver_approved:
        return float("nan"), True, True       # event-sparsity waiver
    return float("nan"), False, False         # data-bug → fail closed


def evaluate(result, *, tier: str = "paper",
             n_trials: Optional[int] = None,
             pbo_perf=None,
             concurrent_paper_sleeves: int = 0,
             live_paper: Optional[dict] = None,
             regime_waiver_approved: bool = False) -> dict:
    """The Ruler-v2 per-criterion detail dict for `tier` ∈ {"paper","capital"}.

    Each value is (observed, ok). `n_trials` is the registry's TRUE trial count for
    the Bayesian prior shrinkage (defaults to N_TRIALS_TESTED). `pbo_perf`, if given,
    is an N_configs×N_blocks per-block OOS matrix → PBO gates when M>1. `live_paper`,
    if given, is {"sr": …, "se": …} realized paper performance folded into the
    posterior. PAPER waives the regime backstop for event-sparsity; CAPITAL waives it
    only with regime_waiver_approved=True (mirrors the significance gate)."""
    from app.ml.retrain_config import (
        N_TRIALS_TESTED, RULERV2_PAPER_MIN_SR, RULERV2_PAPER_IMPLAUSIBLE_SR,
        RULERV2_MIN_DAILY_OBS, RULERV2_MIN_N_FOLDS, RULERV2_PRIOR_SR_SD,
        RULERV2_CAPITAL_MIN_POSTERIOR, RULERV2_BOOTSTRAP_MIN_PSR, RULERV2_PBO_MAX,
        RULERV2_RESIDUAL_ALPHA_MIN_T, RULERV2_CATASTROPHIC_REGIME_SR,
        RULERV2_MAX_PAPER_SLEEVES,
    )
    from scripts.walkforward.gates import MIN_ACTIVE_FOLDS_FOR_GATE

    tier = tier.lower()
    capital = tier == "capital"
    # R7 trial-count resolution: explicit arg > the result's registered family count >
    # the CONSERVATIVE N_TRIALS_TESTED fallback. Defaulting to 300 everywhere would
    # re-import the saturated-DSR multiplicity pathology Ruler v2 exists to kill, so
    # PREFER the registered per-family count when the run recorded one.
    if n_trials is not None:
        nt = int(n_trials)
    elif getattr(result, "n_trials_registered", None) is not None:
        nt = int(result.n_trials_registered)
    else:
        nt = int(N_TRIALS_TESTED)

    r = _oos_return_array(result)
    n_obs = int(r.size)
    hac = inf.hac_sharpe(r)
    point_sr = hac.sr_ann if hac.gating else float("nan")

    n_paths = len(getattr(result, "path_sharpes", []) or [])
    enough_paths = n_paths >= MIN_ACTIVE_FOLDS_FOR_GATE

    # ── PAPER = plausibility ──────────────────────────────────────────────────
    sr_floor_ok = bool(hac.gating and point_sr >= RULERV2_PAPER_MIN_SR)
    ceiling_ok = bool(hac.gating and point_sr <= RULERV2_PAPER_IMPLAUSIBLE_SR)
    reg_obs, reg_ok, reg_waived = _regime_backstop(
        result, floor=RULERV2_CATASTROPHIC_REGIME_SR,
        regime_waiver_approved=(True if not capital else regime_waiver_approved))
    if reg_waived:
        try:
            result.requires_human_review_flag = True
        except Exception:
            pass
    # Concurrent-paper-sleeve cap (caller supplies the current live count).
    sleeve_ok = concurrent_paper_sleeves < RULERV2_MAX_PAPER_SLEEVES

    detail = {
        "n_paths": (n_paths, enough_paths),
        "point_sr_floor": (point_sr, sr_floor_ok),
        "implausibility_ceiling": (point_sr, ceiling_ok),
        "regime_not_catastrophic": (reg_obs, reg_ok),
        "concurrent_paper_sleeves": (concurrent_paper_sleeves, sleeve_ok),
        # report-only (off the AND):
        "hac_t_report": (hac.t_stat, True),
        "pf_report": (getattr(result, "avg_profit_factor", float("nan")), True),
        "calmar_report": (getattr(result, "avg_calmar", float("nan")), True),
        "cost_stress_report": ("not-wired (dark)", True),
    }
    if reg_waived:
        detail["requires_human_review"] = (
            bool(getattr(result, "regime_insufficient_obs", False)), False)

    if not capital:
        return detail

    # ── CAPITAL = significance ────────────────────────────────────────────────
    # Power floor — fails CLOSED below it regardless of the posterior.
    n_folds = int(getattr(result, "n_folds", 0) or 0)
    power_ok = (n_obs >= RULERV2_MIN_DAILY_OBS) and (n_folds >= RULERV2_MIN_N_FOLDS)

    # Bayesian posterior P(SR>0) — backtest (+ optional live-paper) vs a trial-
    # shrunk mean-zero prior. se from the HAC instrument (inf when t≈0 → backtest
    # uninformative → posterior leans on the prior → P≈0.5 → fails the 0.95 gate).
    lp_sr = (live_paper or {}).get("sr") if live_paper else None
    lp_se = (live_paper or {}).get("se") if live_paper else None
    post = bayes_sr.posterior_sr(
        hac.sr_ann if hac.gating else float("nan"),
        hac.se_sr_ann_implied if hac.gating else float("inf"),
        n_trials=nt, prior_sd=RULERV2_PRIOR_SR_SD,
        sr_live=lp_sr, se_live=lp_se)
    posterior_ok = bool(post.gating and post.p_sr_gt_0 >= RULERV2_CAPITAL_MIN_POSTERIOR)
    # STRUCTURAL live-paper requirement — the design's CAPITAL evidence is
    # P(SR>0 | backtest AND live paper). Without it, "capital unreachable on a
    # backtest alone" would rest only on the 0.95 THRESHOLD, which a long/clean
    # backtest (t≈3) clears on its own ~1-in-5 times — re-opening exactly the
    # backtest-only-promotion hole this redesign closes. So make it a hard gating
    # criterion: a CAPITAL verdict REQUIRES a live-paper observation in the posterior.
    live_paper_ok = bool(post.used_live)

    # Multi-factor residual-α t_HAC (primary). Uses the already-computed diagnostic;
    # None (not computed / too few obs) fails closed.
    ra_t = getattr(result, "residual_alpha_t_hac", None)
    ra_ok = bool(ra_t is not None and float(ra_t) >= RULERV2_RESIDUAL_ALPHA_MIN_T)

    # Stationary-bootstrap robustness twin.
    boot = inf.stationary_bootstrap_sr(r)
    boot_ok = bool(boot.gating and boot.p_sr_gt_0 >= RULERV2_BOOTSTRAP_MIN_PSR)

    detail.update({
        "live_paper_present": (post.used_live, live_paper_ok),
        "power_floor": ((n_obs, n_folds), power_ok),
        "posterior_p_sr_gt_0": (post.p_sr_gt_0, posterior_ok),
        "residual_alpha_t_hac": (ra_t, ra_ok),
        "bootstrap_p_sr_gt_0": (boot.p_sr_gt_0, boot_ok),
        "bootstrap_psr_report": (boot.p_sr_gt_0, True),
    })

    # PBO — gates ONLY when a >1-config per-block matrix is supplied (CSCV needs M>1).
    if pbo_perf is not None:
        pbo = inf.pbo_cscv(pbo_perf)
        if pbo.gating:
            detail["pbo"] = (pbo.pbo, pbo.pbo <= RULERV2_PBO_MAX)
        else:
            detail["pbo_report"] = (pbo.pbo, True)   # M<2 / non-finite → non-gating
    else:
        detail["pbo_report"] = (float("nan"), True)  # single config → not applicable
    return detail


def gate_passed(result, *, tier: str = "paper",
                paper_confirmation: bool = False,
                regime_waiver_approved: bool = False,
                **kw) -> bool:
    """Boolean Ruler-v2 verdict: the hard pre-checks (in-sample, true-WF, enough
    paths) then the AND over all GATING detail criteria. `paper_confirmation` is
    accepted for signature-parity with the significance gate (the Bayesian posterior
    already folds live-paper in via `live_paper`, so it is not a separate OR-path)."""
    if getattr(result, "in_sample_override", False):
        return False
    from app.ml.retrain_config import REQUIRE_TRUE_WF_FOR_PROMOTION
    if REQUIRE_TRUE_WF_FOR_PROMOTION and not getattr(result, "is_true_walkforward", False):
        return False
    from scripts.walkforward.gates import MIN_ACTIVE_FOLDS_FOR_GATE
    if len(getattr(result, "path_sharpes", []) or []) < MIN_ACTIVE_FOLDS_FOR_GATE:
        return False
    detail = evaluate(result, tier=tier,
                      regime_waiver_approved=regime_waiver_approved, **kw)
    return all(ok for k, (_, ok) in detail.items() if k not in INFORMATIONAL_KEYS)
