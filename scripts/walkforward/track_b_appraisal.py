"""
track_b_appraisal.py — Track-B v2, the BUDGET-INVARIANT appraisal gate
(Alpha-v7 Phase B, Phase 3). Reached only when TRACKB_MODE="ruler_v2" (ships DARK;
the legacy `book_delta_gate` in book_gate.py is byte-for-byte untouched and stays the
default).

Why v2: the legacy Track-B `min_sharpe_delta ≥ 0.10` is BUDGET-DEPENDENT — the same
sleeve passes or fails purely by the blend fraction `b` (a bigger budget mechanically
moves the book ΔSharpe), so the bar secretly encodes a position-size choice. v2
replaces it with two budget-invariant instruments:

  1. APPRAISAL RATIO (residual-α information ratio) — `multifactor_alpha(cand_vt,
     factors={base})["resid_sharpe"]`: the candidate's factor-HEDGED Sharpe (alpha
     per unit residual vol). Independent of the blend fraction — it is a property of
     the sleeve vs the book, not of how much you allocate. Gate: ≥ RULERV2_TRACKB_MIN_IR.
  2. BLOCK-BOOTSTRAP P(ΔSR > 0) — the stationary-bootstrap probability that adding
     the sleeve at budget `b` RAISES the book Sharpe. This keeps a budget term (ΔSR
     IS measured at `b`) but reports its SIGNIFICANCE, not a raw magnitude bar. Gate:
     ≥ RULERV2_TRACKB_MIN_PDSR.

The budget-invariant backstops from the legacy gate are reused verbatim (corr-to-book,
standalone vol-targeted Sharpe, implausibility ceiling, tail-overlap, risk-budget
ceiling). The WORST-REGIME floor is WAIVED for declared diversifiers / risk premia
(`component_type ∈ {diversifier, risk_premium}`) — failing the book's worst regime is a
crisis-diversifier's whole point; for any other component it gates when a worst-regime
Sharpe is supplied, and FAILS CLOSED when it is absent (unless an explicit human
`regime_waiver_approved` is passed) — mirroring the Track-A gate's data-bug posture so a
missing backstop is never a silent pass.

ΔSR is measured on the SIMPLE budget-b blend `(1-b)*base + b*cand_vt` (NOT the
allocator's combine() with turnover cost) — the v2 instrument reads the SIGNIFICANCE of
the direction of the book-Sharpe change, for which the symmetric ~1bp rebalance cost is
noise; the point estimate and the bootstrap use the identical definition.

PURE: reads its arguments, mutates nothing (not the inputs, not retrain_config), no I/O.
Reuses book_gate._vol_target_candidate / _as_daily_series / _series_sharpe and
inference.multifactor_alpha / _stationary_bootstrap_index_matrix / _auto_block_len.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.walkforward.book_gate import (
    ANN, MIN_ALIGNED_DAYS, ZERO_VARIANCE_STD,
    _as_daily_series, _series_sharpe, _vol_target_candidate,
)
from app.research.inference import (
    multifactor_alpha, _stationary_bootstrap_index_matrix, _auto_block_len,
)

# component_types whose entire purpose is to diverge from the book in a crisis — the
# worst-regime floor is intentionally waived for them (mirrors the Track-A waiver).
WAIVED_REGIME_TYPES = frozenset({"diversifier", "risk_premium"})

# Detail keys that are REPORT-ONLY — excluded from the gate's boolean AND.
INFORMATIONAL_KEYS = frozenset({
    "calmar_delta_report", "max_dd_delta_report", "t_alpha_hac_report",
    "worst_regime_report", "requires_human_review",
})


@dataclass(frozen=True)
class TrackBAppraisalCriteria:
    """Frozen, pre-registered Track-B v2 criteria (a gate run can never edit its own
    bar). Load via from_retrain_config()."""
    min_ir: float
    min_pdsr: float
    max_corr: float
    min_standalone_sr: float
    max_risk_budget: float
    joint_tail_pctl: float
    max_tail_overlap: float
    sharpe_implausibility_ceiling: float
    worst_regime_floor: float

    @classmethod
    def from_retrain_config(cls) -> "TrackBAppraisalCriteria":
        import app.ml.retrain_config as rc
        return cls(
            min_ir=rc.RULERV2_TRACKB_MIN_IR,
            min_pdsr=rc.RULERV2_TRACKB_MIN_PDSR,
            max_corr=rc.TRACKB_MAX_CORR,
            min_standalone_sr=rc.TRACKB_MIN_STANDALONE_SR,
            max_risk_budget=rc.TRACKB_MAX_RISK_BUDGET,
            joint_tail_pctl=rc.TRACKB_JOINT_TAIL_PCTL,
            max_tail_overlap=rc.TRACKB_MAX_TAIL_OVERLAP,
            sharpe_implausibility_ceiling=rc.SHARPE_IMPLAUSIBILITY_CEILING,
            worst_regime_floor=rc.RULERV2_CATASTROPHIC_REGIME_SR,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackBAppraisalResult:
    candidate_label: str
    component_type: str
    risk_budget: float
    n_days: int
    window_start: str
    window_end: str
    appraisal_ir: float          # residual-α information ratio (factor-hedged Sharpe)
    t_alpha_hac: float           # NW-HAC t of the residual alpha (diagnostic)
    delta_sr_point: float        # SR(with) - SR(without) at budget b (point estimate)
    p_delta_sr_gt_0: float       # block-bootstrap P(ΔSR > 0)
    delta_sr_ci_low: float
    delta_sr_ci_high: float
    corr_to_book: float
    standalone_vt_sharpe: float
    calmar_delta: float
    max_dd_delta: float
    n_tail: int
    tail_overlap: int
    tail_overlap_fraction: float
    worst_regime_sharpe: Optional[float]
    regime_waived: bool
    block_len: float
    n_boot: int
    vol_floor_bind_frac: float
    checks: Dict[str, tuple] = field(default_factory=dict)
    failed_criteria: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed: bool = False
    verdict: str = ""
    criteria: Optional[TrackBAppraisalCriteria] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = {k: [v[0], bool(v[1])] for k, v in self.checks.items()}
        d["criteria"] = self.criteria.to_dict() if self.criteria else None
        return d


def _sr_rows(mat: np.ndarray) -> np.ndarray:
    """Annualized Sharpe per row of an (n_boot, n) returns matrix (mean/std ddof=1
    * sqrt(252)); a degenerate (constant) row → 0.0 (matches _series_sharpe)."""
    mu = mat.mean(axis=1)
    sd = mat.std(axis=1, ddof=1)
    return np.where(sd > ZERO_VARIANCE_STD, mu / sd * np.sqrt(ANN), 0.0)


def appraise_track_b(base_daily: pd.Series, candidate_daily: pd.Series, *,
                     component_type: str,
                     criteria: TrackBAppraisalCriteria,
                     candidate_risk_budget: Optional[float] = None,
                     worst_regime_sharpe: Optional[float] = None,
                     regime_waiver_approved: bool = False,
                     n_boot: int = 2000,
                     block_len: Optional[float] = None,
                     seed: int = 0,
                     candidate_label: str = "candidate") -> TrackBAppraisalResult:
    """Budget-invariant Track-B v2 appraisal. PASS iff ALL gating criteria hold.

    base_daily/candidate_daily: daily net returns (pd.DatetimeIndex required; the
    candidate's own costs already inside, as for every sleeve series here).
    component_type: free-form; {diversifier, risk_premium} waive the worst-regime floor.
    candidate_risk_budget: blend fraction b in (0, max_risk_budget]; default = max.
    worst_regime_sharpe: optional; gates (when not waived) or is report-only if absent.
    Raises TypeError on a non-DatetimeIndex; ValueError on an out-of-budget blend or
    too little aligned history. A constant candidate → clean degenerate REJECT.
    """
    b = (criteria.max_risk_budget if candidate_risk_budget is None
         else float(candidate_risk_budget))
    if not (0.0 < b <= criteria.max_risk_budget + 1e-12):
        raise ValueError(
            f"candidate_risk_budget {b} outside the pre-registered budget "
            f"(0, {criteria.max_risk_budget}] — Track B only evaluates SMALL adds")

    base = _as_daily_series(base_daily, "base_daily")
    cand = _as_daily_series(candidate_daily, "candidate_daily")
    aligned = pd.concat([base.rename("base"), cand.rename("cand")], axis=1,
                        join="inner").dropna()
    if len(aligned) < MIN_ALIGNED_DAYS:
        raise ValueError(
            f"only {len(aligned)} aligned days (< {MIN_ALIGNED_DAYS}) — too little "
            "joint history for an appraisal verdict")
    if float(aligned["cand"].std()) < ZERO_VARIANCE_STD:
        return _degenerate_reject(candidate_label, component_type, b, aligned, criteria)

    # Vol-target the candidate to the base book's full-sample ann vol (PIT), then
    # evaluate everything on the shared post-warmup window.
    target_ann_vol = float(aligned["base"].std() * np.sqrt(ANN))
    if target_ann_vol <= 0:
        raise ValueError("base book has zero volatility — appraisal undefined")
    cand_vt, floor_bind = _vol_target_candidate(aligned["cand"], target_ann_vol)
    ev = pd.DataFrame({"base": aligned["base"], "cand": cand_vt}).dropna()
    if len(ev) < MIN_ALIGNED_DAYS:
        raise ValueError(
            f"only {len(ev)} post-warmup days (< {MIN_ALIGNED_DAYS}) after the 60d "
            "vol-targeting warmup — too little joint history")

    base_arr = ev["base"].to_numpy(dtype=float)
    cand_arr = ev["cand"].to_numpy(dtype=float)
    n_days = int(len(ev))

    # (1) APPRAISAL RATIO — residual-α IR of the candidate vs the base book.
    mfa = multifactor_alpha(ev["cand"], ev[["base"]], hac_lag=5)
    appraisal_ir = float(mfa["resid_sharpe"])
    t_alpha_hac = float(mfa["t_alpha_hac"])

    # (2) BLOCK-BOOTSTRAP P(ΔSR > 0) — ΔSR = SR(book_with_b) - SR(book_without) on the
    # simple budget-b blend. Resample PAIRED rows (one index path for both series, to
    # preserve their contemporaneous correlation) via the stationary bootstrap.
    with_arr = (1.0 - b) * base_arr + b * cand_arr
    sr_with_pt = _series_sharpe(pd.Series(with_arr))
    sr_without_pt = _series_sharpe(pd.Series(base_arr))
    delta_sr_point = float(sr_with_pt - sr_without_pt)
    blk = float(block_len) if block_len is not None else _auto_block_len(base_arr)
    blk = max(1.0, blk)
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_index_matrix(n_days, 1.0 / blk, int(n_boot), rng)
    base_bs = base_arr[idx]
    with_bs = with_arr[idx]
    dsr = _sr_rows(with_bs) - _sr_rows(base_bs)
    p_dsr = float(np.mean(dsr > 0.0))
    dsr_lo, dsr_hi = (float(x) for x in np.percentile(dsr, [2.5, 97.5]))

    # Budget-invariant backstops (reused).
    corr = float(ev["base"].corr(ev["cand"]))
    standalone_sr = _series_sharpe(ev["cand"])
    bind = floor_bind.reindex(ev.index)
    vol_floor_bind_frac = float(bind.fillna(False).mean()) if len(bind) else 0.0

    n_tail = max(3, int(np.floor(criteria.joint_tail_pctl * n_days)))
    base_worst = set(ev["base"].nsmallest(n_tail, keep="first").index)
    cand_worst = set(ev["cand"].nsmallest(n_tail, keep="first").index)
    overlap = int(len(base_worst & cand_worst))
    overlap_frac = float(overlap / n_tail)

    # Diagnostics (report-only).
    calmar_delta = float(_calmar(with_arr) - _calmar(base_arr))
    dd_delta = float(_max_dd(with_arr) - _max_dd(base_arr))   # >=0 => no deeper DD

    # Worst-regime backstop — waived for declared diversifiers / risk premia.
    waived = component_type.lower() in WAIVED_REGIME_TYPES

    checks: Dict[str, tuple] = {
        "appraisal_ir": (appraisal_ir, appraisal_ir >= criteria.min_ir),
        "p_delta_sr_gt_0": (p_dsr, p_dsr >= criteria.min_pdsr),
        "corr_to_book": (corr, corr < criteria.max_corr),
        "standalone_vt_sharpe": (standalone_sr, standalone_sr > criteria.min_standalone_sr),
        "standalone_sr_plausible": (
            standalone_sr, standalone_sr <= criteria.sharpe_implausibility_ceiling),
        "risk_budget": (b, b <= criteria.max_risk_budget + 1e-12),
        "tail_overlap": (overlap_frac, overlap_frac <= criteria.max_tail_overlap + 1e-12),
        # report-only (off the AND):
        "calmar_delta_report": (calmar_delta, True),
        "max_dd_delta_report": (dd_delta, True),
        "t_alpha_hac_report": (t_alpha_hac, True),
    }
    if waived:
        # Declared diversifier / risk premium → backstop waived (report-only); the
        # whole point of the sleeve is to diverge from the book in a crisis.
        checks["worst_regime_report"] = (
            (float(worst_regime_sharpe) if worst_regime_sharpe is not None
             else float("nan")), True)
    elif worst_regime_sharpe is not None:
        checks["worst_regime"] = (
            float(worst_regime_sharpe),
            float(worst_regime_sharpe) >= criteria.worst_regime_floor)
    elif regime_waiver_approved:
        # Non-diversifier with NO regime data but an explicit human waiver → report-
        # only WITH a mandatory human-review flag (mirrors the Track-A sign-off path).
        checks["worst_regime_report"] = (float("nan"), True)
        checks["requires_human_review"] = (True, False)
    else:
        # Non-diversifier, no regime data, no waiver → FAIL CLOSED (consistent with the
        # Track-A significance gate's data-bug posture — a missing backstop is not a
        # silent pass).
        checks["worst_regime"] = (float("nan"), False)

    failed = [k for k, (_v, ok) in checks.items()
              if not ok and k not in INFORMATIONAL_KEYS]
    passed = len(failed) == 0

    warnings: List[str] = []
    if vol_floor_bind_frac > 0.10:
        warnings.append(
            f"vol-target floor bound on {vol_floor_bind_frac:.0%} of days "
            "(> 10%) — the candidate's vol-targeting is unreliable")
    if not waived and worst_regime_sharpe is None:
        warnings.append(
            "no worst_regime_sharpe for a non-diversifier component — "
            + ("WAIVED by explicit human sign-off (requires_human_review)"
               if regime_waiver_approved else
               "FAILING CLOSED (pass worst_regime_sharpe, declare it a "
               "diversifier/risk_premium, or set regime_waiver_approved=True)"))

    return TrackBAppraisalResult(
        candidate_label=candidate_label, component_type=component_type,
        risk_budget=b, n_days=n_days,
        window_start=str(ev.index[0].date()), window_end=str(ev.index[-1].date()),
        appraisal_ir=appraisal_ir, t_alpha_hac=t_alpha_hac,
        delta_sr_point=delta_sr_point, p_delta_sr_gt_0=p_dsr,
        delta_sr_ci_low=dsr_lo, delta_sr_ci_high=dsr_hi,
        corr_to_book=corr, standalone_vt_sharpe=standalone_sr,
        calmar_delta=calmar_delta, max_dd_delta=dd_delta,
        n_tail=n_tail, tail_overlap=overlap, tail_overlap_fraction=overlap_frac,
        worst_regime_sharpe=(float(worst_regime_sharpe)
                             if worst_regime_sharpe is not None else None),
        regime_waived=waived, block_len=blk, n_boot=int(n_boot),
        vol_floor_bind_frac=vol_floor_bind_frac,
        checks=checks, failed_criteria=failed, warnings=warnings, passed=passed,
        verdict=("PASS" if passed else "FAIL"), criteria=criteria)


def _max_dd(r: np.ndarray) -> float:
    """Max drawdown (<= 0) of a daily return stream via the cumulative wealth path."""
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def _calmar(r: np.ndarray) -> float:
    """CAGR / |maxDD| (0 when no drawdown), matching sleeve_allocator conventions."""
    n = len(r)
    if n == 0:
        return 0.0
    wealth = float(np.prod(1.0 + r))
    cagr = wealth ** (ANN / n) - 1.0 if wealth > 0 else -1.0
    mdd = _max_dd(r)
    return float(cagr / abs(mdd)) if mdd < 0 else 0.0


def _degenerate_reject(label, component_type, b, aligned, criteria):
    """A constant (zero-variance) candidate → clean REJECT, no NaN math."""
    return TrackBAppraisalResult(
        candidate_label=label, component_type=component_type, risk_budget=b,
        n_days=int(len(aligned)),
        window_start=str(aligned.index[0].date()),
        window_end=str(aligned.index[-1].date()),
        appraisal_ir=0.0, t_alpha_hac=0.0, delta_sr_point=0.0, p_delta_sr_gt_0=0.0,
        delta_sr_ci_low=0.0, delta_sr_ci_high=0.0, corr_to_book=0.0,
        standalone_vt_sharpe=0.0, calmar_delta=0.0, max_dd_delta=0.0,
        n_tail=0, tail_overlap=0, tail_overlap_fraction=0.0,
        worst_regime_sharpe=None, regime_waived=False, block_len=1.0, n_boot=0,
        vol_floor_bind_frac=0.0,
        checks={"appraisal_ir": (0.0, False),
                "p_delta_sr_gt_0": (0.0, False)},
        failed_criteria=["appraisal_ir", "p_delta_sr_gt_0"],
        warnings=["candidate is a constant series (zero variance) — degenerate REJECT"],
        passed=False, verdict="FAIL (degenerate)", criteria=criteria)
