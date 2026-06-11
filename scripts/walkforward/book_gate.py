"""
book_gate.py - Phase 0 (Alpha-v6): Track B (book-delta) acceptance gate.

WHY (NEXT_PHASE_BLUEPRINT_2026-06.md Phase 0 / consensus C2; P0 RESULT
2026-06-10): the standalone significance gate is the WRONG ruler for crisis-
diversifiers and risk premia. The flagship TSMOM sleeve posts t=6.72, passes
the significance CORE, and fails the production PAPER gate ONLY on the
worst_regime_sharpe backstop - because a crisis-diversifier intentionally
whipsaws in one regime. Lowering the t-bar was empirically refuted (3/5 TRUE
zero-SR nulls cleared t>=2.0; the pre-registered recalibration rule returned
NO_ADMISSIBLE_TSTAR). The correct lever is TWO-TRACK acceptance: Track A keeps
the standalone significance gate for alpha strategies; Track B (this module)
judges diversifiers on their PRE-REGISTERED contribution to the COMBINED BOOK
at a small fixed risk budget.

THE QUESTION THE GATE ANSWERS
=============================
"Does adding this candidate sleeve, at a small pre-registered risk budget,
IMPROVE the combined book?" PASS iff ALL of the frozen criteria below hold
(constants live in app/ml/retrain_config.py, registered 2026-06-10 BEFORE any
Track B candidate was evaluated):

  sharpe_delta         book Sharpe (with - without) >= TRACKB_MIN_SHARPE_DELTA (0.10)
  calmar_delta         book Calmar (with - without) >= TRACKB_MIN_CALMAR_DELTA (0.0)
  max_dd_delta         book maxDD must not worsen by more than TRACKB_MAX_DD_DELTA
                       (0.0). SIGN CONVENTION: maxDD <= 0 (sleeve_allocator._maxdd:
                       min of equity/cummax - 1; LESS NEGATIVE = BETTER), so
                       dd_delta = maxDD_with - maxDD_without and the criterion is
                       dd_delta >= -TRACKB_MAX_DD_DELTA (with 0.0: the with-candidate
                       book's drawdown may not be deeper).
  corr_to_book         Pearson corr of the vol-targeted candidate vs the base book's
                       daily returns < TRACKB_MAX_CORR (0.30). One-sided upper bound:
                       negative correlation is a diversifier's whole point and passes.
  standalone_vt_sharpe the candidate's OWN vol-targeted Sharpe > TRACKB_MIN_STANDALONE_SR
                       (0.20) - a floor that rejects negative/zero-SR add-ons that
                       "help" only through noise-cancellation.
  standalone_sr_plausible
                       the same standalone vol-targeted Sharpe must also be <=
                       SHARPE_IMPLAUSIBILITY_CEILING (3.0, the SAME constant Track A
                       uses). An SR above 3.0 net of costs is empirically implausible
                       for any sleeve in this repo - look-ahead or a degenerate
                       (near-constant) series is the likeliest cause, so the gate
                       FAILS rather than admitting it ("implausible standalone
                       Sharpe (look-ahead/degenerate)"). Mirrors Track A's
                       CRITICAL-1 ceiling intent; without it a constant-positive
                       candidate (SR ~ 1e16 from float noise) would PASS.
  risk_budget          the candidate is blended at <= TRACKB_MAX_RISK_BUDGET (0.10)
                       of book risk (enforced at input validation; the blend below).
  tail_overlap         TAIL-OVERLAP TEST, exact definition: on the SAME post-combine
                       index the book metrics use, let
                         n_tail    = max(3, floor(TRACKB_JOINT_TAIL_PCTL * n_days))
                         base_worst = the n_tail dates with the SMALLEST base-book
                                      returns (deterministic tie-break: nsmallest
                                      keep="first")
                         cand_worst = the n_tail dates with the smallest VOL-TARGETED
                                      candidate returns (same tie-break)
                         overlap_fraction = |base_worst intersect cand_worst| / n_tail
                       PASS iff overlap_fraction <= TRACKB_MAX_TAIL_OVERLAP (0.30).
                       Rationale: this IS the blueprint's REGISTERED criterion
                       ("candidate's worst-1% days must not coincide with the book's
                       worst-1%", NEXT_PHASE_BLUEPRINT_2026-06.md Phase 0). An earlier
                       implementation used mean(candidate return on the base's worst
                       days) >= 0 - an UNREGISTERED divergence that was wrong in both
                       directions: (a) MASKABLE - one large positive day flips the
                       mean positive even when the candidate crashes on 13/14 of the
                       book's tail days (false ADMIT of a tail-amplifier); and
                       (b) ~43% false-REJECT on genuinely independent diversifiers
                       (the mean of ~14 noise days is near a coin flip). The overlap
                       form fixes both: under independence the expected overlap
                       fraction is ~ n_tail/n_days (~1%), far below 0.30, so
                       independents essentially never fail; a co-crasher's worst days
                       coincide with the book's, driving the overlap high. The one
                       free parameter (0.30) is pinned in retrain_config as
                       TRACKB_MAX_TAIL_OVERLAP, registered 2026-06-10. The max(3, .)
                       floor keeps the test defined on short histories (a 1-day
                       "tail" is meaningless); note that at the n_tail=3 floor
                       (histories below ~300 evaluated days) a SINGLE coincident
                       worst day = overlap 0.33 > 0.30 fails the test (~8% false
                       reject under independence at ~129 days). This errs strictly
                       (false REJECT, never false ADMIT) and only near the
                       MIN_ALIGNED_DAYS floor; with n_tail >= ~14 the false-reject
                       rate is negligible.

VOL-TARGETING + BLEND (the "risk budget" mechanics)
===================================================
1. ALIGN: base and candidate daily-return series are inner-joined on dates. Both
   inputs MUST carry a pd.DatetimeIndex (anything else raises TypeError - an
   integer index would otherwise be silently coerced to 1970-epoch timestamps).
2. VOL-TARGET the candidate, PIT: leverage_t = target_ann_vol / trailing_vol_{t-1},
   where trailing vol is the SAME estimator the live allocator uses
   (sleeve_allocator._realized_vol: 60d rolling std * sqrt(252), min_periods 30,
   floored at 2% ann) and the shift(1) guarantees day-t exposure uses only data
   through t-1. target_ann_vol = the base book's full-sample annualized vol - a
   single normalization CONSTANT. There is deliberately NO leverage cap: with the
   2% vol floor, leverage is already bounded by target/0.02 (~5-8x for a 10-15%
   book), and an additional cap would BREAK TARGET-INVARIANCE - whenever a cap
   binds, the (full-sample, hence future-dependent) target level changes WHICH
   days are capped, reshaping the series so corr / standalone-SR / tail ordering
   depend on the target (a probe showed standalone SR 1.76 -> 1.17 under target-
   doubling with a 5x cap binding). Uncapped, cand_vt = target * cand_t /
   max(rv_{t-1}, 2%) is an EXACT scalar multiple of a strictly PIT series on
   every non-floor-binding day, so corr, Sharpe and tail ORDERING are invariant
   to the target level. DIAGNOSTIC: vol_floor_bind_frac = fraction of evaluated
   days where the 2% floor binds; above 0.10 a warning is recorded ("vol-target
   unreliable: candidate near-constant vol") - informational, not a hard fail.
   Warmup days (NaN trailing vol) are dropped from BOTH books so the
   with/without comparison is on an identical index. A zero-variance (constant)
   candidate is detected up front and short-circuits to a clean REJECT
   ("degenerate candidate (zero variance)") - vol-targeting noise is undefined
   for it and no warnings/NaNs are emitted.
3. BLEND: with-book = (1-b)*base + b*candidate_vt at constant-mix weights, where
   b = risk budget (default TRACKB_MAX_RISK_BUDGET). Because the candidate is
   vol-matched to the base, b is (to first order, exactly at corr=0) the
   candidate's fraction of book risk. Both the with- and without-book are built
   through app/strategy/sleeve_allocator.combine() - REUSED, not reimplemented -
   so Sharpe / CAGR / maxDD / Calmar / turnover-cost conventions are byte-
   identical to the existing book harness (run_book_allocator.py):
   sharpe = mean/std(ddof=1)*sqrt(252), maxDD = min(eq/cummax - 1) <= 0,
   calmar = CAGR/|maxDD|, weekly rebalance, 1bp allocator cost on weight turnover
   (symmetric across both books -> deltas are net of the blend's own turnover).
   combine() drops its first day (the turnover cost is shift(1)-aligned), so ALL
   gate statistics - corr, standalone SR, the tail sets and n_days - are computed
   on that same post-combine index: the tail days are guaranteed to exist in both
   books and every reported count refers to one and the same window.
   NOTE on metric REUSE: scripts/walkforward/gates.py's compute_calmar /
   compute_profit_factor are PER-FOLD, PER-TRADE metrics with sentinel logic
   (positive-maxDD convention, no-loss/no-DD sentinels) - the wrong fit for a
   continuous book series; BookResult (sleeve_allocator.combine) is the existing
   book-level definition and is what this gate reuses.

DESIGN CAVEATS (known, accepted - documented so the first real run is read right)
==================================================================================
* CALIBRATION TENSION AT b=0.10 (open question for the FIRST REAL RUN): requiring
  Delta-Sharpe >= 0.10 from a 10% risk slice implicitly demands a fairly high
  standalone SR from the candidate - roughly SR >= 0.94 at corr 0, ~0.70 at corr
  -0.3 (first-order: dSR ~ b*(SR_cand - corr*SR_book) for small b). TSMOM - the
  very sleeve this track was built for - shows SR ~ 0.71 with corr ~ +0.25 to the
  PEAD book, so it may NOT clear sharpe_delta at this budget/threshold: the gate
  could structurally reject its motivating candidate. This is an OPEN CALIBRATION
  QUESTION to be resolved AFTER the first real run (TSMOM vs the PEAD book) by a
  REGISTERED amendment to TRACKB_MIN_SHARPE_DELTA and/or the budget if needed -
  never by an ad hoc, post-hoc tweak to admit a specific candidate.
* WEAK INDEPENDENT PROTECTION FROM maxDD/Calmar: the with-book runs at LOWER vol
  than the base (the blend is not re-levered to matched book risk), so maxDD and
  Calmar mechanically improve for almost ANY low-correlation addition. Those two
  criteria therefore provide weak independent discrimination; the corr +
  tail_overlap + sharpe_delta criteria carry the real burden of proof.

SCOPE / PROMOTION
=================
Track B gates PAPER-LEVEL BOOK INCLUSION ONLY. It NEVER auto-promotes to
CAPITAL: any capital allocation additionally requires explicit owner sign-off
plus a codified tail-loss budget (blueprint Phase 0 / section 6). This module is
PURE - no I/O, no network, no config mutation; it reads the frozen constants
through BookGateCriteria.from_retrain_config() and only ever returns a result.

Usage:
    from scripts.walkforward.book_gate import BookGateCriteria, book_delta_gate, format_report
    crit = BookGateCriteria.from_retrain_config()
    res = book_delta_gate(base_daily, candidate_daily, criteria=crit)
    print(format_report(res))
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ANN = 252

# Minimum aligned post-warmup observations for the gate to be meaningful
# (~6 months; below this the Sharpe/maxDD deltas and a 1%-tail of <3 days are
# noise, and the verdict would be unsound either way).
MIN_ALIGNED_DAYS = 120

# Candidate daily-return std below this is treated as zero variance (a constant
# series; float noise on a true constant is ~1e-19, real sleeves are >= 1e-4).
ZERO_VARIANCE_STD = 1e-12

# vol_floor_bind_frac above this records the "vol-target unreliable" warning.
VOL_FLOOR_WARN_FRAC = 0.10


# ==============================================================================
# Criteria (frozen, pre-registered) and result dataclasses
# ==============================================================================
@dataclass(frozen=True)
class BookGateCriteria:
    """The pre-registered Track B criteria. Frozen: a gate run can never edit
    its own bar. Load the registered values via from_retrain_config()."""
    min_sharpe_delta: float
    min_calmar_delta: float
    max_dd_delta: float
    max_corr: float
    min_standalone_sr: float
    max_risk_budget: float
    joint_tail_pctl: float
    max_tail_overlap: float
    # Same ceiling Track A uses (retrain_config.SHARPE_IMPLAUSIBILITY_CEILING).
    sharpe_implausibility_ceiling: float = 3.0

    @classmethod
    def from_retrain_config(cls) -> "BookGateCriteria":
        """Read the registered constants from app/ml/retrain_config.py (the gate
        constants' single source of truth). Mirrors
        gate_calibration.GateConfigSnapshot.from_retrain_config()."""
        import app.ml.retrain_config as rc
        return cls(
            min_sharpe_delta=rc.TRACKB_MIN_SHARPE_DELTA,
            min_calmar_delta=rc.TRACKB_MIN_CALMAR_DELTA,
            max_dd_delta=rc.TRACKB_MAX_DD_DELTA,
            max_corr=rc.TRACKB_MAX_CORR,
            min_standalone_sr=rc.TRACKB_MIN_STANDALONE_SR,
            max_risk_budget=rc.TRACKB_MAX_RISK_BUDGET,
            joint_tail_pctl=rc.TRACKB_JOINT_TAIL_PCTL,
            max_tail_overlap=rc.TRACKB_MAX_TAIL_OVERLAP,
            sharpe_implausibility_ceiling=rc.SHARPE_IMPLAUSIBILITY_CEILING,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BookGateResult:
    """Structured Track B verdict: per-criterion measured value + bool, the
    with/without book metrics, the deltas, and the overall PASS/FAIL."""
    candidate_label: str
    risk_budget: float
    n_days: int
    window_start: str
    window_end: str
    # Book metrics from sleeve_allocator.BookResult.summary() conventions:
    # sharpe, cagr, ann_vol, max_drawdown (<= 0), calmar, ann_turnover, n_days.
    # Empty dicts <=> degenerate short-circuit (no books were built).
    without_book: Dict[str, float]
    with_book: Dict[str, float]
    sharpe_delta: float
    calmar_delta: float
    max_dd_delta: float          # maxDD_with - maxDD_without; >= 0 means no deeper
    corr_to_book: float
    standalone_vt_sharpe: float
    # Tail-overlap test (registered criterion; see module docstring):
    n_tail: int
    tail_overlap: int            # |base_worst intersect cand_worst|
    tail_overlap_fraction: float
    base_worst_dates: List[str]  # the book's n_tail worst days (ISO dates)
    # Diagnostic: fraction of evaluated days where the 2% trailing-vol floor
    # bound the candidate's vol-target leverage (> 0.10 -> warning, not a fail).
    vol_floor_bind_frac: float
    # criterion name -> (measured value, ok). Keys match the criteria docstring.
    checks: Dict[str, tuple] = field(default_factory=dict)
    failed_criteria: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed: bool = False
    verdict: str = ""
    criteria: Optional[BookGateCriteria] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = {k: [v[0], bool(v[1])] for k, v in self.checks.items()}
        d["criteria"] = self.criteria.to_dict() if self.criteria is not None else None
        return d


# ==============================================================================
# Internals (pure helpers)
# ==============================================================================
def _as_daily_series(s: pd.Series, name: str) -> pd.Series:
    """Defensive copy -> DatetimeIndex-verified, sorted, NaN-dropped daily returns."""
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series of daily returns")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(
            f"{name} must have a pd.DatetimeIndex (got {type(s.index).__name__}); "
            "a non-datetime index would be silently coerced to 1970-epoch "
            "timestamps - convert the index explicitly before calling the gate")
    out = s.dropna().copy()
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"{name} has duplicate dates - one return per day required")
    return out.astype(float)


def _series_sharpe(r: pd.Series) -> float:
    """Annualized Sharpe with the SAME convention as sleeve_allocator.combine:
    mean/std(ddof=1) * sqrt(252)."""
    sd = float(r.std())
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def _vol_target_candidate(cand: pd.Series,
                          target_ann_vol: float) -> Tuple[pd.Series, pd.Series]:
    """PIT vol-targeting: scale day-t candidate return by
    target_ann_vol / trailing_vol_{t-1}, trailing vol via the allocator's own
    estimator (sleeve_allocator._realized_vol: 60d, min_periods 30, floor 2%).
    NO leverage cap (see module docstring: a cap breaks target-invariance when
    it binds; the 2% floor already bounds leverage at target/0.02). Warmup days
    return NaN (the caller drops them from BOTH books). The target level is a
    constant, so on non-floor-binding days the result is an exact scalar
    multiple of a strictly PIT series. Returns (vol_targeted_returns,
    floor_bind) where floor_bind is a day-aligned bool Series marking days
    whose applied leverage was floor-bound."""
    from app.strategy.sleeve_allocator import AllocatorConfig, _realized_vol
    acfg = AllocatorConfig()
    rv = _realized_vol(cand.to_frame("cand"), acfg)["cand"]
    lev = (float(target_ann_vol) / rv).shift(1)
    # _realized_vol clips at vol_floor, so rv == floor exactly where it binds.
    floor_bind = (rv <= acfg.vol_floor + 1e-15).shift(1, fill_value=False)
    return cand * lev, floor_bind


def _degenerate_result(label: str, b: float, aligned: pd.DataFrame,
                       criteria: BookGateCriteria) -> BookGateResult:
    """Clean short-circuit REJECT for a zero-variance (constant) candidate:
    vol-targeting / corr / Sharpe are undefined for it, so no books are built
    and no numpy warnings are emitted."""
    return BookGateResult(
        candidate_label=label,
        risk_budget=b,
        n_days=int(len(aligned)),
        window_start=aligned.index[0].date().isoformat(),
        window_end=aligned.index[-1].date().isoformat(),
        without_book={},
        with_book={},
        sharpe_delta=0.0,
        calmar_delta=0.0,
        max_dd_delta=0.0,
        corr_to_book=0.0,
        standalone_vt_sharpe=0.0,
        n_tail=0,
        tail_overlap=0,
        tail_overlap_fraction=0.0,
        base_worst_dates=[],
        vol_floor_bind_frac=1.0,
        checks={"candidate_variance": (0.0, False)},
        failed_criteria=["candidate_variance"],
        warnings=["candidate daily-return std ~ 0: vol-targeting undefined"],
        passed=False,
        verdict="TRACK_B_FAIL - degenerate candidate (zero variance)",
        criteria=criteria,
    )


# ==============================================================================
# The gate (PURE: no I/O, no config mutation, deterministic)
# ==============================================================================
def book_delta_gate(base_daily: pd.Series, candidate_daily: pd.Series, *,
                    criteria: BookGateCriteria,
                    candidate_risk_budget: Optional[float] = None,
                    candidate_label: str = "candidate") -> BookGateResult:
    """Track B book-delta gate. See the module docstring for the full design.

    Args:
      base_daily:      daily net returns of the EXISTING combined book
                       (pd.DatetimeIndex required).
      candidate_daily: daily net returns of the candidate sleeve (its own costs
                       already inside, as for every sleeve series in this repo;
                       pd.DatetimeIndex required).
      criteria:        the frozen pre-registered BookGateCriteria.
      candidate_risk_budget: blend fraction b; default = criteria.max_risk_budget.
                       Must satisfy 0 < b <= criteria.max_risk_budget.
      candidate_label: cosmetic name for the report.

    Returns a BookGateResult; PASS iff ALL criteria are met. PURE: reads its
    arguments, mutates nothing (not the inputs, not retrain_config), no I/O.
    Raises TypeError on a non-DatetimeIndex input; ValueError on an
    out-of-budget blend or insufficient aligned history. A constant (zero-
    variance) candidate returns a clean degenerate REJECT instead of raising.
    """
    b = criteria.max_risk_budget if candidate_risk_budget is None else float(candidate_risk_budget)
    if not (0.0 < b <= criteria.max_risk_budget + 1e-12):
        raise ValueError(
            f"candidate_risk_budget {b} outside the pre-registered budget "
            f"(0, {criteria.max_risk_budget}] - Track B only evaluates SMALL adds")

    base = _as_daily_series(base_daily, "base_daily")
    cand = _as_daily_series(candidate_daily, "candidate_daily")
    aligned = pd.concat([base.rename("base"), cand.rename("cand")], axis=1,
                        join="inner").dropna()
    if len(aligned) < MIN_ALIGNED_DAYS:
        raise ValueError(
            f"only {len(aligned)} aligned days (< {MIN_ALIGNED_DAYS}) - too little "
            "joint history for a book-delta verdict")

    # -- Degenerate candidate: constant series -> clean REJECT (no NaN math) ----
    if float(aligned["cand"].std()) < ZERO_VARIANCE_STD:
        return _degenerate_result(candidate_label, b, aligned, criteria)

    # -- Vol-target the candidate to the base book's full-sample ann vol (PIT) --
    target_ann_vol = float(aligned["base"].std() * np.sqrt(ANN))
    if target_ann_vol <= 0:
        raise ValueError("base book has zero volatility - book-delta gate undefined")
    cand_vt, floor_bind = _vol_target_candidate(aligned["cand"], target_ann_vol)
    cand_vt = cand_vt.dropna()
    df = pd.DataFrame({"base": aligned["base"], "cand": cand_vt}).dropna()
    if len(df) < MIN_ALIGNED_DAYS:
        raise ValueError(
            f"only {len(df)} aligned post-warmup days (< {MIN_ALIGNED_DAYS}) after "
            "the 60d vol-targeting warmup - too little joint history")

    # -- Build WITH / WITHOUT books through the existing allocator (REUSE) -----
    from app.strategy.sleeve_allocator import AllocatorConfig, combine
    acfg = AllocatorConfig()
    w_without = pd.DataFrame({"base": 1.0, "cand": 0.0}, index=df.index)
    w_with = pd.DataFrame({"base": 1.0 - b, "cand": b}, index=df.index)
    res_without = combine(df, w_without, acfg)
    res_with = combine(df, w_with, acfg)
    without_m = res_without.summary()
    with_m = res_with.summary()

    sharpe_delta = float(with_m["sharpe"] - without_m["sharpe"])
    calmar_delta = float(with_m["calmar"] - without_m["calmar"])
    # maxDD <= 0; LESS NEGATIVE = BETTER, so a non-negative delta = no deeper DD.
    dd_delta = float(with_m["max_drawdown"] - without_m["max_drawdown"])

    # -- Evaluate EVERYTHING on the post-combine index (combine() drops day 0
    #    via the shift(1)-aligned turnover cost): corr, SR, tails and n_days all
    #    refer to one and the same window the book metrics were computed on. ---
    ev = df.loc[res_without.returns.index]
    n_days = int(len(ev))
    corr = float(ev["base"].corr(ev["cand"]))
    standalone_sr = _series_sharpe(ev["cand"])
    bind = floor_bind.reindex(ev.index)
    vol_floor_bind_frac = float(bind.fillna(False).mean()) if len(bind) else 0.0

    # -- Tail-overlap test (registered; exact definition in module docstring) ---
    n_tail = max(3, int(np.floor(criteria.joint_tail_pctl * n_days)))
    base_worst = ev["base"].nsmallest(n_tail, keep="first").index
    cand_worst = ev["cand"].nsmallest(n_tail, keep="first").index
    overlap = int(len(set(base_worst) & set(cand_worst)))
    overlap_fraction = float(overlap / n_tail)
    base_worst_dates = [ts.date().isoformat() for ts in sorted(base_worst)]

    checks: Dict[str, tuple] = {
        "sharpe_delta": (sharpe_delta, sharpe_delta >= criteria.min_sharpe_delta),
        "calmar_delta": (calmar_delta, calmar_delta >= criteria.min_calmar_delta),
        "max_dd_delta": (dd_delta, dd_delta >= -criteria.max_dd_delta - 1e-12),
        "corr_to_book": (corr, corr < criteria.max_corr),
        "standalone_vt_sharpe": (standalone_sr, standalone_sr > criteria.min_standalone_sr),
        "standalone_sr_plausible": (
            standalone_sr, standalone_sr <= criteria.sharpe_implausibility_ceiling),
        "risk_budget": (b, b <= criteria.max_risk_budget + 1e-12),
        "tail_overlap": (overlap_fraction,
                         overlap_fraction <= criteria.max_tail_overlap + 1e-12),
    }
    failed = [k for k, (_v, ok) in checks.items() if not ok]
    passed = not failed

    warnings_: List[str] = []
    if vol_floor_bind_frac > VOL_FLOOR_WARN_FRAC:
        warnings_.append(
            "vol-target unreliable: candidate near-constant vol (2% trailing-vol "
            f"floor binds on {vol_floor_bind_frac:.0%} of evaluated days)")

    if passed:
        verdict = ("TRACK_B_PASS - candidate improves the combined book at a "
                   f"{b:.0%} risk budget; eligible for PAPER-level book inclusion "
                   "ONLY (Track B never auto-promotes to CAPITAL - owner sign-off "
                   "+ tail budget required)")
    else:
        verdict = "TRACK_B_FAIL - failed: " + ", ".join(failed)
        if "standalone_sr_plausible" in failed:
            verdict += ("; implausible standalone Sharpe (look-ahead/degenerate) - "
                        "same ceiling Track A enforces (SHARPE_IMPLAUSIBILITY_CEILING)")

    return BookGateResult(
        candidate_label=candidate_label,
        risk_budget=b,
        n_days=n_days,
        window_start=ev.index[0].date().isoformat(),
        window_end=ev.index[-1].date().isoformat(),
        without_book={k: float(v) for k, v in without_m.items()},
        with_book={k: float(v) for k, v in with_m.items()},
        sharpe_delta=sharpe_delta,
        calmar_delta=calmar_delta,
        max_dd_delta=dd_delta,
        corr_to_book=corr,
        standalone_vt_sharpe=standalone_sr,
        n_tail=n_tail,
        tail_overlap=overlap,
        tail_overlap_fraction=overlap_fraction,
        base_worst_dates=base_worst_dates,
        vol_floor_bind_frac=vol_floor_bind_frac,
        checks=checks,
        failed_criteria=failed,
        warnings=warnings_,
        passed=passed,
        verdict=verdict,
        criteria=criteria,
    )


# ==============================================================================
# ASCII report (mirrors the gate_calibration OC-table style)
# ==============================================================================
def format_report(result: BookGateResult) -> str:
    """Deterministic ASCII report for a BookGateResult."""
    c = result.criteria
    bar = "=" * 78
    sub = "-" * 78
    lines = [
        bar,
        "  TRACK B (BOOK-DELTA) ACCEPTANCE GATE - Alpha-v6 Phase 0",
        bar,
        f"  candidate: {result.candidate_label}   risk budget: {result.risk_budget:.1%}",
    ]
    if not result.without_book:           # degenerate short-circuit: no books built
        lines += [
            f"  window:    {result.window_start} -> {result.window_end} "
            f"({result.n_days} aligned days)",
            sub,
            "  VERDICT: FAIL",
            f"  {result.verdict}",
            bar,
        ]
        return "\n".join(lines)

    lines += [
        f"  window:    {result.window_start} -> {result.window_end} "
        f"({result.n_days} evaluated days, post-warmup post-combine)",
        sub,
        f"  {'book metric':<16} {'without':>10} {'with':>10} {'delta':>10}",
    ]
    wo, wi = result.without_book, result.with_book
    rows = [
        ("Sharpe", wo["sharpe"], wi["sharpe"], result.sharpe_delta),
        ("CAGR", wo["cagr"], wi["cagr"], wi["cagr"] - wo["cagr"]),
        ("ann vol", wo["ann_vol"], wi["ann_vol"], wi["ann_vol"] - wo["ann_vol"]),
        ("maxDD", wo["max_drawdown"], wi["max_drawdown"], result.max_dd_delta),
        ("Calmar", wo["calmar"], wi["calmar"], result.calmar_delta),
    ]
    for name, a, bb, d in rows:
        lines.append(f"  {name:<16} {a:>+10.3f} {bb:>+10.3f} {d:>+10.3f}")
    lines += [
        sub,
        f"  {'criterion':<24} {'measured':>10} {'threshold':>26} {'ok':>5}",
    ]
    thr = {
        "sharpe_delta": f">= {c.min_sharpe_delta:+.2f}",
        "calmar_delta": f">= {c.min_calmar_delta:+.2f}",
        "max_dd_delta": f">= {-c.max_dd_delta:+.2f} (no deeper DD)",
        "corr_to_book": f"< {c.max_corr:.2f}",
        "standalone_vt_sharpe": f"> {c.min_standalone_sr:.2f}",
        "standalone_sr_plausible": f"<= {c.sharpe_implausibility_ceiling:.1f} (SR ceiling)",
        "risk_budget": f"<= {c.max_risk_budget:.2f}",
        "tail_overlap": f"<= {c.max_tail_overlap:.2f} (worst-day overlap)",
    }
    for name, (val, ok) in result.checks.items():
        lines.append(f"  {name:<24} {val:>+10.4f} {thr.get(name, ''):>26} "
                     f"{'OK' if ok else 'FAIL':>5}")
    lines += [
        sub,
        f"  tail overlap (registered): {result.tail_overlap}/{result.n_tail} of the "
        "candidate's worst days coincide",
        f"              with the book's worst {result.n_tail} days "
        f"(n_tail = max(3, floor({c.joint_tail_pctl:.0%} of {result.n_days}))) -> "
        f"{result.tail_overlap_fraction:.2f}",
        f"  vol-target: 2% trailing-vol floor bound on "
        f"{result.vol_floor_bind_frac:.1%} of evaluated days",
    ]
    for w in result.warnings:
        lines.append(f"  WARNING: {w}")
    lines += [
        sub,
        f"  VERDICT: {'PASS' if result.passed else 'FAIL'}",
        f"  {result.verdict}",
        "  NOTE: Track B gates PAPER-level book inclusion only. It NEVER",
        "  auto-promotes to CAPITAL (explicit owner sign-off + tail budget required).",
        bar,
    ]
    return "\n".join(lines)
