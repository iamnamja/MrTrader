"""
cpcv.py — Combinatorial Purged Cross-Validation (López de Prado, AFML Chapter 12).

CPCV with k groups and paths test-groups per combination gives C(k, paths) independent
test paths through the data. Each path tests on `paths` of k groups and trains on the
remaining k-paths groups.

Standard 3-fold expanding WF:
  - 3 test periods, right-anchored
  - Single Sharpe point estimate
  - Folds are NOT independent (fold 2 train ⊃ fold 1 test)

CPCV with k=6, paths=2:
  - C(6,2) = 15 independent test combinations
  - Reports Sharpe distribution: mean, std, P5/P95
  - Much higher statistical power; DSR uses distribution shape

Usage:
    from scripts.walkforward.cpcv import run_cpcv
    result = run_cpcv(strategy, engine, n_folds=6, n_paths=2, ...)
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from scripts.walkforward.gates import (
    SHARPE_GATE, MIN_FOLD_SHARPE, MIN_PROFIT_FACTOR, MIN_CALMAR,
    MIN_WORST_REGIME_SHARPE, N_TRIALS_TESTED, deflated_sharpe_ratio,
    MAX_PF_FOR_AVG, CAL_NO_DD_SENTINEL, CAL_TOTAL_LOSS_SENTINEL,
    MIN_TRADES_FOR_CAL_SENTINEL, MIN_ACTIVE_FOLDS_FOR_GATE,
)

logger = logging.getLogger(__name__)


@dataclass
class CPCVResult:
    """Summary of all CPCV path results."""
    model_type: str
    n_folds: int
    n_paths: int
    path_sharpes: List[float] = field(default_factory=list)
    path_profit_factors: List[float] = field(default_factory=list)
    path_calmars: List[float] = field(default_factory=list)
    # Trading-day observation count per path; used for correct DSR n_obs.
    path_n_obs: List[int] = field(default_factory=list)
    # True when the OOS guard was bypassed with allow_in_sample=True.
    in_sample_override: bool = False
    # True only when per-fold retraining was performed (genuine out-of-sample).
    # Set by run_cpcv from strategy.per_fold_retrain. When False the run scored
    # one frozen model across all folds (generalization test, cannot promote when
    # REQUIRE_TRUE_WF_FOR_PROMOTION=True). Mirrors WalkForwardReport.
    is_true_walkforward: bool = False
    # BUG-2: track skipped folds (fold 0 can't be tested — no prior training data).
    # n_skipped > 0 is normal for CPCV; a large fraction indicates a design problem.
    n_skipped: int = 0
    # C1: worst per-regime Sharpe across all paths/folds; None if not populated.
    # Mirrors WalkForwardReport.worst_regime_sharpe so regime gate parity is maintained.
    worst_regime_sharpe: Optional[float] = None
    # CRITICAL-2: per-path deployment diagnostics (mirror WalkForwardReport).
    path_deployments: List[float] = field(default_factory=list)
    path_deployment_adj_sharpes: List[float] = field(default_factory=list)
    # Phase 3 (HIGH-3): when True, gate_passed() requires path_sharpe_tstat >=
    # CPCV_MIN_TSTAT. Off by default — the t-stat is reported and warned but does
    # not block. Enable once baseline t-stat data is collected.
    require_tstat_gate: bool = False

    @property
    def n_combinations(self) -> int:
        return len(self.path_sharpes)

    @property
    def mean_sharpe(self) -> float:
        return float(np.mean(self.path_sharpes)) if self.path_sharpes else 0.0

    @property
    def std_sharpe(self) -> float:
        return float(np.std(self.path_sharpes)) if self.path_sharpes else 0.0

    @property
    def p5_sharpe(self) -> float:
        return float(np.percentile(self.path_sharpes, 5)) if self.path_sharpes else 0.0

    @property
    def p95_sharpe(self) -> float:
        return float(np.percentile(self.path_sharpes, 95)) if self.path_sharpes else 0.0

    @property
    def pct_positive(self) -> float:
        return float(np.mean([s > 0 for s in self.path_sharpes])) if self.path_sharpes else 0.0

    @property
    def path_sharpe_tstat(self) -> float:
        """t = mean_path_sharpe / (std_path_sharpe / sqrt(N_eff)).

        N_eff = n_folds (NOT n_combinations). The C(k,p) paths reuse k folds and
        are strongly positively correlated. Using n_combinations would overstate
        significance. N_eff = n_folds is the conservative, defensible choice.
        Returns 0.0 if < 2 paths or zero dispersion.
        """
        import math
        if len(self.path_sharpes) < 2:
            return 0.0
        sd = self.std_sharpe
        if sd <= 1e-12:
            return 0.0
        n_eff = max(self.n_folds, 1)
        return self.mean_sharpe / (sd / math.sqrt(n_eff))

    @property
    def avg_profit_factor(self) -> float:
        # Cap at MAX_PF_FOR_AVG before averaging — mirrors WalkForwardReport.avg_profit_factor.
        # Include zero-PF paths (unfilled paths should pull the mean down, not be ignored).
        pfs = [min(p, MAX_PF_FOR_AVG) for p in self.path_profit_factors]
        return float(np.mean(pfs)) if pfs else 0.0

    @property
    def avg_calmar(self) -> float:
        # Drop zeros (uncomputed) but include negatives (bad paths) and sentinel
        # values from no-DD profitable paths — mirrors WalkForwardReport.avg_calmar.
        cals = [c for c in self.path_calmars if c != 0]
        return float(np.mean(cals)) if cals else 0.0

    @property
    def total_obs(self) -> int:
        """Raw sum of trading-day observations across all paths.

        NOTE: each unique trading day appears in C(n_folds-1, n_paths-1) paths.
        Use _dsr_n_obs() (not this property) for DSR to avoid inflating T.
        """
        return sum(self.path_n_obs) if self.path_n_obs else 0

    @property
    def unique_obs(self) -> int:
        """Unique trading-day observations, corrected for combinatorial multiplicity.

        BUG-1 fix: in C(k, paths) CPCV, every trading day appears in
        C(k-1, paths-1) paths. total_obs over-counts unique days by that factor,
        which inflates the DSR denominator and makes the gate easier to pass.
        Dividing by C(n_folds-1, n_paths-1) recovers the true unique-day count.

        Example: k=6, paths=2 → C(5,1)=5 → each day appears in 5 paths.
        total_obs = 5 × unique_days; this property returns unique_days.
        """
        import math
        multiplicity = math.comb(max(self.n_folds - 1, 0), max(self.n_paths - 1, 0))
        return max(self.total_obs // max(multiplicity, 1), 1) if self.total_obs > 0 else 0

    def _dsr_n_obs(self) -> int:
        """Unique trading-day count for DSR; fall back to n_combinations for legacy results."""
        return self.unique_obs if self.unique_obs > 0 else max(self.n_combinations, 1)

    # ── CRITICAL-1: DSR ceiling / human-review flags (mirror WalkForwardReport) ──
    def dsr_saturated(self, dsr_n: int = N_TRIALS_TESTED) -> bool:
        """True when DSR p > DSR_SATURATION_P — gate is non-binding at this Sharpe."""
        from app.ml.retrain_config import DSR_SATURATION_P
        _, p = deflated_sharpe_ratio(self.mean_sharpe, dsr_n, self._dsr_n_obs())
        return p > DSR_SATURATION_P

    def requires_human_review(self) -> bool:
        """True when mean_sharpe > SHARPE_IMPLAUSIBILITY_CEILING.
        Does NOT affect gate_passed() — must be checked by the promotion runner."""
        from app.ml.retrain_config import SHARPE_IMPLAUSIBILITY_CEILING
        return self.mean_sharpe > SHARPE_IMPLAUSIBILITY_CEILING

    # ── CRITICAL-2: deployment diagnostics (mirror WalkForwardReport) ──
    @property
    def avg_deployment_pct(self) -> float:
        """Mean capital deployment across paths."""
        return float(np.mean(self.path_deployments)) if self.path_deployments else 0.0

    @property
    def avg_deployment_adjusted_sharpe(self) -> float:
        return float(np.mean(self.path_deployment_adj_sharpes)) if self.path_deployment_adj_sharpes else 0.0

    @property
    def low_deployment(self) -> bool:
        from app.ml.retrain_config import MIN_DEPLOYMENT_PCT_WARN
        return self.avg_deployment_pct < MIN_DEPLOYMENT_PCT_WARN

    # Paper-trade gate thresholds — mirror WalkForwardReport's relaxed values
    PAPER_SHARPE_GATE: float = 0.50
    PAPER_MIN_FOLD_SHARPE: float = -0.40

    def gate_passed(self, dsr_n: int = N_TRIALS_TESTED, paper_gate: bool = False) -> bool:
        # In-sample runs (allow_in_sample override) can never promote past gates.
        if self.in_sample_override:
            return False
        # Frozen-mode (not true per-fold) runs cannot promote when the project-wide
        # flag is set. Default False during Phase 1 rollout — flip to True in Phase 3.
        from app.ml.retrain_config import REQUIRE_TRUE_WF_FOR_PROMOTION
        if REQUIRE_TRUE_WF_FOR_PROMOTION and not self.is_true_walkforward:
            return False
        _, dsr_p = deflated_sharpe_ratio(
            self.mean_sharpe, dsr_n, self._dsr_n_obs()
        )
        sharpe_gate = self.PAPER_SHARPE_GATE if paper_gate else SHARPE_GATE
        min_fold_gate = self.PAPER_MIN_FOLD_SHARPE if paper_gate else MIN_FOLD_SHARPE
        # C11-2: require a minimum number of surviving paths before any gate fires.
        # Without this, a single surviving path can pass mean_sharpe/p5/pct_positive.
        if len(self.path_sharpes) < MIN_ACTIVE_FOLDS_FOR_GATE:
            return False
        # CR-1/C8-9: close "avg==0 → pass" bypass — require MIN_ACTIVE_FOLDS_FOR_GATE paths.
        # C8-6: paper_gate bypasses PF/Calmar (matching WalkForwardReport behaviour).
        n_pf_active = sum(1 for p in self.path_profit_factors if p > 0)
        n_cal_active = sum(1 for c in self.path_calmars if c != 0)
        pf_ok = (paper_gate
                 or n_pf_active < MIN_ACTIVE_FOLDS_FOR_GATE
                 or self.avg_profit_factor >= MIN_PROFIT_FACTOR)
        cal_ok = (paper_gate
                  or n_cal_active < MIN_ACTIVE_FOLDS_FOR_GATE
                  or self.avg_calmar >= MIN_CALMAR)
        wrs = self.worst_regime_sharpe
        # Phase 2: None → fail unless ALLOW_NO_REGIME_GATE=True (mirror WalkForwardReport).
        from app.ml.retrain_config import ALLOW_NO_REGIME_GATE
        import logging as _log
        if wrs is None:
            if ALLOW_NO_REGIME_GATE:
                _log.getLogger(__name__).warning(
                    "CPCVResult: worst_regime_sharpe=None, gate bypassed "
                    "(ALLOW_NO_REGIME_GATE=True). Regime gate NOT enforced."
                )
                regime_ok = True
            else:
                _log.getLogger(__name__).error(
                    "CPCVResult: worst_regime_sharpe=None — regime data insufficient. "
                    "GATE FAILING (set ALLOW_NO_REGIME_GATE=True to bypass). "
                    "Ensure fetch_data loaded _global_regime_map successfully."
                )
                regime_ok = False
        else:
            regime_ok = wrs >= MIN_WORST_REGIME_SHARPE
        # Phase 3 (HIGH-3): CPCV path-Sharpe t-stat significance.
        from app.ml.retrain_config import CPCV_MIN_TSTAT
        tstat = self.path_sharpe_tstat
        tstat_ok = tstat >= CPCV_MIN_TSTAT
        if not tstat_ok:
            import logging as _tlog
            _tlog.getLogger(__name__).warning(
                "CPCV path t-stat %.2f < CPCV_MIN_TSTAT %.2f (N_eff=%d folds, NOT %d paths). "
                "%s",
                tstat, CPCV_MIN_TSTAT, self.n_folds, len(self.path_sharpes),
                "GATING (require_tstat_gate=True)" if self.require_tstat_gate
                else "NOT gating (require_tstat_gate=False)",
            )
        return (
            self.mean_sharpe >= sharpe_gate
            and self.p5_sharpe >= min_fold_gate
            and self.pct_positive >= 0.75
            and dsr_p > 0.95
            and pf_ok
            and cal_ok
            and regime_ok
            and (tstat_ok or not self.require_tstat_gate)
        )

    def gate_detail(self, dsr_n: int = N_TRIALS_TESTED, paper_gate: bool = False) -> dict:
        _, dsr_p = deflated_sharpe_ratio(
            self.mean_sharpe, dsr_n, self._dsr_n_obs()
        )
        sharpe_gate = self.PAPER_SHARPE_GATE if paper_gate else SHARPE_GATE
        min_fold_gate = self.PAPER_MIN_FOLD_SHARPE if paper_gate else MIN_FOLD_SHARPE
        n_pf_active = sum(1 for p in self.path_profit_factors if p > 0)
        n_cal_active = sum(1 for c in self.path_calmars if c != 0)
        n_paths = len(self.path_sharpes)
        enough_paths = n_paths >= MIN_ACTIVE_FOLDS_FOR_GATE
        # Phase 2: regime None → fail unless ALLOW_NO_REGIME_GATE (mirror WalkForwardReport).
        from app.ml.retrain_config import ALLOW_NO_REGIME_GATE as _ALLOW_NRG
        _wrs = self.worst_regime_sharpe
        _wrs_ok = (_wrs is not None and _wrs >= MIN_WORST_REGIME_SHARPE) or (_wrs is None and _ALLOW_NRG)
        # Phase 3 (HIGH-3): path-Sharpe t-stat. ok=True when not gating or above threshold.
        from app.ml.retrain_config import CPCV_MIN_TSTAT
        _tstat = self.path_sharpe_tstat
        _tstat_ok = (_tstat >= CPCV_MIN_TSTAT) or not self.require_tstat_gate
        # C13-1: gate PF/Calmar on enough_paths too, matching gate_passed() early-return.
        # Without this, gate_detail could show PF/Calmar "OK" while gate_passed returns False.
        return {
            "n_paths": (n_paths, enough_paths),
            "mean_sharpe": (self.mean_sharpe, enough_paths and self.mean_sharpe >= sharpe_gate),
            "p5_sharpe": (self.p5_sharpe, enough_paths and self.p5_sharpe >= min_fold_gate),
            "pct_positive": (self.pct_positive, enough_paths and self.pct_positive >= 0.75),
            "dsr_p": (dsr_p, enough_paths and dsr_p > 0.95),
            "avg_profit_factor": (self.avg_profit_factor,
                                  enough_paths and (
                                      paper_gate
                                      or n_pf_active < MIN_ACTIVE_FOLDS_FOR_GATE
                                      or self.avg_profit_factor >= MIN_PROFIT_FACTOR)),
            "avg_calmar": (self.avg_calmar,
                           enough_paths and (
                               paper_gate
                               or n_cal_active < MIN_ACTIVE_FOLDS_FOR_GATE
                               or self.avg_calmar >= MIN_CALMAR)),
            "worst_regime_sharpe": (self.worst_regime_sharpe, enough_paths and _wrs_ok),
            "path_sharpe_tstat": (_tstat, _tstat_ok),
            # CRITICAL-1/2: informational keys — NOT part of gate_passed() boolean AND.
            # ok=True when NOT triggered (appear in failed-keys list only when fired).
            "human_review_required": (self.mean_sharpe, not self.requires_human_review()),
            "low_deployment_warning": (self.avg_deployment_pct, not self.low_deployment),
        }

    def print(self) -> None:
        from scripts.walkforward.reports import _ok, _err, _header
        _header(f"CPCV Report - {self.model_type.upper()} "
                f"C({self.n_folds},{self.n_paths})={self.n_combinations} paths")
        if self.in_sample_override:
            print("  *** IN-SAMPLE RUN (--allow-in-sample): results cannot promote past gates ***")
        print(f"  Mean Sharpe:  {self.mean_sharpe:+.3f}  (gate: > {SHARPE_GATE})  "
              f"{'OK' if self.mean_sharpe >= SHARPE_GATE else 'FAIL'}")
        print(f"  Std Sharpe:   {self.std_sharpe:.3f}")
        print(f"  P5 Sharpe:    {self.p5_sharpe:+.3f}  (gate: > {MIN_FOLD_SHARPE})  "
              f"{'OK' if self.p5_sharpe >= MIN_FOLD_SHARPE else 'FAIL'}")
        print(f"  P95 Sharpe:   {self.p95_sharpe:+.3f}")
        print(f"  % positive:   {self.pct_positive:.1%}  (gate: >= 75%)  "
              f"{'OK' if self.pct_positive >= 0.75 else 'FAIL'}")
        from app.ml.retrain_config import CPCV_MIN_TSTAT
        tstat = self.path_sharpe_tstat
        tstat_ok = tstat >= CPCV_MIN_TSTAT
        print(f"  Path Sharpe t-stat: {tstat:+.2f}  "
              f"(gate: > {CPCV_MIN_TSTAT}; N_eff={self.n_folds} folds, NOT {len(self.path_sharpes)} paths)  "
              f"{'OK' if tstat_ok else 'WARN'}"
              f"{'  [GATING]' if self.require_tstat_gate and not tstat_ok else ''}")
        print(f"    NOTE: N_eff={self.n_folds} (folds), NOT {len(self.path_sharpes)} (paths). "
              f"C(k,p) paths reuse k folds and are correlated.")
        _, dsr_p = deflated_sharpe_ratio(self.mean_sharpe, N_TRIALS_TESTED, self._dsr_n_obs())
        print(f"  DSR p:        {dsr_p:.3f}  (gate: > 0.95)  {'OK' if dsr_p > 0.95 else 'FAIL'}")
        from app.ml.retrain_config import (
            DSR_SATURATION_P, SHARPE_IMPLAUSIBILITY_CEILING, MIN_DEPLOYMENT_PCT_WARN,
        )
        if dsr_p > DSR_SATURATION_P:
            print(f"  *** DSR SATURATED (p > {DSR_SATURATION_P}) — NO selection-bias "
                  f"screening at this Sharpe; rely on deployment-adjusted Sharpe ***")
        if self.requires_human_review():
            print(f"  *** HUMAN REVIEW REQUIRED: mean Sharpe {self.mean_sharpe:.3f} > "
                  f"ceiling {SHARPE_IMPLAUSIBILITY_CEILING} — verify no deployment artifact ***")
        if self.avg_deployment_pct > 0:
            print(f"  Avg capital deployed:    {self.avg_deployment_pct:.1%}")
            print(f"  Deployment-adj Sharpe:   {self.avg_deployment_adjusted_sharpe:+.3f}  "
                  f"(diagnostic: scaled to 100% deployment)")
            if self.low_deployment:
                print(f"  *** LOW DEPLOYMENT ({self.avg_deployment_pct:.1%} < "
                      f"{MIN_DEPLOYMENT_PCT_WARN:.0%}) — raw Sharpe not comparable to "
                      f"fully-invested benchmarks ***")
        if self.avg_profit_factor > 0:
            print(f"  Avg PF:       {self.avg_profit_factor:.3f}  "
                  f"(gate: > {MIN_PROFIT_FACTOR})  "
                  f"{'OK' if self.avg_profit_factor >= MIN_PROFIT_FACTOR else 'FAIL'}")
        if self.avg_calmar != 0:
            print(f"  Avg Calmar:   {self.avg_calmar:.3f}  "
                  f"(gate: > {MIN_CALMAR})  "
                  f"{'OK' if self.avg_calmar >= MIN_CALMAR else 'FAIL'}")
        print()
        if self.gate_passed():
            _ok(f"CPCV GATE PASSED - mean Sharpe {self.mean_sharpe:.3f}, "
                f"P5={self.p5_sharpe:.3f}, pct_pos={self.pct_positive:.1%}")
        else:
            detail = self.gate_detail()
            failed = [k for k, (v, ok) in detail.items() if not ok]
            _err(f"CPCV GATE NOT MET - failed: {', '.join(failed)}")


def run_cpcv(
    strategy,
    purge_days: int,
    embargo_days: Optional[int],
    n_folds: int = 6,
    n_paths: int = 2,
    total_years: Optional[int] = None,
    total_days: Optional[int] = None,
    train_years: Optional[int] = None,
    allow_sacred_holdout: bool = False,  # P0
) -> CPCVResult:
    """
    Run Combinatorial Purged Cross-Validation.

    strategy must already have data fetched (strategy.fetch_data called).
    """
    from scripts.walkforward.engine import FoldEngine
    from app.ml.retrain_config import assert_no_sacred_holdout as _assert_holdout

    # P0: hard guard against using sacred holdout data in CPCV runs.
    from datetime import datetime as _dt_now
    _assert_holdout(
        _dt_now.now().date(),
        allow_sacred_holdout=allow_sacred_holdout,
        context="cpcv.run_cpcv",
    )

    _embargo = embargo_days if embargo_days is not None else purge_days
    engine = FoldEngine(strategy=strategy, purge_days=purge_days, embargo_days=_embargo,
                        parallel=False)

    # Build the k fold segments
    if total_years is not None:
        from datetime import datetime, timedelta
        # IM-6: anchor end_all to retrain_as_of() for reproducibility — datetime.now()
        # drifts across reruns so two consecutive runs produce different fold boundaries,
        # enabling "temporal multiple testing" (re-run until a favourable boundary).
        try:
            from app.ml.retrain_config import retrain_as_of
            end_all = datetime.combine(retrain_as_of(), datetime.min.time())
        except Exception:
            end_all = datetime.now()
        start_all = end_all - timedelta(days=total_years * 365 + 30)
        all_boundaries = engine._build_calendar_folds(
            n_folds, start_all, end_all, total_years, train_years
        )
    else:
        all_days = getattr(strategy, "all_days_sorted", [])
        all_boundaries = engine._build_trading_day_folds(n_folds, all_days)

    if len(all_boundaries) < n_folds:
        logger.warning("CPCV: only %d boundaries available, need %d", len(all_boundaries), n_folds)
        return CPCVResult(
            model_type=getattr(strategy, "model_type", "unknown"),
            n_folds=n_folds, n_paths=n_paths,
        )

    # MEDIUM-3 / H-1: data span gate — fail early when data is too short to
    # construct non-degenerate folds.
    # Intraday (total_years is None): gate on len(all_days_sorted).
    # Swing (total_years is not None): gate on distinct dates in symbols_data
    #   (the calendar-fold path does not populate all_days_sorted, so we count
    #   distinct trading days from the loaded bars instead).
    from scripts.walkforward.gates import DataSpanError
    from app.ml.retrain_config import MIN_DATA_SPAN_TRADING_DAYS, ENFORCE_MIN_DATA_SPAN

    if total_years is None:
        _all_days_check = getattr(strategy, "all_days_sorted", None)
        if _all_days_check is not None:
            _n_span = len(_all_days_check)
            if _n_span < MIN_DATA_SPAN_TRADING_DAYS:
                _src = getattr(strategy, "data_source", "unknown")
                msg = (f"CPCV DATA SPAN TOO SHORT: {_n_span} days < {MIN_DATA_SPAN_TRADING_DAYS} "
                       f"[source={_src}]. Re-fetch data.")
                if ENFORCE_MIN_DATA_SPAN:
                    raise DataSpanError(msg)
                logger.warning(msg)
    else:
        # Swing calendar-fold path: count distinct dates from symbols_data (H-1 fix).
        import pandas as _pd
        _sd = getattr(strategy, "symbols_data", None) or {}
        if _sd:
            _all_dates_set: set = set()
            for _df in _sd.values():
                if hasattr(_df, "index"):
                    for _d in _pd.DatetimeIndex(_df.index).date:
                        _all_dates_set.add(_d)
            _n_swing_span = len(_all_dates_set)
            if _n_swing_span > 0 and _n_swing_span < MIN_DATA_SPAN_TRADING_DAYS:
                msg = (f"CPCV (swing) DATA SPAN TOO SHORT: {_n_swing_span} distinct trading days "
                       f"< MIN_DATA_SPAN_TRADING_DAYS={MIN_DATA_SPAN_TRADING_DAYS}. Re-fetch data.")
                if ENFORCE_MIN_DATA_SPAN:
                    raise DataSpanError(msg)
                logger.warning(msg)

    # C14-1: intraday (trading-day) coverage guard. The fold boundaries are built
    # from strategy.all_days_sorted, but the per-fold TRAIN matrix is built from
    # strategy.symbols_data (build_train_matrix_for_window derives train_days from
    # the loaded 5-min bars). If all_days_sorted spans EARLIER than the actual
    # 5-min bars, the first folds get train_start before any bars exist →
    # train_days={} → empty matrix → "no training samples" (the 2nd per-fold
    # empty-matrix bug). Fail loudly here with the concrete span numbers instead
    # of emitting N silent per-fold warnings.
    if total_years is None:
        import pandas as _pd
        _sd = getattr(strategy, "symbols_data", None) or {}
        _data_dates: set = set()
        for _df in _sd.values():
            if hasattr(_df, "index"):
                for _d in _pd.DatetimeIndex(_df.index).date:
                    _data_dates.add(_d)
        if _data_dates:
            _data_start = min(_data_dates)
            _fold_start = all_boundaries[0][0]
            _fold_start = _fold_start.date() if hasattr(_fold_start, "date") else _fold_start
            if _data_start > _fold_start:
                from app.ml.retrain_config import ENFORCE_MIN_DATA_SPAN
                msg = (
                    f"CPCV (intraday) DATA-SPAN MISMATCH: fold train windows start "
                    f"{_fold_start} but 5-min bars start {_data_start}. Folds before "
                    f"the data start produce EMPTY train matrices ('no training "
                    f"samples'). all_days_sorted must be clamped to the loaded bars "
                    f"(IntradayStrategy.fetch_data). Re-fetch with a consistent window."
                )
                if ENFORCE_MIN_DATA_SPAN:
                    raise DataSpanError(msg)
                logger.warning(msg)

    # C11-10: verify strategy data range covers the constructed fold window so that
    # a caller who fetched a shorter range does not silently produce empty-fold results.
    if total_years is not None:
        _sd = getattr(strategy, "symbols_data", None) or {}
        if _sd:
            import pandas as pd
            _all_dates = []
            for _df in _sd.values():
                if hasattr(_df, "index"):
                    _all_dates.extend(_df.index.tolist())
            if _all_dates:
                _data_start = min(_all_dates)
                _data_start = _data_start.date() if hasattr(_data_start, "date") else _data_start
                _fold_start = all_boundaries[0][0]
                _fold_start = _fold_start.date() if hasattr(_fold_start, "date") else _fold_start
                if _data_start > _fold_start:
                    logger.warning(
                        "C11-10: strategy data starts %s but first fold train-start is %s. "
                        "Folds before data start will produce empty results. "
                        "Re-fetch with a longer window.",
                        _data_start, _fold_start,
                    )

    # Per-fold mode plumbing: the strategy retrains a fresh model inside each
    # run_fold and runs the per-fold OOS guard there, so the global single-cutoff
    # guard below is meaningless. Pass purge/embargo through for the per-fold guard.
    _per_fold = bool(getattr(strategy, "per_fold_retrain", False))
    strategy._purge_days = purge_days
    strategy._embargo_days = _embargo

    # OOS-guard: every test fold must start strictly after the model's training cutoff.
    _allow_in_sample = getattr(strategy, "allow_in_sample", False)
    if not _per_fold:
        from scripts.walkforward.oos_guard import assert_model_oos
        _trained_through = getattr(getattr(strategy, "model", None), "trained_through", None)
        _model_label = (
            f"{getattr(strategy, 'model_type', 'unknown')} "
            f"v{getattr(strategy, 'version', '?')}"
        )
        # C12-2: pass trading_day_set for intraday so purge_days is in trading days.
        _all_days = getattr(strategy, "all_days_sorted", None)
        _trading_day_set = set(_all_days) if _all_days else None
        assert_model_oos(
            trained_through=_trained_through,
            fold_boundaries=all_boundaries,
            purge_days=purge_days,
            model_label=_model_label,
            allow_in_sample=_allow_in_sample,
            trading_day_set=_trading_day_set,
        )

    result = CPCVResult(
        model_type=getattr(strategy, "model_type", "unknown"),
        n_folds=n_folds,
        n_paths=n_paths,
        in_sample_override=_allow_in_sample,
        is_true_walkforward=_per_fold,
    )

    # Generate C(k, paths) combinations
    fold_indices = list(range(len(all_boundaries)))
    combinations = list(itertools.combinations(fold_indices, n_paths))
    logger.info("CPCV: %d combinations (C(%d,%d))", len(combinations), n_folds, n_paths)

    # IM-3/C-2: track per-regime sharpes across all folds so we aggregate
    # regime-by-regime (mean across folds) then take the min over regimes,
    # avoiding the raw-min-over-all-pairs noise bias.
    all_regime_sharpes_by_regime: dict = {}  # regime_name -> [sharpe, ...]

    for combo_idx, test_indices in enumerate(combinations):
        train_indices = [i for i in fold_indices if i not in test_indices]
        if not train_indices:
            continue

        # Test on the selected folds
        combo_sharpes = []
        combo_pfs = []
        combo_cals = []
        combo_n_obs: List[int] = []
        combo_deps: List[float] = []
        combo_dep_adj: List[float] = []
        # C8-5: accumulate per-regime within this combo to average before global append,
        # so each combo contributes equally (not proportional to its fold count).
        combo_regime_by_name: dict = {}  # regime -> [sharpe, ...]

        for ti in test_indices:
            tr_start, tr_end, te_start, te_end = all_boundaries[ti]
            # WF deep-review pass 4 fix: previously used
            #   best_train = sorted(train_indices)[-1]
            #   real_tr_end = all_boundaries[best_train][1]
            # which selected the LATEST train fold index regardless of whether it
            # came before or after the test fold. For test_indices like (0, 1) it
            # extended training all the way to the end of fold 5 — a massive
            # look-ahead leak (training data contained future folds, including
            # other tests in the same combo).
            #
            # Correct CPCV requires training only on fold indices strictly less
            # than this test fold's index, then leaving a purge gap before
            # te_start. Other train folds at indices > ti are dropped for this
            # particular test fold (they remain usable as training when the
            # paired test fold in the combo is processed).
            prior_train = [j for j in train_indices if j < ti]
            if not prior_train:
                # BUG-2: no causal training history before this test fold (ti == 0).
                # Skipping is the only causally valid option — training on folds AFTER
                # ti would be look-ahead. Count skips for the completeness metric;
                # a large n_skipped fraction would indicate CPCV is biased toward
                # later (potentially stronger) regimes.
                result.n_skipped += 1
                continue
            best_train = max(prior_train)
            tr_end_candidate = all_boundaries[best_train][1]
            # Apply purge gap: train must end at least `purge_days` calendar days
            # before te_start. The engine's per-fold purge is bypassed here because
            # we override train_end manually, so enforce the gap explicitly.
            from datetime import timedelta as _td
            max_tr_end = te_start - _td(days=max(purge_days, 0) + 1)
            real_tr_end = min(tr_end_candidate, max_tr_end)
            # BUG-3 fix: honor train_years for rolling window CPCV.
            # Previously always used all_boundaries[0][0] (expanding from start).
            # With train_years set, roll the train start forward so only the most
            # recent train_years of data are used — matching WF rolling behavior.
            if train_years:
                rolling_start = real_tr_end - _td(days=int(train_years * 365))
                real_tr_start = max(all_boundaries[0][0], rolling_start)
            else:
                real_tr_start = all_boundaries[0][0]  # expanding: always from start
            if real_tr_end <= real_tr_start:
                result.n_skipped += 1
                continue

            # BUG-23 fix: with rolling train_years, the training window
            # [real_tr_start, real_tr_end] can overlap a prior test fold's
            # [te_start, te_end] — training on data that was tested elsewhere
            # in this combination is a form of in-sample contamination.
            # Purge_days is applied symmetrically: enforce it AFTER each prior
            # test fold's end too (not just before this test's start), preventing
            # label-horizon leakage where training data starts within purge_days
            # of a prior test fold ending.
            prior_test_folds = [j for j in test_indices if j < ti]
            # C12-1: use _embargo (not purge_days) for the post-test buffer.
            # embargo_days >= purge_days is the typical setup; using the smaller
            # purge_days would allow training windows within the embargo zone.
            _embargo_td = _td(days=max(_embargo, 0))
            overlap = any(
                real_tr_start < all_boundaries[j][3] + _embargo_td  # te_end + embargo
                and real_tr_end > all_boundaries[j][2]               # te_start of prior test
                for j in prior_test_folds
            )
            if overlap:
                logger.debug(
                    "CPCV overlap guard: combo %d ti=%d rolling window [%s, %s] "
                    "overlaps prior test fold (including %d-day post-test embargo) — skipping",
                    combo_idx, ti, real_tr_start, real_tr_end, _embargo,
                )
                result.n_skipped += 1
                continue

            try:
                fold = strategy.run_fold(
                    combo_idx * len(all_boundaries) + ti + 1,
                    n_folds,
                    real_tr_start,
                    real_tr_end,
                    te_start,
                    te_end,
                )
                combo_sharpes.append(fold.sharpe)
                combo_pfs.append(fold.profit_factor)
                combo_cals.append(fold.calmar_ratio)
                combo_n_obs.append(getattr(fold, "n_obs", 0) or 0)
                combo_deps.append(getattr(fold, "avg_capital_deployed_pct", 0.0) or 0.0)
                combo_dep_adj.append(getattr(fold, "deployment_adjusted_sharpe", 0.0) or 0.0)
                for regime, sh in getattr(fold, "regime_sharpes", {}).items():
                    combo_regime_by_name.setdefault(regime, []).append(sh)
            except Exception as exc:
                logger.warning("CPCV combo %d fold %d failed: %s", combo_idx, ti, exc)

        if combo_sharpes:
            path_sharpe = float(np.mean(combo_sharpes))
            result.path_sharpes.append(path_sharpe)
            # I3/I4: include all PFs (don't drop zero-PF folds), cap at MAX_PF_FOR_AVG.
            capped_pfs = [min(p, MAX_PF_FOR_AVG) for p in combo_pfs]
            result.path_profit_factors.append(float(np.mean(capped_pfs)) if capped_pfs else 0.0)
            # I2: drop only uncomputed zeros; keep negatives (CAL_TOTAL_LOSS_SENTINEL).
            valid_cals = [c for c in combo_cals if c != 0]
            result.path_calmars.append(float(np.mean(valid_cals)) if valid_cals else 0.0)
            # C-1 fix: each path's Sharpe is a mean of n_paths fold-Sharpes, so
            # the independent observation count for that path is the per-fold mean,
            # not the sum. Summing inflated _dsr_n_obs by n_paths, making the DSR
            # gate easier to pass (smaller sr_var → higher dsr_z).
            result.path_n_obs.append(int(np.mean(combo_n_obs)) if combo_n_obs else 0)
            # CRITICAL-2: per-path deployment diagnostics (mean of fold values).
            result.path_deployments.append(float(np.mean(combo_deps)) if combo_deps else 0.0)
            result.path_deployment_adj_sharpes.append(
                float(np.mean(combo_dep_adj)) if combo_dep_adj else 0.0)
            # C8-5: append per-combo regime mean (not per-fold raw value) so each
            # combo contributes equally to worst_regime_sharpe regardless of fold count.
            for regime, vals in combo_regime_by_name.items():
                all_regime_sharpes_by_regime.setdefault(regime, []).append(float(np.mean(vals)))

        if (combo_idx + 1) % 5 == 0 or combo_idx == len(combinations) - 1:
            logger.info("CPCV: %d/%d combinations done, mean Sharpe so far: %.3f",
                        combo_idx + 1, len(combinations),
                        float(np.mean(result.path_sharpes)) if result.path_sharpes else 0)

    # C-2 / IM-3: populate worst_regime_sharpe using per-regime mean (not raw min)
    # so the gate is stable and not dominated by single-fold noise in one regime.
    if all_regime_sharpes_by_regime:
        per_regime_means = [float(np.mean(v)) for v in all_regime_sharpes_by_regime.values()]
        result.worst_regime_sharpe = float(min(per_regime_means))
    else:
        logger.warning(
            "CPCV: no regime_sharpes recorded across any fold — "
            "worst_regime_sharpe is None, regime gate inactive. "
            "Populate FoldResult.regime_sharpes in your strategy to enable it."
        )

    # BUG-2: log completeness — how many fold evaluations were skipped due to
    # having no causal training history. For k=6, paths=2, exactly (k-1)=5 folds
    # will be skipped (one per combo containing fold 0). This is expected and does
    # not indicate a bug, but a large skip fraction beyond that is worth investigating.
    total_fold_evals = len(combinations) * n_paths
    expected_skips = len(combinations) - len(result.path_sharpes)
    if result.n_skipped > 0:
        skip_pct = result.n_skipped / max(total_fold_evals, 1) * 100
        if skip_pct > 20:
            logger.warning(
                "CPCV: %d/%d fold evaluations skipped (%.0f%%) — CPCV distribution "
                "is biased toward later regimes. Consider increasing n_folds or "
                "checking fold boundary construction.",
                result.n_skipped, total_fold_evals, skip_pct,
            )
        else:
            logger.info("CPCV: %d fold evaluations skipped (fold 0 no-prior-train; expected).",
                        result.n_skipped)

    return result
