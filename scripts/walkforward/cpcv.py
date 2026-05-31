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
    N_TRIALS_TESTED, deflated_sharpe_ratio,
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
    # BUG-2: track skipped folds (fold 0 can't be tested — no prior training data).
    # n_skipped > 0 is normal for CPCV; a large fraction indicates a design problem.
    n_skipped: int = 0

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
    def avg_profit_factor(self) -> float:
        pfs = [p for p in self.path_profit_factors if p > 0]
        return float(np.mean(pfs)) if pfs else 0.0

    @property
    def avg_calmar(self) -> float:
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

    def gate_passed(self) -> bool:
        # In-sample runs (allow_in_sample override) can never promote past gates.
        if self.in_sample_override:
            return False
        _, dsr_p = deflated_sharpe_ratio(
            self.mean_sharpe, N_TRIALS_TESTED, self._dsr_n_obs()
        )
        pf_ok = self.avg_profit_factor == 0 or self.avg_profit_factor >= MIN_PROFIT_FACTOR
        cal_ok = self.avg_calmar == 0 or self.avg_calmar >= MIN_CALMAR
        return (
            self.mean_sharpe >= SHARPE_GATE
            and self.p5_sharpe >= MIN_FOLD_SHARPE
            and self.pct_positive >= 0.75
            and dsr_p > 0.95
            and pf_ok
            and cal_ok
        )

    def gate_detail(self) -> dict:
        _, dsr_p = deflated_sharpe_ratio(
            self.mean_sharpe, N_TRIALS_TESTED, self._dsr_n_obs()
        )
        return {
            "mean_sharpe": (self.mean_sharpe, self.mean_sharpe >= SHARPE_GATE),
            "p5_sharpe": (self.p5_sharpe, self.p5_sharpe >= MIN_FOLD_SHARPE),
            "pct_positive": (self.pct_positive, self.pct_positive >= 0.75),
            "dsr_p": (dsr_p, dsr_p > 0.95),
            "avg_profit_factor": (self.avg_profit_factor,
                                  self.avg_profit_factor == 0 or self.avg_profit_factor >= MIN_PROFIT_FACTOR),
            "avg_calmar": (self.avg_calmar,
                           self.avg_calmar == 0 or self.avg_calmar >= MIN_CALMAR),
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
        _, dsr_p = deflated_sharpe_ratio(self.mean_sharpe, N_TRIALS_TESTED, self._dsr_n_obs())
        print(f"  DSR p:        {dsr_p:.3f}  (gate: > 0.95)  {'OK' if dsr_p > 0.95 else 'FAIL'}")
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
        from datetime import datetime
        end_all = datetime.now()
        start_all = end_all - __import__("datetime").timedelta(days=total_years * 365 + 30)
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

    # OOS-guard: every test fold must start strictly after the model's training cutoff.
    from scripts.walkforward.oos_guard import assert_model_oos
    _trained_through = getattr(getattr(strategy, "model", None), "trained_through", None)
    _allow_in_sample = getattr(strategy, "allow_in_sample", False)
    _model_label = (
        f"{getattr(strategy, 'model_type', 'unknown')} "
        f"v{getattr(strategy, 'version', '?')}"
    )
    assert_model_oos(
        trained_through=_trained_through,
        fold_boundaries=all_boundaries,
        purge_days=purge_days,
        model_label=_model_label,
        allow_in_sample=_allow_in_sample,
    )

    result = CPCVResult(
        model_type=getattr(strategy, "model_type", "unknown"),
        n_folds=n_folds,
        n_paths=n_paths,
        in_sample_override=_allow_in_sample,
    )

    # Generate C(k, paths) combinations
    fold_indices = list(range(len(all_boundaries)))
    combinations = list(itertools.combinations(fold_indices, n_paths))
    logger.info("CPCV: %d combinations (C(%d,%d))", len(combinations), n_folds, n_paths)

    for combo_idx, test_indices in enumerate(combinations):
        train_indices = [i for i in fold_indices if i not in test_indices]
        if not train_indices:
            continue

        # Test on the selected folds
        combo_sharpes = []
        combo_pfs = []
        combo_cals = []
        combo_n_obs: List[int] = []

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
            except Exception as exc:
                logger.warning("CPCV combo %d fold %d failed: %s", combo_idx, ti, exc)

        if combo_sharpes:
            path_sharpe = float(np.mean(combo_sharpes))
            result.path_sharpes.append(path_sharpe)
            result.path_profit_factors.append(float(np.mean([p for p in combo_pfs if p > 0]) if combo_pfs else 0))
            result.path_calmars.append(float(np.mean([c for c in combo_cals if c != 0]) if combo_cals else 0))
            result.path_n_obs.append(int(sum(combo_n_obs)))

        if (combo_idx + 1) % 5 == 0 or combo_idx == len(combinations) - 1:
            logger.info("CPCV: %d/%d combinations done, mean Sharpe so far: %.3f",
                        combo_idx + 1, len(combinations),
                        float(np.mean(result.path_sharpes)) if result.path_sharpes else 0)

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
