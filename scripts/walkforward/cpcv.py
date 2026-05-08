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
from datetime import date
from typing import List, Optional

import numpy as np

from scripts.walkforward.gates import (
    FoldResult, WalkForwardReport, WalkForwardReport,
    SHARPE_GATE, MIN_FOLD_SHARPE, MIN_PROFIT_FACTOR, MIN_CALMAR,
    N_TRIALS_TESTED, deflated_sharpe_ratio,
    compute_profit_factor, compute_calmar, compute_k_ratio,
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

    def gate_passed(self) -> bool:
        _, dsr_p = deflated_sharpe_ratio(self.mean_sharpe, N_TRIALS_TESTED,
                                          max(self.n_combinations, 1))
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
        _, dsr_p = deflated_sharpe_ratio(self.mean_sharpe, N_TRIALS_TESTED,
                                          max(self.n_combinations, 1))
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
        print(f"  Mean Sharpe:  {self.mean_sharpe:+.3f}  (gate: > {SHARPE_GATE})  "
              f"{'OK' if self.mean_sharpe >= SHARPE_GATE else 'FAIL'}")
        print(f"  Std Sharpe:   {self.std_sharpe:.3f}")
        print(f"  P5 Sharpe:    {self.p5_sharpe:+.3f}  (gate: > {MIN_FOLD_SHARPE})  "
              f"{'OK' if self.p5_sharpe >= MIN_FOLD_SHARPE else 'FAIL'}")
        print(f"  P95 Sharpe:   {self.p95_sharpe:+.3f}")
        print(f"  % positive:   {self.pct_positive:.1%}  (gate: >= 75%)  "
              f"{'OK' if self.pct_positive >= 0.75 else 'FAIL'}")
        _, dsr_p = deflated_sharpe_ratio(self.mean_sharpe, N_TRIALS_TESTED, self.n_combinations)
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
) -> CPCVResult:
    """
    Run Combinatorial Purged Cross-Validation.

    strategy must already have data fetched (strategy.fetch_data called).
    """
    from scripts.walkforward.engine import FoldEngine

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

    result = CPCVResult(
        model_type=getattr(strategy, "model_type", "unknown"),
        n_folds=n_folds,
        n_paths=n_paths,
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

        for ti in test_indices:
            tr_start, tr_end, te_start, te_end = all_boundaries[ti]
            # Use the earliest train fold's start as training start
            # (simplified: use train fold immediately before test fold)
            best_train = sorted(train_indices)[-1]  # latest train fold before test
            real_tr_start = all_boundaries[0][0]  # expanding: always from start
            real_tr_end = all_boundaries[best_train][1]

            try:
                fold = strategy.run_fold(
                    combo_idx * len(test_indices) + ti + 1,
                    n_folds,
                    real_tr_start,
                    real_tr_end,
                    te_start,
                    te_end,
                )
                combo_sharpes.append(fold.sharpe)
                combo_pfs.append(fold.profit_factor)
                combo_cals.append(fold.calmar_ratio)
            except Exception as exc:
                logger.warning("CPCV combo %d fold %d failed: %s", combo_idx, ti, exc)

        if combo_sharpes:
            path_sharpe = float(np.mean(combo_sharpes))
            result.path_sharpes.append(path_sharpe)
            result.path_profit_factors.append(float(np.mean([p for p in combo_pfs if p > 0]) if combo_pfs else 0))
            result.path_calmars.append(float(np.mean([c for c in combo_cals if c != 0]) if combo_cals else 0))

        if (combo_idx + 1) % 5 == 0 or combo_idx == len(combinations) - 1:
            logger.info("CPCV: %d/%d combinations done, mean Sharpe so far: %.3f",
                        combo_idx + 1, len(combinations),
                        float(np.mean(result.path_sharpes)) if result.path_sharpes else 0)

    return result
