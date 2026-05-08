"""
engine.py — FoldEngine: strategy-agnostic fold construction, purge, and embargo.

The engine knows nothing about swing vs. intraday specifics. It calls
strategy.fetch_data() once, then strategy.run_fold() per fold.

Usage:
    from scripts.walkforward import FoldEngine
    from scripts.walkforward.strategies.swing import SwingStrategy

    strategy = SwingStrategy(model=..., version=..., symbols=[...])
    engine = FoldEngine(strategy=strategy, purge_days=10, embargo_days=10)
    report = engine.run(n_folds=3, total_years=5)
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, Optional

from scripts.walkforward.gates import WalkForwardReport

logger = logging.getLogger(__name__)


class FoldEngine:
    """Strategy-agnostic fold construction with purge + embargo on both sides."""

    def __init__(
        self,
        strategy,
        purge_days: int = 10,
        embargo_days: Optional[int] = None,
        parallel: bool = True,
        regime_map: Optional[Dict] = None,
    ):
        self.strategy = strategy
        self.purge_days = purge_days
        self.embargo_days = embargo_days if embargo_days is not None else purge_days
        self.parallel = parallel
        self.regime_map: Dict = regime_map or {}

    def run(
        self,
        n_folds: int,
        total_years: Optional[int] = None,
        total_days: Optional[int] = None,
        train_years: Optional[int] = None,
    ) -> WalkForwardReport:
        """Run the walk-forward. Exactly one of total_years or total_days must be given."""
        if total_years is None and total_days is None:
            raise ValueError("Provide either total_years or total_days.")

        end_all = datetime.now()
        if total_years is not None:
            start_all = end_all - timedelta(days=total_years * 365 + 30)
            self.strategy.fetch_data(start_all, end_all)
            fold_boundaries = self._build_calendar_folds(
                n_folds, start_all, end_all, total_years, train_years
            )
        else:
            end_date = end_all.date()
            start_date = end_date - timedelta(days=total_days + 10)
            self.strategy.fetch_data(start_date, end_date)
            fold_boundaries = self._build_trading_day_folds(
                n_folds, getattr(self.strategy, "all_days_sorted", [])
            )

        report = WalkForwardReport(model_type=getattr(self.strategy, "model_type", "unknown"))

        # Regime diversity check (warns only; never blocks fold construction)
        if self.regime_map:
            self._check_fold_diversity(fold_boundaries, n_folds)

        def _run_one(args):
            idx, tr_start, tr_end, te_start, te_end = args
            t0 = time.time()
            logger.info("Fold %d/%d  train:%s->%s  test:%s->%s",
                        idx, n_folds, tr_start, tr_end, te_start, te_end)
            fold = self.strategy.run_fold(idx, n_folds, tr_start, tr_end, te_start, te_end)
            logger.info("Fold %d done in %.1fs — %d trades, Sharpe %.2f",
                        idx, time.time() - t0, fold.trades, fold.sharpe)
            return fold

        fold_args = [
            (i + 1, tr_start, tr_end, te_start, te_end)
            for i, (tr_start, tr_end, te_start, te_end) in enumerate(fold_boundaries)
        ]
        if self.parallel and len(fold_args) > 1:
            with ThreadPoolExecutor(max_workers=n_folds) as pool:
                results = list(pool.map(_run_one, fold_args))
        else:
            results = [_run_one(args) for args in fold_args]
        report.folds = sorted(results, key=lambda f: f.fold)
        return report

    def _build_calendar_folds(self, n_folds, start_all, end_all, total_years, train_years):
        """Build fold boundaries using calendar days (swing)."""
        segment_days = int(total_years * 365 / (n_folds + 1))
        fold_boundaries = []
        for fold_idx in range(n_folds):
            train_end_dt = end_all - timedelta(days=segment_days * (n_folds - fold_idx))
            test_start_dt = train_end_dt + timedelta(days=self.purge_days + 1)
            raw_test_end_dt = train_end_dt + timedelta(days=segment_days)
            # Next fold's train starts after embargo gap
            test_end_dt = raw_test_end_dt - timedelta(days=self.embargo_days)
            if train_years is not None:
                fold_train_start = max(
                    start_all.date(),
                    (train_end_dt - timedelta(days=train_years * 365)).date()
                )
            else:
                fold_train_start = start_all.date()
            fold_boundaries.append((
                fold_train_start,
                train_end_dt.date(),
                test_start_dt.date(),
                min(test_end_dt.date(), end_all.date()),
            ))
        return fold_boundaries

    def _check_fold_diversity(self, fold_boundaries, n_folds: int) -> None:
        """Log regime distribution per fold test window. Warns if < 2 distinct regime labels."""
        from datetime import timedelta as _td
        from scripts.walkforward.regime import regime_diversity

        for i, (_, _, te_start, te_end) in enumerate(fold_boundaries):
            # Enumerate calendar days in test window
            if hasattr(te_start, "date"):
                te_start = te_start.date()
            if hasattr(te_end, "date"):
                te_end = te_end.date()
            days = []
            d = te_start
            while d <= te_end:
                days.append(d)
                d += _td(days=1)
            n_regimes = regime_diversity(days, self.regime_map)
            from collections import Counter
            from scripts.walkforward.regime import label_days as _ldays
            counts = Counter(_ldays(days, self.regime_map))
            counts.pop("UNK", None)
            logger.info(
                "Fold %d/%d regime diversity: %d distinct labels — %s",
                i + 1, n_folds, n_regimes,
                dict(sorted(counts.items())),
            )
            if n_regimes < 2:
                logger.warning(
                    "Fold %d/%d test window is regime-homogeneous (%d label). "
                    "Weak fold Sharpe may be regime-specific, not a model failure.",
                    i + 1, n_folds, n_regimes,
                )

    def _build_trading_day_folds(self, n_folds, all_days_sorted):
        """Build fold boundaries using trading days (intraday)."""
        if not all_days_sorted:
            return []
        segment_size = max(1, len(all_days_sorted) // (n_folds + 1))
        fold_boundaries = []
        for fi in range(n_folds):
            tr_end_idx = segment_size * (fi + 1) - 1
            te_start_idx = min(tr_end_idx + self.purge_days + 1, len(all_days_sorted) - 1)
            raw_te_end_idx = min(segment_size * (fi + 1) + segment_size - 1, len(all_days_sorted) - 1)
            # Subtract embargo days from test end (in trading days)
            te_end_idx = max(te_start_idx, raw_te_end_idx - self.embargo_days)
            fold_boundaries.append((
                all_days_sorted[0],
                all_days_sorted[tr_end_idx],
                all_days_sorted[te_start_idx],
                all_days_sorted[te_end_idx],
            ))
        return fold_boundaries
