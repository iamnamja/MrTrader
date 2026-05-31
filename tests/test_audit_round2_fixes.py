"""
Tests for audit round-2 fixes (19 remaining bugs from Opus 4.7 adversarial audit).

BUG-1:  CPCV DSR n_obs combinatorial multiplicity correction (unique_obs property)
BUG-2:  CPCV n_skipped completeness metric
BUG-3:  CPCV honors train_years per-fold (rolling window)
BUG-6:  WalkForwardReport.total_obs drops total_trades fallback
BUG-8:  OOS guard trading-day purge (not calendar-day) for intraday
BUG-17/18: retrain_config assert_purge_horizon_invariant enforced at import
BUG-20: retrain_config.retrain_as_of() returns deterministic last-Friday date
"""
from __future__ import annotations

import math
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# BUG-1: CPCV unique_obs corrects for combinatorial multiplicity
# ---------------------------------------------------------------------------

class TestCPCVUniqueObs:
    def _result(self, n_folds, n_paths, path_n_obs):
        from scripts.walkforward.cpcv import CPCVResult
        r = CPCVResult(model_type="swing", n_folds=n_folds, n_paths=n_paths)
        r.path_n_obs = path_n_obs
        r.path_sharpes = [1.0] * len(path_n_obs)
        return r

    def test_k6_paths2_divides_by_5(self):
        """k=6, paths=2 → C(5,1)=5 — total_obs/5 = unique_obs."""
        r = self._result(n_folds=6, n_paths=2, path_n_obs=[252] * 15)
        assert r.total_obs == 15 * 252
        # C(5,1) = 5
        assert math.comb(5, 1) == 5
        assert r.unique_obs == (15 * 252) // 5

    def test_k6_paths3_divides_by_c52(self):
        """k=6, paths=3 → C(5,2)=10 — total_obs/10 = unique_obs."""
        r = self._result(n_folds=6, n_paths=3, path_n_obs=[252] * 20)
        assert r.unique_obs == (20 * 252) // 10

    def test_dsr_uses_unique_obs_not_total(self):
        """_dsr_n_obs() must return unique_obs, not total_obs."""
        r = self._result(n_folds=6, n_paths=2, path_n_obs=[252] * 15)
        assert r._dsr_n_obs() == r.unique_obs
        assert r._dsr_n_obs() < r.total_obs

    def test_dsr_n_obs_fallback_when_no_path_n_obs(self):
        """Legacy results with no path_n_obs fall back to n_combinations."""
        r = self._result(n_folds=6, n_paths=2, path_n_obs=[])
        r.path_sharpes = [1.0] * 5
        assert r._dsr_n_obs() == 5  # max(n_combinations, 1)

    def test_unique_obs_zero_when_no_path_n_obs(self):
        r = self._result(n_folds=6, n_paths=2, path_n_obs=[])
        assert r.unique_obs == 0


# ---------------------------------------------------------------------------
# BUG-2: CPCV n_skipped completeness metric
# ---------------------------------------------------------------------------

class TestCPCVNSkipped:
    def test_n_skipped_field_exists(self):
        from scripts.walkforward.cpcv import CPCVResult
        r = CPCVResult(model_type="swing", n_folds=6, n_paths=2)
        assert hasattr(r, "n_skipped")
        assert r.n_skipped == 0

    def test_run_cpcv_counts_skipped_folds(self):
        """run_cpcv must increment n_skipped for fold 0 (no prior training data)."""
        from scripts.walkforward.cpcv import CPCVResult
        from datetime import date as _date
        from unittest.mock import patch

        mock_strategy = MagicMock()
        mock_strategy.model_type = "swing"
        mock_strategy.version = 1
        mock_strategy.allow_in_sample = False
        mock_strategy.model = MagicMock()
        mock_strategy.model.trained_through = _date(2018, 1, 1)
        mock_strategy.all_days_sorted = []

        from scripts.walkforward.gates import FoldResult

        def _fake_run_fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
            return FoldResult(
                fold=fold_idx, train_start=tr_start, train_end=tr_end,
                test_start=te_start, test_end=te_end,
                trades=10, win_rate=0.5, sharpe=1.0, max_drawdown=0.05,
                total_return=0.10, stop_exit_rate=0.2,
            )

        mock_strategy.run_fold.side_effect = _fake_run_fold

        with patch("app.ml.retrain_config.assert_no_sacred_holdout"):
            from scripts.walkforward.cpcv import run_cpcv
            result = run_cpcv(
                strategy=mock_strategy,
                purge_days=10,
                embargo_days=10,
                n_folds=4,
                n_paths=2,
                total_years=5,
                allow_sacred_holdout=True,
            )
        # k=4, paths=2: combos containing fold 0 = C(3,1) = 3; each skips once → n_skipped ≥ 1
        assert result.n_skipped >= 1


# ---------------------------------------------------------------------------
# BUG-3: CPCV honors train_years per-fold
# ---------------------------------------------------------------------------

class TestCPCVTrainYears:
    def test_train_years_limits_tr_start(self):
        """With train_years=2, real_tr_start must not go further back than 2 years before tr_end."""
        from datetime import date as _date, timedelta as _td
        from unittest.mock import patch, MagicMock
        from scripts.walkforward.gates import FoldResult

        captured = []

        mock_strategy = MagicMock()
        mock_strategy.model_type = "swing"
        mock_strategy.version = 1
        mock_strategy.allow_in_sample = False
        mock_strategy.model = MagicMock()
        mock_strategy.model.trained_through = _date(2018, 1, 1)
        mock_strategy.all_days_sorted = []

        def _capture_fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
            captured.append((tr_start, tr_end))
            return FoldResult(
                fold=fold_idx, train_start=tr_start, train_end=tr_end,
                test_start=te_start, test_end=te_end,
                trades=10, win_rate=0.5, sharpe=1.0, max_drawdown=0.05,
                total_return=0.10, stop_exit_rate=0.2,
            )

        mock_strategy.run_fold.side_effect = _capture_fold

        with patch("app.ml.retrain_config.assert_no_sacred_holdout"):
            from scripts.walkforward.cpcv import run_cpcv
            run_cpcv(
                strategy=mock_strategy,
                purge_days=10,
                embargo_days=10,
                n_folds=4,
                n_paths=2,
                total_years=5,
                train_years=2,
                allow_sacred_holdout=True,
            )

        # Every captured fold's training window must be <= 2 years long
        for tr_start, tr_end in captured:
            window_days = (tr_end - tr_start).days
            assert window_days <= 2 * 365 + 10, (
                f"train_years=2 violated: tr_start={tr_start} tr_end={tr_end} "
                f"= {window_days} days > {2*365+10}"
            )

    def test_train_window_never_overlaps_prior_test_fold(self):
        """BUG-23: rolling window must not overlap any prior test fold's [te_start, te_end]."""
        from datetime import date as _date, timedelta as _td
        from unittest.mock import patch, MagicMock
        from scripts.walkforward.gates import FoldResult
        from scripts.walkforward.cpcv import run_cpcv

        # Track (combo_idx, tr_start, tr_end, te_start, te_end) per run_fold call
        captured = []
        call_counter = [0]

        mock_strategy = MagicMock()
        mock_strategy.model_type = "swing"
        mock_strategy.version = 1
        mock_strategy.allow_in_sample = False
        mock_strategy.model = MagicMock()
        mock_strategy.model.trained_through = _date(2018, 1, 1)
        mock_strategy.all_days_sorted = []

        def _capture(fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
            # fold_idx encodes combo_idx * n_folds + ti + 1 (from cpcv.py)
            combo_idx = (fold_idx - 1) // n_folds
            captured.append((combo_idx, tr_start, tr_end, te_start, te_end))
            return FoldResult(
                fold=fold_idx, train_start=tr_start, train_end=tr_end,
                test_start=te_start, test_end=te_end,
                trades=10, win_rate=0.5, sharpe=1.0, max_drawdown=0.05,
                total_return=0.10, stop_exit_rate=0.2,
            )

        mock_strategy.run_fold.side_effect = _capture

        with patch("app.ml.retrain_config.assert_no_sacred_holdout"):
            result = run_cpcv(
                strategy=mock_strategy,
                purge_days=10,
                embargo_days=10,
                n_folds=6,
                n_paths=2,
                total_years=5,
                train_years=1,  # short window forces potential overlap
                allow_sacred_holdout=True,
            )

        # Within the same combination, a fold's training window must not overlap
        # any earlier test fold's [te_start, te_end]. Cross-combo overlaps are fine.
        from itertools import groupby
        by_combo = {}
        for combo_idx, tr_start, tr_end, te_start, te_end in captured:
            by_combo.setdefault(combo_idx, []).append((tr_start, tr_end, te_start, te_end))

        for combo_idx, folds in by_combo.items():
            folds_sorted = sorted(folds, key=lambda f: f[2])  # sort by te_start
            for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds_sorted):
                for otr_start, otr_end, ote_start, ote_end in folds_sorted[:i]:
                    assert not (tr_start < ote_end and tr_end > ote_start), (
                        f"BUG-23 combo {combo_idx}: training [{tr_start},{tr_end}] "
                        f"overlaps prior test [{ote_start},{ote_end}]"
                    )


# ---------------------------------------------------------------------------
# BUG-6: WalkForwardReport.total_obs no longer falls back to total_trades
# ---------------------------------------------------------------------------

class TestTotalObsNoFallback:
    def _make_report(self, n_obs_values, n_trades_per_fold=50):
        from scripts.walkforward.gates import WalkForwardReport, FoldResult
        r = WalkForwardReport(model_type="swing")
        for i, n_obs in enumerate(n_obs_values):
            r.folds.append(FoldResult(
                fold=i + 1,
                train_start=date(2021, 1, 1), train_end=date(2022, 1, 1),
                test_start=date(2022, 1, 1), test_end=date(2023, 1, 1),
                trades=n_trades_per_fold, win_rate=0.5, sharpe=1.0,
                max_drawdown=0.05, total_return=0.10, stop_exit_rate=0.2,
                n_obs=n_obs,
            ))
        return r

    def test_total_obs_uses_n_obs_field(self):
        r = self._make_report([252, 248, 250])
        assert r.total_obs == 252 + 248 + 250

    def test_total_obs_zero_when_no_n_obs(self):
        """BUG-6: must NOT fall back to total_trades."""
        r = self._make_report([0, 0, 0], n_trades_per_fold=100)
        assert r.total_obs == 0
        assert r.total_trades == 300  # trades exist but total_obs must not use them

    def test_total_obs_partial_folds(self):
        """Folds with n_obs=0 contribute 0 to the sum."""
        r = self._make_report([252, 0, 248])
        assert r.total_obs == 252 + 248


# ---------------------------------------------------------------------------
# BUG-8: OOS guard trading-day purge
# ---------------------------------------------------------------------------

class TestOOSGuardTradingDayPurge:
    def _trading_days(self, from_date: date, n: int) -> set:
        """Generate n trading days (Mon-Fri) starting from from_date."""
        days = set()
        d = from_date
        while len(days) < n:
            if d.weekday() < 5:
                days.add(d)
            d += timedelta(days=1)
        return days

    def test_calendar_purge_passes_with_weekend_gap(self):
        """Without trading_day_set, Friday trained_through + purge=2 + Monday te_start PASSES
        (3 calendar days > purge=2). This is the bug we're testing against."""
        from scripts.walkforward.oos_guard import assert_model_oos
        trained_through = date(2023, 1, 6)   # Friday
        te_start = date(2023, 1, 9)          # Monday (3 calendar days later)
        # Calendar-day mode: cutoff = Jan 6 + 2 = Jan 8 (Sunday). Jan 9 > Jan 8 → passes
        # But only 1 trading day (Jan 9 itself is te_start; gap is 0 trading days)
        assert_model_oos(
            trained_through=trained_through,
            fold_boundaries=[(date(2022, 1, 1), trained_through, te_start, date(2023, 6, 1))],
            purge_days=2,
        )  # passes with calendar (this is the pre-fix behavior)

    def test_trading_day_purge_fails_with_weekend_gap(self):
        """With trading_day_set, Friday trained_through + purge=2 + Monday te_start FAILS
        because there are 0 trading days between them (weekend)."""
        from scripts.walkforward.oos_guard import assert_model_oos, OOSViolation
        trained_through = date(2023, 1, 6)  # Friday
        te_start = date(2023, 1, 9)         # Monday — only 0 trading days gap (weekend)
        trading_days = self._trading_days(date(2022, 1, 3), 600)

        with pytest.raises(OOSViolation):
            assert_model_oos(
                trained_through=trained_through,
                fold_boundaries=[(date(2022, 1, 1), trained_through, te_start, date(2023, 6, 1))],
                purge_days=2,
                trading_day_set=trading_days,
            )

    def test_trading_day_purge_passes_with_sufficient_gap(self):
        """With trading_day_set and 3+ trading days gap, purge=2 passes."""
        from scripts.walkforward.oos_guard import assert_model_oos
        trained_through = date(2023, 1, 6)   # Friday
        # Skip to following Wednesday: Mon Jan 9, Tue Jan 10, Wed Jan 11 = 3 trading days
        te_start = date(2023, 1, 11)
        trading_days = self._trading_days(date(2022, 1, 3), 600)

        assert_model_oos(
            trained_through=trained_through,
            fold_boundaries=[(date(2022, 1, 1), trained_through, te_start, date(2023, 6, 1))],
            purge_days=2,
            trading_day_set=trading_days,
        )  # 3 trading days gap >= purge=2 → passes


# ---------------------------------------------------------------------------
# BUG-17/18: purge horizon invariant enforced at import
# ---------------------------------------------------------------------------

class TestPurgeHorizonInvariant:
    def test_invariant_holds_with_current_config(self):
        """retrain_config imports cleanly — invariant currently holds."""
        import app.ml.retrain_config as rc
        # If the invariant check raised, the import would have failed.
        # We can also verify the values directly.
        required = rc.FEATURE_LOOKBACK_DAYS + rc.LABEL_HORIZON_DAYS + 5
        assert rc.SWING_PURGE_DAYS >= required, (
            f"Purge invariant violated in config: {rc.SWING_PURGE_DAYS} < {required}"
        )

    def test_invariant_function_raises_on_bad_config(self):
        """_assert_purge_horizon_invariant() raises RuntimeError when purge < requirement."""
        import app.ml.retrain_config as rc
        original = rc.SWING_PURGE_DAYS
        try:
            rc.SWING_PURGE_DAYS = 5  # way too small
            with pytest.raises(RuntimeError, match="Purge invariant"):
                rc._assert_purge_horizon_invariant()
        finally:
            rc.SWING_PURGE_DAYS = original


# ---------------------------------------------------------------------------
# BUG-20: retrain_as_of() returns deterministic last-Friday date
# ---------------------------------------------------------------------------

class TestRetrainAsOf:
    def test_returns_a_friday_or_earlier(self):
        from app.ml.retrain_config import retrain_as_of
        d = retrain_as_of()
        assert d.weekday() <= 4  # Mon=0 ... Fri=4

    def test_returns_friday_when_today_is_friday(self):
        """When today is Friday, retrain_as_of returns today (if not in holdout)."""
        from app.ml.retrain_config import retrain_as_of, _parse_sacred_holdout_start
        from unittest.mock import patch
        from datetime import date as _date
        # Find the most recent Friday
        today = _date.today()
        offset = (today.weekday() - 4) % 7
        last_friday = today - timedelta(days=offset if offset else 0)
        holdout = _parse_sacred_holdout_start()
        expected = min(last_friday, holdout - timedelta(days=1))
        result = retrain_as_of()
        assert result == expected

    def test_strictly_before_sacred_holdout(self):
        from app.ml.retrain_config import retrain_as_of, _parse_sacred_holdout_start
        holdout = _parse_sacred_holdout_start()
        assert retrain_as_of() < holdout
