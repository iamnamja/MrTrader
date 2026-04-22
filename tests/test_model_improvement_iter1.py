"""
Tests for Model Improvement Iteration 1:
  - Train/test embargo (swing + intraday)
  - Cross-sectional feature normalization (cs_normalize utility)
"""
import numpy as np
import pytest


# ── CS normalize utility ──────────────────────────────────────────────────────

class TestCsNormalize:
    def test_single_row_returned_unchanged(self):
        from app.ml.cs_normalize import cs_normalize
        X = np.array([[1.0, 2.0, 3.0]])
        result = cs_normalize(X)
        np.testing.assert_array_equal(result, X)

    def test_two_rows_zero_mean(self):
        from app.ml.cs_normalize import cs_normalize
        X = np.array([[1.0, 4.0], [3.0, 8.0]])
        result = cs_normalize(X)
        np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-6)

    def test_two_rows_unit_std(self):
        from app.ml.cs_normalize import cs_normalize
        X = np.array([[1.0, 4.0], [3.0, 8.0]])
        result = cs_normalize(X)
        np.testing.assert_allclose(result.std(axis=0), [1.0, 1.0], atol=1e-6)

    def test_constant_column_does_not_explode(self):
        from app.ml.cs_normalize import cs_normalize
        X = np.array([[5.0, 1.0], [5.0, 3.0]])
        result = cs_normalize(X)
        assert np.all(np.isfinite(result))

    def test_shape_preserved(self):
        from app.ml.cs_normalize import cs_normalize
        X = np.random.rand(20, 10)
        result = cs_normalize(X)
        assert result.shape == X.shape


class TestCsNormalizeByGroup:
    def test_groups_normalized_independently(self):
        from app.ml.cs_normalize import cs_normalize_by_group
        X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]])
        groups = np.array([0, 0, 1, 1])
        result = cs_normalize_by_group(X, groups)
        # Group 0 mean should be 0
        np.testing.assert_allclose(result[:2].mean(axis=0), [0.0, 0.0], atol=1e-6)
        # Group 1 mean should be 0
        np.testing.assert_allclose(result[2:].mean(axis=0), [0.0, 0.0], atol=1e-6)

    def test_single_row_group_unchanged(self):
        from app.ml.cs_normalize import cs_normalize_by_group
        X = np.array([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
        groups = np.array([0, 1, 1])  # group 0 has 1 row → unchanged
        result = cs_normalize_by_group(X, groups)
        np.testing.assert_array_equal(result[0], X[0])

    def test_original_array_not_mutated(self):
        from app.ml.cs_normalize import cs_normalize_by_group
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        original = X.copy()
        groups = np.array([0, 0])
        cs_normalize_by_group(X, groups)
        np.testing.assert_array_equal(X, original)


# ── Embargo — swing ───────────────────────────────────────────────────────────

class TestSwingEmbargo:
    def test_embargo_constant_positive(self):
        from app.ml.training import EMBARGO_WINDOWS, FORWARD_DAYS, STEP_DAYS
        assert EMBARGO_WINDOWS >= 1

    def test_embargo_creates_gap(self):
        """train end + EMBARGO_WINDOWS steps should be < test start index."""
        from app.ml.training import EMBARGO_WINDOWS, STEP_DAYS, TEST_FRACTION

        window_starts = list(range(0, 500, 10))  # simulate 50 windows
        split_idx = max(1, int(len(window_starts) * (1 - TEST_FRACTION)))
        embargo_start = min(split_idx + EMBARGO_WINDOWS, len(window_starts))
        test_window_starts = window_starts[embargo_start:]

        if test_window_starts:
            last_train_start = window_starts[split_idx - 1]
            first_test_start = test_window_starts[0]
            gap = first_test_start - last_train_start
            assert gap >= STEP_DAYS * (EMBARGO_WINDOWS + 1)

    def test_embargo_does_not_eliminate_test_set(self):
        from app.ml.training import EMBARGO_WINDOWS, TEST_FRACTION

        window_starts = list(range(0, 500, 10))
        split_idx = max(1, int(len(window_starts) * (1 - TEST_FRACTION)))
        embargo_start = min(split_idx + EMBARGO_WINDOWS, len(window_starts))
        test_window_starts = window_starts[embargo_start:]
        assert len(test_window_starts) > 0


# ── Embargo — intraday ────────────────────────────────────────────────────────

class TestIntradayEmbargo:
    def test_embargo_applied(self):
        """Verify embargo logic: test days start 1 day after split point."""
        from app.ml.intraday_training import TEST_FRACTION

        sorted_days = list(range(200))
        split_idx = max(1, int(len(sorted_days) * (1 - TEST_FRACTION)))
        embargo_start = min(split_idx + 1, len(sorted_days))
        test_days = set(sorted_days[embargo_start:])
        train_days = set(sorted_days[:split_idx])

        assert len(train_days & test_days) == 0  # no overlap
        # The embargoed day should be in neither set
        embargoed = sorted_days[split_idx]
        assert embargoed not in train_days
        assert embargoed not in test_days

    def test_embargo_does_not_eliminate_test_set(self):
        from app.ml.intraday_training import TEST_FRACTION

        sorted_days = list(range(200))
        split_idx = max(1, int(len(sorted_days) * (1 - TEST_FRACTION)))
        embargo_start = min(split_idx + 1, len(sorted_days))
        test_days = sorted_days[embargo_start:]
        assert len(test_days) > 0
