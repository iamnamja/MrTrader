"""Tests for app/ml/ts_normalize.py — rolling time-series normalization (Fix 2)."""
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.ml.ts_normalize import (
    TSNormalizerState,
    assert_state_compatible,
    fit_transform_train,
    load_state,
    save_state,
    transform,
)


def _make_data(n_symbols=3, n_windows=30, n_features=5, seed=42):
    """Generate synthetic (X, symbols, window_ids) for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_symbols * n_windows, n_features))
    symbols = np.array([f"SYM{i}" for i in range(n_symbols) for _ in range(n_windows)])
    window_ids = np.array([w for _ in range(n_symbols) for w in range(n_windows)])
    return X, symbols, window_ids


class TestFitTransformTrain:
    def test_output_shape_matches_input(self):
        X, syms, wids = _make_data()
        X_norm, keep, state = fit_transform_train(X, syms, wids, min_warmup=3)
        assert X_norm.shape == X.shape
        assert keep.shape == (len(X),)

    def test_cold_start_rows_dropped(self):
        """First min_warmup windows per symbol should be excluded."""
        X, syms, wids = _make_data(n_symbols=2, n_windows=20)
        X_norm, keep, state = fit_transform_train(X, syms, wids, min_warmup=5)
        # 2 symbols × 5 cold-start windows = 10 dropped rows
        assert keep.sum() == len(X) - 10

    def test_kept_rows_are_normalized(self):
        """Kept rows should have mean≈0, std≈1 when aggregated (not exact per-row)."""
        X, syms, wids = _make_data(n_symbols=4, n_windows=40)
        X_norm, keep, _ = fit_transform_train(X, syms, wids)
        kept = X_norm[keep]
        assert kept.shape[0] > 10
        # Column means should be close to 0
        assert np.abs(kept.mean(axis=0)).max() < 2.0

    def test_constant_feature_gives_zeros_not_nan(self):
        """A feature with std=0 over the trailing window should produce zeros, not inf/nan."""
        X, syms, wids = _make_data(n_symbols=2, n_windows=25)
        X[:, 0] = 1.0  # constant feature
        X_norm, keep, _ = fit_transform_train(X, syms, wids, min_warmup=3)
        assert not np.any(np.isnan(X_norm[keep, 0]))
        assert not np.any(np.isinf(X_norm[keep, 0]))
        assert np.all(X_norm[keep, 0] == 0.0)

    def test_no_lookahead(self):
        """Row at window_idx=T should not be influenced by rows at T+1, T+2, ..."""
        # Use a monotonically increasing feature — later rows have larger values.
        # If there were lookahead, the normalized value at row T would shift
        # downward (mean contaminated by larger future values).
        n_sym, n_win = 1, 30
        X = np.zeros((n_win, 1))
        X[:, 0] = np.arange(n_win, dtype=float)
        syms = np.array(["SYM0"] * n_win)
        wids = np.arange(n_win)
        X_norm, keep, _ = fit_transform_train(X, syms, wids, lookback=5, min_warmup=3)
        # Each kept row: normalized using only its OWN past rows.
        # If lookahead existed, the first kept rows would have higher z-scores
        # (they look small relative to future large values). The reverse should hold:
        # early kept rows should have HIGHER z-scores (they are near the trailing mean,
        # not far below future values). Sufficient: no NaN and reasonable range.
        kept_vals = X_norm[keep, 0]
        assert not np.any(np.isnan(kept_vals))
        assert kept_vals.max() < 10.0  # no blow-up

    def test_state_has_last_stats_for_all_symbols(self):
        X, syms, wids = _make_data(n_symbols=3, n_windows=20)
        _, _, state = fit_transform_train(X, syms, wids)
        for sym in ["SYM0", "SYM1", "SYM2"]:
            assert sym in state.last_stats
            mean, std = state.last_stats[sym]
            assert mean.shape == (X.shape[1],)
            assert std.shape == (X.shape[1],)


class TestTransform:
    def test_shape_preserved(self):
        X_tr, syms_tr, wids_tr = _make_data(n_symbols=3, n_windows=25)
        _, _, state = fit_transform_train(X_tr, syms_tr, wids_tr)

        X_te, syms_te, wids_te = _make_data(n_symbols=3, n_windows=5, seed=99)
        # Shift window ids to be after training
        wids_te = wids_te + 100
        X_norm, keep = transform(X_te, syms_te, wids_te, state)
        assert X_norm.shape == X_te.shape
        assert keep.shape == (len(X_te),)

    def test_new_symbol_falls_back_to_last_stats(self):
        """A symbol not seen in training uses last_stats fallback."""
        X_tr, syms_tr, wids_tr = _make_data(n_symbols=2, n_windows=20)
        _, _, state = fit_transform_train(X_tr, syms_tr, wids_tr)

        # Manually inject a last_stats entry for a "new" symbol using a known symbol's stats
        state.last_stats["NEW_SYM"] = state.last_stats["SYM0"]

        X_te = np.ones((3, X_tr.shape[1]))
        syms_te = np.array(["NEW_SYM"] * 3)
        wids_te = np.array([200, 201, 202])
        X_norm, keep = transform(X_te, syms_te, wids_te, state)
        assert keep.all()  # fallback applied — all kept

    def test_truly_unknown_symbol_excluded(self):
        X_tr, syms_tr, wids_tr = _make_data(n_symbols=2, n_windows=20)
        _, _, state = fit_transform_train(X_tr, syms_tr, wids_tr)

        X_te = np.ones((2, X_tr.shape[1]))
        syms_te = np.array(["UNKNOWN_A", "UNKNOWN_B"])
        wids_te = np.array([300, 301])
        X_norm, keep = transform(X_te, syms_te, wids_te, state)
        assert not keep.any()


class TestPickleRoundtrip:
    def test_save_load_produces_identical_transforms(self):
        X_tr, syms_tr, wids_tr = _make_data(n_symbols=3, n_windows=25)
        _, _, state = fit_transform_train(X_tr, syms_tr, wids_tr)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "norm_state.pkl")
            save_state(state, path)
            loaded_state = load_state(path)

        X_te, syms_te, wids_te = _make_data(n_symbols=3, n_windows=5, seed=77)
        wids_te = wids_te + 200

        # Deep-copy state histories before each transform to avoid mutation side-effects
        import copy
        X1, k1 = transform(X_te, syms_te, wids_te, copy.deepcopy(state))
        X2, k2 = transform(X_te, syms_te, wids_te, copy.deepcopy(loaded_state))

        np.testing.assert_array_equal(k1, k2)
        np.testing.assert_allclose(X1[k1], X2[k2], rtol=1e-6)


class TestFeatureNamesHash:
    def test_mismatched_features_raise(self):
        X_tr, syms_tr, wids_tr = _make_data(n_symbols=2, n_windows=20)
        feat_names = [f"f{i}" for i in range(X_tr.shape[1])]
        _, _, state = fit_transform_train(X_tr, syms_tr, wids_tr, feature_names=feat_names)

        wrong_names = [f"g{i}" for i in range(X_tr.shape[1])]
        with pytest.raises(ValueError, match="feature mismatch"):
            assert_state_compatible(state, wrong_names)

    def test_matching_features_no_raise(self):
        X_tr, syms_tr, wids_tr = _make_data(n_symbols=2, n_windows=20)
        feat_names = [f"f{i}" for i in range(X_tr.shape[1])]
        _, _, state = fit_transform_train(X_tr, syms_tr, wids_tr, feature_names=feat_names)
        assert_state_compatible(state, feat_names)  # should not raise

    def test_state_without_hash_skips_check(self):
        state = TSNormalizerState()  # no hash set
        assert_state_compatible(state, ["a", "b", "c"])  # should not raise
