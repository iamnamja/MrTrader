"""
Tests for ML campaign Options B (regime split) and C (label horizon).

Option C: LABEL_HORIZON_DAYS configurable in retrain_config.py.
Option B: REGIME_SPLIT_VIX_THRESHOLD trains separate low/high-VIX models.
"""

import importlib

import numpy as np
import pytest


# ── Option C: Label horizon ──────────────────────────────────────────────────


class TestLabelHorizon:
    def test_default_horizon_is_5(self):
        """Default LABEL_HORIZON_DAYS preserves legacy 5d label."""
        from app.ml import retrain_config
        importlib.reload(retrain_config)
        assert retrain_config.LABEL_HORIZON_DAYS == 5

    def test_10d_label_uses_correct_shift(self, monkeypatch):
        """When LABEL_HORIZON_DAYS=10, train_model mutates FORWARD_DAYS to 10."""
        from app.ml import training, retrain_config
        # Save originals
        orig_fwd = training.FORWARD_DAYS
        orig_step = training.STEP_DAYS
        try:
            monkeypatch.setattr(retrain_config, "LABEL_HORIZON_DAYS", 10)
            monkeypatch.setattr(training, "_CFG_LABEL_HORIZON_DAYS", 10)
            # Reset module FORWARD_DAYS so the override path triggers
            training.FORWARD_DAYS = 5
            training.STEP_DAYS = 5

            trainer = training.ModelTrainer(use_feature_store=False)
            trainer._fetch_data = lambda *a, **k: {}  # short-circuit
            with pytest.raises(RuntimeError):
                trainer.train_model(symbols=["AAPL"], years=1)

            assert training.FORWARD_DAYS == 10
            assert training.STEP_DAYS == 10
        finally:
            training.FORWARD_DAYS = orig_fwd
            training.STEP_DAYS = orig_step

    def test_absolute_hurdle_scales_with_horizon(self, monkeypatch):
        """abs_hurdle scales linearly with horizon: 5d:0.003 → 10d:0.006 → 15d:0.009."""
        from app.ml import training, retrain_config
        orig_fwd = training.FORWARD_DAYS
        try:
            monkeypatch.setattr(training, "_CFG_LABEL_HORIZON_DAYS", 10)
            monkeypatch.setattr(training, "_CFG_LABEL_ABS_HURDLE_5D", 0.003)
            training.FORWARD_DAYS = 5
            trainer = training.ModelTrainer(use_feature_store=False)
            trainer._fetch_data = lambda *a, **k: {}
            with pytest.raises(RuntimeError):
                trainer.train_model(symbols=["AAPL"], years=1)
            # After the override, hurdle should be 0.003 * (10/5) = 0.006
            assert trainer._label_abs_hurdle == pytest.approx(0.006)
        finally:
            training.FORWARD_DAYS = orig_fwd

    def test_15d_horizon_hurdle_is_0_009(self, monkeypatch):
        from app.ml import training
        orig_fwd = training.FORWARD_DAYS
        try:
            monkeypatch.setattr(training, "_CFG_LABEL_HORIZON_DAYS", 15)
            monkeypatch.setattr(training, "_CFG_LABEL_ABS_HURDLE_5D", 0.003)
            training.FORWARD_DAYS = 5
            trainer = training.ModelTrainer(use_feature_store=False)
            trainer._fetch_data = lambda *a, **k: {}
            with pytest.raises(RuntimeError):
                trainer.train_model(symbols=["AAPL"], years=1)
            assert trainer._label_abs_hurdle == pytest.approx(0.009)
        finally:
            training.FORWARD_DAYS = orig_fwd


# ── Option B: Regime-split inference ────────────────────────────────────────


class _StubBase:
    """Stand-in for an XGBClassifier-like primary estimator."""
    def __init__(self, fixed_proba: float):
        self._p = fixed_proba

    def predict_proba(self, X):
        n = X.shape[0]
        return np.column_stack([np.full(n, 1 - self._p), np.full(n, self._p)])


def _make_loaded_model(low_p: float, high_p: float, threshold: float = 20.0):
    """Hand-construct a PortfolioSelectorModel with a simulated regime-split sibling."""
    from app.ml.model import PortfolioSelectorModel
    from sklearn.preprocessing import StandardScaler

    low = PortfolioSelectorModel(model_type="xgboost")
    low.model = _StubBase(low_p)
    low.scaler = StandardScaler()
    low.scaler.fit(np.zeros((2, 3)) + np.array([[0, 0, 0], [1, 1, 1]]))
    low.is_trained = True
    low.feature_names = ["a", "b", "c"]
    low.predict_threshold = 0.5

    high = PortfolioSelectorModel(model_type="xgboost")
    high.model = _StubBase(high_p)
    high.scaler = StandardScaler()
    high.scaler.fit(np.zeros((2, 3)) + np.array([[0, 0, 0], [1, 1, 1]]))
    high.is_trained = True
    high.feature_names = ["a", "b", "c"]
    high.predict_threshold = 0.5

    low._highvix_sibling = high
    low._regime_split_threshold = threshold
    return low


class TestRegimeSplit:
    def test_low_vix_model_selected_below_threshold(self):
        """VIX below threshold → low-vix model probabilities used."""
        m = _make_loaded_model(low_p=0.80, high_p=0.20, threshold=20.0)
        X = np.array([[0.5, 0.5, 0.5]])
        _, probs = m.predict_with_vix(X, vix_level=15.0)
        assert probs[0] == pytest.approx(0.80)

    def test_high_vix_model_selected_above_threshold(self):
        """VIX at or above threshold → high-vix model probabilities used."""
        m = _make_loaded_model(low_p=0.80, high_p=0.20, threshold=20.0)
        X = np.array([[0.5, 0.5, 0.5]])
        _, probs = m.predict_with_vix(X, vix_level=25.0)
        assert probs[0] == pytest.approx(0.20)
        # Boundary: VIX == threshold also routes to high-vix
        _, probs_eq = m.predict_with_vix(X, vix_level=20.0)
        assert probs_eq[0] == pytest.approx(0.20)

    def test_missing_vix_falls_back_to_high_vix(self):
        """Conservative default: unknown VIX uses high-vix (cautious) model."""
        m = _make_loaded_model(low_p=0.80, high_p=0.20, threshold=20.0)
        X = np.array([[0.5, 0.5, 0.5]])
        _, probs = m.predict_with_vix(X, vix_level=None)
        assert probs[0] == pytest.approx(0.20)

    def test_no_sibling_falls_back_to_predict(self):
        """When sibling not loaded, predict_with_vix delegates to predict()."""
        from app.ml.model import PortfolioSelectorModel
        from sklearn.preprocessing import StandardScaler
        m = PortfolioSelectorModel(model_type="xgboost")
        m.model = _StubBase(0.55)
        m.scaler = StandardScaler()
        m.scaler.fit(np.zeros((2, 3)) + np.array([[0, 0, 0], [1, 1, 1]]))
        m.is_trained = True
        m.feature_names = ["a", "b", "c"]
        # No _highvix_sibling attribute set
        X = np.array([[0.5, 0.5, 0.5]])
        _, probs = m.predict_with_vix(X, vix_level=15.0)
        assert probs[0] == pytest.approx(0.55)
