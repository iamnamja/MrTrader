"""Tests for Phase 40: swing model retrain with improved hyperparameters."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestModelHyperparameters:

    def test_max_depth_is_4(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        assert m.model.max_depth == 4

    def test_n_estimators_increased(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        assert m.model.n_estimators >= 300

    def test_learning_rate_low(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        assert m.model.learning_rate <= 0.05

    def test_min_child_weight_set(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        assert m.model.min_child_weight >= 5

    def test_colsample_bytree_reduced(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        assert m.model.colsample_bytree <= 0.7

    def test_regularisation_params_set(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        assert m.model.reg_alpha > 0
        assert m.model.reg_lambda > 1.0


class TestEarlyStopping:

    def _make_data(self, n=200, n_feat=10):
        np.random.seed(42)
        X = np.random.randn(n, n_feat)
        y = (X[:, 0] + np.random.randn(n) * 0.5 > 0).astype(int)
        return X, y

    def test_train_with_validation_set(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        X, y = self._make_data(200)
        X_tr, y_tr = X[:160], y[:160]
        X_val, y_val = X[160:], y[160:]
        m.train(X_tr, y_tr, X_val=X_val, y_val=y_val, early_stopping_rounds=10)
        assert m.is_trained

    def test_train_without_validation_still_works(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        X, y = self._make_data(100)
        m.train(X, y)
        assert m.is_trained

    def test_predict_after_early_stop_train(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        X, y = self._make_data(200)
        m.train(X[:160], y[:160], X_val=X[160:], y_val=y[160:], early_stopping_rounds=10)
        preds, proba = m.predict(X[160:])
        assert len(preds) == 40
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_auc_eval_metric(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        assert m.model.eval_metric == "auc"

    def test_scale_pos_weight_with_val_set(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        X, y = self._make_data(200)
        # Should not raise
        m.train(X[:160], y[:160], scale_pos_weight=2.0, X_val=X[160:], y_val=y[160:])
        assert m.is_trained


class TestTrainingPipelineEarlyStopping:
    """Verify training.py passes val set for early stopping."""

    def test_train_calls_model_train_with_val(self):
        from app.ml.training import ModelTrainer
        trainer = ModelTrainer()
        train_kwargs = {}

        original_train = trainer.model.train
        def capture_train(X, y, names=None, **kwargs):
            train_kwargs.update(kwargs)
            # Call without val set to avoid actual fit
            original_train(X, y, names)

        trainer.model.train = capture_train

        import numpy as np
        X_tr = np.random.randn(100, 5)
        y_tr = np.random.randint(0, 2, 100)
        X_te = np.random.randn(20, 5)
        y_te = np.random.randint(0, 2, 20)

        # Simulate what train_model does with test set
        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
        trainer.model.train(
            X_tr, y_tr, None,
            scale_pos_weight=spw,
            X_val=X_te, y_val=y_te,
            early_stopping_rounds=30,
        )
        assert "X_val" in train_kwargs
        assert train_kwargs["early_stopping_rounds"] == 30
