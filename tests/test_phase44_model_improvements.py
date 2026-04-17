"""Tests for Phase 44: ATR-adaptive labels, 5-year default, ensemble model."""
import numpy as np
import pandas as pd
import pytest


class TestAtrLabelThresholds:

    def _make_df(self, n=30, atr_pct=0.02):
        """Synthetic OHLC where ATR ≈ atr_pct * close."""
        np.random.seed(0)
        closes = 100 + np.cumsum(np.random.randn(n) * 0.1)
        highs = closes * (1 + atr_pct / 2)
        lows = closes * (1 - atr_pct / 2)
        opens = closes * 0.999
        return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})

    def test_high_vol_stock_gets_larger_target(self):
        from app.ml.training import _atr_label_thresholds
        df_high = self._make_df(atr_pct=0.04)  # TSLA-like: 4% ATR
        df_low = self._make_df(atr_pct=0.005)  # PG-like: 0.5% ATR
        t_high, _ = _atr_label_thresholds(df_high, 100.0)
        t_low, _ = _atr_label_thresholds(df_low, 100.0)
        assert t_high > t_low

    def test_target_clamped_to_max(self):
        from app.ml.training import _atr_label_thresholds, ATR_MAX_TARGET
        df = self._make_df(atr_pct=0.10)  # extreme 10% ATR
        target, _ = _atr_label_thresholds(df, 100.0)
        assert target <= ATR_MAX_TARGET

    def test_target_clamped_to_min(self):
        from app.ml.training import _atr_label_thresholds, ATR_MIN_TARGET
        df = self._make_df(atr_pct=0.001)  # near-zero ATR
        target, _ = _atr_label_thresholds(df, 100.0)
        assert target >= ATR_MIN_TARGET

    def test_stop_smaller_than_target(self):
        from app.ml.training import _atr_label_thresholds
        df = self._make_df(atr_pct=0.02)
        target, stop = _atr_label_thresholds(df, 100.0)
        assert stop < target

    def test_fallback_on_insufficient_data(self):
        from app.ml.training import _atr_label_thresholds, LABEL_TARGET_PCT, LABEL_STOP_PCT
        df = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]})
        target, stop = _atr_label_thresholds(df, 100.0)
        assert target == LABEL_TARGET_PCT
        assert stop == LABEL_STOP_PCT


class TestEnsembleModel:

    def test_ensemble_trains_both_components(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("ensemble")
        np.random.seed(0)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        m.train(X, y)
        assert m.is_trained
        assert m._lr_model is not None

    def test_ensemble_predict_returns_blended_proba(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("ensemble")
        np.random.seed(1)
        X = np.random.randn(120, 10)
        y = (X[:, 0] > 0).astype(int)
        m.train(X[:100], y[:100])
        preds, proba = m.predict(X[100:])
        assert len(preds) == 20
        assert all(0 <= p <= 1 for p in proba)

    def test_ensemble_proba_between_xgb_and_lr(self):
        """Blended proba should be between XGBoost and LR outputs (within float tolerance)."""
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("ensemble")
        np.random.seed(2)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        m.train(X[:80], y[:80])
        X_test = X[80:]
        X_scaled = m.scaler.transform(X_test)
        xgb_p = m.model.predict_proba(X_scaled)[:, 1]
        lr_p = m._lr_model.predict_proba(X_scaled)[:, 1]
        expected = 0.70 * xgb_p + 0.30 * lr_p
        _, blended = m.predict(X_test)
        np.testing.assert_allclose(blended, expected, rtol=1e-5)

    def test_ensemble_save_load_roundtrip(self, tmp_path):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("ensemble")
        np.random.seed(3)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        m.train(X[:80], y[:80])
        m.save(str(tmp_path), version=99, model_name="test")
        m2 = PortfolioSelectorModel("ensemble")
        m2.load(str(tmp_path), version=99, model_name="test")
        assert m2.is_trained
        assert m2.model_type == "ensemble"
        assert m2._lr_model is not None
        _, p1 = m.predict(X[80:])
        _, p2 = m2.predict(X[80:])
        np.testing.assert_allclose(p1, p2, rtol=1e-5)

    def test_xgboost_mode_still_works(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        np.random.seed(4)
        X = np.random.randn(80, 5)
        y = (X[:, 0] > 0).astype(int)
        m.train(X[:60], y[:60])
        preds, proba = m.predict(X[60:])
        assert len(preds) == 20
        assert m._lr_model is None
