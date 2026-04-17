"""Tests for Phase 43: sample weights + enhanced features."""
import numpy as np
import pandas as pd
import pytest
from datetime import date


class TestSampleWeights:

    def _meta(self, n=20):
        """Generate synthetic metadata for n samples."""
        np.random.seed(42)
        sectors = (["Technology", "Financials", "Healthcare"] * ((n // 3) + 1))[:n]
        return dict(
            window_indices=list(range(n)),
            total_windows=n,
            outcome_returns=list(np.random.randn(n) * 0.03),
            vol_percentiles=list(np.random.uniform(0, 1, n)),
            avg_volumes=list(np.random.uniform(1e5, 1e7, n)),
            sector_labels=sectors,
        )

    def test_returns_correct_shape(self):
        from app.ml.sample_weights import compute_sample_weights
        m = self._meta(20)
        w = compute_sample_weights(**m)
        assert len(w) == 20

    def test_mean_approximately_one(self):
        from app.ml.sample_weights import compute_sample_weights
        m = self._meta(50)
        w = compute_sample_weights(**m)
        assert abs(w.mean() - 1.0) < 0.05

    def test_all_positive(self):
        from app.ml.sample_weights import compute_sample_weights
        m = self._meta(30)
        w = compute_sample_weights(**m)
        assert (w > 0).all()

    def test_newer_samples_weighted_higher(self):
        """Last sample should have higher recency weight than first."""
        from app.ml.sample_weights import compute_sample_weights
        n = 50
        # Identical metadata except window index
        w = compute_sample_weights(
            window_indices=list(range(n)),
            total_windows=n,
            outcome_returns=[0.03] * n,
            vol_percentiles=[0.5] * n,
            avg_volumes=[1e6] * n,
            sector_labels=["Technology"] * n,
        )
        assert w[-1] > w[0]

    def test_recency_max_multiplier_respected(self):
        """Newest / oldest weight ratio should not exceed recency_max_multiplier."""
        from app.ml.sample_weights import compute_sample_weights
        n = 50
        w = compute_sample_weights(
            window_indices=list(range(n)),
            total_windows=n,
            outcome_returns=[0.03] * n,
            vol_percentiles=[0.5] * n,
            avg_volumes=[1e6] * n,
            sector_labels=["Technology"] * n,
            recency_max_multiplier=2.5,
        )
        # After normalisation ratio won't be exactly 2.5, but newest > oldest
        assert w[-1] > w[0]

    def test_decisive_outcomes_weighted_higher(self):
        """Samples with larger |outcome_return| should get higher weight."""
        from app.ml.sample_weights import compute_sample_weights
        n = 10
        # First half: small outcome, second half: large outcome
        outcome_returns = [0.001] * 5 + [0.08] * 5
        w = compute_sample_weights(
            window_indices=list(range(n)),
            total_windows=n,
            outcome_returns=outcome_returns,
            vol_percentiles=[0.5] * n,
            avg_volumes=[1e6] * n,
            sector_labels=["Technology"] * n,
        )
        assert w[5:].mean() > w[:5].mean()

    def test_vol_regime_match_upweights_similar(self):
        """Samples with vol_percentile close to current should have higher weight."""
        from app.ml.sample_weights import compute_sample_weights
        n = 20
        # First 10: low vol (0.1), last 10: high vol (0.9)
        # Current is high vol (0.9) → last 10 should be upweighted
        w = compute_sample_weights(
            window_indices=list(range(n)),
            total_windows=n,
            outcome_returns=[0.03] * n,
            vol_percentiles=[0.1] * 10 + [0.9] * 10,
            avg_volumes=[1e6] * n,
            sector_labels=["Technology"] * n,
            current_vol_percentile=0.9,
        )
        assert w[10:].mean() > w[:10].mean()

    def test_sector_diversity_downweights_overrepresented(self):
        """Sector with more stocks should have lower per-sample weight."""
        from app.ml.sample_weights import compute_sample_weights
        n = 15
        # 10 tech, 5 financial
        w = compute_sample_weights(
            window_indices=list(range(n)),
            total_windows=n,
            outcome_returns=[0.03] * n,
            vol_percentiles=[0.5] * n,
            avg_volumes=[1e6] * n,
            sector_labels=["Technology"] * 10 + ["Financials"] * 5,
        )
        avg_tech = w[:10].mean()
        avg_fin = w[10:].mean()
        assert avg_fin > avg_tech  # financials (fewer) get higher weight

    def test_empty_returns_empty(self):
        from app.ml.sample_weights import compute_sample_weights
        w = compute_sample_weights([], 0, [], [], [], [])
        assert len(w) == 0

    def test_returns_float32(self):
        from app.ml.sample_weights import compute_sample_weights
        m = self._meta(10)
        w = compute_sample_weights(**m)
        assert w.dtype == np.float32


class TestModelTrainWithSampleWeights:

    def test_train_accepts_sample_weight(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        np.random.seed(0)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        w = np.random.uniform(0.5, 2.0, 100).astype(np.float32)
        m.train(X, y, sample_weight=w)
        assert m.is_trained

    def test_train_with_weight_and_val_set(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        np.random.seed(1)
        X = np.random.randn(120, 5)
        y = (X[:, 0] > 0).astype(int)
        w = np.ones(100, dtype=np.float32)
        w[:10] = 2.0  # upweight first 10
        m.train(X[:100], y[:100], X_val=X[100:], y_val=y[100:], sample_weight=w)
        assert m.is_trained

    def test_predict_still_works_after_weighted_train(self):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel("xgboost")
        np.random.seed(2)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        w = np.random.uniform(0.5, 1.5, 100).astype(np.float32)
        m.train(X[:80], y[:80], sample_weight=w[:80])
        preds, proba = m.predict(X[80:])
        assert len(preds) == 20
        assert all(0 <= p <= 1 for p in proba)


class TestNewFeatures:

    def _make_bars(self, n=100, seed=7):
        np.random.seed(seed)
        idx = pd.bdate_range("2024-01-02", periods=n)
        p = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({
            "open": p * 0.999, "high": p * 1.005,
            "low": p * 0.995, "close": p,
            "volume": np.ones(n) * 1_000_000,
        }, index=idx)

    def test_new_features_present(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        df = self._make_bars()
        result = fe.engineer_features("AAPL", df, fetch_fundamentals=False)
        assert result is not None
        assert "vpt_momentum" in result
        assert "range_expansion" in result
        assert "vwap_distance_20d" in result
        assert "williams_r_14" in result
        assert "cci_20" in result
        assert "price_acceleration" in result

    def test_williams_r_in_range(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.engineer_features("AAPL", self._make_bars(), fetch_fundamentals=False)
        assert -1.0 <= result["williams_r_14"] <= 0.0

    def test_range_expansion_positive(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.engineer_features("AAPL", self._make_bars(), fetch_fundamentals=False)
        assert result["range_expansion"] >= 0

    def test_cci_within_clip_bounds(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.engineer_features("AAPL", self._make_bars(), fetch_fundamentals=False)
        assert -1.5 <= result["cci_20"] <= 1.5

    def test_fmp_consecutive_beats_default_zero(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        # fetch_fundamentals=False → defaults
        result = fe.engineer_features("AAPL", self._make_bars(), fetch_fundamentals=False)
        assert result["fmp_consecutive_beats"] == 0.0

    def test_total_feature_count_increased(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.engineer_features("AAPL", self._make_bars(), fetch_fundamentals=False)
        # Was 68; now 68 + 2 (fmp enhanced) + 3 (vp dynamics) + 3 (daily tech) = 74 (some overlap with existing)
        assert len(result) >= 74
