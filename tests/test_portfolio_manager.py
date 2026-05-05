"""
Unit tests for Phase 4: Feature engineering, ML model, and training pipeline.

All tests are pure-Python — no network, database, Redis, or Alpaca calls.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml.features import FeatureEngineer, _normalise_columns
from app.ml.model import PortfolioSelectorModel
from app.utils.constants import SECTOR_MAP, SECTOR_LIST, SP_100_TICKERS


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_bars(n: int = 100, start_price: float = 100.0, trend: float = 0.5) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with a mild upward or downward trend."""
    np.random.seed(42)
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(max(1.0, closes[-1] * (1 + np.random.normal(trend / 252, 0.01))))
    closes = np.array(closes)
    return pd.DataFrame({
        "close": closes,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "open": closes * 0.998,
        "volume": np.random.randint(500_000, 2_000_000, n).astype(float),
    })


def _make_yfinance_bars(n: int = 100) -> pd.DataFrame:
    """Simulate capitalized yfinance-style column names."""
    df = _make_bars(n)
    df.columns = [c.capitalize() for c in df.columns]
    return df


# ─── Column Normalisation ─────────────────────────────────────────────────────

class TestNormaliseColumns:
    def test_lowercase_passthrough(self):
        df = _make_bars(60)
        result = _normalise_columns(df)
        assert all(c == c.lower() for c in result.columns)

    def test_capitalised_columns(self):
        df = _make_yfinance_bars(60)
        result = _normalise_columns(df)
        assert "close" in result.columns
        assert "high" in result.columns

    def test_adj_close_renamed(self):
        df = pd.DataFrame({"Adj Close": [100.0], "High": [101.0], "Low": [99.0], "Volume": [1e6]})
        result = _normalise_columns(df)
        assert "close" in result.columns
        assert "adj close" not in result.columns


# ─── Feature Engineering ──────────────────────────────────────────────────────

class TestFeatureEngineer:
    def setup_method(self):
        self.engineer = FeatureEngineer()

    def test_returns_dict(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_required_features_present(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars)
        required = [
            "rsi_14", "rsi_7", "macd", "macd_signal", "macd_histogram",
            "ema_20", "ema_50", "price_above_ema20", "price_above_ema50",
            "price_change_pct", "price_to_52w_high", "price_to_52w_low",
            "volume_ratio", "uptrend", "downtrend", "volatility",
            "momentum_5d", "momentum_20d", "sentiment",
        ]
        for feat in required:
            assert feat in features, f"Missing feature: {feat}"

    def test_returns_none_when_insufficient_data(self):
        bars = _make_bars(30)  # below MIN_BARS=52
        features = self.engineer.engineer_features("AAPL", bars)
        assert features is None

    def test_returns_none_when_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        features = self.engineer.engineer_features("AAPL", df)
        assert features is None

    def test_rsi_in_valid_range(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars)
        assert 0 <= features["rsi_14"] <= 100
        assert 0 <= features["rsi_7"] <= 100

    def test_binary_flags_are_0_or_1(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars)
        assert features["price_above_ema20"] in (0.0, 1.0)
        assert features["price_above_ema50"] in (0.0, 1.0)
        assert features["uptrend"] in (0.0, 1.0)
        assert features["downtrend"] in (0.0, 1.0)

    def test_sentiment_defaults_to_zero(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars, sentiment=None)
        assert features["sentiment"] == 0.0

    def test_sentiment_passed_through(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars, sentiment=0.75)
        assert features["sentiment"] == pytest.approx(0.75)

    def test_volume_ratio_positive(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars)
        assert features["volume_ratio"] > 0

    def test_volatility_non_negative(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars)
        assert features["volatility"] >= 0

    def test_yfinance_columns_accepted(self):
        bars = _make_yfinance_bars(100)
        features = self.engineer.engineer_features("AAPL", bars)
        assert features is not None

    def test_all_values_are_finite(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars)
        _nan_ok = {"nis_direction_score", "nis_materiality_score", "nis_already_priced_in",
                   "nis_sizing_mult", "nis_downside_risk",
                   "macro_avg_direction", "macro_pct_bearish", "macro_pct_bullish",
                   "macro_avg_materiality", "macro_pct_high_risk"}
        for name, val in features.items():
            if name in _nan_ok:
                continue
            assert np.isfinite(val), f"Feature '{name}' is not finite: {val}"

    def test_feature_count_at_least_15(self):
        bars = _make_bars(100)
        features = self.engineer.engineer_features("AAPL", bars)
        assert len(features) >= 15


# ─── ML Model ────────────────────────────────────────────────────────────────

class TestPortfolioSelectorModel:
    def _trained_model(self, n: int = 100, n_features: int = 19):
        model = PortfolioSelectorModel(model_type="xgboost")
        X = np.random.rand(n, n_features)
        y = np.random.randint(0, 2, n)
        names = [f"feature_{i}" for i in range(n_features)]
        model.train(X, y, names)
        return model, X, names

    def test_train_sets_is_trained(self):
        model, _, _ = self._trained_model()
        assert model.is_trained is True

    def test_predict_returns_correct_shapes(self):
        model, X, _ = self._trained_model()
        preds, probs = model.predict(X[:10])
        assert len(preds) == 10
        assert len(probs) == 10

    def test_probabilities_in_0_1(self):
        model, X, _ = self._trained_model()
        _, probs = model.predict(X)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_predictions_binary(self):
        model, X, _ = self._trained_model()
        preds, _ = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_raises_if_not_trained(self):
        model = PortfolioSelectorModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(np.random.rand(5, 10))

    def test_random_forest_also_works(self):
        model = PortfolioSelectorModel(model_type="random_forest")
        X = np.random.rand(80, 10)
        y = np.random.randint(0, 2, 80)
        model.train(X, y, [f"f{i}" for i in range(10)])
        preds, probs = model.predict(X[:5])
        assert len(preds) == 5

    def test_save_and_load(self, tmp_path):
        model, X, names = self._trained_model()
        model.save(str(tmp_path), version=1)

        model2 = PortfolioSelectorModel(model_type="xgboost")
        model2.load(str(tmp_path), version=1)
        assert model2.is_trained

        preds1, probs1 = model.predict(X[:5])
        preds2, probs2 = model2.predict(X[:5])
        np.testing.assert_array_equal(preds1, preds2)
        np.testing.assert_allclose(probs1, probs2, rtol=1e-5)

    def test_load_raises_if_file_missing(self, tmp_path):
        model = PortfolioSelectorModel()
        with pytest.raises(FileNotFoundError):
            model.load(str(tmp_path), version=99)

    def test_feature_importance_returns_sorted_list(self):
        model, _, names = self._trained_model()
        importance = model.feature_importance()
        assert importance is not None
        assert len(importance) == len(names)
        # Check sorted descending
        scores = [score for _, score in importance]
        assert scores == sorted(scores, reverse=True)


# ─── Constants ───────────────────────────────────────────────────────────────

class TestConstants:
    def test_sp100_has_entries(self):
        assert len(SP_100_TICKERS) >= 50

    def test_sp100_no_duplicates(self):
        assert len(SP_100_TICKERS) == len(set(SP_100_TICKERS))

    def test_sector_map_covers_all_tickers(self):
        missing = [t for t in SP_100_TICKERS if t not in SECTOR_MAP]
        assert missing == [], f"Tickers missing from SECTOR_MAP: {missing}"

    def test_sector_list_is_unique(self):
        assert len(SECTOR_LIST) == len(set(SECTOR_LIST))

    def test_sector_list_not_empty(self):
        assert len(SECTOR_LIST) > 0

    def test_brk_b_excluded(self):
        # BRK-B excluded from universe — invalid Alpaca symbol
        assert "BRK-B" not in SP_100_TICKERS
        assert "BRK.B" not in SP_100_TICKERS
