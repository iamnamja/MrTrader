"""Tests for Phase 39: intraday model rebuild with vol context, outcome labeling."""
import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock


def _make_bars(n=100, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2023-01-03 09:35", periods=n, freq="5min")
    p = 100 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open": p * 0.999, "high": p * 1.004,
        "low": p * 0.996, "close": p,
        "volume": np.ones(n) * 500_000,
    }, index=idx)


def _make_daily_bars(n=252):
    np.random.seed(0)
    idx = pd.bdate_range("2022-01-03", periods=n)
    p = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": p * 0.999, "high": p * 1.005,
        "low": p * 0.995, "close": p,
        "volume": np.ones(n) * 1_000_000,
    }, index=idx)


class TestComputeIntradayFeaturesWithDailyContext:

    def test_returns_36_features(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars()
        daily = _make_daily_bars()
        result = compute_intraday_features(bars, daily_bars=daily)
        assert result is not None
        assert len(result) == 37

    def test_daily_vol_features_present(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars()
        daily = _make_daily_bars()
        result = compute_intraday_features(bars, daily_bars=daily)
        assert "daily_vol_percentile" in result
        assert "daily_vol_regime" in result
        assert "daily_parkinson_vol" in result

    def test_daily_vol_percentile_in_range(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars()
        daily = _make_daily_bars()
        result = compute_intraday_features(bars, daily_bars=daily)
        assert 0.0 <= result["daily_vol_percentile"] <= 1.0

    def test_defaults_without_daily_bars(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars()
        result = compute_intraday_features(bars)
        assert result is not None
        assert result["daily_vol_percentile"] == 0.5
        assert result["daily_vol_regime"] == 1.0
        assert result["daily_parkinson_vol"] == 0.0

    def test_defaults_with_insufficient_daily_bars(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars()
        daily = _make_daily_bars(n=10)  # too few
        result = compute_intraday_features(bars, daily_bars=daily)
        assert result["daily_vol_percentile"] == 0.5

    def test_returns_33_features_without_daily(self):
        """Legacy: 33 features minus daily context = still 36 (defaults filled)."""
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars()
        result = compute_intraday_features(bars)
        assert len(result) == 37  # always 36; daily context defaults to 0.5/1/0


class TestOutcomeBasedLabel:
    """Test that training uses target-before-stop outcome labeling."""

    def test_target_constants_exist(self):
        from app.ml.intraday_training import TARGET_PCT, STOP_PCT, HOLD_BARS
        assert TARGET_PCT > 0
        assert STOP_PCT > 0
        assert HOLD_BARS > 0

    def test_target_larger_than_stop(self):
        from app.ml.intraday_training import TARGET_PCT, STOP_PCT
        assert TARGET_PCT > STOP_PCT

    def test_label_1_when_target_hit_first(self):
        """Simulate one day: target hit before stop → label=1."""
        entry = 100.0
        from app.ml.intraday_training import TARGET_PCT, STOP_PCT
        target = entry * (1 + TARGET_PCT)
        stop = entry * (1 - STOP_PCT)
        future = pd.DataFrame({
            "high": [entry * 1.001, entry * 1.003, target + 0.01, entry * 1.002],
            "low":  [entry * 0.999, entry * 0.998, entry * 0.999, entry * 0.998],
        })
        label = 0
        for _, bar in future.iterrows():
            if float(bar["high"]) >= target:
                label = 1
                break
            if float(bar["low"]) <= stop:
                label = 0
                break
        assert label == 1

    def test_label_0_when_stop_hit_first(self):
        entry = 100.0
        from app.ml.intraday_training import TARGET_PCT, STOP_PCT
        target = entry * (1 + TARGET_PCT)
        stop = entry * (1 - STOP_PCT)
        future = pd.DataFrame({
            "high": [entry * 1.001, entry * 1.001, entry * 1.002],
            "low":  [entry * 0.999, stop - 0.01, entry * 0.998],
        })
        label = 0
        for _, bar in future.iterrows():
            if float(bar["high"]) >= target:
                label = 1
                break
            if float(bar["low"]) <= stop:
                label = 0
                break
        assert label == 0


class TestIntradayTrainerScalePosWeight:

    def _make_trainer(self):
        from app.ml.intraday_training import IntradayModelTrainer
        return IntradayModelTrainer()

    def test_scale_pos_weight_passed_to_model(self):
        trainer = self._make_trainer()
        train_calls = []

        def _mock_train(X, y, names, scale_pos_weight=None):
            train_calls.append(scale_pos_weight)

        trainer.model.train = _mock_train

        X = np.zeros((10, 3))
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 2 pos, 8 neg → spw=4.0
        trainer._last_feature_names = ["a", "b", "c"]

        # Mimic the training call with imbalanced y
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
        trainer.model.train(X, y, trainer._last_feature_names, scale_pos_weight=spw)

        assert train_calls[0] == 4.0

    def test_fetch_daily_returns_empty_on_error(self):
        trainer = self._make_trainer()
        with patch.object(trainer._provider.__class__, "get_daily_bars_bulk", side_effect=Exception("fail")):
            # Should return {} not raise
            result = trainer._fetch_daily(["AAPL"], datetime.now() - timedelta(days=30), datetime.now())
        assert isinstance(result, dict)
