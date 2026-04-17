"""Tests for Phase 41: feature store and automated retraining pipeline."""
import json
import tempfile
import os
import pytest
from datetime import date


class TestFeatureStore:

    def _store(self, tmp_path):
        from app.ml.feature_store import FeatureStore
        return FeatureStore(str(tmp_path / "test_features.db"))

    def test_put_and_get(self, tmp_path):
        store = self._store(tmp_path)
        feats = {"rsi_14": 0.6, "momentum_20d": 0.05}
        store.put("AAPL", date(2024, 3, 15), feats)
        result = store.get("AAPL", date(2024, 3, 15))
        assert result is not None
        assert result["rsi_14"] == pytest.approx(0.6)

    def test_get_miss_returns_none(self, tmp_path):
        store = self._store(tmp_path)
        assert store.get("AAPL", date(2024, 3, 15)) is None

    def test_put_overwrites(self, tmp_path):
        store = self._store(tmp_path)
        store.put("AAPL", date(2024, 3, 15), {"rsi_14": 0.6})
        store.put("AAPL", date(2024, 3, 15), {"rsi_14": 0.9})
        result = store.get("AAPL", date(2024, 3, 15))
        assert result["rsi_14"] == pytest.approx(0.9)

    def test_different_symbols_independent(self, tmp_path):
        store = self._store(tmp_path)
        store.put("AAPL", date(2024, 3, 15), {"rsi_14": 0.6})
        store.put("MSFT", date(2024, 3, 15), {"rsi_14": 0.3})
        assert store.get("AAPL", date(2024, 3, 15))["rsi_14"] == pytest.approx(0.6)
        assert store.get("MSFT", date(2024, 3, 15))["rsi_14"] == pytest.approx(0.3)

    def test_count(self, tmp_path):
        store = self._store(tmp_path)
        store.put("AAPL", date(2024, 3, 15), {"x": 1.0})
        store.put("AAPL", date(2024, 3, 16), {"x": 2.0})
        store.put("MSFT", date(2024, 3, 15), {"x": 3.0})
        assert store.count() == 3

    def test_put_batch(self, tmp_path):
        store = self._store(tmp_path)
        rows = [
            ("AAPL", date(2024, 3, 15), {"rsi_14": 0.6}),
            ("AAPL", date(2024, 3, 16), {"rsi_14": 0.7}),
            ("MSFT", date(2024, 3, 15), {"rsi_14": 0.4}),
        ]
        store.put_batch(rows)
        assert store.count() == 3
        assert store.get("AAPL", date(2024, 3, 16))["rsi_14"] == pytest.approx(0.7)

    def test_get_batch(self, tmp_path):
        store = self._store(tmp_path)
        store.put("AAPL", date(2024, 3, 15), {"rsi_14": 0.6})
        store.put("AAPL", date(2024, 3, 16), {"rsi_14": 0.7})
        store.put("AAPL", date(2024, 3, 17), {"rsi_14": 0.8})
        result = store.get_batch("AAPL", [date(2024, 3, 15), date(2024, 3, 17)])
        assert len(result) == 2
        assert result[date(2024, 3, 15)]["rsi_14"] == pytest.approx(0.6)

    def test_get_batch_empty_list(self, tmp_path):
        store = self._store(tmp_path)
        assert store.get_batch("AAPL", []) == {}

    def test_evict_before(self, tmp_path):
        store = self._store(tmp_path)
        store.put("AAPL", date(2024, 1, 1), {"x": 1.0})
        store.put("AAPL", date(2024, 6, 1), {"x": 2.0})
        store.put("AAPL", date(2024, 12, 1), {"x": 3.0})
        deleted = store.evict_before(date(2024, 6, 1))
        assert deleted == 1
        assert store.count() == 2
        assert store.get("AAPL", date(2024, 1, 1)) is None
        assert store.get("AAPL", date(2024, 6, 1)) is not None

    def test_clear(self, tmp_path):
        store = self._store(tmp_path)
        store.put("AAPL", date(2024, 3, 15), {"x": 1.0})
        store.put("MSFT", date(2024, 3, 15), {"x": 2.0})
        store.clear()
        assert store.count() == 0

    def test_persists_across_instances(self, tmp_path):
        from app.ml.feature_store import FeatureStore
        db = str(tmp_path / "store.db")
        s1 = FeatureStore(db)
        s1.put("AAPL", date(2024, 3, 15), {"rsi_14": 0.55})
        s2 = FeatureStore(db)
        assert s2.get("AAPL", date(2024, 3, 15))["rsi_14"] == pytest.approx(0.55)


class TestTrainerFeatureStoreIntegration:

    def test_trainer_has_feature_store(self, tmp_path):
        from app.ml.training import ModelTrainer
        t = ModelTrainer(model_dir=str(tmp_path), use_feature_store=True)
        assert t._feature_store is not None

    def test_trainer_no_feature_store(self, tmp_path):
        from app.ml.training import ModelTrainer
        t = ModelTrainer(model_dir=str(tmp_path), use_feature_store=False)
        assert t._feature_store is None

    def test_cache_hit_skips_engineer_features(self, tmp_path):
        """When a cache entry exists, engineer_features should not be called."""
        from app.ml.training import ModelTrainer
        t = ModelTrainer(model_dir=str(tmp_path), use_feature_store=True)

        # Pre-populate cache with fake features
        feats = {"rsi_14": 0.5, "ema_20": 0.01}
        t._feature_store.put("AAPL", date(2024, 3, 15), feats)

        call_count = [0]
        original = t.feature_engineer.engineer_features
        def mock_fe(*a, **kw):
            call_count[0] += 1
            return original(*a, **kw)
        t.feature_engineer.engineer_features = mock_fe

        result = t._feature_store.get("AAPL", date(2024, 3, 15))
        # Cache hit — no engineer_features call needed
        assert result is not None
        assert call_count[0] == 0
