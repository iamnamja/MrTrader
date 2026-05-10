"""Tests for the per-fold feature cache used to speed up WF simulation."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.backtesting.feature_cache import FeatureCache, build_feature_cache


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ohlcv(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n_rows, freq="B")
    close = 100.0 + rng.standard_normal(n_rows).cumsum()
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n_rows).astype(float),
    }, index=idx)


def _make_symbols_data(n_syms: int = 5, n_rows: int = 300) -> dict:
    return {f"SYM{i:02d}": _make_ohlcv(n_rows, seed=i) for i in range(n_syms)}


# ── FeatureCache unit tests ────────────────────────────────────────────────────

class TestFeatureCacheDataStructure:
    def test_get_row_returns_correct_row(self):
        feature_names = ["f1", "f2", "f3"]
        mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        d1, d2 = date(2023, 1, 3), date(2023, 1, 4)
        cache = FeatureCache(
            feature_names=feature_names,
            matrix={"AAPL": mat},
            dates={"AAPL": np.array([d1, d2])},
            date_index={"AAPL": {d1: 0, d2: 1}},
        )
        row = cache.get_row("AAPL", d1)
        assert row is not None
        np.testing.assert_array_equal(row, [1.0, 2.0, 3.0])

    def test_get_row_missing_sym_returns_none(self):
        cache = FeatureCache(feature_names=["f1"], matrix={}, dates={}, date_index={})
        assert cache.get_row("MISSING", date(2023, 1, 3)) is None

    def test_get_row_missing_date_returns_none(self):
        feature_names = ["f1"]
        mat = np.array([[1.0]], dtype=np.float32)
        d = date(2023, 1, 3)
        cache = FeatureCache(
            feature_names=feature_names,
            matrix={"AAPL": mat},
            dates={"AAPL": np.array([d])},
            date_index={"AAPL": {d: 0}},
        )
        assert cache.get_row("AAPL", date(2023, 1, 4)) is None

    def test_get_features_returns_dict(self):
        feature_names = ["f1", "f2"]
        mat = np.array([[1.5, 2.5]], dtype=np.float32)
        d = date(2023, 1, 3)
        cache = FeatureCache(
            feature_names=feature_names,
            matrix={"AAPL": mat},
            dates={"AAPL": np.array([d])},
            date_index={"AAPL": {d: 0}},
        )
        feats = cache.get_features("AAPL", d)
        assert feats == {"f1": pytest.approx(1.5, abs=1e-4), "f2": pytest.approx(2.5, abs=1e-4)}

    def test_symbols_with_returns_correct_symbols(self):
        d = date(2023, 1, 3)
        cache = FeatureCache(
            feature_names=["f1"],
            matrix={},
            dates={},
            date_index={"AAPL": {d: 0}, "MSFT": {}, "GOOG": {d: 0}},
        )
        result = set(cache.symbols_with(d))
        assert result == {"AAPL", "GOOG"}

    def test_memory_mb_property(self):
        mat = np.zeros((100, 80), dtype=np.float32)
        cache = FeatureCache(
            feature_names=list(range(80)),
            matrix={"AAPL": mat},
            dates={},
            date_index={},
        )
        expected_mb = mat.nbytes / 1_048_576
        assert abs(cache.memory_mb - expected_mb) < 0.001


# ── build_feature_cache tests ──────────────────────────────────────────────────

class TestBuildFeatureCache:
    def _make_mock_fe(self, feature_names):
        """Return a FeatureEngineer mock that returns deterministic features."""
        fe = MagicMock()
        fe.engineer_features.side_effect = lambda sym, bars, **kw: {
            f: float(hash(sym + f) % 100) / 100.0 for f in feature_names
        }
        return fe

    def test_build_returns_feature_cache(self):
        symbols_data = _make_symbols_data(n_syms=3, n_rows=150)
        trading_days = [date(2020, 7, 1), date(2020, 7, 2), date(2020, 7, 6)]
        feature_names = ["rsi_14", "ema_ratio", "vol_percentile_52w"]

        with patch("app.backtesting.feature_cache._WORKER_FE", None), \
             patch("app.ml.features.FeatureEngineer") as MockFE:
            MockFE.return_value = self._make_mock_fe(feature_names)
            cache = build_feature_cache(
                symbols_data=symbols_data,
                trading_days=trading_days,
                feature_names=feature_names,
                workers=1,
                executor="thread",
            )

        assert isinstance(cache, FeatureCache)
        assert cache.feature_names == feature_names

    def test_cache_skips_synthetic_symbols(self):
        symbols_data = _make_symbols_data(n_syms=2, n_rows=150)
        symbols_data["^VIX"] = _make_ohlcv()
        symbols_data["SPY"] = _make_ohlcv()
        trading_days = [date(2020, 7, 1)]
        feature_names = ["f1"]

        with patch("app.backtesting.feature_cache._WORKER_FE", None), \
             patch("app.ml.features.FeatureEngineer") as MockFE:
            MockFE.return_value = self._make_mock_fe(feature_names)
            cache = build_feature_cache(
                symbols_data=symbols_data,
                trading_days=trading_days,
                feature_names=feature_names,
                workers=1,
                executor="thread",
            )

        assert "^VIX" not in cache.matrix
        assert "SPY" not in cache.matrix

    def test_cache_feature_values_are_float32(self):
        symbols_data = _make_symbols_data(n_syms=2, n_rows=150)
        trading_days = [date(2020, 7, 1)]
        feature_names = ["f1", "f2"]

        with patch("app.backtesting.feature_cache._WORKER_FE", None), \
             patch("app.ml.features.FeatureEngineer") as MockFE:
            MockFE.return_value = self._make_mock_fe(feature_names)
            cache = build_feature_cache(
                symbols_data=symbols_data,
                trading_days=trading_days,
                feature_names=feature_names,
                workers=1,
                executor="thread",
            )

        for mat in cache.matrix.values():
            assert mat.dtype == np.float32


# ── AgentSimulator backward-compat tests ───────────────────────────────────────

class TestAgentSimulatorCacheParam:
    def test_default_feature_cache_is_none(self):
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator()
        assert sim.feature_cache is None

    def test_accepts_feature_cache_kwarg(self):
        from app.backtesting.agent_simulator import AgentSimulator
        cache = FeatureCache(feature_names=[], matrix={}, dates={}, date_index={})
        sim = AgentSimulator(feature_cache=cache)
        assert sim.feature_cache is cache

    def test_sim_scan_interval_days_default(self):
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator()
        assert sim.sim_scan_interval_days == 1

    def test_sim_scan_interval_days_configurable(self):
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator(sim_scan_interval_days=5)
        assert sim.sim_scan_interval_days == 5
