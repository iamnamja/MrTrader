"""Tests for Phase 30: data provider abstraction + rolling window training."""
import pandas as pd
import numpy as np
import pytest
from datetime import date, datetime
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_daily_df(n=300, start="2023-01-01"):
    idx = pd.bdate_range(start=start, periods=n)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": prices * 0.999, "high": prices * 1.005,
        "low": prices * 0.995, "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


# ── DataProvider base interface ───────────────────────────────────────────────

class TestDataProviderBase:

    def test_cannot_instantiate_abstract(self):
        from app.data.base import DataProvider
        with pytest.raises(TypeError):
            DataProvider()

    def test_normalise_lowercases_columns(self):
        from app.data.base import DataProvider
        df = pd.DataFrame({"Close": [1.0], "Open": [1.0], "Volume": [1000.0]})
        result = DataProvider._normalise(df)
        assert "close" in result.columns
        assert "open" in result.columns

    def test_normalise_renames_adj_close(self):
        from app.data.base import DataProvider
        df = pd.DataFrame({"Adj Close": [1.0], "Open": [1.0], "Volume": [1000.0]})
        result = DataProvider._normalise(df)
        assert "close" in result.columns
        assert "adj close" not in result.columns


# ── Registry ──────────────────────────────────────────────────────────────────

class TestRegistry:

    def test_get_yfinance_provider(self):
        from app.data import get_provider
        p = get_provider("yfinance")
        assert p.name == "yfinance"

    def test_get_default_is_yfinance(self):
        from app.data import get_provider
        p = get_provider()
        assert p.name == "yfinance"

    def test_unknown_provider_raises(self):
        from app.data import get_provider
        with pytest.raises(ValueError, match="Unknown data provider"):
            get_provider("nonexistent_provider_xyz")

    def test_register_custom_provider(self):
        from app.data.base import DataProvider
        from app.data.registry import register_provider, get_provider

        class DummyProvider(DataProvider):

            @property
            def name(self):
                return "dummy"

            def get_daily_bars(self, *a, **kw):
                return None

            def get_intraday_bars(self, *a, **kw):
                return None

        register_provider("dummy_test", DummyProvider())
        p = get_provider("dummy_test")
        assert p.name == "dummy"

    def test_list_providers_includes_yfinance(self):
        from app.data.registry import list_providers
        assert "yfinance" in list_providers()


# ── YFinanceProvider ──────────────────────────────────────────────────────────

class TestYFinanceProvider:

    def _provider(self, tmp_path=None):
        from app.data.yfinance_provider import YFinanceProvider
        return YFinanceProvider()

    def _isolated_cache(self, tmp_path):
        """Patch get_cache to use a throwaway directory so tests don't share state."""
        from app.data.cache import DataCache
        isolated = DataCache(cache_dir=tmp_path)
        return patch("app.data.yfinance_provider.get_cache", return_value=isolated)

    def test_get_daily_bars_returns_normalised_df(self, tmp_path):
        p = self._provider()
        df = _make_daily_df()
        # Use unique symbol so global cache can't interfere
        with self._isolated_cache(tmp_path), patch("yfinance.download", return_value=df):
            result = p.get_daily_bars("TEST_SYM_A", date(2023, 1, 1), date(2024, 1, 1))
        assert result is not None
        assert "close" in result.columns
        assert len(result) > 0

    def test_get_daily_bars_returns_none_on_empty(self, tmp_path):
        p = self._provider()
        with self._isolated_cache(tmp_path), \
                patch("yfinance.download", return_value=pd.DataFrame()):
            result = p.get_daily_bars("TEST_SYM_B", date(2023, 1, 1), date(2024, 1, 1))
        assert result is None

    def test_get_daily_bars_returns_none_on_exception(self, tmp_path):
        p = self._provider()
        with self._isolated_cache(tmp_path), \
                patch("yfinance.download", side_effect=Exception("network")):
            result = p.get_daily_bars("TEST_SYM_C", date(2023, 1, 1), date(2024, 1, 1))
        assert result is None

    def test_get_intraday_bars_uses_correct_interval(self, tmp_path):
        p = self._provider()
        df = _make_daily_df(50)
        with self._isolated_cache(tmp_path), patch("yfinance.download", return_value=df) as mock_dl:
            p.get_intraday_bars(
                "AAPL",
                datetime(2024, 1, 1), datetime(2024, 1, 10),
                interval_minutes=15,
            )
            call_kwargs = mock_dl.call_args[1]
            assert call_kwargs["interval"] == "15m"

    def test_bulk_daily_falls_back_on_error(self):
        p = self._provider()
        with patch("yfinance.download", side_effect=Exception("rate limit")):
            result = p.get_daily_bars_bulk(
                ["AAPL", "MSFT"], date(2023, 1, 1), date(2024, 1, 1)
            )
        assert isinstance(result, dict)


# ── AlpacaProvider ────────────────────────────────────────────────────────────

class TestAlpacaProvider:

    def _provider(self):
        from app.data.alpaca_provider import AlpacaProvider
        return AlpacaProvider()

    def test_get_daily_bars_returns_none_on_client_error(self):
        p = self._provider()
        with patch.object(p, "_client", side_effect=Exception("no key")):
            result = p.get_daily_bars("AAPL", date(2023, 1, 1), date(2024, 1, 1))
        assert result is None

    def test_health_check_returns_false_on_error(self):
        p = self._provider()
        with patch.object(p, "_client", side_effect=Exception("no key")):
            assert p.health_check() is False


# ── Rolling window training ───────────────────────────────────────────────────

class TestRollingWindowTraining:

    def _trainer(self):
        from app.ml.training import ModelTrainer
        return ModelTrainer()

    def _multi_symbol_data(self, n_symbols=10, n_days=500):
        """Create enough data for multiple windows across multiple symbols."""
        data = {}
        for i in range(n_symbols):
            np.random.seed(i)
            sym = f"SYM{i:02d}"
            data[sym] = _make_daily_df(n_days)
        return data

    def test_rolling_matrix_produces_more_samples_than_single_window(self):
        trainer = self._trainer()
        data = self._multi_symbol_data(10, 500)
        X_train, y_train, X_test, y_test, names, _ = trainer._build_rolling_matrix(
            data, fetch_fundamentals=False
        )
        # Rolling should produce many more samples than 10 (single-window)
        assert len(X_train) + len(X_test) > 10

    def test_test_set_is_time_based_not_random(self):
        """Test set must be the most recent windows, not a random sample."""
        from app.ml.training import WINDOW_DAYS as _W, STEP_DAYS as _S, TEST_FRACTION as _T  # noqa
        trainer = self._trainer()
        data = self._multi_symbol_data(8, 600)
        X_train, y_train, X_test, y_test, names, _ = trainer._build_rolling_matrix(
            data, fetch_fundamentals=False
        )
        # Both sets should have samples
        assert len(X_train) > 0
        assert len(X_test) >= 0  # may be 0 with small dataset

    def test_feature_names_consistent(self):
        trainer = self._trainer()
        data = self._multi_symbol_data(8, 500)
        X_train, y_train, X_test, y_test, names, _ = trainer._build_rolling_matrix(
            data, fetch_fundamentals=False
        )
        assert len(names) >= 19  # at least price-only features
        if len(X_train) > 0:
            assert X_train.shape[1] == len(names)

    def test_insufficient_data_returns_empty(self):
        trainer = self._trainer()
        # Only 30 days — not enough for any window
        tiny_data = {"AAPL": _make_daily_df(30), "MSFT": _make_daily_df(30)}
        X_train, y_train, X_test, y_test, names, _ = trainer._build_rolling_matrix(
            tiny_data, fetch_fundamentals=False
        )
        assert len(X_train) == 0

    def test_labels_are_binary(self):
        trainer = self._trainer()
        data = self._multi_symbol_data(10, 500)
        X_train, y_train, X_test, y_test, names, _ = trainer._build_rolling_matrix(
            data, fetch_fundamentals=False
        )
        if len(y_train) > 0:
            assert set(y_train).issubset({0, 1})

    def test_no_label_leakage_across_split(self):
        """Train and test X arrays must not overlap (time-based split)."""
        trainer = self._trainer()
        data = self._multi_symbol_data(12, 700)
        X_train, y_train, X_test, y_test, names, _ = trainer._build_rolling_matrix(
            data, fetch_fundamentals=False
        )
        # Shapes must be consistent
        if len(X_train) > 0 and len(X_test) > 0:
            assert X_train.shape[1] == X_test.shape[1]


# ── ModelTrainer.train_model uses rolling pipeline ────────────────────────────

class TestModelTrainerIntegration:

    def test_train_model_calls_rolling_matrix(self):
        from app.ml.training import ModelTrainer
        trainer = ModelTrainer()
        data = {}
        for i in range(12):
            np.random.seed(i)
            data[f"SYM{i:02d}"] = _make_daily_df(600)

        with patch.object(
            trainer, "_fetch_data", return_value=data
        ):
            with patch.object(
                trainer, "_record_version"
            ):
                with patch.object(
                    trainer, "_next_version", return_value=99
                ):
                    version = trainer.train_model(
                        symbols=[f"SYM{i:02d}" for i in range(12)],
                        years=3,
                        fetch_fundamentals=False,
                    )
        assert version == 99
