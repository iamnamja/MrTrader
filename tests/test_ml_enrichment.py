"""Tests for Phase 28: ML feature enrichment."""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bars(n=100):
    """Minimal OHLCV DataFrame with enough rows for all features."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "close": prices,
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    })


# ── FeatureEngineer — new features present ────────────────────────────────────

class TestFeatureEngineerEnriched:

    def _fe(self):
        from app.ml.features import FeatureEngineer
        return FeatureEngineer()

    def test_returns_dict_without_fundamentals(self):
        fe = self._fe()
        result = fe.engineer_features("AAPL", _make_bars(), fetch_fundamentals=False)
        assert result is not None
        assert isinstance(result, dict)

    def test_new_fundamental_keys_present(self):
        fe = self._fe()
        result = fe.engineer_features("AAPL", _make_bars(), fetch_fundamentals=False)
        for key in ["pe_ratio", "pb_ratio", "profit_margin", "revenue_growth",
                    "debt_to_equity", "earnings_proximity_days"]:
            assert key in result

    def test_new_enrichment_keys_present(self):
        fe = self._fe()
        result = fe.engineer_features("AAPL", _make_bars(), fetch_fundamentals=False)
        for key in ["sector_momentum", "insider_score", "earnings_surprise", "regime_score"]:
            assert key in result

    def test_total_feature_count_at_least_25(self):
        fe = self._fe()
        result = fe.engineer_features("AAPL", _make_bars(), fetch_fundamentals=False)
        assert len(result) >= 25

    def test_original_technical_features_still_present(self):
        fe = self._fe()
        result = fe.engineer_features("AAPL", _make_bars(), fetch_fundamentals=False)
        for key in ["rsi_14", "macd", "ema_20", "momentum_20d", "volatility"]:
            assert key in result

    def test_regime_score_used_when_provided(self):
        fe = self._fe()
        result = fe.engineer_features(
            "AAPL", _make_bars(), fetch_fundamentals=False, regime_score=0.75
        )
        assert result["regime_score"] == pytest.approx(0.75)

    def test_sector_momentum_zero_without_fetch(self):
        fe = self._fe()
        result = fe.engineer_features(
            "AAPL", _make_bars(), fetch_fundamentals=False, sector="Technology"
        )
        assert result["sector_momentum"] == 0.0

    def test_returns_none_insufficient_bars(self):
        fe = self._fe()
        assert fe.engineer_features("AAPL", _make_bars(10), fetch_fundamentals=False) is None

    def test_defaults_when_fundamentals_fetcher_raises(self):
        fe = self._fe()
        with patch("app.ml.fundamental_fetcher.get_fundamentals", side_effect=Exception("fail")):
            result = fe.engineer_features("AAPL", _make_bars(), fetch_fundamentals=True)
        assert result is not None
        assert result["pe_ratio"] == 0.0

    def test_yfinance_columns_normalised(self):
        fe = self._fe()
        bars = _make_bars()
        bars.columns = [c.capitalize() for c in bars.columns]  # yfinance style
        result = fe.engineer_features("AAPL", bars, fetch_fundamentals=False)
        assert result is not None


# ── get_fundamentals ──────────────────────────────────────────────────────────

class TestGetFundamentals:

    def test_returns_dict_with_all_keys(self):
        from app.ml.fundamental_fetcher import get_fundamentals
        mock_info = {
            "trailingPE": 25.0, "priceToBook": 3.0,
            "profitMargins": 0.20, "revenueGrowth": 0.10,
            "debtToEquity": 50.0,
        }
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_ticker.calendar = None
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_fundamentals("TEST")
        assert result["pe_ratio"] == pytest.approx(25.0)
        assert result["pb_ratio"] == pytest.approx(3.0)
        assert result["profit_margin"] == pytest.approx(0.20)
        assert result["revenue_growth"] == pytest.approx(0.10)
        assert result["debt_to_equity"] == pytest.approx(0.5)  # /100

    def test_safe_defaults_on_exception(self):
        from app.ml.fundamental_fetcher import get_fundamentals
        with patch("yfinance.Ticker", side_effect=Exception("network")):
            result = get_fundamentals("FAIL")
        assert result["pe_ratio"] == 0.0
        assert result["earnings_proximity_days"] == 90.0

    def test_caps_pe_at_200(self):
        from app.ml.fundamental_fetcher import get_fundamentals
        mock_ticker = MagicMock()
        mock_ticker.info = {"trailingPE": 5000.0}
        mock_ticker.calendar = None
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_fundamentals("HIGH_PE")
        assert result["pe_ratio"] == pytest.approx(200.0)


# ── get_sector_momentum ───────────────────────────────────────────────────────

class TestGetSectorMomentum:

    def setup_method(self):
        import app.ml.fundamental_fetcher as ff
        ff._etf_cache.clear()

    def test_returns_float(self):
        import app.ml.fundamental_fetcher as ff
        ff._etf_cache.clear()
        import pandas as pd
        prices = list(range(100, 130))
        df = pd.DataFrame({
            "close": prices,
            "open": prices, "high": prices, "low": prices, "volume": [1e6] * 30,
        })
        with patch("yfinance.download", return_value=df):
            result = ff.get_sector_momentum("Technology")
        assert isinstance(result, float)

    def test_unknown_sector_returns_zero(self):
        from app.ml.fundamental_fetcher import get_sector_momentum
        assert get_sector_momentum("Unknown Sector XYZ") == 0.0

    def test_safe_on_download_failure(self):
        import app.ml.fundamental_fetcher as ff
        ff._etf_cache.clear()
        with patch("yfinance.download", side_effect=Exception("timeout")):
            result = ff.get_sector_momentum("Technology")
        assert result == 0.0


# ── get_earnings_surprise ─────────────────────────────────────────────────────

class TestGetEarningsSurprise:

    def setup_method(self):
        import app.ml.fundamental_fetcher as ff
        ff._av_cache.clear()

    def test_returns_zero_without_api_key(self):
        import app.ml.fundamental_fetcher as ff
        ff._av_cache.clear()
        with patch("app.ml.fundamental_fetcher.requests.get"):
            result = ff.get_earnings_surprise("AV_NO_KEY", api_key=None)
        assert result == 0.0

    def test_positive_surprise(self):
        import app.ml.fundamental_fetcher as ff
        ff._av_cache.clear()
        payload = {
            "quarterlyEarnings": [{"estimatedEPS": "2.00", "reportedEPS": "2.20"}]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        with patch("app.ml.fundamental_fetcher.requests.get", return_value=mock_resp):
            result = ff.get_earnings_surprise("AV_POS", api_key="test_key")
        assert result == pytest.approx(0.10)

    def test_negative_surprise(self):
        import app.ml.fundamental_fetcher as ff
        ff._av_cache.clear()
        payload = {
            "quarterlyEarnings": [{"estimatedEPS": "2.00", "reportedEPS": "1.80"}]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        with patch("app.ml.fundamental_fetcher.requests.get", return_value=mock_resp):
            result = ff.get_earnings_surprise("AV_NEG", api_key="test_key")
        assert result == pytest.approx(-0.10)

    def test_safe_on_request_failure(self):
        import app.ml.fundamental_fetcher as ff
        ff._av_cache.clear()
        with patch("app.ml.fundamental_fetcher.requests.get", side_effect=Exception("timeout")):
            result = ff.get_earnings_surprise("AV_FAIL", api_key="test_key")
        assert result == 0.0


# ── get_insider_score ─────────────────────────────────────────────────────────

class TestGetInsiderScore:

    def test_safe_on_request_failure(self):
        from app.ml.fundamental_fetcher import get_insider_score
        with patch("app.ml.fundamental_fetcher.requests.get", side_effect=Exception("timeout")):
            result = get_insider_score("AAPL")
        assert result == 0.0

    def test_returns_float_in_range(self):
        from app.ml.fundamental_fetcher import get_insider_score
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"hits": {"hits": []}}
        with patch("app.ml.fundamental_fetcher.requests.get", return_value=mock_resp):
            result = get_insider_score("AAPL")
        assert -1.0 <= result <= 1.0


# ── ModelTrainer integration ──────────────────────────────────────────────────

class TestModelTrainerEnriched:

    def test_build_feature_matrix_passes_sector(self):
        from app.ml.training import ModelTrainer
        trainer = ModelTrainer()

        bars = _make_bars(200)
        symbols_data = {"AAPL": bars, "MSFT": bars}
        labels = {"AAPL": 1, "MSFT": 0}

        with patch.object(
            trainer.feature_engineer, "engineer_features",
            wraps=trainer.feature_engineer.engineer_features
        ) as mock_fe:
            trainer._build_feature_matrix(
                symbols_data, labels, fetch_fundamentals=False
            )
            # Verify sector was passed for known symbols
            calls = mock_fe.call_args_list
            aapl_call = next(c for c in calls if c.args[0] == "AAPL")
            assert aapl_call.kwargs.get("sector") == "Technology"

    def test_feature_matrix_shape_correct(self):
        from app.ml.training import ModelTrainer
        trainer = ModelTrainer()
        bars = _make_bars(200)
        symbols_data = {"AAPL": bars, "MSFT": bars}
        labels = {"AAPL": 1, "MSFT": 0}

        X, y, names = trainer._build_feature_matrix(
            symbols_data, labels, fetch_fundamentals=False
        )
        assert len(X) == 2
        assert len(y) == 2
        assert len(names) >= 25
