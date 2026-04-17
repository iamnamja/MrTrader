"""Tests for Phase 37: options/volatility features."""
import numpy as np
import pandas as pd
import pytest
from datetime import date
from unittest.mock import patch, MagicMock


def _prices(n=252, seed=42):
    np.random.seed(seed)
    p = 100 + np.cumsum(np.random.randn(n) * 0.5)
    h = p * 1.005
    lo = p * 0.995
    return p, h, lo


class TestComputeVolFeatures:

    def test_returns_all_5_keys(self):
        from app.ml.options_features import compute_vol_features
        p, h, lo = _prices()
        result = compute_vol_features(p, h, lo)
        assert set(result.keys()) == {
            "vol_percentile_52w", "vol_regime", "vol_of_vol", "atr_trend", "parkinson_vol"
        }

    def test_vol_percentile_in_range(self):
        from app.ml.options_features import compute_vol_features
        p, h, lo = _prices()
        result = compute_vol_features(p, h, lo)
        assert 0.0 <= result["vol_percentile_52w"] <= 1.0

    def test_vol_regime_positive(self):
        from app.ml.options_features import compute_vol_features
        p, h, lo = _prices()
        result = compute_vol_features(p, h, lo)
        assert result["vol_regime"] > 0

    def test_atr_trend_positive(self):
        from app.ml.options_features import compute_vol_features
        p, h, lo = _prices()
        result = compute_vol_features(p, h, lo)
        assert result["atr_trend"] > 0

    def test_parkinson_vol_positive(self):
        from app.ml.options_features import compute_vol_features
        p, h, lo = _prices()
        result = compute_vol_features(p, h, lo)
        assert result["parkinson_vol"] > 0

    def test_insufficient_data_returns_defaults(self):
        from app.ml.options_features import compute_vol_features
        p = np.array([100.0, 101.0, 99.0])
        result = compute_vol_features(p, p * 1.01, p * 0.99)
        assert result["vol_percentile_52w"] == 0.5
        assert result["vol_regime"] == 1.0

    def test_low_vol_period_has_low_percentile(self):
        from app.ml.options_features import compute_vol_features
        # 252 days quiet then suddenly quiet at end
        np.random.seed(0)
        # High vol earlier, low vol recent
        p_high = 100 + np.cumsum(np.random.randn(200) * 2.0)
        p_low = p_high[-1] + np.cumsum(np.random.randn(52) * 0.1)
        p = np.concatenate([p_high, p_low])
        h, lo = p * 1.005, p * 0.995
        result = compute_vol_features(p, h, lo)
        # Recent vol is much lower than 52w high → percentile should be low
        assert result["vol_percentile_52w"] < 0.4

    def test_high_vol_period_has_high_percentile(self):
        from app.ml.options_features import compute_vol_features
        np.random.seed(1)
        p_low = 100 + np.cumsum(np.random.randn(200) * 0.1)
        p_high = p_low[-1] + np.cumsum(np.random.randn(52) * 3.0)
        p = np.concatenate([p_low, p_high])
        h, lo = p * 1.01, p * 0.99
        result = compute_vol_features(p, h, lo)
        assert result["vol_percentile_52w"] > 0.6

    def test_vol_regime_reflects_short_long_ratio(self):
        """vol_regime = rv10 / rv60; verify it tracks the ratio direction."""
        from app.ml.options_features import compute_vol_features
        import numpy as np
        # 252 days of calm then 10 very volatile days at the end
        p = np.ones(252) * 100.0
        p[-10:] = 100 + np.cumsum(np.random.RandomState(99).randn(10) * 10)
        result = compute_vol_features(p, p * 1.01, p * 0.99)
        # rv10 >> rv60 so regime > 1
        returns = np.diff(np.log(p))
        rv10 = float(np.std(returns[-10:]) * np.sqrt(252))
        rv60 = float(np.std(returns[-60:]) * np.sqrt(252))
        if rv60 > 0:
            expected_regime = min(3.0, rv10 / rv60)
            assert abs(result["vol_regime"] - expected_regime) < 0.01


class TestGetLiveOptionsFeatures:

    def _mock_chain(self, spot=150.0):
        calls = pd.DataFrame({
            "strike": [145, 150, 155],
            "volume": [100, 500, 200],
            "openInterest": [1000, 5000, 2000],
            "impliedVolatility": [0.30, 0.28, 0.32],
        })
        puts = pd.DataFrame({
            "strike": [145, 150, 155],
            "volume": [80, 300, 150],
            "openInterest": [800, 3000, 1500],
            "impliedVolatility": [0.31, 0.29, 0.33],
        })
        chain = MagicMock()
        chain.calls = calls
        chain.puts = puts
        return chain

    def test_returns_3_keys(self):
        from app.ml.options_features import get_live_options_features
        import app.ml.options_features as opt_mod
        opt_mod._options_cache.clear()
        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-05-01", "2026-05-15")
        mock_ticker.option_chain.return_value = self._mock_chain()
        mock_ticker.fast_info.last_price = 150.0
        mock_ticker.history.return_value = pd.DataFrame({
            "Close": 150 + np.random.randn(30) * 1
        })
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_live_options_features("AAPL_OPT_TEST")
        assert "options_put_call_ratio" in result
        assert "options_iv_atm" in result
        assert "options_iv_premium" in result

    def test_put_call_ratio_calculated(self):
        from app.ml.options_features import get_live_options_features
        import app.ml.options_features as opt_mod
        opt_mod._options_cache.clear()
        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-05-01",)
        chain = self._mock_chain()
        mock_ticker.option_chain.return_value = chain
        mock_ticker.fast_info.last_price = 150.0
        mock_ticker.history.return_value = pd.DataFrame({"Close": np.ones(30) * 150})
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_live_options_features("AAPL_PCR_TEST")
        total_call = chain.calls["volume"].sum()
        total_put = chain.puts["volume"].sum()
        expected = total_put / total_call
        assert abs(result["options_put_call_ratio"] - expected) < 0.01

    def test_safe_on_exception(self):
        from app.ml.options_features import get_live_options_features
        import app.ml.options_features as opt_mod
        opt_mod._options_cache.clear()
        with patch("yfinance.Ticker", side_effect=Exception("no data")):
            result = get_live_options_features("FAIL_OPT")
        assert result["options_put_call_ratio"] == 1.0
        assert result["options_iv_atm"] == 0.0

    def test_uses_cache(self):
        from app.ml.options_features import get_live_options_features
        import app.ml.options_features as opt_mod
        opt_mod._options_cache.clear()
        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-05-01",)
        mock_ticker.option_chain.return_value = self._mock_chain()
        mock_ticker.fast_info.last_price = 150.0
        mock_ticker.history.return_value = pd.DataFrame({"Close": np.ones(30) * 150})
        with patch("yfinance.Ticker", return_value=mock_ticker) as mock_yf:
            get_live_options_features("CACHE_OPT_TEST")
            get_live_options_features("CACHE_OPT_TEST")
        assert mock_yf.call_count == 1


class TestFeatureEngineerOptionsIntegration:

    def _make_bars(self, n=252):
        np.random.seed(42)
        idx = pd.bdate_range("2023-01-02", periods=n)
        p = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({
            "open": p * 0.999, "high": p * 1.005,
            "low": p * 0.995, "close": p,
            "volume": np.ones(n) * 1_000_000,
        }, index=idx)

    def test_vol_features_in_output_training_mode(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        df = self._make_bars()
        # Training mode: as_of_date set → no live options call
        result = fe.engineer_features(
            "AAPL", df, fetch_fundamentals=False, as_of_date=date(2023, 12, 31)
        )
        assert result is not None
        assert "vol_percentile_52w" in result
        assert "vol_regime" in result
        assert "vol_of_vol" in result
        assert "atr_trend" in result
        assert "parkinson_vol" in result
        # Live options should be defaults
        assert result["options_put_call_ratio"] == 1.0

    def test_vol_percentile_in_0_1(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        df = self._make_bars()
        result = fe.engineer_features("AAPL", df, fetch_fundamentals=False, as_of_date=date(2023, 12, 31))
        assert 0.0 <= result["vol_percentile_52w"] <= 1.0

    def test_total_feature_count(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        df = self._make_bars()
        result = fe.engineer_features("AAPL", df, fetch_fundamentals=False, as_of_date=date(2023, 12, 31))
        # Should have at least 56 features (48 + 8 FMP defaults + vol features)
        assert len(result) >= 56
