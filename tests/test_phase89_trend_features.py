"""Tests for Phase 89 trend-persistence features added to features.py."""
import numpy as np
import pytest
from app.ml.features import _aroon, _hurst_exponent, FeatureEngineer


class TestAroon:
    def test_aroon_up_at_recent_high(self):
        """When the most recent bar is the 25-period high, aroon_up should be 1.0."""
        highs = np.array([float(i) for i in range(1, 27)])  # strictly ascending, last is max
        lows = np.ones(26)
        up, down = _aroon(highs, lows, period=25)
        assert up == pytest.approx(1.0)

    def test_aroon_down_at_recent_low(self):
        """When the most recent bar is the 25-period low, aroon_down should be 1.0."""
        highs = np.ones(26) * 100.0
        lows = np.array([float(26 - i) for i in range(26)])  # strictly descending, last is min
        up, down = _aroon(highs, lows, period=25)
        assert down == pytest.approx(1.0)

    def test_aroon_returns_half_when_insufficient_data(self):
        highs = np.array([1.0, 2.0])
        lows = np.array([0.5, 1.0])
        up, down = _aroon(highs, lows, period=25)
        assert up == 0.5
        assert down == 0.5

    def test_aroon_clipped_to_0_1(self):
        highs = np.linspace(1, 26, 26)
        lows = np.linspace(0.5, 25.5, 26)
        up, down = _aroon(highs, lows, period=25)
        assert 0.0 <= up <= 1.0
        assert 0.0 <= down <= 1.0


class TestHurstExponent:
    def test_hurst_returns_value_in_range(self):
        """Hurst exponent should always return a value in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            prices = 100 + np.cumsum(rng.normal(0, 1, 120))
            h = _hurst_exponent(prices, max_lag=20)
            assert 0.0 <= h <= 1.0

    def test_returns_half_on_insufficient_data(self):
        prices = np.linspace(100, 110, 10)
        h = _hurst_exponent(prices, max_lag=20)
        assert h == 0.5

    def test_clipped_to_0_1(self):
        prices = np.linspace(100, 200, 200)
        h = _hurst_exponent(prices, max_lag=20)
        assert 0.0 <= h <= 1.0


class TestFeatureEngineerPhase89:
    """Integration tests: engineer_features includes Phase 89 keys."""

    def _make_df(self, n=260):
        import pandas as pd
        rng = np.random.default_rng(7)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        prices = np.abs(prices) + 1
        return pd.DataFrame({
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }, index=dates)

    def test_phase89_keys_present(self):
        fe = FeatureEngineer()
        df = self._make_df(260)
        feats = fe.engineer_features("AAPL", df, sector="Technology")
        assert feats is not None
        for key in ("aroon_up_25", "aroon_down_25", "aroon_oscillator_25",
                    "adx_rising", "adx_14_pct", "pct_closes_above_ema20",
                    "drawdown_from_20d_high", "hurst_exponent_60d",
                    "volatility_adj_dist_52wk_high"):
            assert key in feats, f"Missing feature: {key}"

    def test_aroon_up_range(self):
        fe = FeatureEngineer()
        df = self._make_df(260)
        feats = fe.engineer_features("AAPL", df, sector="Technology")
        assert 0.0 <= feats["aroon_up_25"] <= 1.0
        assert 0.0 <= feats["aroon_down_25"] <= 1.0
        assert -1.0 <= feats["aroon_oscillator_25"] <= 1.0

    def test_pct_closes_above_ema20_range(self):
        fe = FeatureEngineer()
        df = self._make_df(260)
        feats = fe.engineer_features("AAPL", df, sector="Technology")
        assert 0.0 <= feats["pct_closes_above_ema20"] <= 1.0

    def test_drawdown_from_20d_high_nonpositive(self):
        fe = FeatureEngineer()
        df = self._make_df(260)
        feats = fe.engineer_features("AAPL", df, sector="Technology")
        assert feats["drawdown_from_20d_high"] <= 0.0
        assert feats["drawdown_from_20d_high"] >= -1.0

    def test_hurst_range(self):
        fe = FeatureEngineer()
        df = self._make_df(260)
        feats = fe.engineer_features("AAPL", df, sector="Technology")
        assert 0.0 <= feats["hurst_exponent_60d"] <= 1.0

    def test_vol_adj_dist_nonpositive(self):
        fe = FeatureEngineer()
        df = self._make_df(260)
        feats = fe.engineer_features("AAPL", df, sector="Technology")
        assert feats["volatility_adj_dist_52wk_high"] <= 0.0

    def test_short_df_returns_none(self):
        """With only 30 bars (< MIN_BARS=52), engineer_features returns None — no crash."""
        fe = FeatureEngineer()
        df = self._make_df(30)
        feats = fe.engineer_features("AAPL", df, sector="Technology")
        assert feats is None  # expected: insufficient data
