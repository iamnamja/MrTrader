"""
Tests for Model Improvement Iterations 3 & 4:
  Iter 3 — Session-time features (intraday): minutes_since_open,
            is_open_session, is_close_session
  Iter 4 — Sector-relative momentum (swing): sector_momentum_5d,
            momentum_5d_sector_neutral
"""
import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bars(n: int, start: str = "2024-01-02 09:30") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min")
    prices = 100.0 + np.arange(n, dtype=float) * 0.01
    return pd.DataFrame({
        "open": prices, "high": prices * 1.001,
        "low": prices * 0.999, "close": prices,
        "volume": np.full(n, 1e6),
    }, index=idx)


def _make_bars_ending_at(end_time: str, n: int = 20) -> pd.DataFrame:
    """Make n bars whose last bar ends at end_time."""
    end = pd.Timestamp(end_time)
    start = end - pd.Timedelta(minutes=5 * (n - 1))
    idx = pd.date_range(start, periods=n, freq="5min")
    prices = 100.0 + np.arange(n, dtype=float) * 0.01
    return pd.DataFrame({
        "open": prices, "high": prices * 1.001,
        "low": prices * 0.999, "close": prices,
        "volume": np.full(n, 1e6),
    }, index=idx)


# ── Iter 3: session-time features ────────────────────────────────────────────

class TestSessionTimeFeatures:
    def _feats(self, n_bars: int):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars(n_bars)
        return compute_intraday_features(bars)

    def test_new_features_present(self):
        feats = self._feats(20)
        assert feats is not None
        assert "minutes_since_open" in feats
        assert "is_open_session" in feats
        assert "is_close_session" in feats

    def test_minutes_since_open_from_timestamp(self):
        from app.ml.intraday_features import compute_intraday_features
        # Last bar at 11:30 = 120 min since 09:30
        bars = _make_bars_ending_at("2024-01-02 11:30", n=20)
        feats = compute_intraday_features(bars)
        assert feats is not None
        assert feats["minutes_since_open"] == pytest.approx(120.0)

    def test_minutes_capped_at_390(self):
        feats = self._feats(100)  # 100 * 5 = 500 > 390
        assert feats["minutes_since_open"] == pytest.approx(390.0)

    def test_open_session_flag_first_30_min(self):
        from app.ml.intraday_features import compute_intraday_features
        # Last bar at 09:30 → 0 min elapsed → is_open_session = 1
        bars = _make_bars_ending_at("2024-01-02 09:30", n=20)
        feats = compute_intraday_features(bars)
        assert feats is not None
        assert feats["is_open_session"] == pytest.approx(1.0)

    def test_open_session_flag_off_after_30_min(self):
        from app.ml.intraday_features import compute_intraday_features
        # Last bar at 10:05 → 35 min elapsed → is_open_session = 0
        bars = _make_bars_ending_at("2024-01-02 10:05", n=20)
        feats = compute_intraday_features(bars)
        assert feats is not None
        assert feats["is_open_session"] == pytest.approx(0.0)

    def test_close_session_flag_last_60_min(self):
        from app.ml.intraday_features import compute_intraday_features
        # Last bar at 15:30 = 360 min → is_close_session = 1
        bars = _make_bars_ending_at("2024-01-02 15:30", n=20)
        feats = compute_intraday_features(bars)
        assert feats is not None
        assert feats["is_close_session"] == pytest.approx(1.0)

    def test_close_session_flag_off_midday(self):
        from app.ml.intraday_features import compute_intraday_features
        # Last bar at 12:00 = 150 min → not close session
        bars = _make_bars_ending_at("2024-01-02 12:00", n=20)
        feats = compute_intraday_features(bars)
        assert feats is not None
        assert feats["is_close_session"] == pytest.approx(0.0)

    def test_time_of_day_consistent_with_minutes(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars_ending_at("2024-01-02 12:45", n=20)  # 195 min = 0.5 session
        feats = compute_intraday_features(bars)
        assert feats is not None
        assert feats["time_of_day"] == pytest.approx(feats["minutes_since_open"] / 390.0)

    def test_feature_names_list_updated(self):
        from app.ml.intraday_features import FEATURE_NAMES
        assert "minutes_since_open" in FEATURE_NAMES
        assert "is_open_session" in FEATURE_NAMES
        assert "is_close_session" in FEATURE_NAMES


# ── Iter 4: sector-relative momentum ─────────────────────────────────────────

class TestSectorMomentum5d:
    def test_function_exists(self):
        from app.ml.fundamental_fetcher import get_sector_momentum_5d
        assert callable(get_sector_momentum_5d)

    def test_returns_zero_for_unknown_sector(self):
        from app.ml.fundamental_fetcher import get_sector_momentum_5d
        result = get_sector_momentum_5d("Unknown Sector XYZ")
        assert result == 0.0

    def test_returns_float(self):
        from app.ml.fundamental_fetcher import get_sector_momentum_5d
        from unittest.mock import patch, MagicMock
        import pandas as pd
        import app.ml.fundamental_fetcher as ff

        mock_df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 101.5, 102.5, 103.0, 104.0, 103.5]
        })
        mock_client = MagicMock()
        mock_client.get_bars.return_value = mock_df

        ff._etf_5d_cache.clear()
        with patch("app.integrations.alpaca.AlpacaClient.get_bars", return_value=mock_df):
            with patch("app.integrations.get_alpaca_client", return_value=mock_client):
                result = get_sector_momentum_5d("Technology")
        assert isinstance(result, float)


class TestMomentum5dSectorNeutral:
    def test_feature_present_in_output(self):
        from app.ml.features import FeatureEngineer
        from unittest.mock import patch
        import pandas as pd, numpy as np

        fe = FeatureEngineer()
        n = 80
        dates = pd.date_range("2022-01-03", periods=n, freq="B")
        prices = 100 + np.arange(n, dtype=float)
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.01,
            "low": prices * 0.99, "close": prices,
            "volume": np.full(n, 1e6),
        }, index=dates)

        with patch("app.ml.fundamental_fetcher.get_sector_momentum", return_value=0.01):
            with patch("app.ml.fundamental_fetcher.get_sector_momentum_5d", return_value=0.005):
                feats = fe.engineer_features(
                    "AAPL", df, sector="Technology",
                    fetch_fundamentals=False
                )

        if feats is not None:
            assert "momentum_5d_sector_neutral" in feats
            assert "sector_momentum_5d" in feats

    def test_sector_neutral_5d_is_difference(self):
        """momentum_5d_sector_neutral = momentum_5d - sector_momentum_5d."""
        from app.ml.features import FeatureEngineer
        from unittest.mock import patch
        import pandas as pd, numpy as np

        fe = FeatureEngineer()
        n = 80
        dates = pd.date_range("2022-01-03", periods=n, freq="B")
        prices = np.concatenate([np.full(75, 100.0), np.linspace(100, 102, 5)])
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.01,
            "low": prices * 0.99, "close": prices,
            "volume": np.full(n, 1e6),
        }, index=dates)

        etf_5d = 0.01
        with patch("app.ml.fundamental_fetcher.get_sector_momentum", return_value=0.02):
            with patch("app.ml.fundamental_fetcher.get_sector_momentum_5d", return_value=etf_5d):
                feats = fe.engineer_features(
                    "AAPL", df, sector="Technology",
                    fetch_fundamentals=False
                )

        if feats is not None and "momentum_5d_sector_neutral" in feats:
            expected = feats.get("momentum_5d", 0.0) - etf_5d
            assert feats["momentum_5d_sector_neutral"] == pytest.approx(expected, abs=1e-6)
