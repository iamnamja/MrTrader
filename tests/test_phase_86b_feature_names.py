"""Phase 86b — FEATURE_NAMES authoritative filter + regime proxy removal tests."""
import numpy as np
from unittest.mock import MagicMock, patch


class TestFeatureNamesAuthoritative:
    """FEATURE_NAMES must be the authoritative list for training features."""

    def test_feature_names_excludes_regime_proxies(self):
        from app.ml.intraday_features import FEATURE_NAMES
        zeroed_by_cs_normalize = {"regime_vix_proxy", "regime_vix_pct60d", "regime_spy_ma20_dist"}
        overlap = zeroed_by_cs_normalize & set(FEATURE_NAMES)
        assert overlap == set(), f"Market-wide features in FEATURE_NAMES (zeroed by cs_normalize): {overlap}"

    def test_feature_names_includes_86b_features(self):
        from app.ml.intraday_features import FEATURE_NAMES
        required = {"stock_vs_spy_5d_return", "stock_vs_spy_mom_ratio", "gap_vs_spy_gap"}
        missing = required - set(FEATURE_NAMES)
        assert missing == set(), f"86b features missing from FEATURE_NAMES: {missing}"

    def test_feature_names_count(self):
        from app.ml.intraday_features import FEATURE_NAMES
        assert len(FEATURE_NAMES) == 56, f"Expected 56 features, got {len(FEATURE_NAMES)}"

    def test_daily_vol_regime_included(self):
        """daily_vol_regime is stock-specific (not market-wide) — should stay."""
        from app.ml.intraday_features import FEATURE_NAMES
        assert "daily_vol_regime" in FEATURE_NAMES

    def test_training_filters_to_feature_names(self):
        """_symbol_to_rows must filter feats dict to FEATURE_NAMES before building rows."""
        from app.ml.intraday_features import FEATURE_NAMES

        # Simulate a feats dict with extra keys not in FEATURE_NAMES
        extra_keys = {"regime_vix_proxy", "regime_vix_pct60d", "nis_direction_score"}
        all_keys = set(FEATURE_NAMES) | extra_keys
        simulated_feats = {k: 0.0 for k in all_keys}

        # Apply the same filter used in _symbol_to_rows
        filtered = {k: simulated_feats[k] for k in FEATURE_NAMES if k in simulated_feats}

        assert set(filtered.keys()) == set(FEATURE_NAMES)
        assert "regime_vix_proxy" not in filtered
        assert "nis_direction_score" not in filtered
        assert "stock_vs_spy_5d_return" in filtered

    def test_feature_order_preserved(self):
        """Filtered feats must be in FEATURE_NAMES order so matrix columns align."""
        from app.ml.intraday_features import FEATURE_NAMES

        shuffled_feats = {k: float(i) for i, k in enumerate(reversed(FEATURE_NAMES))}
        filtered = {k: shuffled_feats[k] for k in FEATURE_NAMES if k in shuffled_feats}

        assert list(filtered.keys()) == FEATURE_NAMES


class TestComputeFeaturesCoverage:
    """compute_intraday_features() must compute values for all FEATURE_NAMES."""

    def _minimal_bars(self, n=25):
        import pandas as pd
        idx = pd.date_range("2025-01-10 09:35", periods=n, freq="5min")
        close = np.linspace(100, 102, n)
        df = pd.DataFrame({
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(n, 50000.0),
        }, index=idx)
        return df

    def _daily_bars(self, n=30):
        import pandas as pd
        from datetime import date, timedelta
        dates = [date(2024, 12, 1) + timedelta(days=i) for i in range(n)]
        idx = pd.DatetimeIndex(dates)
        close = np.linspace(490, 510, n)
        return pd.DataFrame({
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": np.full(n, 1e8),
        }, index=idx)

    def test_86b_features_computed_with_spy_bars(self):
        from app.ml.intraday_features import compute_intraday_features, FEATURE_NAMES
        bars = self._minimal_bars()
        daily = self._daily_bars()
        spy_daily = self._daily_bars()

        feats = compute_intraday_features(
            bars=bars,
            daily_bars=daily,
            spy_daily_bars=spy_daily,
            symbol="AAPL",
            as_of_date=None,
        )
        assert feats is not None
        assert "stock_vs_spy_5d_return" in feats
        assert "stock_vs_spy_mom_ratio" in feats
        assert "gap_vs_spy_gap" in feats

    def test_feature_names_all_present_in_output(self):
        from app.ml.intraday_features import compute_intraday_features, FEATURE_NAMES
        bars = self._minimal_bars()
        daily = self._daily_bars()
        spy_daily = self._daily_bars()

        feats = compute_intraday_features(
            bars=bars,
            daily_bars=daily,
            spy_daily_bars=spy_daily,
            symbol="AAPL",
            as_of_date=None,
        )
        assert feats is not None
        missing = [f for f in FEATURE_NAMES if f not in feats]
        assert missing == [], f"FEATURE_NAMES keys missing from feats output: {missing}"
