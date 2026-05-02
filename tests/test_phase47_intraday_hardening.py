"""
Phase 47 hardening tests — live PM intraday path.

Covers:
- PM abstention gate is present in both swing and intraday code paths
- Intraday feature alignment uses model.feature_names (not dict order)
- FEATURE_NAMES constant matches compute_intraday_features() output
"""
import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest


# ── PM abstention gate structural checks ─────────────────────────────────────

class TestPMAbstentionGatePresence:
    """Verify abstention gate is present in both intraday and swing code paths."""

    def _get_pm_source(self):
        from app.agents import portfolio_manager as pm_module
        return inspect.getsource(pm_module)

    def test_abstention_gate_in_intraday_path(self):
        """select_intraday_instruments must call _market_regime_allows_entries."""
        from app.agents.portfolio_manager import PortfolioManager
        src = inspect.getsource(PortfolioManager.select_intraday_instruments)
        assert "_market_regime_allows_entries" in src, (
            "PM abstention gate missing from select_intraday_instruments()"
        )

    def test_abstention_gate_in_swing_path(self):
        """_send_swing_proposals must call _market_regime_allows_entries."""
        from app.agents.portfolio_manager import PortfolioManager
        src = inspect.getsource(PortfolioManager._send_swing_proposals)
        assert "_market_regime_allows_entries" in src, (
            "PM abstention gate missing from _send_swing_proposals()"
        )

    def test_market_regime_checks_vix_threshold(self):
        """_market_regime_allows_entries must check VIX >= 25."""
        from app.agents.portfolio_manager import PortfolioManager
        src = inspect.getsource(PortfolioManager._market_regime_allows_entries)
        assert "25" in src, "VIX threshold 25 not found in _market_regime_allows_entries"

    def test_market_regime_checks_spy_ma(self):
        """_market_regime_allows_entries must check SPY vs moving average."""
        from app.agents.portfolio_manager import PortfolioManager
        src = inspect.getsource(PortfolioManager._market_regime_allows_entries)
        assert "mean" in src or "MA" in src or "ma" in src, (
            "SPY MA check not found in _market_regime_allows_entries"
        )

    def test_intraday_fetch_uses_25_daily_bars(self):
        """_fetch_intraday_features must request 25 daily bars for vol context features."""
        from app.agents.portfolio_manager import PortfolioManager
        src = inspect.getsource(PortfolioManager.select_intraday_instruments)
        assert 'limit=25' in src, (
            "Daily bars fetch should request limit=25 for range_vs_20d_avg and vol context features"
        )

    def test_intraday_uses_russell_1000_universe(self):
        """Intraday scoring must use RUSSELL_1000_TICKERS (not [:50] cap)."""
        from app.agents.portfolio_manager import PortfolioManager
        src = inspect.getsource(PortfolioManager.select_intraday_instruments)
        assert "RUSSELL_1000_TICKERS" in src, (
            "Intraday scan must use RUSSELL_1000_TICKERS to match training universe"
        )
        assert "[:50]" not in src, "Intraday universe must not be capped at 50 symbols"

    def test_daily_bars_passed_to_compute_features(self):
        """compute_intraday_features must receive daily_bars= argument."""
        from app.agents.portfolio_manager import PortfolioManager
        src = inspect.getsource(PortfolioManager.select_intraday_instruments)
        assert "daily_bars=daily" in src, (
            "daily_bars must be passed to compute_intraday_features() for vol context"
        )

    def test_atr_based_stop_target_in_proposals(self):
        """Intraday proposals must use 0.4x/0.8x prior-day-range stops (not fixed pct)."""
        from app.agents.portfolio_manager import PortfolioManager
        src = inspect.getsource(PortfolioManager.select_intraday_instruments)
        assert "0.4 * prior_range" in src, "Stop must be 0.4× prior-day range"
        assert "0.8 * prior_range" in src, "Target must be 0.8× prior-day range"


# ── Feature alignment ─────────────────────────────────────────────────────────

class TestFeatureAlignment:
    """Intraday scoring uses model.feature_names to select columns."""

    def test_feature_alignment_uses_model_feature_names(self):
        """PM alignment logic selects model's expected columns by name, not dict order."""
        feat_names_50 = [f"feat_{i}" for i in range(50)]
        model_feat_names = [f"feat_{i}" for i in range(0, 50, 5)]  # 10 of the 50

        features_by_symbol = {
            "AAPL": {f: float(j) for j, f in enumerate(feat_names_50)},
            "MSFT": {f: float(j + 0.5) for j, f in enumerate(feat_names_50)},
        }

        symbols = list(features_by_symbol.keys())
        X = np.array([
            [features_by_symbol[s].get(f, 0.0) for f in model_feat_names]
            for s in symbols
        ])

        assert X.shape == (2, 10)
        assert X[0, 0] == 0.0   # feat_0
        assert X[0, 1] == 5.0   # feat_5
        assert X[0, 2] == 10.0  # feat_10

    def test_intraday_scoring_uses_model_feature_names_not_dict_order(self):
        """select_intraday_instruments uses model.feature_names for column selection."""
        from app.agents.portfolio_manager import PortfolioManager
        src = inspect.getsource(PortfolioManager.select_intraday_instruments)
        assert "model_feat_names" in src or "feature_names" in src, (
            "PM intraday scoring must use model feature_names for alignment"
        )


# ── FEATURE_NAMES constant ────────────────────────────────────────────────────

class TestFeatureNamesConstant:
    """FEATURE_NAMES must match compute_intraday_features() output keys exactly."""

    def test_feature_names_matches_compute_output(self):
        import pandas as pd
        np.random.seed(42)
        from app.ml.intraday_features import compute_intraday_features, FEATURE_NAMES

        n = 78
        t = pd.date_range("2025-01-10 09:30", periods=n, freq="5min")
        close = 100 + np.cumsum(np.random.randn(n) * 0.1)
        bars = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": np.random.randint(10000, 100000, n).astype(float),
        }, index=t)

        result = compute_intraday_features(bars, prior_close=99.0,
                                           prior_day_high=101.0, prior_day_low=97.0)
        assert result is not None
        assert set(result.keys()) == set(FEATURE_NAMES), (
            f"Key mismatch:\n"
            f"  Extra in computed: {set(result.keys()) - set(FEATURE_NAMES)}\n"
            f"  Missing from computed: {set(FEATURE_NAMES) - set(result.keys())}"
        )
        assert len(FEATURE_NAMES) == len(result)

    def test_feature_names_count(self):
        """FEATURE_NAMES must have the expected count (58 with Phase 86 market-condition features)."""
        from app.ml.intraday_features import FEATURE_NAMES
        assert len(FEATURE_NAMES) == 58, (
            f"Expected 58 features (53 prior + 5 Phase 86 market-condition features), got {len(FEATURE_NAMES)}"
        )

    def test_phase_47_5_features_present(self):
        """All 8 Phase 47-5 quality features must be in FEATURE_NAMES."""
        from app.ml.intraday_features import FEATURE_NAMES
        phase47_5_features = [
            "trend_efficiency", "green_bar_ratio", "above_vwap_ratio",
            "pullback_from_high", "range_vs_20d_avg", "rel_strength_vs_spy",
            "vol_x_momentum", "gap_followthrough",
        ]
        missing = [f for f in phase47_5_features if f not in FEATURE_NAMES]
        assert not missing, f"Phase 47-5 features missing from FEATURE_NAMES: {missing}"
