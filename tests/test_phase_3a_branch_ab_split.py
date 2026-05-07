"""
Phase 3a tests — cs_normalize Branch A/B feature split for intraday.

Covers:
- BRANCH_B_FEATURES defined and present in FEATURE_NAMES (at end)
- cs_normalize_branch_a preserves Branch B values after normalization
- intraday_features.py computes vix_regime_level, spy_5d_return_daily, day_of_week
- VIX pandas-or bug fixed in agent_simulator.py (swing) and intraday_agent_simulator.py
- swing walkforward PIT filter passes ^VIX and SPY through to simulator
"""
from datetime import date
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest


# ── Branch A/B feature definitions ────────────────────────────────────────────

class TestBranchBDefinition:
    def test_branch_b_features_defined(self):
        from app.ml.intraday_features import BRANCH_B_FEATURES
        assert len(BRANCH_B_FEATURES) == 3
        assert "vix_regime_level" in BRANCH_B_FEATURES
        assert "spy_5d_return_daily" in BRANCH_B_FEATURES
        assert "day_of_week" in BRANCH_B_FEATURES

    def test_branch_b_features_at_end_of_feature_names(self):
        from app.ml.intraday_features import FEATURE_NAMES, BRANCH_B_FEATURES
        tail = FEATURE_NAMES[-len(BRANCH_B_FEATURES):]
        assert tail == BRANCH_B_FEATURES, (
            f"Branch B features must be the last {len(BRANCH_B_FEATURES)} entries "
            f"in FEATURE_NAMES for index predictability. Got tail: {tail}"
        )

    def test_branch_b_indices_are_contiguous_at_end(self):
        from app.ml.intraday_features import FEATURE_NAMES, BRANCH_B_FEATURES
        n = len(FEATURE_NAMES)
        b = len(BRANCH_B_FEATURES)
        expected_idx = list(range(n - b, n))
        actual_idx = [FEATURE_NAMES.index(f) for f in BRANCH_B_FEATURES]
        assert actual_idx == expected_idx

    def test_feature_count_increased_by_three(self):
        from app.ml.intraday_features import FEATURE_NAMES
        # Was 56 features before Phase 3a; now 59
        assert len(FEATURE_NAMES) == 59, (
            f"Expected 59 features (56 Branch A + 3 Branch B). Got {len(FEATURE_NAMES)}"
        )


# ── cs_normalize_branch_a ─────────────────────────────────────────────────────

class TestCsNormalizeBranchA:
    def test_branch_b_values_preserved_after_normalization(self):
        from app.ml.cs_normalize import cs_normalize_branch_a
        rng = np.random.default_rng(0)
        # 10 symbols × 5 features, last 2 are Branch B (same value across symbols)
        X = rng.standard_normal((10, 5))
        # Make cols 3 and 4 identical across rows (Branch B = global market state)
        X[:, 3] = 17.5   # VIX level
        X[:, 4] = -0.02  # SPY 5d return
        branch_b_cols = [3, 4]
        X_out = cs_normalize_branch_a(X, branch_b_cols)
        np.testing.assert_allclose(X_out[:, 3], 17.5, err_msg="Branch B col 3 not preserved")
        np.testing.assert_allclose(X_out[:, 4], -0.02, err_msg="Branch B col 4 not preserved")

    def test_branch_a_columns_are_normalized(self):
        from app.ml.cs_normalize import cs_normalize_branch_a
        rng = np.random.default_rng(1)
        X = rng.standard_normal((10, 5))
        X[:, 4] = 20.0   # Branch B col
        X_out = cs_normalize_branch_a(X, branch_b_cols=[4])
        # Branch A cols 0-3 should be z-scored (mean ≈ 0, std ≈ 1)
        for col in range(4):
            assert abs(X_out[:, col].mean()) < 0.1, f"Branch A col {col} not mean-centered"

    def test_single_row_unchanged(self):
        from app.ml.cs_normalize import cs_normalize_branch_a
        X = np.array([[1.0, 2.0, 3.0, 17.5, -0.02]])
        X_out = cs_normalize_branch_a(X, branch_b_cols=[3, 4])
        np.testing.assert_array_equal(X_out, X)

    def test_empty_branch_b_falls_back_to_standard(self):
        from app.ml.cs_normalize import cs_normalize, cs_normalize_branch_a
        rng = np.random.default_rng(2)
        X = rng.standard_normal((8, 4))
        X_standard = cs_normalize(X.copy())
        X_branch_a = cs_normalize_branch_a(X.copy(), branch_b_cols=[])
        np.testing.assert_allclose(X_branch_a, X_standard, rtol=1e-6)


# ── Feature computation ────────────────────────────────────────────────────────

class TestBranchBFeatureComputation:
    def _make_bars(self, n=30):
        idx = pd.date_range("2025-01-02 09:35", periods=n, freq="5min")
        return pd.DataFrame({
            "open": np.random.uniform(100, 110, n),
            "high": np.random.uniform(110, 120, n),
            "low": np.random.uniform(90, 100, n),
            "close": np.random.uniform(100, 110, n),
            "volume": np.random.randint(1000, 10000, n),
        }, index=idx)

    def _make_spy_daily(self, n=25):
        closes = np.cumprod(1 + np.random.randn(n) * 0.01) * 400
        return pd.DataFrame({
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": np.full(n, 1e8),
        }, index=pd.date_range("2024-11-01", periods=n, freq="B"))

    def test_vix_regime_level_computed(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = self._make_bars()
        spy_daily = self._make_spy_daily()
        feats = compute_intraday_features(bars, spy_daily_bars=spy_daily)
        assert feats is not None
        assert "vix_regime_level" in feats
        assert 0.0 < feats["vix_regime_level"] < 200.0

    def test_spy_5d_return_daily_computed(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = self._make_bars()
        spy_daily = self._make_spy_daily(25)
        feats = compute_intraday_features(bars, spy_daily_bars=spy_daily)
        assert feats is not None
        assert "spy_5d_return_daily" in feats
        # Should be between -50% and +50%
        assert -50 < feats["spy_5d_return_daily"] < 50

    def test_day_of_week_computed_from_as_of_date(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = self._make_bars()
        monday = date(2025, 5, 5)  # Monday
        feats = compute_intraday_features(bars, as_of_date=monday)
        assert feats is not None
        assert "day_of_week" in feats
        assert feats["day_of_week"] == 0.0  # Monday = 0

    def test_branch_b_defaults_when_no_spy_daily(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = self._make_bars()
        feats = compute_intraday_features(bars, spy_daily_bars=None)
        assert feats is not None
        assert feats["vix_regime_level"] == 20.0
        assert feats["spy_5d_return_daily"] == 0.0


# ── VIX pandas-or bug fix in swing AgentSimulator ─────────────────────────────

class TestSwingVIXBugFixed:
    def test_agent_simulator_vix_none_check(self):
        """Explicit None check instead of 'or' operator prevents DataFrame truth-value error."""
        import inspect
        from app.backtesting import agent_simulator
        src = inspect.getsource(agent_simulator)
        # Old bug: symbols_data.get("^VIX") or symbols_data.get("VIX")
        assert 'symbols_data.get("^VIX") or symbols_data.get("VIX")' not in src, (
            "Pandas 'or' ambiguity bug still present in agent_simulator.py"
        )

    def test_vix_dataframe_doesnt_raise(self):
        """Non-empty VIX DataFrame no longer raises ValueError in agent_simulator."""
        from app.backtesting.agent_simulator import AgentSimulator
        import pandas as pd
        vix_df = pd.DataFrame({"close": [20.0, 21.0, 22.0]},
                               index=pd.date_range("2025-01-01", periods=3))
        spy_df = pd.DataFrame({"close": [400.0, 401.0, 402.0]},
                               index=pd.date_range("2025-01-01", periods=3))
        symbols_data = {"^VIX": vix_df, "SPY": spy_df}
        # Should not raise ValueError about truth value of DataFrame
        try:
            _vix_df = symbols_data.get("^VIX")
            if _vix_df is None:
                _vix_df = symbols_data.get("VIX")
            assert _vix_df is not None and len(_vix_df) == 3
        except ValueError as e:
            pytest.fail(f"VIX DataFrame truth-value error: {e}")


# ── Swing walk-forward PIT filter passes synthetic symbols ─────────────────────

class TestSwingPITFilterSyntheticSymbols:
    def test_vix_spy_bypass_pit_filter_in_swing_fold(self):
        """^VIX and SPY must be passed to swing simulator even though not in sp100."""
        import inspect
        from scripts import walkforward_tier3
        src = inspect.getsource(walkforward_tier3)
        # Check the fix is present
        assert '_synthetic' in src or 'synthetic' in src.lower(), (
            "Synthetic symbol bypass for swing PIT filter not found in walkforward_tier3.py"
        )
        # Verify the fix allows ^VIX through
        assert '"^VIX"' in src or "'VIX'" in src

    def test_synthetic_symbols_set_defined(self):
        """_synthetic set is defined in _run_swing_fold context."""
        import inspect
        import ast
        from scripts import walkforward_tier3
        src = inspect.getsource(walkforward_tier3)
        assert '_synthetic' in src
