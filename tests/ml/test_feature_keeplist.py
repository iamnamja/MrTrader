"""Tests for P1 feature keep-list: BENIGN_SWING_FEATURES and BENIGN_INTRADAY_FEATURES."""
import pytest

# v217 feature set — 17 cross-regime features selected via IC audit (2026-05-24)
V217_FEATURES = frozenset({
    "momentum_252d_ex1m", "ix_momentum_vol", "price_to_52w_high", "price_to_52w_low",
    "reversal_5d_vol_weighted", "downtrend", "range_expansion",
    "operating_margin", "profit_margin", "pe_ratio",
    "vol_regime", "vrp", "vol_percentile_52w",
    "volume_trend",
    "wq_alpha35", "wq_alpha40", "wq_alpha43",
})


class TestFeatureKeeplist:
    def test_swing_keeplist_count(self):
        from app.ml.retrain_config import BENIGN_SWING_FEATURES
        assert len(BENIGN_SWING_FEATURES) == 17, (
            f"BENIGN_SWING_FEATURES should have 17 features (v217), got {len(BENIGN_SWING_FEATURES)}"
        )

    def test_intraday_keeplist_count(self):
        from app.ml.retrain_config import BENIGN_INTRADAY_FEATURES
        assert len(BENIGN_INTRADAY_FEATURES) == 25, (
            f"BENIGN_INTRADAY_FEATURES should have 25 features, got {len(BENIGN_INTRADAY_FEATURES)}"
        )

    def test_swing_keeplist_no_duplicates(self):
        from app.ml.retrain_config import BENIGN_SWING_FEATURES
        assert len(BENIGN_SWING_FEATURES) == len(set(BENIGN_SWING_FEATURES)), \
            "BENIGN_SWING_FEATURES contains duplicate feature names"

    def test_intraday_keeplist_no_duplicates(self):
        from app.ml.retrain_config import BENIGN_INTRADAY_FEATURES
        assert len(BENIGN_INTRADAY_FEATURES) == len(set(BENIGN_INTRADAY_FEATURES)), \
            "BENIGN_INTRADAY_FEATURES contains duplicate feature names"

    def test_swing_keeplist_is_v217(self):
        """v217 feature set must exactly match the IC-audited 17-feature list."""
        from app.ml.retrain_config import BENIGN_SWING_FEATURES
        actual = set(BENIGN_SWING_FEATURES)
        missing = V217_FEATURES - actual
        extra = actual - V217_FEATURES
        assert not missing, f"v217 features missing from keeplist: {missing}"
        assert not extra, f"Unexpected features in keeplist (not in v217): {extra}"

    def test_swing_keeplist_has_counter_trend_features(self):
        """Counter-trend features are essential for cross-regime stability (v217 design)."""
        from app.ml.retrain_config import BENIGN_SWING_FEATURES
        counter_trend = {"reversal_5d_vol_weighted", "downtrend", "range_expansion"}
        missing = counter_trend - set(BENIGN_SWING_FEATURES)
        assert not missing, f"Counter-trend features missing from swing keeplist: {missing}"

    def test_swing_keeplist_has_quality_features(self):
        """Quality fundamentals required for bear-regime IC (v217 design)."""
        from app.ml.retrain_config import BENIGN_SWING_FEATURES
        quality = {"operating_margin", "profit_margin", "pe_ratio"}
        missing = quality - set(BENIGN_SWING_FEATURES)
        assert not missing, f"Quality features missing from swing keeplist: {missing}"

    def test_intraday_keeplist_contains_regime_features(self):
        from app.ml.retrain_config import BENIGN_INTRADAY_FEATURES
        regime_required = {
            "vix_term_ratio", "spy_above_ma50", "spy_above_ma200",
            "breadth_rsp_spy_ratio_20d", "credit_hyg_ief_20d",
        }
        missing = regime_required - set(BENIGN_INTRADAY_FEATURES)
        assert not missing, f"Missing regime features in intraday keeplist: {missing}"

    def test_benign_filter_disabled_by_default(self):
        from app.ml.retrain_config import BENIGN_FILTER_ENABLED
        assert BENIGN_FILTER_ENABLED is False, \
            "BENIGN_FILTER_ENABLED must default to False (opt-in only)"

    def test_benign_threshold_is_half(self):
        from app.ml.retrain_config import BENIGN_REGIME_THRESHOLD
        assert BENIGN_REGIME_THRESHOLD == 0.5, \
            "BENIGN_REGIME_THRESHOLD should be 0.5 (3 of 5 components bullish)"

    def test_feature_pruning_logic(self):
        """Verify keep-list pruning retains only listed features."""
        from app.ml.retrain_config import BENIGN_SWING_FEATURES
        all_features = {
            "momentum_252d_ex1m": 0.15,
            "vol_regime": 0.7,
            "volume_trend": 0.12,
            "some_deprecated_feature": 0.99,   # NOT in v217 keeplist
            "rsi_14": 55.0,                    # removed in v217
        }
        filtered = {k: v for k, v in all_features.items() if k in BENIGN_SWING_FEATURES}
        assert "some_deprecated_feature" not in filtered
        assert "rsi_14" not in filtered  # removed in v217
        assert "momentum_252d_ex1m" in filtered
        assert "volume_trend" in filtered
