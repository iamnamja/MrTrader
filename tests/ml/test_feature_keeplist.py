"""Tests for P1 feature keep-list: BENIGN_SWING_FEATURES and BENIGN_INTRADAY_FEATURES."""
import pytest


class TestFeatureKeeplist:
    def test_swing_keeplist_count(self):
        from app.ml.retrain_config import BENIGN_SWING_FEATURES
        assert len(BENIGN_SWING_FEATURES) == 35, (
            f"BENIGN_SWING_FEATURES should have 35 features, got {len(BENIGN_SWING_FEATURES)}"
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

    def test_swing_keeplist_contains_regime_features(self):
        from app.ml.retrain_config import BENIGN_SWING_FEATURES
        regime_required = {
            "vix_term_ratio", "spy_above_ma50", "spy_above_ma200",
            "breadth_rsp_spy_ratio_20d", "credit_hyg_ief_20d",
        }
        missing = regime_required - set(BENIGN_SWING_FEATURES)
        assert not missing, f"Missing regime features in swing keeplist: {missing}"

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
            "rsi_14": 55.0,
            "volume_ratio": 1.2,
            "sector_momentum": 0.03,
            "some_deprecated_feature": 0.99,  # NOT in keeplist
            "another_old_feature": -0.5,       # NOT in keeplist
        }
        filtered = {k: v for k, v in all_features.items() if k in BENIGN_SWING_FEATURES}
        assert "some_deprecated_feature" not in filtered
        assert "another_old_feature" not in filtered
        # Features that ARE in the keeplist should survive
        assert "rsi_14" in filtered
        assert "volume_ratio" in filtered
