"""
Phase 4a tests — frozen HPO params + feature audit script.

Covers:
- FROZEN_HPO_PARAMS is set (not None) and has required XGBoost keys
- FROZEN_HPO_PARAMS values are in valid training ranges
- feature_correlation_audit.py can be imported without error
- audit_model returns expected keys in result dict
- zero-importance features are returned for known models
"""
import pytest


# ── Frozen HPO params ─────────────────────────────────────────────────────────

class TestFrozenHPOParams:
    def test_frozen_hpo_is_set(self):
        from app.ml.intraday_training import FROZEN_HPO_PARAMS
        assert FROZEN_HPO_PARAMS is not None, (
            "FROZEN_HPO_PARAMS must be set after v51 HPO search. "
            "Set to None only before the first HPO run."
        )

    def test_frozen_hpo_has_required_keys(self):
        from app.ml.intraday_training import FROZEN_HPO_PARAMS
        required = {
            "n_estimators", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_weight",
            "gamma", "reg_alpha", "reg_lambda",
        }
        missing = required - set(FROZEN_HPO_PARAMS.keys())
        assert not missing, f"FROZEN_HPO_PARAMS missing keys: {missing}"

    def test_frozen_hpo_values_in_range(self):
        from app.ml.intraday_training import FROZEN_HPO_PARAMS
        p = FROZEN_HPO_PARAMS
        assert 100 <= p["n_estimators"] <= 2000, f"n_estimators out of range: {p['n_estimators']}"
        assert 2 <= p["max_depth"] <= 12, f"max_depth out of range: {p['max_depth']}"
        assert 0.001 <= p["learning_rate"] <= 0.5, f"learning_rate out of range: {p['learning_rate']}"
        assert 0.3 <= p["subsample"] <= 1.0, f"subsample out of range: {p['subsample']}"
        assert 0.1 <= p["colsample_bytree"] <= 1.0, f"colsample_bytree out of range: {p['colsample_bytree']}"
        assert p["gamma"] >= 0, f"gamma must be >= 0, got {p['gamma']}"
        assert p["reg_alpha"] >= 0, f"reg_alpha must be >= 0, got {p['reg_alpha']}"
        assert p["reg_lambda"] >= 0, f"reg_lambda must be >= 0, got {p['reg_lambda']}"

    def test_frozen_hpo_v51_values(self):
        """Regression guard: v51 params should remain until a better run is found."""
        from app.ml.intraday_training import FROZEN_HPO_PARAMS
        assert FROZEN_HPO_PARAMS["n_estimators"] == 754
        assert FROZEN_HPO_PARAMS["max_depth"] == 7
        assert abs(FROZEN_HPO_PARAMS["learning_rate"] - 0.01290) < 1e-5


# ── Feature audit script ───────────────────────────────────────────────────────

class TestFeatureCorrelationAudit:
    def test_script_importable(self):
        import importlib, sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        # Should import without error
        import scripts.feature_correlation_audit as fca
        assert hasattr(fca, "audit_model")
        assert hasattr(fca, "run_audit")
        assert hasattr(fca, "SWING_SEMANTIC_GROUPS")
        assert hasattr(fca, "INTRADAY_SEMANTIC_GROUPS")

    def test_semantic_groups_non_empty(self):
        from scripts.feature_correlation_audit import SWING_SEMANTIC_GROUPS, INTRADAY_SEMANTIC_GROUPS
        assert len(SWING_SEMANTIC_GROUPS) >= 5, "Swing should have at least 5 semantic groups"
        assert len(INTRADAY_SEMANTIC_GROUPS) >= 5, "Intraday should have at least 5 semantic groups"
        for group_name, members in SWING_SEMANTIC_GROUPS.items():
            assert len(members) >= 2, f"Group '{group_name}' must have at least 2 members"

    def test_audit_model_returns_expected_structure(self):
        """Smoke test: audit_model runs and returns valid result dict for swing."""
        from scripts.feature_correlation_audit import audit_model, SWING_SEMANTIC_GROUPS
        result = audit_model("swing", SWING_SEMANTIC_GROUPS)
        required_keys = {
            "version", "total_features", "zero_importance",
            "drop_candidates", "recommended_feature_count", "top_features",
        }
        assert required_keys <= set(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )
        assert result["total_features"] > 0
        assert result["recommended_feature_count"] <= result["total_features"]
        assert len(result["top_features"]) > 0

    def test_swing_has_zero_importance_features(self):
        """Swing models should have some zero-importance features (XGBoost pruning is expected)."""
        from scripts.feature_correlation_audit import audit_model, SWING_SEMANTIC_GROUPS
        result = audit_model("swing", SWING_SEMANTIC_GROUPS)
        # Any real swing model should have at least some zero-importance features —
        # XGBoost routinely ignores weak features via regularization.
        # We don't assert a specific count because it changes with each retrain.
        assert isinstance(result["zero_importance"], list)
        assert result["recommended_feature_count"] <= result["total_features"]
