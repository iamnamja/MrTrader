"""R4 — EXPERIMENT_OVERRIDES hyperparameter injection tests."""
from __future__ import annotations


def test_experiment_overrides_default_empty():
    """EXPERIMENT_OVERRIDES must be empty dict by default."""
    from app.ml.model import EXPERIMENT_OVERRIDES
    assert EXPERIMENT_OVERRIDES == {}


def test_experiment_overrides_applied_to_model():
    """Non-empty EXPERIMENT_OVERRIDES are applied to XGBClassifier params."""
    import app.ml.model as model_mod
    original = dict(model_mod.EXPERIMENT_OVERRIDES)
    try:
        model_mod.EXPERIMENT_OVERRIDES = {"reg_alpha": 2.0, "reg_lambda": 2.0, "colsample_bytree": 0.5}
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel(model_type="xgboost")
        params = m.model.get_params()
        assert params["reg_alpha"] == 2.0
        assert params["reg_lambda"] == 2.0
        assert params["colsample_bytree"] == 0.5
    finally:
        model_mod.EXPERIMENT_OVERRIDES = original


def test_default_params_preserved_without_overrides():
    """Default XGBoost params are intact when EXPERIMENT_OVERRIDES is empty."""
    import app.ml.model as model_mod
    assert model_mod.EXPERIMENT_OVERRIDES == {}
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel(model_type="xgboost")
    params = m.model.get_params()
    assert params["reg_alpha"] == 0.1
    assert params["reg_lambda"] == 1.5
    assert params["colsample_bytree"] == 0.6
    assert params["max_depth"] == 4
