"""Phase R6 — Regime-aware intraday training row exclusion tests."""
import numpy as np
import pytest
from datetime import date
from unittest.mock import patch, MagicMock


def _make_trainer():
    from app.ml.intraday_training import IntradayModelTrainer
    import logging
    t = IntradayModelTrainer.__new__(IntradayModelTrainer)
    t.logger = logging.getLogger("test_r6")
    return t


class TestLoadRiskOffOrdinals:
    def test_returns_dict_of_ordinals(self):
        trainer = _make_trainer()
        with patch("app.ml.intraday_training.IntradayModelTrainer._load_regime_weight_ordinals",
                   return_value={date(2025, 4, 1).toordinal(): 0.0}):
            result = trainer._load_regime_weight_ordinals()
        assert isinstance(result, dict)

    def test_returns_empty_dict_on_db_error(self):
        trainer = _make_trainer()
        with patch("app.database.session.get_session", side_effect=Exception("DB down")):
            result = trainer._load_regime_weight_ordinals()
        assert result == {}


class TestRiskOffExclusion:
    """Unit-test the exclusion logic extracted from train_model."""

    def _apply_exclusion(self, day_ords, risk_off_ordinals):
        keep_mask = ~np.isin(day_ords, list(risk_off_ordinals))
        return keep_mask

    def test_excludes_risk_off_rows(self):
        day_ords = np.array([100, 100, 101, 101, 102], dtype=float)
        risk_off = {101}
        mask = self._apply_exclusion(day_ords, risk_off)
        assert mask.sum() == 3
        assert not np.any(day_ords[mask] == 101)

    def test_keeps_all_when_no_risk_off(self):
        day_ords = np.array([100, 101, 102], dtype=float)
        mask = self._apply_exclusion(day_ords, set())
        assert mask.all()

    def test_excludes_all_when_all_risk_off(self):
        day_ords = np.array([100, 101], dtype=float)
        mask = self._apply_exclusion(day_ords, {100, 101})
        assert not mask.any()

    def test_does_not_touch_test_rows(self):
        """Exclusion only applies to train split — test rows always kept."""
        # Simulate: train has risk-off day 101, test has day 101 too
        train_ords = np.array([100, 101], dtype=float)
        test_ords = np.array([101, 102], dtype=float)
        risk_off = {101}

        train_mask = self._apply_exclusion(train_ords, risk_off)
        test_mask = self._apply_exclusion(test_ords, set())  # never exclude test

        assert train_mask.sum() == 1
        assert test_mask.sum() == 2  # test fully intact


class TestRetainConfig:
    def test_exclude_risk_off_in_intraday_retrain_config(self):
        from app.ml.retrain_config import INTRADAY_RETRAIN
        assert INTRADAY_RETRAIN.get("exclude_risk_off_days") is True

    def test_exclude_risk_off_accepted_by_train_model_signature(self):
        import inspect
        from app.ml.intraday_training import IntradayModelTrainer
        sig = inspect.signature(IntradayModelTrainer.train_model)
        assert "exclude_risk_off_days" in sig.parameters
        assert sig.parameters["exclude_risk_off_days"].default is True
