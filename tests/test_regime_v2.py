"""Tests for Phase R7 — Regime V2 (3-class labels, temperature scaling, proportional R6)."""
from __future__ import annotations

import pickle
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# label_regime_day
# ---------------------------------------------------------------------------

class TestLabelRegimeDay:
    def _row(self, **kwargs):
        base = {
            "vix_level": 18.0, "vix_pct_1y": 0.40, "vix_term_ratio": 0.95,
            "spy_ma50_dist": 0.03, "spy_ma200_dist": 0.05,
            "credit_hyg_ief_20d": 0.002, "breadth_rsp_spy_ratio_20d": 0.01,
            "spy_20d_return": 0.03,
        }
        base.update(kwargs)
        return base

    def test_risk_on_clean_tape(self):
        from app.ml.regime_features import label_regime_day
        assert label_regime_day(self._row()) == 2  # RISK_ON

    def test_risk_off_high_vix(self):
        from app.ml.regime_features import label_regime_day
        assert label_regime_day(self._row(vix_level=35.0)) == 0

    def test_risk_off_broken_mas(self):
        from app.ml.regime_features import label_regime_day
        assert label_regime_day(self._row(spy_ma50_dist=-0.06, spy_ma200_dist=-0.02)) == 0

    def test_risk_off_credit_stress(self):
        from app.ml.regime_features import label_regime_day
        assert label_regime_day(self._row(credit_hyg_ief_20d=-0.04)) == 0

    def test_risk_off_breadth_collapse(self):
        from app.ml.regime_features import label_regime_day
        assert label_regime_day(self._row(breadth_rsp_spy_ratio_20d=-0.05, spy_20d_return=-0.07)) == 0

    def test_risk_caution_elevated_vix(self):
        from app.ml.regime_features import label_regime_day
        # vix > 20 but < 28 → not RISK_OFF but fails RISK_ON vix condition
        assert label_regime_day(self._row(vix_level=23.0)) == 1

    def test_risk_caution_below_ma50(self):
        from app.ml.regime_features import label_regime_day
        assert label_regime_day(self._row(spy_ma50_dist=-0.01)) == 1  # fails RISK_ON ma50_dist>0

    def test_none_values_default_to_neutral(self):
        from app.ml.regime_features import label_regime_day
        # All None → defaults → falls into RISK_CAUTION (vix defaults to 20 which is not <20)
        result = label_regime_day({})
        assert result in (1, 2)  # neutral defaults → caution or on

    def test_vix_term_backwardation_with_high_pct(self):
        from app.ml.regime_features import label_regime_day
        # vix_pct > 0.85 and vix_term > 1.05 → RISK_OFF even if VIX < 28
        assert label_regime_day(self._row(vix_level=25.0, vix_pct_1y=0.90, vix_term_ratio=1.10)) == 0


# ---------------------------------------------------------------------------
# label_name
# ---------------------------------------------------------------------------

def test_label_name():
    from app.ml.regime_features import label_name
    assert label_name(0) == "RISK_OFF"
    assert label_name(1) == "RISK_CAUTION"
    assert label_name(2) == "RISK_ON"
    assert label_name(99) == "UNKNOWN"


# ---------------------------------------------------------------------------
# score_from_probs (regime_training)
# ---------------------------------------------------------------------------

def test_score_from_probs_pure_risk_on():
    from app.ml.regime_training import score_from_probs
    probs = np.array([[0.0, 0.0, 1.0]])
    assert score_from_probs(probs)[0] == pytest.approx(1.0)


def test_score_from_probs_pure_risk_off():
    from app.ml.regime_training import score_from_probs
    probs = np.array([[1.0, 0.0, 0.0]])
    assert score_from_probs(probs)[0] == pytest.approx(0.0)


def test_score_from_probs_pure_caution():
    from app.ml.regime_training import score_from_probs
    probs = np.array([[0.0, 1.0, 0.0]])
    assert score_from_probs(probs)[0] == pytest.approx(0.5)


def test_score_from_probs_uniform():
    from app.ml.regime_training import score_from_probs
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    score = score_from_probs(probs)[0]
    assert 0.4 < score < 0.6


# ---------------------------------------------------------------------------
# RegimeModel._score_v2
# ---------------------------------------------------------------------------

class TestRegimeModelScoreV2:
    def _make_model(self):
        from app.ml.regime_model import RegimeModel
        m = RegimeModel()
        # Fake XGB booster that returns fixed logits
        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([[0.5, 2.0, 1.0]])  # CAUTION wins
        mock_xgb = MagicMock()
        mock_xgb.get_booster.return_value = mock_booster
        m._xgb_model = mock_xgb
        m._temperature = 1.0
        m._model_version = 2
        m._feature_names = []
        m._version = "1"
        return m

    def test_score_v2_returns_three_probs(self):
        m = self._make_model()
        import xgboost as xgb
        with patch("xgboost.DMatrix"):
            probs, score, label = m._score_v2(np.zeros((1, 0)))
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_score_v2_score_in_range(self):
        m = self._make_model()
        with patch("xgboost.DMatrix"):
            _, score, _ = m._score_v2(np.zeros((1, 0)))
        assert 0.0 <= score <= 1.0

    def test_score_v2_label_is_argmax(self):
        m = self._make_model()
        with patch("xgboost.DMatrix"):
            _, _, label = m._score_v2(np.zeros((1, 0)))
        # logit[1]=2.0 is highest → RISK_CAUTION
        assert label == "RISK_CAUTION"

    def test_temperature_scaling_high_T_softens(self):
        from app.ml.regime_model import RegimeModel
        m = self._make_model()
        m._temperature = 10.0  # high T → uniform probs
        with patch("xgboost.DMatrix"):
            probs, score, _ = m._score_v2(np.zeros((1, 0)))
        # All probs should be closer to 1/3
        assert all(abs(p - 1 / 3) < 0.15 for p in probs)


# ---------------------------------------------------------------------------
# Proportional R6 weight map (regime_model)
# ---------------------------------------------------------------------------

class TestRegimeWeightMap:
    def test_risk_off_gets_zero(self):
        from app.ml.training import ModelTrainer
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.logger = MagicMock()

        rows = [
            (date(2024, 1, 2), "RISK_OFF"),
            (date(2024, 1, 3), "RISK_CAUTION"),
            (date(2024, 1, 4), "RISK_ON"),
        ]
        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = rows
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: mock_db
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch("app.database.session.get_session", return_value=mock_session):
            weight_map = trainer._load_regime_weight_map()

        assert weight_map[date(2024, 1, 2)] == 0.0   # RISK_OFF
        assert weight_map[date(2024, 1, 3)] == 0.5   # RISK_CAUTION
        assert weight_map[date(2024, 1, 4)] == 1.0   # RISK_ON

    def test_fail_open_returns_empty(self):
        from app.ml.training import ModelTrainer
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.logger = MagicMock()

        with patch("app.database.session.get_session", side_effect=Exception("DB down")):
            result = trainer._load_regime_weight_map()
        assert result == {}


# ---------------------------------------------------------------------------
# RegimeModel legacy fallback
# ---------------------------------------------------------------------------

def test_regime_model_fallback_when_not_loaded():
    from app.ml.regime_model import RegimeModel
    m = RegimeModel()
    result = m._legacy_fallback(date(2024, 1, 1), "test")
    assert result["regime_label"] == "UNKNOWN"
    assert result["regime_score"] == 0.5
    assert result["version"] == "legacy_fallback"


# ---------------------------------------------------------------------------
# Model pickle V2 format
# ---------------------------------------------------------------------------

def test_regime_model_load_v2_pickle():
    """RegimeModel.load() correctly reads V2 pickle format."""
    from app.ml.regime_model import RegimeModel
    import xgboost as xgb

    real_xgb = xgb.XGBClassifier(n_estimators=1, objective="multi:softprob", num_class=3)
    real_xgb.fit(
        np.array([[1, 2], [3, 4], [5, 6]]),
        np.array([0, 1, 2]),
    )

    payload = {
        "xgb_model": real_xgb,
        "feature_names": ["vix_level", "vix_pct_1y"],
        "version": "3",
        "model_version": 2,
        "temperature": 1.23,
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(payload, f)
        tmp_path = Path(f.name)

    try:
        m = RegimeModel()
        ok = m.load(tmp_path)
        assert ok
        assert m._model_version == 2
        assert m._temperature == pytest.approx(1.23)
        assert m._version == "3"
        assert m._iso_model is None
    finally:
        tmp_path.unlink(missing_ok=True)
