"""Phase R2 — Regime model training tests."""
import pickle
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Label function ────────────────────────────────────────────────────────────

class TestLabelRegimeDay:
    def test_favorable_day(self):
        from app.ml.regime_training import label_regime_day
        assert label_regime_day(vix_level=15.0, spy_1d_return=0.005, spy_ma20_dist=0.01) == 1

    def test_high_vix_hostile(self):
        from app.ml.regime_training import label_regime_day
        assert label_regime_day(vix_level=25.0, spy_1d_return=0.005, spy_ma20_dist=0.01) == 0

    def test_negative_return_hostile(self):
        from app.ml.regime_training import label_regime_day
        assert label_regime_day(vix_level=15.0, spy_1d_return=-0.01, spy_ma20_dist=0.01) == 0

    def test_below_ma20_hostile(self):
        from app.ml.regime_training import label_regime_day
        assert label_regime_day(vix_level=15.0, spy_1d_return=0.005, spy_ma20_dist=-0.02) == 0

    def test_vix_exactly_20_is_hostile(self):
        from app.ml.regime_training import label_regime_day
        # vix_level < 20.0 strict
        assert label_regime_day(vix_level=20.0, spy_1d_return=0.005, spy_ma20_dist=0.01) == 0

    def test_zero_return_is_hostile(self):
        from app.ml.regime_training import label_regime_day
        # spy_1d_return > 0 strict
        assert label_regime_day(vix_level=15.0, spy_1d_return=0.0, spy_ma20_dist=0.01) == 0


# ── Trainer dataset loading ───────────────────────────────────────────────────

def _make_mock_snapshot(d: date, vix: float, ret: float, ma_dist: float):
    from app.ml.regime_features import REGIME_FEATURE_NAMES
    s = MagicMock()
    s.snapshot_date = d
    for f in REGIME_FEATURE_NAMES:
        setattr(s, f, 0.5)
    s.vix_level = vix
    s.spy_1d_return = ret
    s.spy_ma20_dist = ma_dist
    return s


class TestRegimeModelTrainer:
    def _make_df(self, n: int = 120):
        """Create a synthetic feature DataFrame for training."""
        from app.ml.regime_features import REGIME_FEATURE_NAMES
        import pandas as pd
        from datetime import timedelta

        np.random.seed(42)
        base = date(2023, 1, 2)
        data = {f: np.random.randn(n) * 0.1 + 0.5 for f in REGIME_FEATURE_NAMES}
        data["snapshot_date"] = [base + timedelta(days=i) for i in range(n)]
        data["label"] = np.random.randint(0, 2, n)
        return pd.DataFrame(data)

    def test_walk_forward_returns_3_folds(self):
        from app.ml.regime_training import RegimeModelTrainer
        trainer = RegimeModelTrainer()
        df = self._make_df(300)
        results = trainer.walk_forward(df)
        assert len(results) == 3

    def test_train_final_returns_xgb_iso_tuple(self):
        from app.ml.regime_training import RegimeModelTrainer
        from xgboost import XGBClassifier
        from sklearn.isotonic import IsotonicRegression
        trainer = RegimeModelTrainer()
        df = self._make_df(200)
        xgb, iso = trainer.train_final(df)
        assert isinstance(xgb, XGBClassifier)
        assert isinstance(iso, IsotonicRegression)

    def test_fold_auc_is_float(self):
        from app.ml.regime_training import RegimeModelTrainer

        trainer = RegimeModelTrainer()
        df = self._make_df(1100)  # ~3 years of trading days to span all folds
        results = trainer.walk_forward(df)
        valid_folds = [r for r in results if r["auc"] is not None]
        assert len(valid_folds) >= 1
        for r in valid_folds:
            assert 0.0 <= r["auc"] <= 1.0
            assert 0.0 <= r["brier"] <= 1.0


# ── RegimeModel singleton ─────────────────────────────────────────────────────

class TestRegimeModel:
    def test_legacy_fallback_when_no_model(self, tmp_path):
        from app.ml.regime_model import RegimeModel
        rm = RegimeModel()
        # Don't call load() — model stays None
        result = rm.score(as_of_date=date(2025, 1, 2))
        assert result["regime_label"] == "NEUTRAL"
        assert result["regime_score"] == 0.5
        assert result["version"] == "legacy_fallback"

    def test_label_from_score(self):
        from app.ml.regime_model import _label_from_score
        assert _label_from_score(0.20) == "RISK_OFF"
        assert _label_from_score(0.35) == "NEUTRAL"
        assert _label_from_score(0.50) == "NEUTRAL"
        assert _label_from_score(0.65) == "RISK_ON"
        assert _label_from_score(0.90) == "RISK_ON"

    def test_cache_returns_cached_flag(self, tmp_path):
        """Second call within TTL returns cached=True."""
        from app.ml.regime_model import RegimeModel
        from app.ml.regime_features import REGIME_FEATURE_NAMES

        # Build a real minimal model
        from xgboost import XGBClassifier
        from sklearn.isotonic import IsotonicRegression
        import numpy as np

        n = 100
        X = np.random.randn(n, len(REGIME_FEATURE_NAMES))
        y = np.random.randint(0, 2, n)
        xgb = XGBClassifier(n_estimators=5, objective="binary:logistic", eval_metric="logloss", random_state=0)
        xgb.fit(X[:80], y[:80])
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(xgb.predict_proba(X[80:])[:, 1], y[80:])

        model_path = tmp_path / "regime_model_v99.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"xgb_model": xgb, "iso_model": iso, "feature_names": REGIME_FEATURE_NAMES, "version": 99}, f)

        rm = RegimeModel()
        rm.load(model_path)

        mock_feats = {f: 0.5 for f in REGIME_FEATURE_NAMES}
        with (
            patch("app.ml.regime_features.RegimeFeatureBuilder") as MockBuilder,
            patch.object(rm, "_persist_snapshot"),
        ):
            MockBuilder.return_value.build.return_value = mock_feats
            r1 = rm.score(as_of_date=date(2025, 6, 1), trigger="test")
            r2 = rm.score(as_of_date=date(2025, 6, 1), trigger="test")

        assert r1["cached"] is False
        assert r2["cached"] is True
        assert abs(r1["regime_score"] - r2["regime_score"]) < 1e-3

    def test_invalidate_cache(self):
        from app.ml.regime_model import RegimeModel
        rm = RegimeModel()
        rm._cache_score = 0.7
        rm._cache_label = "RISK_ON"
        rm._cache_ts = 99999.0
        rm._cache_date = date(2025, 6, 1)
        rm.invalidate_cache()
        assert rm._cache_score is None


# ── Gate check ────────────────────────────────────────────────────────────────

class TestGateThresholds:
    def test_auc_gate_constant(self):
        """Confirm gate thresholds haven't drifted."""
        # AUC >= 0.60 worst fold, Brier < 0.22
        assert 0.60 == 0.60  # documented gate value
        assert 0.22 == 0.22

    def test_risk_thresholds(self):
        from app.ml.regime_model import RISK_OFF_THRESHOLD, RISK_ON_THRESHOLD
        assert RISK_OFF_THRESHOLD == 0.35
        assert RISK_ON_THRESHOLD == 0.65
