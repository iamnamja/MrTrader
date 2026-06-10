"""Phase R2 — Regime model training tests."""
import pickle
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Label function ────────────────────────────────────────────────────────────

class TestLabelRegimeDay:
    """V2 label tests — delegates to regime_features.label_regime_day."""

    def _row(self, **kwargs):
        base = {
            "vix_level": 18.0, "vix_pct_1y": 0.40, "vix_term_ratio": 0.95,
            "spy_ma50_dist": 0.03, "spy_ma200_dist": 0.05,
            "credit_hyg_ief_20d": 0.002, "breadth_rsp_spy_ratio_20d": 0.01,
            "spy_20d_return": 0.03,
        }
        base.update(kwargs)
        return base

    def test_favorable_day(self):
        from app.ml.regime_features import label_regime_day
        assert label_regime_day(self._row()) == 2  # RISK_ON clean tape

    def test_high_vix_hostile(self):
        from app.ml.regime_features import label_regime_day
        assert label_regime_day(self._row(vix_level=35.0)) == 0  # RISK_OFF

    def test_negative_return_hostile(self):
        from app.ml.regime_features import label_regime_day
        # Broken MAs → RISK_OFF
        assert label_regime_day(self._row(spy_ma50_dist=-0.06, spy_ma200_dist=-0.02)) == 0

    def test_below_ma20_hostile(self):
        from app.ml.regime_features import label_regime_day
        # Below MA50 → RISK_CAUTION (not full RISK_OFF unless also below MA200)
        assert label_regime_day(self._row(spy_ma50_dist=-0.02, spy_ma200_dist=0.01)) == 1

    def test_vix_exactly_20_is_caution(self):
        from app.ml.regime_features import label_regime_day
        # vix==20 fails RISK_ON (needs vix<20); not RISK_OFF either → RISK_CAUTION
        assert label_regime_day(self._row(vix_level=20.0)) == 1

    def test_credit_stress_is_risk_off(self):
        from app.ml.regime_features import label_regime_day
        assert label_regime_day(self._row(credit_hyg_ief_20d=-0.05)) == 0


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

    def test_train_final_returns_xgb_temperature_tuple(self):
        from app.ml.regime_training import RegimeModelTrainer
        from xgboost import XGBClassifier
        trainer = RegimeModelTrainer()
        df = self._make_df(200)
        df["label"] = np.random.randint(0, 3, len(df))  # 3-class labels
        xgb, temperature = trainer.train_final(df)
        assert isinstance(xgb, XGBClassifier)
        assert isinstance(temperature, float)
        assert temperature > 0.0

    def test_fold_metrics_are_floats(self):
        from app.ml.regime_training import RegimeModelTrainer

        trainer = RegimeModelTrainer()
        df = self._make_df(1100)
        df["label"] = np.random.randint(0, 3, len(df))
        results = trainer.walk_forward(df)
        valid_folds = [r for r in results if r["log_loss"] is not None]
        assert len(valid_folds) >= 1
        for r in valid_folds:
            assert r["log_loss"] >= 0.0
            assert 0.0 <= r["macro_f1"] <= 1.0


# ── RegimeModel singleton ─────────────────────────────────────────────────────

class TestRegimeModel:
    def test_legacy_fallback_when_no_model(self, tmp_path):
        from app.ml.regime_model import RegimeModel
        rm = RegimeModel()
        # Don't call load() — model stays None
        result = rm.score(as_of_date=date(2025, 1, 2))
        assert result["regime_label"] == "UNKNOWN"
        assert result["regime_score"] == 0.5
        assert result["version"] == "legacy_fallback"

    def test_label_from_score(self):
        from app.ml.regime_training import label_from_score
        assert label_from_score(0.20) == "RISK_OFF"
        assert label_from_score(0.35) == "RISK_CAUTION"
        assert label_from_score(0.50) == "RISK_CAUTION"
        assert label_from_score(0.65) == "RISK_ON"
        assert label_from_score(0.90) == "RISK_ON"

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
    def test_risk_thresholds(self):
        from app.ml.regime_model import RISK_OFF_THRESHOLD, RISK_ON_THRESHOLD
        assert RISK_OFF_THRESHOLD == 0.30
        assert RISK_ON_THRESHOLD == 0.60

    def test_train_script_gate_values(self):
        """The regime gate must use the reviewed 3-class values (macro_F1 >= 0.60,
        log_loss < 0.45) — now centralized in retrain_config + the shared regime_gate(),
        NOT inline literals in the CLI script. (The old 0.22 was a 2-class Brier cutoff
        wrongly applied to 3-class cross-entropy log-loss.)
        """
        from pathlib import Path
        from app.ml.retrain_config import REGIME_GATE_MACRO_F1_MIN, REGIME_GATE_LOG_LOSS_MAX
        assert REGIME_GATE_MACRO_F1_MIN == 0.60
        assert REGIME_GATE_LOG_LOSS_MAX == 0.45
        # The CLI must delegate to the shared evaluator, and the stale 0.22 must be gone.
        src = (Path(__file__).parent.parent / "scripts" / "train_regime_model.py").read_text()
        assert "regime_gate" in src, "train_regime_model.py must use the shared regime_gate()"
        assert "0.22" not in src, "stale 2-class Brier threshold 0.22 must be removed"


class TestPklPayloadKeys:
    """Ensure the pkl written by RegimeModelTrainer.train() includes required keys."""

    def _run_train(self, tmp_path):
        from app.ml.regime_training import RegimeModelTrainer, REGIME_FEATURE_NAMES
        import numpy as np
        import pandas as pd
        from unittest.mock import patch

        n = 120
        rng = np.random.default_rng(42)
        df_mock = pd.DataFrame(
            rng.standard_normal((n, len(REGIME_FEATURE_NAMES))),
            columns=REGIME_FEATURE_NAMES,
        )
        df_mock["label"] = rng.integers(0, 3, n)
        dates = pd.date_range("2019-01-01", periods=n, freq="B").date
        df_mock["snapshot_date"] = dates

        trainer = RegimeModelTrainer()
        with (
            patch.object(trainer, "load_dataset", return_value=df_mock),
            patch("app.ml.regime_training.MODEL_DIR", tmp_path),
            patch.object(trainer, "_write_model_version"),
        ):
            path = trainer.train(version=99)
        return path

    def test_required_keys_present(self, tmp_path):
        import pickle
        path = self._run_train(tmp_path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        for key in ("xgb_model", "feature_names", "version", "model_version",
                    "wf_results", "wf_log_loss_mean", "wf_macro_f1_mean"):
            assert key in payload, f"pkl missing key: {key}"

    def test_model_version_marker_is_2(self, tmp_path):
        import pickle
        path = self._run_train(tmp_path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert payload["model_version"] == 2, "model_version marker must be 2 for V2 regime model"

    def test_wf_results_is_list_of_dicts(self, tmp_path):
        import pickle
        path = self._run_train(tmp_path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        wf = payload["wf_results"]
        assert isinstance(wf, list) and len(wf) > 0
        assert "macro_f1" in wf[0] and "log_loss" in wf[0]
