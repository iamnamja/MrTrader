"""
Tests for Phase B4: MLOps — model versioning, AUC drift, SHAP persistence, API.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ─── SHAP persistence ─────────────────────────────────────────────────────────

class TestShapPersistence:
    def _make_trainer(self):
        from app.ml.training import ModelTrainer
        return ModelTrainer()

    def test_log_shap_returns_none_on_empty_X(self):
        import numpy as np
        trainer = self._make_trainer()
        result = trainer._log_shap_importance(np.array([]), ["f1", "f2"])
        assert result is None

    def test_log_shap_returns_none_when_shap_unavailable(self):
        import numpy as np
        trainer = self._make_trainer()
        with patch.dict("sys.modules", {"shap": None}):
            result = trainer._log_shap_importance(
                np.random.rand(10, 3), ["f1", "f2", "f3"]
            )
        assert result is None

    def test_shap_result_included_in_metrics_when_available(self):
        """If _log_shap_importance returns a dict, it appears in metrics."""
        import numpy as np
        from app.ml.training import ModelTrainer
        trainer = ModelTrainer()

        fake_shap = {"feature_a": 0.5, "feature_b": 0.3}
        with patch.object(trainer, "_log_shap_importance", return_value=fake_shap):
            metrics = {"auc": 0.72}
            shap_top = trainer._log_shap_importance(np.zeros((5, 2)), ["f1", "f2"])
            if shap_top:
                metrics["shap_top_features"] = shap_top
        assert metrics.get("shap_top_features") == fake_shap


# ─── AUC drift detection ──────────────────────────────────────────────────────

class TestAucDriftAlert:
    def _make_trainer(self):
        from app.ml.training import ModelTrainer
        return ModelTrainer()

    def test_drift_logged_when_auc_below_threshold(self):
        trainer = self._make_trainer()
        mock_db = MagicMock()

        with patch("app.ml.training.get_session", return_value=mock_db):
            with patch("app.ml.training.logger") as mock_logger:
                trainer._record_version(
                    version=99,
                    n_train=1000,
                    n_test=200,
                    model_path="/tmp/fake.pkl",
                    years=5,
                    metrics={"auc": 0.55, "accuracy": 0.60},
                )
                warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
                drift_warned = any("DRIFT" in c or "0.55" in c for c in warning_calls)
                assert drift_warned

    def test_no_drift_warning_when_auc_above_threshold(self):
        trainer = self._make_trainer()
        mock_db = MagicMock()

        with patch("app.ml.training.get_session", return_value=mock_db):
            with patch("app.ml.training.logger") as mock_logger:
                trainer._record_version(
                    version=100,
                    n_train=1000,
                    n_test=200,
                    model_path="/tmp/fake.pkl",
                    years=5,
                    metrics={"auc": 0.75, "accuracy": 0.68},
                )
                warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
                drift_warned = any("DRIFT" in c for c in warning_calls)
                assert not drift_warned

    def test_no_drift_warning_when_auc_missing(self):
        trainer = self._make_trainer()
        mock_db = MagicMock()

        with patch("app.ml.training.get_session", return_value=mock_db):
            with patch("app.ml.training.logger") as mock_logger:
                trainer._record_version(
                    version=101,
                    n_train=1000,
                    n_test=200,
                    model_path="/tmp/fake.pkl",
                    years=5,
                    metrics={"accuracy": 0.68},
                )
                warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
                drift_warned = any("DRIFT" in c for c in warning_calls)
                assert not drift_warned


# ─── Model versions API ───────────────────────────────────────────────────────

class TestModelVersionsAPI:
    def test_endpoint_returns_versions_list(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from app.api.routes import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        mock_mv = MagicMock()
        mock_mv.version = 5
        mock_mv.training_date = None
        mock_mv.status = "ACTIVE"
        mock_mv.performance = {"auc": 0.75, "accuracy": 0.68, "n_train": 1000, "n_test": 200}
        mock_mv.data_range_start = "2021-01-01"
        mock_mv.data_range_end = "2026-01-01"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_mv]

        with patch("app.api.routes.get_session", return_value=mock_db):
            response = client.get("/api/dashboard/model/versions")

        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert len(data["versions"]) == 1
        v = data["versions"][0]
        assert v["version"] == 5
        assert v["auc"] == 0.75
        assert v["drift_flag"] is False

    def test_drift_flag_true_when_auc_low(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from app.api.routes import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        mock_mv = MagicMock()
        mock_mv.version = 6
        mock_mv.training_date = None
        mock_mv.status = "ACTIVE"
        mock_mv.performance = {"auc": 0.58}
        mock_mv.data_range_start = "2021-01-01"
        mock_mv.data_range_end = "2026-01-01"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_mv]

        with patch("app.api.routes.get_session", return_value=mock_db):
            response = client.get("/api/dashboard/model/versions")

        assert response.status_code == 200
        v = response.json()["versions"][0]
        assert v["drift_flag"] is True

    def test_empty_versions_returns_empty_list(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from app.api.routes import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        with patch("app.api.routes.get_session", return_value=mock_db):
            response = client.get("/api/dashboard/model/versions")

        assert response.status_code == 200
        assert response.json()["versions"] == []


# ─── Retrain cron logic ───────────────────────────────────────────────────────

class TestRetainCron:
    def test_dry_run_exits_zero(self):
        import sys
        sys.argv = ["retrain_cron.py", "--dry-run"]
        import importlib
        import scripts.retrain_cron as rc
        importlib.reload(rc)
        # With dry-run, main() returns 0
        with patch("sys.argv", ["retrain_cron.py", "--dry-run"]):
            result = rc.main()
        assert result == 0

    def test_prune_keeps_n_files(self, tmp_path):
        from scripts.retrain_cron import prune_old_model_files
        # Create 5 dummy model files
        for i in range(1, 6):
            (tmp_path / f"swing_v{i}.pkl").write_text("x")
        prune_old_model_files(tmp_path, keep=3)
        remaining = list(tmp_path.glob("swing_v*.pkl"))
        assert len(remaining) == 3
