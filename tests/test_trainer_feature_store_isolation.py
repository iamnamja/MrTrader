"""
Regression: ModelTrainer's feature-store path must honor MRTRADER_FEATURE_STORE_DB.

The module-level `portfolio_manager = PortfolioManager()` singleton constructs a
ModelTrainer at import time, which opens the feature-store SQLite DB. ModelTrainer
used to hard-code `f"{model_dir}/feature_store.db"` — the real, shared on-disk file
— ignoring the MRTRADER_FEATURE_STORE_DB override that conftest sets per xdist
worker. Under `--dist=loadscope` every worker then opened the SAME physical file
concurrently → intermittent "database is locked" (surfaced in
test_agent_simulator_rebalance.py). The trainer now reads the env override, so each
worker is isolated; production (env unset) keeps {model_dir}/feature_store.db.
"""
from __future__ import annotations

from app.ml.training import ModelTrainer


def test_feature_store_honors_env_override(monkeypatch, tmp_path):
    target = tmp_path / "isolated_fs.db"
    monkeypatch.setenv("MRTRADER_FEATURE_STORE_DB", str(target))
    trainer = ModelTrainer(use_feature_store=True)
    assert trainer._feature_store is not None
    assert trainer._feature_store._db_path == str(target)


def test_feature_store_falls_back_to_model_dir_when_env_unset(monkeypatch, tmp_path):
    # conftest sets the env per-worker; remove it to exercise the production path.
    monkeypatch.delenv("MRTRADER_FEATURE_STORE_DB", raising=False)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    trainer = ModelTrainer(use_feature_store=True, model_dir=str(model_dir))
    assert trainer._feature_store._db_path == f"{model_dir}/feature_store.db"
