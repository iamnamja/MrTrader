"""Tests for the weekly regime-model retrain + the shared regime gate.

Context: `scripts/train_regime_model.py` was broken on main — it read
`payload["wf_auc_min"]`/`["brier_score"]` from the saved PICKLE, but those keys were
only written to the DB row (the pickle had `wf_log_loss_mean`/`wf_macro_f1_mean`) → KeyError.
And the 0.22 gate cutoff was a 2-class Brier value wrongly applied to the 3-class
cross-entropy log-loss. The fix: write the gate inputs into the pickle, evaluate the gate
via ONE shared `regime_gate()` (CLI + PM), and add a weekly file-age-gated PM retrain.
"""
import os
import pickle
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ml.regime_training import regime_gate
from app.ml import retrain_config as rc


_PASS = {"version": 9, "wf_auc_min": 0.728, "wf_log_loss_mean": 0.358}
_FAIL_F1 = {"version": 9, "wf_auc_min": 0.55, "wf_log_loss_mean": 0.358}
_FAIL_LL = {"version": 9, "wf_auc_min": 0.728, "wf_log_loss_mean": 0.50}


# ───────────────────────── config + gate ───────────────────────────────────────

def test_config_constants():
    assert rc.REGIME_RETRAIN_INTERVAL_DAYS == 7
    assert rc.REGIME_GATE_MACRO_F1_MIN == 0.60
    assert rc.REGIME_GATE_LOG_LOSS_MAX == 0.45  # 3-class CE, NOT the old 0.22 Brier


def test_regime_gate_pass():
    ok, failures = regime_gate(_PASS)
    assert ok is True and failures == []


def test_regime_gate_fail_f1():
    ok, failures = regime_gate(_FAIL_F1)
    assert ok is False and any("macro_F1" in f for f in failures)


def test_regime_gate_fail_log_loss():
    ok, failures = regime_gate(_FAIL_LL)
    assert ok is False and any("log_loss" in f for f in failures)


def test_regime_gate_missing_keys_fails_safely():
    """THE regression: the old code did payload['wf_auc_min'] → KeyError when the pickle
    lacked it. regime_gate must use safe defaults and FAIL (not raise) on an empty payload."""
    ok, failures = regime_gate({})
    assert ok is False
    assert len(failures) == 2  # both thresholds reported


def test_regime_gate_non_numeric_fails_safely():
    ok, failures = regime_gate({"wf_auc_min": MagicMock(), "wf_log_loss_mean": MagicMock()})
    assert ok is False


# ───────────────────────── PM _retrain_regime ──────────────────────────────────

def _pm():
    """A PortfolioManager with __init__ bypassed — _retrain_regime only needs these two."""
    from app.agents.portfolio_manager import PortfolioManager
    pm = PortfolioManager.__new__(PortfolioManager)
    pm.logger = MagicMock()
    pm.log_decision = AsyncMock()
    return pm


def _write_pkl(path, payload, age_days=0.0):
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))


@pytest.mark.asyncio
async def test_retrain_regime_skips_when_fresh(tmp_path):
    """A model younger than the interval → skip; trainer must NOT be invoked."""
    _write_pkl(tmp_path / "regime_model_v1.pkl", _PASS, age_days=1.0)  # fresh
    pm = _pm()
    with patch("app.ml.regime_training.MODEL_DIR", tmp_path), \
         patch("app.ml.regime_training.RegimeModelTrainer") as Trainer:
        await pm._retrain_regime()
    Trainer.assert_not_called()
    pm.log_decision.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrain_regime_gate_pass(tmp_path):
    """Stale model → retrain; new model PASSES gate → logged active, file kept."""
    _write_pkl(tmp_path / "regime_model_v1.pkl", _PASS, age_days=30.0)  # stale
    new_path = tmp_path / "regime_model_v2.pkl"

    def _train():  # what the executor runs
        _write_pkl(new_path, _PASS)
        return new_path

    trainer = MagicMock()
    trainer.train.side_effect = _train
    pm = _pm()
    with patch("app.ml.regime_training.MODEL_DIR", tmp_path), \
         patch("app.ml.regime_training.RegimeModelTrainer", return_value=trainer):
        await pm._retrain_regime()

    assert new_path.exists()  # passing model kept
    pm.log_decision.assert_awaited_once()
    assert pm.log_decision.await_args.args[0] == "REGIME_MODEL_RETRAINED"


@pytest.mark.asyncio
async def test_retrain_regime_gate_fail_deletes_new_file(tmp_path):
    """Stale model → retrain; new model FAILS gate → new pickle deleted (loader falls back
    to the prior version), failure logged."""
    _write_pkl(tmp_path / "regime_model_v1.pkl", _PASS, age_days=30.0)
    new_path = tmp_path / "regime_model_v2.pkl"

    def _train():
        _write_pkl(new_path, _FAIL_LL)  # fails the log-loss gate
        return new_path

    trainer = MagicMock()
    trainer.train.side_effect = _train
    pm = _pm()
    with patch("app.ml.regime_training.MODEL_DIR", tmp_path), \
         patch("app.ml.regime_training.RegimeModelTrainer", return_value=trainer):
        await pm._retrain_regime()

    assert not new_path.exists()  # gate-failed model removed
    pm.log_decision.assert_awaited_once()
    assert pm.log_decision.await_args.args[0] == "REGIME_RETRAIN_GATE_FAILED"
