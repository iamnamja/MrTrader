"""Alpha-v4 P0 — freeze dead XS-ML retrain.

The dead swing (XS LambdaRank) + intraday 5-min ML must NOT retrain nightly.
These guard the three enforcement points: the per-model flags in retrain_cron,
and the orchestrator's RETRAIN_WEEKDAY global guard (the root-cause fix — the daily
trigger previously ignored the flag and retrained the dead models every night).
"""
from __future__ import annotations

import asyncio

import pytest


class _FakeDB:
    def close(self):
        pass


def test_run_swing_frozen_returns_true_without_training(monkeypatch):
    import scripts.retrain_cron as rc
    monkeypatch.setattr("app.ml.retrain_config.SWING_ENABLED", False)

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("ModelTrainer instantiated despite SWING_ENABLED=False")

    monkeypatch.setattr("app.ml.training.ModelTrainer", _Boom)
    # True (not False) so the orchestrator doesn't log a spurious gate failure.
    assert rc.run_swing(dry_run=False) is True


def test_run_intraday_frozen_returns_true_without_training(monkeypatch):
    import scripts.retrain_cron as rc
    monkeypatch.setattr("app.ml.retrain_config.INTRADAY_ENABLED", False)

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("IntradayModelTrainer instantiated despite INTRADAY_ENABLED=False")

    monkeypatch.setattr("app.ml.intraday_training.IntradayModelTrainer", _Boom)
    assert rc.run_intraday(dry_run=False) is True


def test_run_swing_enabled_passes_freeze_guard(monkeypatch):
    """When re-enabled, the freeze guard must NOT short-circuit — flow reaches the trainer."""
    import scripts.retrain_cron as rc
    monkeypatch.setattr("app.ml.retrain_config.SWING_ENABLED", True)
    monkeypatch.setattr("app.database.session.get_session", lambda: _FakeDB())
    monkeypatch.setattr(rc, "_previous_active", lambda db, kind: None)

    class _Reached(Exception):
        pass

    class _Trainer:
        def __init__(self, *a, **k):
            raise _Reached()

    monkeypatch.setattr("app.ml.training.ModelTrainer", _Trainer)
    with pytest.raises(_Reached):
        rc.run_swing(dry_run=False)


def test_orchestrator_retrain_disabled_by_weekday(monkeypatch):
    """RETRAIN_WEEKDAY=-1 must stop _trigger_retraining before spawning the subprocess."""
    from app.orchestrator import orchestrator
    monkeypatch.setattr("app.ml.retrain_config.RETRAIN_WEEKDAY", -1)

    async def _boom(*a, **k):
        raise AssertionError("retrain subprocess spawned despite RETRAIN_WEEKDAY=-1")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)
    asyncio.run(orchestrator._trigger_retraining())  # returns early, no spawn
