"""Tests for app/ml/retrain_config.py — single source of truth enforcement."""
from __future__ import annotations

import json
from pathlib import Path


def test_n_trials_tested_defined_and_positive():
    from app.ml.retrain_config import N_TRIALS_TESTED
    assert isinstance(N_TRIALS_TESTED, int)
    assert N_TRIALS_TESTED >= 100, \
        f"N_TRIALS_TESTED={N_TRIALS_TESTED} looks too low; expected >= 100 after R1 correction"


def test_n_trials_tested_imported_in_gates():
    """gates.py must import N_TRIALS_TESTED from retrain_config, not redefine it."""
    from app.ml.retrain_config import N_TRIALS_TESTED as config_val
    from scripts.walkforward.gates import N_TRIALS_TESTED as gates_val
    assert config_val is gates_val, (
        "gates.py N_TRIALS_TESTED is not the same object as retrain_config.N_TRIALS_TESTED. "
        "gates.py must import from retrain_config, not redefine."
    )


def test_n_trials_tested_imported_in_walkforward_tier3():
    """walkforward_tier3.py must import N_TRIALS_TESTED from retrain_config."""
    import scripts.walkforward_tier3 as wf3
    from app.ml.retrain_config import N_TRIALS_TESTED as config_val
    tier3_val = getattr(wf3, "N_TRIALS_TESTED", None)
    assert tier3_val is not None, "N_TRIALS_TESTED not found in walkforward_tier3 namespace"
    assert tier3_val is config_val, (
        "walkforward_tier3.N_TRIALS_TESTED is not the same object as retrain_config.N_TRIALS_TESTED"
    )


def test_n_trials_monotonically_nondecreasing():
    """N_TRIALS_TESTED must never decrease (would make DSR less conservative)."""
    from app.ml.retrain_config import N_TRIALS_TESTED
    history_file = Path(__file__).parent / "fixtures" / "n_trials_history.json"
    if not history_file.exists():
        return  # fixture not present — skip
    history = json.loads(history_file.read_text())
    historical_max = max(entry["value"] for entry in history)
    assert N_TRIALS_TESTED >= historical_max, (
        f"N_TRIALS_TESTED={N_TRIALS_TESTED} is less than historical maximum {historical_max}. "
        f"DSR becomes less conservative when this decreases. "
        f"Update the value and document in ML_EXPERIMENT_LOG.md."
    )


def test_swing_retrain_keys():
    from app.ml.retrain_config import SWING_RETRAIN
    required = {"model_type", "label_scheme", "hpo_trials", "walk_forward_folds", "walk_forward_years"}
    assert required.issubset(SWING_RETRAIN.keys())


def test_swing_gate_keys():
    from app.ml.retrain_config import SWING_GATE
    assert "min_avg_sharpe" in SWING_GATE
    assert "min_fold_sharpe" in SWING_GATE
    assert SWING_GATE["min_avg_sharpe"] >= 0.5, "Gate should be non-trivial"
    assert SWING_GATE["min_fold_sharpe"] < 0, "Min fold Sharpe allows mild negatives"


def test_max_workers_positive():
    from app.ml.retrain_config import MAX_WORKERS, MAX_THREADS, MAX_FOLD_WORKERS
    assert MAX_WORKERS >= 1
    assert MAX_THREADS >= 1
    assert MAX_FOLD_WORKERS >= 1


def test_cost_from_turnover():
    """cost_from_turnover helper in cost_models.py matches expected formula."""
    from scripts.walkforward.cost_models import cost_from_turnover
    # Full portfolio replacement at 5 bps per side
    assert abs(cost_from_turnover(1.0, 5.0) - 0.0005) < 1e-10
    # Zero turnover = zero cost
    assert cost_from_turnover(0.0, 5.0) == 0.0
    # Scaling is linear
    assert abs(cost_from_turnover(0.5, 10.0) - 0.0005) < 1e-10
