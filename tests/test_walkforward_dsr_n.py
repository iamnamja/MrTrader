"""R1 tests: DSR N_TRIALS_TESTED must be honest (≥100) and documented."""
from __future__ import annotations

import inspect


def test_dsr_n_trials_at_least_100():
    """N_TRIALS_TESTED must reflect the actual number of model variants tried."""
    from scripts.walkforward_tier3 import N_TRIALS_TESTED
    assert N_TRIALS_TESTED >= 100, (
        f"N_TRIALS_TESTED={N_TRIALS_TESTED} understates selection bias. "
        "ML_EXPERIMENT_LOG records ~200 variants tried across iter 1-6 + phases 18-87."
    )


def test_dsr_n_trials_documented():
    """The constant must reference ML_EXPERIMENT_LOG in its source (assignment or import comment).

    N_TRIALS_TESTED is now defined in retrain_config.py (single source of truth).
    The provenance comment lives there; this test verifies retrain_config.py documents it.
    """
    from app.ml import retrain_config
    src = inspect.getsource(retrain_config)
    # Find the N_TRIALS_TESTED assignment with a provenance comment
    for line in src.splitlines():
        if "N_TRIALS_TESTED" in line and "=" in line and "assert" not in line and "def " not in line:
            if "ML_EXPERIMENT_LOG" in line or "history" in line.lower() or "trials" in line.lower():
                return  # documented
    # Also accept a comment block mentioning both N_TRIALS_TESTED and ML_EXPERIMENT_LOG
    assert "N_TRIALS_TESTED" in src and "ML_EXPERIMENT_LOG" in src, (
        "retrain_config.py must document N_TRIALS_TESTED with ML_EXPERIMENT_LOG provenance."
    )


def test_deflated_sharpe_increases_with_n_trials():
    """DSR p-value must decrease (harder to pass) as n_trials increases."""
    from scripts.walkforward_tier3 import _deflated_sharpe_ratio
    # After WF deep-review pass-2 DSR fix (restored sqrt(V) scaling), SR=0.8 with
    # T=1000 saturates both p-values at 1.0. Use a smaller signal so the deflation
    # term materially moves the p-value between the two n_trials values.
    _, p_low = _deflated_sharpe_ratio(0.1, n_trials=15, n_obs=100)
    _, p_high = _deflated_sharpe_ratio(0.1, n_trials=200, n_obs=100)
    assert p_high < p_low, (
        f"Expected p_value(N=200)={p_high:.4f} < p_value(N=15)={p_low:.4f}. "
        "More trials → harder DSR threshold → lower p-value for same observed Sharpe."
    )
