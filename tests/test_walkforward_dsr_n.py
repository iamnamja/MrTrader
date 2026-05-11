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
    """The constant must reference ML_EXPERIMENT_LOG in its inline comment."""
    import scripts.walkforward_tier3 as wf_mod
    src = inspect.getsource(wf_mod)
    # Find the line with N_TRIALS_TESTED assignment
    for line in src.splitlines():
        if "N_TRIALS_TESTED" in line and "=" in line and "assert" not in line and "args" not in line:
            assert "ML_EXPERIMENT_LOG" in line, (
                f"N_TRIALS_TESTED line has no ML_EXPERIMENT_LOG provenance comment: {line!r}"
            )
            return
    raise AssertionError("N_TRIALS_TESTED assignment line not found in walkforward_tier3.py")


def test_deflated_sharpe_increases_with_n_trials():
    """DSR p-value must decrease (harder to pass) as n_trials increases."""
    from scripts.walkforward_tier3 import _deflated_sharpe_ratio
    _, p_low = _deflated_sharpe_ratio(0.8, n_trials=15, n_obs=1000)
    _, p_high = _deflated_sharpe_ratio(0.8, n_trials=200, n_obs=1000)
    assert p_high < p_low, (
        f"Expected p_value(N=200)={p_high:.4f} < p_value(N=15)={p_low:.4f}. "
        "More trials → harder DSR threshold → lower p-value for same observed Sharpe."
    )
