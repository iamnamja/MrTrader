"""Tests that CPCV DSR uses total trading-day count (n_obs), not n_combinations."""
from scripts.walkforward.cpcv import CPCVResult


def test_total_obs_sums_path_n_obs():
    r = CPCVResult(model_type="intraday", n_folds=6, n_paths=2)
    r.path_n_obs = [80] * 15
    assert r.total_obs == 1200


def test_total_obs_zero_when_path_n_obs_empty():
    r = CPCVResult(model_type="intraday", n_folds=6, n_paths=2)
    assert r.total_obs == 0


def test_dsr_n_obs_uses_total_obs_when_available():
    r = CPCVResult(model_type="intraday", n_folds=6, n_paths=2)
    r.path_sharpes = [1.2] * 15
    r.path_n_obs = [80] * 15
    assert r._dsr_n_obs() == 1200


def test_dsr_n_obs_falls_back_to_n_combinations_for_legacy():
    r = CPCVResult(model_type="intraday", n_folds=6, n_paths=2)
    r.path_sharpes = [1.2] * 14
    # path_n_obs empty = legacy result with no obs tracking
    assert r._dsr_n_obs() == 14


def test_cpcv_dsr_uses_correct_n_obs_in_gate():
    """With n_obs=1200, DSR p-value differs from the buggy n_obs=15 version."""
    from scripts.walkforward.gates import deflated_sharpe_ratio, N_TRIALS_TESTED

    r = CPCVResult(model_type="intraday", n_folds=6, n_paths=2)
    r.path_sharpes = [1.2] * 15
    r.path_profit_factors = [1.5] * 15
    r.path_calmars = [0.4] * 15
    r.path_n_obs = [80] * 15  # 1200 total

    assert r.total_obs == 1200

    _, p_correct = deflated_sharpe_ratio(1.2, N_TRIALS_TESTED, 1200)
    _, p_buggy = deflated_sharpe_ratio(1.2, N_TRIALS_TESTED, 15)
    # Correct n_obs gives higher statistical confidence (more observations)
    assert p_correct > p_buggy


def test_in_sample_override_prevents_gate_pass():
    r = CPCVResult(model_type="intraday", n_folds=6, n_paths=2, in_sample_override=True)
    r.path_sharpes = [10.0] * 15  # absurdly high in-sample
    r.path_profit_factors = [5.0] * 15
    r.path_calmars = [10.0] * 15
    r.path_n_obs = [100] * 15
    assert r.gate_passed() is False


def test_no_in_sample_override_allows_gate_pass():
    r = CPCVResult(model_type="intraday", n_folds=6, n_paths=2, in_sample_override=False)
    r.path_sharpes = [2.0] * 15
    r.path_profit_factors = [1.5] * 15
    r.path_calmars = [1.0] * 15
    r.path_n_obs = [120] * 15
    assert r.gate_passed() is True
