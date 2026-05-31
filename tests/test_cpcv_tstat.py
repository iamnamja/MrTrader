"""Phase 3 (HIGH-3) — CPCV path-Sharpe t-stat tests.

t = mean_path_sharpe / (std_path_sharpe / sqrt(N_eff)), N_eff = n_folds (NOT
n_paths — the C(k,p) paths reuse k folds and are correlated). Gate is off by
default (require_tstat_gate=False) — reported and warned but not blocking.
"""
from unittest.mock import patch

from scripts.walkforward.cpcv import CPCVResult


def _cpcv(path_sharpes, n_folds=6, require_tstat_gate=False, **kw):
    return CPCVResult(
        model_type="test", n_folds=n_folds, n_paths=2,
        path_sharpes=list(path_sharpes),
        path_profit_factors=[1.5] * len(path_sharpes),
        path_calmars=[1.0] * len(path_sharpes),
        path_n_obs=[250] * len(path_sharpes),
        worst_regime_sharpe=0.5,  # so regime gate isn't the blocker
        require_tstat_gate=require_tstat_gate,
        **kw,
    )


# 1. zero std → tstat 0.0, no div-by-zero
def test_tstat_zero_dispersion():
    r = _cpcv([1.0] * 5, n_folds=6)
    assert r.std_sharpe == 0.0
    assert r.path_sharpe_tstat == 0.0


# 2. known value: mean=1.0, std=0.5, n_folds=4 → t = 1.0/(0.5/2) = 4.0
def test_tstat_known_value():
    # Build paths with population std (np.std ddof=0) == 0.5 and mean == 1.0.
    # values 0.5 and 1.5 → mean 1.0, std 0.5.
    r = _cpcv([0.5, 1.5], n_folds=4)
    assert r.mean_sharpe == 1.0
    assert abs(r.std_sharpe - 0.5) < 1e-9
    assert abs(r.path_sharpe_tstat - 4.0) < 1e-9


# 3. require_tstat_gate=False + low tstat → gate not blocked by tstat; WARNING logged
def test_tstat_not_gating_when_flag_false(caplog):
    # Low tstat (< 2.0), all paths positive so pct_positive=1.0 and mean >= 0.8:
    # [0.1, 1.9, 0.1, 1.9], n_folds=2 → mean 1.0, std 0.9 → t ≈ 1.57 < 2.0.
    r = _cpcv([0.1, 1.9, 0.1, 1.9], n_folds=2, require_tstat_gate=False)
    assert r.path_sharpe_tstat < 2.0
    with caplog.at_level("WARNING"):
        passed = r.gate_passed()
    assert passed is True  # not blocked by the (sub-threshold) t-stat
    assert any("path t-stat" in rec.message for rec in caplog.records)


# 4. require_tstat_gate=True + tstat < threshold → gate False
def test_tstat_gating_blocks_low_tstat():
    # Same sub-threshold t-stat as above, but now gating is ON → gate must fail.
    r = _cpcv([0.1, 1.9, 0.1, 1.9], n_folds=2, require_tstat_gate=True)
    assert r.path_sharpe_tstat < 2.0
    assert r.gate_passed() is False


# 5. require_tstat_gate=True + tstat exactly 2.0 → passes (>= boundary)
def test_tstat_gating_boundary_passes():
    # Construct so tstat == 2.0 exactly: t = mean/(std/sqrt(n_folds)).
    # Pick mean=1.0, n_folds=4, want t=2.0 → std/2 = 0.5 → std = 1.0.
    # values 0.0 and 2.0 → mean 1.0, std 1.0 (ddof=0). n_folds=4 → t = 1.0/(1.0/2)=2.0.
    # But pct_positive of [0,2] = 0.5 < 0.75. Need pct_positive >= 0.75 AND tstat==2.
    # Use 4 values: a,a,b,b with all > 0 to keep pct_positive=1.0.
    # mean=m, std (ddof0) = (b-a)/2 for two-level split of equal counts.
    # want m=1.0, std=1.0 → (b-a)/2 = 1.0, (a+b)/2 = 1.0 → a=0.0... that's not >0.
    # Instead set n_folds higher so a smaller std reaches t=2 with all-positive paths.
    # mean=1.0, want std = mean*sqrt(n_folds)/2. n_folds=16 → std = 0.5*4/... let's
    # just solve: t=2 → std = mean*sqrt(n_folds)/t = 1.0*sqrt(16)/2 = 2.0. Too big.
    # Simpler: directly verify the boundary via two positive levels.
    # values 0.5,0.5,1.5,1.5 → mean 1.0, std 0.5, n_folds=16 → t=1.0/(0.5/4)=8 (too high).
    # Use n_folds=1 not allowed (<2 folds n/a). Find n_folds with std 0.5: t=1/(0.5/sqrt(n))=2
    #   → sqrt(n)=1 → n=1. Not usable. So use std 1.0 all-positive: values 0.5,0.5,2.5,2.5?
    # mean=1.5, std=1.0, n_folds=... t=2 → 1.5/(1.0/sqrt(n))=2 → sqrt(n)=2/1.5=1.333 → n=1.78.
    # Cleanest: just assert the property at the threshold using direct numbers and
    # check gate logic via the property, decoupled from pct_positive by patching it.
    r = _cpcv([0.5, 0.5, 1.5, 1.5], n_folds=16, require_tstat_gate=True)
    # mean 1.0, std 0.5, n_folds 16 → t = 1.0/(0.5/4) = 8.0 (>= 2.0) → passes.
    assert r.path_sharpe_tstat >= 2.0
    assert r.gate_passed() is True

    # Now force a tstat exactly at 2.0 and confirm >= boundary passes.
    with patch.object(CPCVResult, "path_sharpe_tstat", property(lambda self: 2.0)):
        assert r.gate_passed() is True


# 6. gate_detail always contains path_sharpe_tstat with correct value/flag
def test_gate_detail_contains_tstat():
    r = _cpcv([0.85, 0.95, 0.9, 1.4, 0.82], n_folds=6, require_tstat_gate=False)
    detail = r.gate_detail()
    assert "path_sharpe_tstat" in detail
    val, ok = detail["path_sharpe_tstat"]
    assert val == r.path_sharpe_tstat
    # Not gating → ok is always True.
    assert ok is True

    r_gate = _cpcv([0.3, 0.4, 1.5, 1.4, 0.5], n_folds=6, require_tstat_gate=True)
    val2, ok2 = r_gate.gate_detail()["path_sharpe_tstat"]
    assert ok2 == (val2 >= 2.0)


# 7. construction without require_tstat_gate still works (default False)
def test_construct_without_require_tstat_gate():
    r = CPCVResult(model_type="test", n_folds=6, n_paths=2, path_sharpes=[1.0] * 5)
    assert r.require_tstat_gate is False
    assert isinstance(r.path_sharpe_tstat, float)
