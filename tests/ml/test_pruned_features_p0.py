"""P0 tests: macro features pruned from ranker; cross-sectional unprune override."""
from __future__ import annotations

import inspect

from app.ml.training import _BASE_PRUNED, PRUNED_FEATURES, _resolve_pruned_features

P0_MACRO_PRUNED = frozenset({
    "vix_term_ratio",
    "breadth_rsp_spy_ratio_20d",
    "credit_hyg_ief_20d",
    "sector_dispersion_20d",
    "spy_above_ma50",
    "spy_above_ma200",
    "regime_score",
})


# ── P0: macro features pruned ─────────────────────────────────────────────────

def test_p0_macro_features_pruned_from_base():
    missing = P0_MACRO_PRUNED - _BASE_PRUNED
    assert not missing, f"P0 expects these in _BASE_PRUNED but they are absent: {missing}"


def test_p0_macro_features_pruned_from_resolved():
    resolved = _resolve_pruned_features()
    missing = P0_MACRO_PRUNED - resolved
    assert not missing, f"_resolve_pruned_features() missing P0 pruned features: {missing}"


def test_p0_macro_features_in_pruned_features_constant():
    missing = P0_MACRO_PRUNED - PRUNED_FEATURES
    assert not missing, f"PRUNED_FEATURES missing P0 pruned features: {missing}"


def test_features_module_still_computes_macro():
    """Compute paths stay in features.py — P2 regime-gate layer depends on them."""
    from app.ml import features as feat_mod
    src = inspect.getsource(feat_mod)
    for name in ("vix_term_ratio", "spy_above_ma200", "credit_hyg_ief_20d",
                 "breadth_rsp_spy_ratio_20d", "sector_dispersion_20d"):
        assert name in src, (
            f"{name} compute removed from features.py — P2 regime gate depends on it"
        )


# ── P1 unprune override helper ────────────────────────────────────────────────

def _get_unprune_fn():
    """Import _apply_unprune_overrides; skip test if not yet implemented."""
    try:
        from app.ml.training import _apply_unprune_overrides
        return _apply_unprune_overrides
    except ImportError:
        return None


def test_unprune_override_removes_named_features():
    fn = _get_unprune_fn()
    if fn is None:
        import pytest
        pytest.skip("_apply_unprune_overrides not yet implemented (P1)")
    out = fn(_BASE_PRUNED, ["adx_14_pct", "rs_vs_spy_5d"])
    assert "adx_14_pct" not in out
    assert "rs_vs_spy_5d" not in out
    assert "rsi_14" in out  # unrelated pruned feature stays pruned


def test_unprune_override_ignores_unknown():
    fn = _get_unprune_fn()
    if fn is None:
        import pytest
        pytest.skip("_apply_unprune_overrides not yet implemented (P1)")
    out = fn(_BASE_PRUNED, ["__not_a_real_feature__"])
    assert out == _BASE_PRUNED


def test_unprune_none_is_noop():
    fn = _get_unprune_fn()
    if fn is None:
        import pytest
        pytest.skip("_apply_unprune_overrides not yet implemented (P1)")
    assert fn(_BASE_PRUNED, None) == _BASE_PRUNED
