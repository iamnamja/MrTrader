"""
P1 tests: --unprune ablation flag for training.py.

Tests cover:
- apply_unprune_overrides removes named features from PRUNED_FEATURES
- Named bucket aliases (rs, adx_aroon, phase89, all) expand correctly
- Unknown names are silently accepted (no crash)
- The override does not mutate _BASE_PRUNED
"""
from __future__ import annotations

import importlib


def _reload_training():
    """Reload app.ml.training to get a fresh PRUNED_FEATURES."""
    import app.ml.training as m
    importlib.reload(m)
    return m


def test_unprune_rs_bucket_removes_rs_features():
    m = _reload_training()
    assert "rs_vs_spy_5d" in m.PRUNED_FEATURES
    m.apply_unprune_overrides(["rs"])
    assert "rs_vs_spy_5d" not in m.PRUNED_FEATURES
    assert "rs_vs_spy_10d" not in m.PRUNED_FEATURES
    assert "rs_vs_spy_60d" not in m.PRUNED_FEATURES


def test_unprune_adx_aroon_bucket():
    m = _reload_training()
    m.apply_unprune_overrides(["adx_aroon"])
    for feat in ("adx_14_pct", "adx_rising", "aroon_up_25", "aroon_down_25", "aroon_oscillator_25"):
        assert feat not in m.PRUNED_FEATURES, f"{feat} still in PRUNED_FEATURES after unprune"


def test_unprune_phase89_bucket():
    m = _reload_training()
    m.apply_unprune_overrides(["phase89"])
    for feat in ("drawdown_from_20d_high", "hurst_exponent_60d",
                 "pct_closes_above_ema20", "volatility_adj_dist_52wk_high"):
        assert feat not in m.PRUNED_FEATURES


def test_unprune_all_bucket_covers_all_three():
    m = _reload_training()
    rs = {"rs_vs_spy_5d", "rs_vs_spy_10d", "rs_vs_spy_60d"}
    adx = {"adx_14_pct", "adx_rising", "aroon_up_25", "aroon_down_25", "aroon_oscillator_25"}
    ph89 = {"drawdown_from_20d_high", "hurst_exponent_60d",
            "pct_closes_above_ema20", "volatility_adj_dist_52wk_high"}
    m.apply_unprune_overrides(["all"])
    for feat in rs | adx | ph89:
        assert feat not in m.PRUNED_FEATURES


def test_unprune_individual_feature_name():
    m = _reload_training()
    assert "adx_14_pct" in m.PRUNED_FEATURES
    m.apply_unprune_overrides(["adx_14_pct"])
    assert "adx_14_pct" not in m.PRUNED_FEATURES


def test_unprune_unknown_name_does_not_crash():
    m = _reload_training()
    before = len(m.PRUNED_FEATURES)
    m.apply_unprune_overrides(["nonexistent_feature_xyz"])
    # No crash; unknown name has no effect
    assert len(m.PRUNED_FEATURES) == before


def test_unprune_does_not_mutate_base_pruned():
    m = _reload_training()
    base_before = frozenset(m._BASE_PRUNED)
    m.apply_unprune_overrides(["all"])
    assert m._BASE_PRUNED == base_before


def test_unprune_none_is_noop():
    m = _reload_training()
    before = frozenset(m.PRUNED_FEATURES)
    # Calling with empty list should be a noop
    m.apply_unprune_overrides([])
    assert m.PRUNED_FEATURES == before
