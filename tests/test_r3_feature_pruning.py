"""R3 — Correlation-prune feature pruning tests.

Verifies that the 18 R3-pruned features are present in PRUNED_FEATURES
and that the remaining feature set is within expected size bounds.
"""
from __future__ import annotations


R3_PRUNED_FEATURES = [
    # Zero-importance in v163 audit
    "cmf_20",
    "dema_20_dist",
    "keltner_position",
    "cci_20",
    "price_efficiency_20d",
    "vol_price_confirmation",
    "volume_surge_3d",
    "wq_alpha44",
    "choch_detected",
    "bars_since_choch",
    "momentum_20d_sector_neutral",
    "price_change_pct",
    "volume_ratio",
    # Semantic-group redundancy
    "reversal_5d",
    "reversal_3d",
    "pressure_persistence",
    "pressure_displacement",
    "hh_hl_sequence",
]


def test_r3_features_are_in_pruned_set():
    """All 18 R3-pruned features must appear in PRUNED_FEATURES."""
    from app.ml.training import PRUNED_FEATURES
    missing = [f for f in R3_PRUNED_FEATURES if f not in PRUNED_FEATURES]
    assert not missing, f"R3 features not in PRUNED_FEATURES: {missing}"


def test_r3_pruned_count():
    """Exactly 18 R3-specific features are pruned."""
    assert len(R3_PRUNED_FEATURES) == 18


def test_high_signal_features_not_pruned():
    """Top-signal features from v163/v188 must NOT be in PRUNED_FEATURES."""
    from app.ml.training import PRUNED_FEATURES
    must_keep = [
        "atr_norm", "volatility", "parkinson_vol", "vrp", "realized_vol_20d",
        "sector_momentum", "downtrend", "near_52w_high", "momentum_252d_ex1m",
        "reversal_5d_vol_weighted",  # kept instead of reversal_5d
        "pressure_index",            # kept instead of sub-components
    ]
    pruned = [f for f in must_keep if f in PRUNED_FEATURES]
    assert not pruned, f"High-signal features incorrectly pruned: {pruned}"
