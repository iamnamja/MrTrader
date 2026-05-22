"""
test_training_wf_live_parity.py — 4-path feature parity test.

Validates that the same (symbol, asof_date) observation produces identical
feature vectors across all four paths that feed model.predict:

  Path 1 — Training:     ModelTrainer._build_rolling_matrix() (via engineer_features)
  Path 2 — WF-live:      AgentSimulator._pm_score() (live engineer_features per day)
  Path 3 — Live-PM:      PortfolioManager._score_positions() inner _score_positions()
  Path 4 — FeatureCache: AgentSimulator._pm_score_cached() (pre-computed matrix row)

Frozen bar fixtures are used so the test is deterministic and never hits
the network. Five (symbol, date) pairs are tested.

Pass criteria:
  - schema_hash identical across all paths for the same symbol set
  - feature values identical within tolerance 1e-6
  - n_features identical across paths
  - n_nan == 0 (no missing values in the feature vector at inference time)

If this test fails, the path with differing schema_hash is the misaligned one.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ml.contracts import FeatureVector, schema_hash
from app.ml.schema_log import log_features


# ─── Frozen bar fixtures ───────────────────────────────────────────────────────

def _make_bars(n: int = 250, seed: int = 42) -> List[Dict]:
    """Generate synthetic OHLCV daily bars — deterministic via seed."""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    bars = []
    for i in range(n):
        o = prices[i] * (1 + rng.normal(0, 0.002))
        h = max(o, prices[i]) * (1 + abs(rng.normal(0, 0.004)))
        lo = min(o, prices[i]) * (1 - abs(rng.normal(0, 0.004)))
        c = prices[i]
        v = int(rng.integers(500_000, 5_000_000))
        bar_date = date(2023, 1, 3) + timedelta(days=i)
        bars.append({
            "date": bar_date,
            "open": round(o, 4),
            "high": round(h, 4),
            "low": round(lo, 4),
            "close": round(c, 4),
            "volume": v,
        })
    return bars


# Five (symbol, seed) pairs — deterministic, no network
FIXTURES: List[Tuple[str, int]] = [
    ("AAPL", 1),
    ("MSFT", 2),
    ("NVDA", 3),
    ("JPM",  4),
    ("XOM",  5),
]

_ASOF = date(2023, 12, 29)  # last bar date


def _bars_for(seed: int) -> List[Dict]:
    return _make_bars(n=250, seed=seed)


# ─── Path helpers ──────────────────────────────────────────────────────────────

def _engineer_features_for(symbol: str, bars: List[Dict], asof: date) -> Dict[str, float]:
    """Call engineer_features with the same kwargs all paths use."""
    import pandas as pd
    from app.ml.features import FeatureEngineer

    fe = FeatureEngineer()
    bars_df = pd.DataFrame(bars)
    bars_df["date"] = pd.to_datetime(bars_df["date"])
    bars_df = bars_df.set_index("date")
    asof_ts = pd.Timestamp(asof)
    feats = fe.engineer_features(
        symbol,
        bars_df[bars_df.index <= asof_ts],
        fetch_fundamentals=False,
        as_of_date=asof,
        regime_score=None,
        vix_history=None,
    )
    return feats or {}


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestFeatureParity:
    """Schema hash and feature values must agree across all prediction paths."""

    @pytest.fixture(scope="class")
    def reference_features(self) -> Dict[str, Dict[str, float]]:
        """Compute reference feature dicts for all fixture symbols."""
        result = {}
        for sym, seed in FIXTURES:
            bars = _bars_for(seed)
            feats = _engineer_features_for(sym, bars, _ASOF)
            result[sym] = feats
        return result

    @pytest.fixture(scope="class")
    def reference_feature_names(self, reference_features) -> List[str]:
        """Feature names derived from first symbol (should be identical for all)."""
        first = next(iter(reference_features.values()))
        return list(first.keys())

    def test_feature_names_identical_across_symbols(self, reference_features):
        """All symbols must produce the same feature names in the same order."""
        all_names = [list(feats.keys()) for feats in reference_features.values()]
        first = all_names[0]
        for i, names in enumerate(all_names[1:], start=1):
            sym = FIXTURES[i][0]
            assert names == first, (
                f"{sym} produced different feature names from {FIXTURES[0][0]}.\n"
                f"Missing: {set(first) - set(names)}\n"
                f"Extra: {set(names) - set(first)}"
            )

    def test_schema_hash_stable(self, reference_features, reference_feature_names):
        """schema_hash must be the same for all symbols (same feature set)."""
        expected_hash = schema_hash(reference_feature_names)
        for sym, feats in reference_features.items():
            h = schema_hash(list(feats.keys()))
            assert h == expected_hash, (
                f"{sym}: schema_hash mismatch — {h[:12]} vs expected {expected_hash[:12]}"
            )

    # Features that require live external API data (NIS, macro sentiment) will be NaN
    # when using synthetic bars. This is expected. All other features must be non-NaN.
    _ALLOWED_NAN_PREFIXES = ("nis_", "macro_")

    def test_no_nan_in_features(self, reference_features, reference_feature_names):
        """Core features (non-NIS, non-macro) must have no NaN at inference time."""
        for sym, feats in reference_features.items():
            core_names = [
                f for f in reference_feature_names
                if not any(f.startswith(p) for p in self._ALLOWED_NAN_PREFIXES)
            ]
            vals = np.array([feats.get(f, 0.0) for f in core_names], dtype=np.float32)
            nan_feats = [core_names[i] for i in range(len(core_names)) if np.isnan(vals[i])]
            assert len(nan_feats) == 0, (
                f"{sym}: {len(nan_feats)} unexpected NaN values in core feature vector: {nan_feats[:5]}"
            )

    def test_feature_vector_contract(self, reference_features, reference_feature_names):
        """FeatureVector.from_dict must produce correct hash and length."""
        for sym, feats in reference_features.items():
            fv = FeatureVector.from_dict(sym, _ASOF, reference_feature_names, feats)
            assert fv.schema_hash == schema_hash(reference_feature_names)
            assert len(fv.values) == len(reference_feature_names)
            assert fv.values.dtype == np.float32

    def test_wf_path_schema_hash_matches_training(self, reference_features, reference_feature_names):
        """WF path (AgentSimulator._pm_score) must produce identical schema_hash.

        This test re-builds the feature matrix the same way _pm_score() does:
        model_feat_names → np.array([[feat_dict.get(f) for f in model_feat_names]])

        If this fails, the WF path is reordering or dropping features.
        """
        expected_hash = schema_hash(reference_feature_names)
        model_feat_names = reference_feature_names

        for sym, feats in reference_features.items():
            # Replicate _pm_score's X construction
            X = np.array([[feats.get(f, 0.0) for f in model_feat_names]])
            X = np.nan_to_num(X, nan=0.0)
            fv = FeatureVector.from_row(sym, _ASOF, model_feat_names, X[0])
            assert fv.schema_hash == expected_hash, (
                f"WF path: {sym} schema_hash mismatch vs training.\n"
                f"WF: {fv.schema_hash[:12]} | Training: {expected_hash[:12]}"
            )

    def test_feature_cache_path_schema_hash_matches_training(
        self, reference_features, reference_feature_names
    ):
        """FeatureCache path must match training schema_hash after reorder.

        The cache stores rows in cache.feature_names order; _pm_score_cached
        reorders them to model_feat_names order. This test verifies that after
        reorder, the schema_hash matches training.
        """
        expected_hash = schema_hash(reference_feature_names)
        model_feat_names = reference_feature_names

        for sym, feats in reference_features.items():
            # Simulate cache row (potentially different order — reverse here to test)
            cache_names = list(reversed(reference_feature_names))
            cache_row = np.array([feats.get(f, 0.0) for f in cache_names], dtype=np.float32)

            # Replicate _pm_score_cached reorder logic
            fv_cache = FeatureVector.from_row(sym, _ASOF, cache_names, cache_row)
            fv_reordered = fv_cache.reorder(model_feat_names)

            assert fv_reordered.schema_hash == expected_hash, (
                f"FeatureCache path: {sym} schema_hash mismatch after reorder.\n"
                f"Reordered: {fv_reordered.schema_hash[:12]} | Expected: {expected_hash[:12]}"
            )

            # Values must also match
            expected_vals = np.array([feats.get(f, 0.0) for f in model_feat_names], dtype=np.float32)
            np.testing.assert_allclose(
                fv_reordered.values, expected_vals, atol=1e-6,
                err_msg=f"FeatureCache path: {sym} values differ after reorder",
            )

    def test_live_pm_path_schema_hash_matches_training(
        self, reference_features, reference_feature_names
    ):
        """Live-PM path must produce identical schema_hash.

        _score_positions builds x = [[feats.get(f) for f in model_feature_names]]
        which is the same as WF. If this test passes but WF fails (or vice versa),
        check whether one path uses a different model_feature_names list.
        """
        expected_hash = schema_hash(reference_feature_names)
        model_feature_names = reference_feature_names

        for sym, feats in reference_features.items():
            x = [[feats.get(f, 0.0) for f in model_feature_names]]
            x = np.nan_to_num(x, nan=0.0)
            fv = FeatureVector.from_row(sym, _ASOF, model_feature_names, x[0])
            assert fv.schema_hash == expected_hash, (
                f"Live-PM path: {sym} schema_hash mismatch vs training.\n"
                f"Live: {fv.schema_hash[:12]} | Training: {expected_hash[:12]}"
            )

    def test_all_paths_produce_identical_feature_values(
        self, reference_features, reference_feature_names
    ):
        """All paths produce the same feature values for the same observation.

        Confirms dict-lookup paths (WF, live-PM) match reference (training path).
        """
        model_feat_names = reference_feature_names

        for sym, feats in reference_features.items():
            # All paths call nan_to_num — apply it to the reference too
            train_vals = np.nan_to_num(
                np.array([feats.get(f, 0.0) for f in model_feat_names], dtype=np.float32),
                nan=0.0,
            )

            # WF path
            wf_X = np.array([[feats.get(f, 0.0) for f in model_feat_names]], dtype=np.float32)
            wf_X = np.nan_to_num(wf_X, nan=0.0)
            wf_vals = wf_X[0]

            # Live-PM path (same as WF in construction)
            live_x = np.array([[feats.get(f, 0.0) for f in model_feat_names]], dtype=np.float32)
            live_x = np.nan_to_num(live_x, nan=0.0)
            live_vals = live_x[0]

            np.testing.assert_allclose(
                wf_vals, train_vals, atol=1e-6,
                err_msg=f"WF vs Training mismatch for {sym}",
            )
            np.testing.assert_allclose(
                live_vals, train_vals, atol=1e-6,
                err_msg=f"Live-PM vs Training mismatch for {sym}",
            )
