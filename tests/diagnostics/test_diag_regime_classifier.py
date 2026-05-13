"""Tests for scripts/diag_regime_classifier.py — Phase A4 regime validation."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.diag_regime_classifier import (
    RISK_OFF_THRESHOLD,
    RISK_ON_THRESHOLD,
    VIX_ELEVATED,
    VIX_HIGH,
    _class_distribution,
    _compute_expected_labels,
    _confusion_matrix,
)


# ── _compute_expected_labels ──────────────────────────────────────────────────

class TestComputeExpectedLabels:
    def _make_vix(self, values: list) -> pd.Series:
        idx = pd.bdate_range("2025-01-02", periods=len(values))
        return pd.Series(values, index=idx)

    def test_low_vix_maps_to_risk_on(self):
        vix = self._make_vix([12.0, 14.0, 15.0])
        labels = _compute_expected_labels(vix)
        assert (labels == "RISK_ON").all()

    def test_high_vix_maps_to_risk_off(self):
        vix = self._make_vix([30.0, 35.0, 40.0])
        labels = _compute_expected_labels(vix)
        assert (labels == "RISK_OFF").all()

    def test_elevated_vix_maps_to_neutral(self):
        vix = self._make_vix([20.0, 22.0, 24.0])
        labels = _compute_expected_labels(vix)
        assert (labels == "NEUTRAL").all()

    def test_mixed_vix_correct_labels(self):
        vix = self._make_vix([10.0, 21.0, 30.0])
        labels = _compute_expected_labels(vix)
        assert labels.iloc[0] == "RISK_ON"
        assert labels.iloc[1] == "NEUTRAL"
        assert labels.iloc[2] == "RISK_OFF"

    def test_boundary_vix_elevated(self):
        """VIX == VIX_ELEVATED should be RISK_ON (<=)."""
        vix = self._make_vix([VIX_ELEVATED])
        labels = _compute_expected_labels(vix)
        assert labels.iloc[0] == "RISK_ON"

    def test_boundary_vix_high(self):
        """VIX == VIX_HIGH should be RISK_OFF (>=)."""
        vix = self._make_vix([VIX_HIGH])
        labels = _compute_expected_labels(vix)
        assert labels.iloc[0] == "RISK_OFF"


# ── _class_distribution ───────────────────────────────────────────────────────

class TestClassDistribution:
    def test_single_class(self):
        dist = _class_distribution(["RISK_ON"] * 10)
        assert abs(dist["RISK_ON"] - 1.0) < 1e-9

    def test_equal_split(self):
        dist = _class_distribution(["RISK_ON", "RISK_OFF"] * 5)
        assert abs(dist["RISK_ON"] - 0.5) < 1e-9
        assert abs(dist["RISK_OFF"] - 0.5) < 1e-9

    def test_fractions_sum_to_one(self):
        labels = ["RISK_ON"] * 6 + ["RISK_OFF"] * 3 + ["NEUTRAL"] * 1
        dist = _class_distribution(labels)
        assert abs(sum(dist.values()) - 1.0) < 1e-9

    def test_trivial_majority_detection(self):
        """Simulates the R5 trivial-AUC problem: 98% RISK_ON."""
        labels = ["RISK_ON"] * 98 + ["RISK_OFF"] * 2
        dist = _class_distribution(labels)
        risk_on_pct = dist.get("RISK_ON", 0.0)
        assert risk_on_pct > 0.80, "Should detect trivial majority class"


# ── _confusion_matrix ─────────────────────────────────────────────────────────

class TestConfusionMatrix:
    ALL_LABELS = ["RISK_ON", "NEUTRAL", "RISK_OFF"]

    def test_perfect_agreement(self):
        labels = ["RISK_ON", "NEUTRAL", "RISK_OFF"]
        cm = _confusion_matrix(labels, labels, self.ALL_LABELS)
        # Diagonal should be 1, off-diagonal 0
        for lbl in self.ALL_LABELS:
            assert cm.loc[lbl, lbl] == 1
        assert cm.values.sum() == len(labels)

    def test_all_mismatch(self):
        predicted = ["RISK_ON", "RISK_ON", "RISK_ON"]
        expected = ["RISK_OFF", "NEUTRAL", "RISK_OFF"]
        cm = _confusion_matrix(predicted, expected, self.ALL_LABELS)
        # All predictions = RISK_ON → entire row RISK_ON is populated
        assert cm.loc["RISK_ON", "RISK_OFF"] == 2
        assert cm.loc["RISK_ON", "NEUTRAL"] == 1
        assert cm.loc["RISK_OFF"].sum() == 0
        assert cm.loc["NEUTRAL"].sum() == 0

    def test_output_shape(self):
        cm = _confusion_matrix(
            ["RISK_ON", "RISK_OFF"],
            ["RISK_ON", "RISK_OFF"],
            self.ALL_LABELS,
        )
        assert cm.shape == (3, 3)

    def test_unknown_labels_ignored(self):
        predicted = ["UNKNOWN_LABEL", "RISK_ON"]
        expected = ["RISK_ON", "RISK_ON"]
        cm = _confusion_matrix(predicted, expected, self.ALL_LABELS)
        # UNKNOWN_LABEL not in all_labels → skipped; only RISK_ON/RISK_ON counted
        assert cm.loc["RISK_ON", "RISK_ON"] == 1
        assert cm.values.sum() == 1

    def test_trivial_risk_on_bias_visible(self):
        """Simulates a classifier that always predicts RISK_ON."""
        predicted = ["RISK_ON"] * 30
        expected = ["RISK_ON"] * 20 + ["RISK_OFF"] * 7 + ["NEUTRAL"] * 3
        cm = _confusion_matrix(predicted, expected, self.ALL_LABELS)
        # All predictions in RISK_ON row
        assert cm.loc["RISK_ON"].sum() == 30
        assert cm.loc["RISK_OFF"].sum() == 0
        assert cm.loc["NEUTRAL"].sum() == 0


# ── Constants sanity ──────────────────────────────────────────────────────────

class TestConstants:
    def test_thresholds_in_0_1(self):
        assert 0 < RISK_OFF_THRESHOLD < RISK_ON_THRESHOLD < 1

    def test_vix_thresholds_ordered(self):
        assert VIX_ELEVATED < VIX_HIGH

    def test_risk_on_threshold_above_half(self):
        assert RISK_ON_THRESHOLD > 0.5

    def test_risk_off_threshold_below_half(self):
        assert RISK_OFF_THRESHOLD < 0.5
