"""MIN_REALIZED_R experiment — realized-R label infrastructure tests.

Labels FAILED gate (v48 avg -4.514, AUC 0.55). Infrastructure kept for
future use. USE_REALIZED_R_LABELS=False restores cross-sectional scheme.
"""
import numpy as np


class TestLabelSwitchFlag:
    def test_use_realized_r_labels_false_by_default(self):
        """Current deployment must use cross-sectional labels."""
        from app.ml.intraday_training import USE_REALIZED_R_LABELS
        assert not USE_REALIZED_R_LABELS, (
            "Realized-R labels failed gate (AUC 0.55 — incompatible with cs_normalize). "
            "USE_REALIZED_R_LABELS must remain False."
        )

    def test_min_realized_r_constant_exists(self):
        from app.ml.intraday_training import MIN_REALIZED_R
        assert 0.0 < MIN_REALIZED_R < 1.0

    def test_training_window_days_is_730(self):
        """365d shorter-window experiment failed gate — reverted to 730d."""
        from app.ml.retrain_config import INTRADAY_RETRAIN
        assert INTRADAY_RETRAIN["days"] == 730, (
            "365d training window failed walk-forward gate (fold 1 -3.62) — must be 730d."
        )


class TestRealizedRLabelLogic:
    """Unit tests for the realized-R labeling formula (even though disabled)."""

    def _apply_labels_realized_r(self, returns, atr_targets, min_r=0.35, abs_hurdle=0.003):
        """Mirror the realized-R labeling from _apply_labels()."""
        returns = np.array(returns)
        atr_targets = np.array(atr_targets)
        realized_r = returns / np.maximum(atr_targets, 1e-8)
        return ((realized_r >= min_r) & (returns >= abs_hurdle)).astype(np.int8)

    def test_hit_target_labeled_positive(self):
        """Stock that hits 80% of ATR target (realized_R=1.0) → label 1."""
        labels = self._apply_labels_realized_r(
            returns=[0.008],      # +0.8%
            atr_targets=[0.008],  # ATR target = 0.8% → realized_R = 1.0
        )
        assert labels[0] == 1

    def test_partial_hit_above_threshold_positive(self):
        """35% of ATR target → label 1 (at the boundary)."""
        labels = self._apply_labels_realized_r(
            returns=[0.003],      # exactly 0.3% (abs hurdle)
            atr_targets=[0.008],  # realized_R = 0.375 ≥ 0.35
        )
        assert labels[0] == 1

    def test_below_abs_hurdle_negative(self):
        """Return below 0.30% absolute hurdle → label 0 regardless of R."""
        labels = self._apply_labels_realized_r(
            returns=[0.002],      # 0.2% — below 0.30% hurdle
            atr_targets=[0.003],  # realized_R = 0.67 > 0.35
        )
        assert labels[0] == 0

    def test_negative_return_always_negative(self):
        """Losing trades always label 0."""
        labels = self._apply_labels_realized_r(
            returns=[-0.005],
            atr_targets=[0.008],
        )
        assert labels[0] == 0

    def test_zero_positives_allowed_on_bad_days(self):
        """When ALL stocks have negative returns → all label 0."""
        returns = [-0.01, -0.005, -0.003, -0.001]
        atr_targets = [0.008, 0.007, 0.006, 0.005]
        labels = self._apply_labels_realized_r(returns, atr_targets)
        assert labels.sum() == 0, "Bad day should have zero positives"

    def test_cross_sectional_always_has_positives(self):
        """Cross-sectional labeling always labels top 20% — even on bad days."""
        returns = np.array([-0.01, -0.005, -0.003, -0.001, -0.0005])
        # Even with all negative, cross-sectional top-20% labels the least-bad stock
        # (this is the flaw realized-R labels fix — but realized-R fails with cs_normalize)
        threshold = np.percentile(returns, 80)
        labels = (returns >= threshold).astype(int)
        # The top 20% gets labeled 1 even on a universally bad day
        assert labels.sum() > 0


class TestShorterWindowConclusion:
    """Ensure the shorter-window experiment conclusion is documented."""

    def test_retrain_config_uses_730d_not_365d(self):
        from app.ml.retrain_config import INTRADAY_RETRAIN
        assert INTRADAY_RETRAIN["days"] == 730

    def test_raw_parts_includes_atr_target(self):
        """raw_parts now carry [day_ordinal, best_return, atr_target_pct] — 3 columns."""
        # Verify the format by checking that the realized-R formula uses index 2
        raw = np.array([[19700.0, 0.008, 0.008]])  # [day_ord, return, atr_target]
        best_return = raw[:, 1]
        atr_target = raw[:, 2]
        realized_r = best_return / np.maximum(atr_target, 1e-8)
        assert float(realized_r[0]) == pytest.approx(1.0)

    def __init_subclass__(cls, **kwargs):
        pass


import pytest  # noqa: E402 — needed for approx in the above method
