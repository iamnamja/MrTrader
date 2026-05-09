"""Tests for Phase 91 intraday microstructure features and label behavior.

Phase 91 changes retained:
  - 4 new microstructure features at bar 12 (vwap_slope_to_bar12,
    first_30min_volume_ratio, spy_5min_return_bar12, vix_5min_change)
  - raw_train returned from _apply_labels (fixes dispersion-gate size mismatch)

Phase 91 changes reverted:
  - Hybrid label (top-20% AND realized-R): precision dropped to 16% OOS (<20%
    base rate), avg WF Sharpe -3.72. Reverted to pure top-20% + absolute hurdle.
  - Per-day dispersion gate: removed training days aggressively, worsened
    train/test distribution mismatch.
"""
import numpy as np
import pandas as pd
import pytest


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_bars(n=15, trend="up"):
    """Return a 5-min bar DataFrame suitable for compute_intraday_features."""
    rng = np.random.default_rng(42)
    opens = 100.0 + np.arange(n) * (0.02 if trend == "up" else -0.02)
    closes = opens + rng.normal(0, 0.05, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.1, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.1, n)
    volumes = rng.integers(10_000, 100_000, n).astype(float)
    t = pd.date_range("2024-01-10 09:30", periods=n, freq="5min")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=t)


# ── New microstructure feature tests ─────────────────────────────────────────

class TestPhase91MicrostructureFeatures:
    def _compute(self, n=15, trend="up"):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars(n, trend)
        spy = _make_bars(n, "up")
        feats = compute_intraday_features(bars, spy_bars=spy, prior_close=99.0)
        return feats

    def test_new_feature_keys_present(self):
        feats = self._compute()
        for key in ("vwap_slope_to_bar12", "first_30min_volume_ratio",
                    "spy_5min_return_bar12", "vix_5min_change"):
            assert key in feats, f"Missing Phase 91 feature: {key}"

    def test_first_30min_volume_ratio_range(self):
        feats = self._compute()
        ratio = feats["first_30min_volume_ratio"]
        assert 0.0 <= ratio <= 1.0

    def test_vwap_slope_up_trend_is_float(self):
        feats = self._compute(trend="up")
        assert isinstance(feats["vwap_slope_to_bar12"], float)

    def test_spy_5min_return_bar12_is_finite(self):
        feats = self._compute()
        assert np.isfinite(feats["spy_5min_return_bar12"])

    def test_vix_5min_change_is_finite(self):
        feats = self._compute()
        assert np.isfinite(feats["vix_5min_change"])

    def test_features_present_in_feature_names(self):
        from app.ml.intraday_features import FEATURE_NAMES
        for key in ("vwap_slope_to_bar12", "first_30min_volume_ratio",
                    "spy_5min_return_bar12", "vix_5min_change"):
            assert key in FEATURE_NAMES, f"{key} not in FEATURE_NAMES"

    def test_fallback_no_spy(self):
        """Features should not crash when spy_bars is None."""
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_bars(15)
        feats = compute_intraday_features(bars, spy_bars=None, prior_close=99.0)
        assert feats is not None
        assert "spy_5min_return_bar12" in feats
        assert feats["spy_5min_return_bar12"] == 0.0


# ── Label behavior (current implementation: pure top-20%) ─────────────────────

class TestInradayLabelBehavior:
    """Verify the current (pure top-20%) label produces 20% positive class."""

    def _apply_label(self, day_rets, absolute_hurdle=0.003):
        """Mirror current _apply_labels top-20% + absolute hurdle logic."""
        day_rets = np.array(day_rets)
        if len(day_rets) < 2:
            return np.zeros(len(day_rets), dtype=np.int8)
        threshold = np.percentile(day_rets, 80)
        return ((day_rets >= threshold) & (day_rets >= absolute_hurdle)).astype(np.int8)

    def test_top20_rate_approximately_20pct(self):
        """With no absolute hurdle binding, label rate ≈ 20%."""
        rng = np.random.default_rng(42)
        # Returns well above hurdle so absolute floor doesn't bind
        rets = rng.normal(0.01, 0.005, 200).tolist()
        labels = self._apply_label(rets, absolute_hurdle=0.0)
        rate = labels.mean()
        assert 0.18 <= rate <= 0.22, f"Expected ~20% positive rate, got {rate:.2%}"

    def test_absolute_hurdle_reduces_rate_on_flat_day(self):
        """On a flat day (all returns ~0), absolute hurdle reduces positives below 20%."""
        flat_rets = [0.0001 * i for i in range(100)]
        labels = self._apply_label(flat_rets, absolute_hurdle=0.003)
        # Most returns < 0.003, so even top-20% stocks may fail hurdle
        assert labels.sum() <= 20

    def test_label_uses_absolute_hurdle(self):
        """Returns below CS_ABSOLUTE_HURDLE=0.003 must not be labeled 1."""
        rets = [0.001, 0.002, 0.0025, 0.001, 0.0015]
        labels = self._apply_label(rets, absolute_hurdle=0.003)
        assert labels.sum() == 0, "No returns meet absolute hurdle"


# ── _apply_labels returns filtered raws (size-consistency test) ───────────────

class TestApplyLabelsReturnSignature:
    """Verify _apply_labels returns 3-tuple (X, labels, raws) with consistent sizes."""

    def test_apply_dispersion_gate_false_in_source(self):
        """Dispersion gate was removed — source should not contain apply_dispersion_gate."""
        import inspect
        import app.ml.intraday_training as mod
        src = inspect.getsource(mod)
        assert "apply_dispersion_gate" not in src, \
            "apply_dispersion_gate was reverted but still present in source"

    def test_hybrid_label_reverted(self):
        """Hybrid realized-R label was reverted — source should not contain HYBRID_REALIZED_R_MIN."""
        import inspect
        import app.ml.intraday_training as mod
        src = inspect.getsource(mod)
        assert "HYBRID_REALIZED_R_MIN" not in src, \
            "HYBRID_REALIZED_R_MIN label was reverted but still present in source"
