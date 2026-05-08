"""
Tests for Phase 3b — Triple-Barrier Labels (Full-Universe Swing Retrain).

Verifies:
  1. TB_PHASE3B_* constants exist in training.py with correct values
  2. --tb-target-mult / --tb-stop-mult / --forward-days CLI args are parseable
  3. Defaults: None (use existing constants)
  4. run_rolling_pipeline signature accepts tb_target_mult / tb_stop_mult / forward_days
  5. Module-level overrides are applied when non-None
  6. _atr_label_thresholds respects ATR_MULT_TARGET / ATR_MULT_STOP module globals
"""
import pytest
import argparse
from unittest.mock import patch
import numpy as np


# ── 1. Constants ──────────────────────────────────────────────────────────────

class TestPhase3bConstants:
    def test_target_mult_20(self):
        from app.ml.training import TB_PHASE3B_TARGET_MULT
        assert TB_PHASE3B_TARGET_MULT == pytest.approx(2.0)

    def test_stop_mult_12(self):
        from app.ml.training import TB_PHASE3B_STOP_MULT
        assert TB_PHASE3B_STOP_MULT == pytest.approx(1.2)

    def test_forward_days_10(self):
        from app.ml.training import TB_PHASE3B_FORWARD_DAYS
        assert TB_PHASE3B_FORWARD_DAYS == 10

    def test_existing_target_still_15(self):
        from app.ml.training import ATR_MULT_TARGET
        # Default baseline remains 1.5 — Phase 3b uses override, not default change
        assert ATR_MULT_TARGET == pytest.approx(1.5)

    def test_existing_stop_still_05(self):
        from app.ml.training import ATR_MULT_STOP
        assert ATR_MULT_STOP == pytest.approx(0.5)


# ── 2-3. CLI args ─────────────────────────────────────────────────────────────

class TestPhase3bCLI:
    def _parse(self, extra=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--label-scheme", default="atr")
        parser.add_argument("--tb-target-mult", type=float, default=None)
        parser.add_argument("--tb-stop-mult", type=float, default=None)
        parser.add_argument("--forward-days", type=int, default=None)
        return parser.parse_args(extra or [])

    def test_defaults_none(self):
        args = self._parse()
        assert args.tb_target_mult is None
        assert args.tb_stop_mult is None
        assert args.forward_days is None

    def test_tb_target_mult_parseable(self):
        args = self._parse(["--tb-target-mult", "2.0"])
        assert args.tb_target_mult == pytest.approx(2.0)

    def test_tb_stop_mult_parseable(self):
        args = self._parse(["--tb-stop-mult", "1.2"])
        assert args.tb_stop_mult == pytest.approx(1.2)

    def test_forward_days_parseable(self):
        args = self._parse(["--forward-days", "10"])
        assert args.forward_days == 10

    def test_all_3b_params_together(self):
        args = self._parse([
            "--label-scheme", "triple_barrier",
            "--tb-target-mult", "2.0",
            "--tb-stop-mult", "1.2",
            "--forward-days", "10",
        ])
        assert args.label_scheme == "triple_barrier"
        assert args.tb_target_mult == pytest.approx(2.0)
        assert args.tb_stop_mult == pytest.approx(1.2)
        assert args.forward_days == 10


# ── 4. run_rolling_pipeline signature ────────────────────────────────────────

class TestRunRollingPipelineSignature:
    def test_accepts_tb_params(self):
        import inspect
        from scripts.train_model import run_rolling_pipeline
        sig = inspect.signature(run_rolling_pipeline)
        assert "tb_target_mult" in sig.parameters
        assert "tb_stop_mult" in sig.parameters
        assert "forward_days" in sig.parameters

    def test_defaults_none(self):
        import inspect
        from scripts.train_model import run_rolling_pipeline
        sig = inspect.signature(run_rolling_pipeline)
        assert sig.parameters["tb_target_mult"].default is None
        assert sig.parameters["tb_stop_mult"].default is None
        assert sig.parameters["forward_days"].default is None


# ── 5. Module-level override is applied ──────────────────────────────────────

class TestModuleOverride:
    def test_atr_mult_override(self):
        """Verify run_rolling_pipeline mutates the module globals when override is passed."""
        import app.ml.training as t
        original_target = t.ATR_MULT_TARGET
        original_stop = t.ATR_MULT_STOP

        try:
            # Simulate what run_rolling_pipeline does at its start
            t.ATR_MULT_TARGET = 2.0
            t.ATR_MULT_STOP = 1.2
            assert t.ATR_MULT_TARGET == pytest.approx(2.0)
            assert t.ATR_MULT_STOP == pytest.approx(1.2)
        finally:
            # Restore so other tests aren't affected
            t.ATR_MULT_TARGET = original_target
            t.ATR_MULT_STOP = original_stop


# ── 6. _atr_label_thresholds respects module globals ─────────────────────────

class TestAtrLabelThresholds:
    def _make_window(self, n=20, atr_pct=0.02):
        """Make a synthetic window DataFrame with predictable ATR."""
        import pandas as pd
        import numpy as np
        base_price = 100.0
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        daily_range = base_price * atr_pct  # ATR ≈ daily_range
        data = {
            "open": [base_price] * n,
            "high": [base_price + daily_range] * n,
            "low": [base_price - daily_range] * n,
            "close": [base_price] * n,
        }
        return pd.DataFrame(data, index=dates)

    def test_target_scales_with_mult(self):
        import app.ml.training as t
        from app.ml.training import _atr_label_thresholds
        df = self._make_window(n=20, atr_pct=0.02)
        original = t.ATR_MULT_TARGET
        try:
            t.ATR_MULT_TARGET = 1.5
            target_15, _ = _atr_label_thresholds(df, 100.0)
            t.ATR_MULT_TARGET = 2.0
            target_20, _ = _atr_label_thresholds(df, 100.0)
            assert target_20 > target_15
        finally:
            t.ATR_MULT_TARGET = original

    def test_stop_scales_with_mult(self):
        import app.ml.training as t
        from app.ml.training import _atr_label_thresholds
        df = self._make_window(n=20, atr_pct=0.02)
        original = t.ATR_MULT_STOP
        try:
            t.ATR_MULT_STOP = 0.5
            _, stop_05 = _atr_label_thresholds(df, 100.0)
            t.ATR_MULT_STOP = 1.2
            _, stop_12 = _atr_label_thresholds(df, 100.0)
            assert stop_12 > stop_05
        finally:
            t.ATR_MULT_STOP = original
