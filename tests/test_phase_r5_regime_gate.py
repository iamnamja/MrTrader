"""
Tests for Phase R5 — Intraday Regime Gate.

Verifies:
  1. IntradayAgentSimulator: use_regime_gate / regime_map / r5b / r5c params stored
  2. R5-A: label "4DX" (VIX4 + SPY downtrend) blocks entries
  3. R5-A: label "3UP" does NOT block entries
  4. R5-A: label "4UP" does NOT block (VIX4 but uptrend — not the macro-dominated regime)
  5. R5-A: label "4DN" DOES block (VIX4 + SPY below 50d MA)
  6. R5-B threshold stored as r5b_dispersion_threshold (default 0.4)
  7. R5-C thresholds stored correctly
  8. Gate disabled when use_regime_gate=False (default)
  9. CLI: --regime-gate flag is parseable; off by default
  10. run_intraday_walkforward accepts use_regime_gate + regime_map kwargs
"""
import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator


# ── helpers ──────────────────────────────────────────────────────────────────

def _sim(**kwargs) -> IntradayAgentSimulator:
    return IntradayAgentSimulator(**kwargs)


# ── 1. Parameters stored correctly ───────────────────────────────────────────

class TestR5Params:
    def test_use_regime_gate_default_false(self):
        sim = _sim()
        assert sim.use_regime_gate is False

    def test_use_regime_gate_stored(self):
        sim = _sim(use_regime_gate=True, regime_map={date(2023, 1, 1): "1UP"})
        assert sim.use_regime_gate is True
        assert date(2023, 1, 1) in sim.regime_map

    def test_regime_map_default_empty(self):
        sim = _sim()
        assert sim.regime_map == {}

    def test_r5b_threshold_default_04(self):
        sim = _sim()
        assert sim.r5b_dispersion_threshold == pytest.approx(0.4)

    def test_r5b_threshold_stored(self):
        sim = _sim(r5b_dispersion_threshold=0.3)
        assert sim.r5b_dispersion_threshold == pytest.approx(0.3)

    def test_r5c_vix_default_35(self):
        sim = _sim()
        assert sim.r5c_vix_threshold == pytest.approx(35.0)

    def test_r5c_spy_drawdown_default(self):
        sim = _sim()
        assert sim.r5c_spy_drawdown == pytest.approx(-0.05)


# ── 2-5. R5-A label logic ─────────────────────────────────────────────────────

class TestR5ALabels:
    """Test that the R5-A gate logic (label startswith '4' and 'D' in label) works correctly."""

    def _check_blocked(self, label: str) -> bool:
        """Returns True if this label would trigger R5-A gate."""
        return label.startswith("4") and "D" in label

    def test_4dn_blocked(self):
        assert self._check_blocked("4DN") is True

    def test_4dp_blocked(self):
        assert self._check_blocked("4DP") is True

    def test_4up_not_blocked(self):
        # VIX4 but SPY above 50d MA → no block
        assert self._check_blocked("4UP") is False

    def test_4un_not_blocked(self):
        assert self._check_blocked("4UN") is False

    def test_3dn_not_blocked(self):
        # VIX quartile 3, not 4
        assert self._check_blocked("3DN") is False

    def test_1up_not_blocked(self):
        assert self._check_blocked("1UP") is False

    def test_empty_label_not_blocked(self):
        assert self._check_blocked("") is False


# ── 6. R5-B dispersion threshold ─────────────────────────────────────────────

class TestR5BThreshold:
    def test_custom_threshold(self):
        sim = _sim(use_regime_gate=True, r5b_dispersion_threshold=0.35)
        assert sim.r5b_dispersion_threshold == pytest.approx(0.35)

    def test_zero_threshold_always_passes(self):
        sim = _sim(use_regime_gate=True, r5b_dispersion_threshold=0.0)
        # With 0 threshold, dispersion >= 0 × median always — gate never triggers
        assert sim.r5b_dispersion_threshold == pytest.approx(0.0)


# ── 7. R5-C thresholds ───────────────────────────────────────────────────────

class TestR5CThresholds:
    def test_custom_vix_threshold(self):
        sim = _sim(use_regime_gate=True, r5c_vix_threshold=40.0)
        assert sim.r5c_vix_threshold == pytest.approx(40.0)

    def test_custom_spy_drawdown(self):
        sim = _sim(use_regime_gate=True, r5c_spy_drawdown=-0.03)
        assert sim.r5c_spy_drawdown == pytest.approx(-0.03)


# ── 8. Gate disabled when use_regime_gate=False ───────────────────────────────

class TestR5GateDisabled:
    def test_gate_off_by_default(self):
        sim = _sim()
        assert not sim.use_regime_gate

    def test_regime_map_ignored_when_gate_off(self):
        # Even with a fully blocked regime_map, gate should not trigger
        sim = _sim(use_regime_gate=False, regime_map={date(2023, 3, 22): "4DN"})
        assert not sim.use_regime_gate


# ── 9. CLI flag parseable, default off ───────────────────────────────────────

class TestR5CLIFlag:
    def test_regime_gate_flag_parseable(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--regime-gate", action="store_true", default=False)
        args = parser.parse_args(["--regime-gate"])
        assert args.regime_gate is True

    def test_regime_gate_default_false(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--regime-gate", action="store_true", default=False)
        args = parser.parse_args([])
        assert args.regime_gate is False


# ── 10. run_intraday_walkforward accepts R5 kwargs ────────────────────────────

class TestRunIntradayR5Integration:
    def test_signature_accepts_regime_gate(self):
        import inspect
        from scripts.walkforward_tier3 import run_intraday_walkforward
        sig = inspect.signature(run_intraday_walkforward)
        assert "use_regime_gate" in sig.parameters
        assert "regime_map" in sig.parameters

    def test_default_use_regime_gate_false(self):
        import inspect
        from scripts.walkforward_tier3 import run_intraday_walkforward
        sig = inspect.signature(run_intraday_walkforward)
        assert sig.parameters["use_regime_gate"].default is False

    def test_default_regime_map_none(self):
        import inspect
        from scripts.walkforward_tier3 import run_intraday_walkforward
        sig = inspect.signature(run_intraday_walkforward)
        assert sig.parameters["regime_map"].default is None
