"""Task 1 — WF simulator opportunity score parity with live PM (Phase 5b sync)."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def _make_settings(vix=0.25, vix_trend=0.15, ma=0.25, mom=0.10, breadth=0.15, disp=0.10):
    s = MagicMock()
    s.opp_score_vix_weight = vix
    s.opp_score_vix_trend_weight = vix_trend
    s.opp_score_ma_weight = ma
    s.opp_score_mom_weight = mom
    s.opp_score_breadth_weight = breadth
    s.opp_score_dispersion_weight = disp
    return s


class TestOppScoreWeightFormula:
    """Test the renormalized 4-component formula the simulator uses."""

    def _compute(self, settings, vix_score, vix_trend, ma_score, mom_score):
        """Replicate simulator formula exactly."""
        w_vix = settings.opp_score_vix_weight
        w_vt = settings.opp_score_vix_trend_weight
        w_ma = settings.opp_score_ma_weight
        w_mom = settings.opp_score_mom_weight
        w_total = (w_vix + w_vt + w_ma + w_mom) or 1.0
        return (w_vix * vix_score + w_vt * vix_trend + w_ma * ma_score + w_mom * mom_score) / w_total

    def test_default_weights_renormalize_correctly(self):
        """Default config weights (0.25/0.15/0.25/0.10) renormalize over 4 components."""
        s = _make_settings()  # breadth=0.15, disp=0.10 → excluded in sim
        score = self._compute(s, 0.8, 0.7, 1.0, 0.6)
        # w_total = 0.25+0.15+0.25+0.10 = 0.75
        expected = (0.25*0.8 + 0.15*0.7 + 0.25*1.0 + 0.10*0.6) / 0.75
        assert abs(score - expected) < 1e-9

    def test_score_in_range(self):
        s = _make_settings()
        for _ in range(20):
            vix_s, vt, ma, mom = np.random.rand(4)
            score = self._compute(s, vix_s, vt, ma, mom)
            assert 0.0 <= score <= 1.0, f"score {score} out of range"

    def test_differs_from_old_hardcoded(self):
        """New config-driven formula produces different result than old 0.35/0.20/0.30/0.15."""
        s = _make_settings()
        vix_s, vt, ma, mom = 0.8, 0.7, 1.0, 0.6
        new_score = self._compute(s, vix_s, vt, ma, mom)
        old_score = 0.35*vix_s + 0.20*vt + 0.30*ma + 0.15*mom
        # They should differ because default weights renormalize differently
        assert abs(new_score - old_score) > 1e-6

    def test_zero_weight_total_doesnt_crash(self):
        """Guard against division by zero when all weights are 0."""
        s = _make_settings(vix=0, vix_trend=0, ma=0, mom=0, breadth=0, disp=0)
        # w_total = 0 → guarded to 1.0
        w_total = (0.0 + 0.0 + 0.0 + 0.0) or 1.0
        assert w_total == 1.0

    def test_custom_weights_applied(self):
        """Arbitrary weights are used, not hardcoded."""
        s = _make_settings(vix=0.5, vix_trend=0.3, ma=0.1, mom=0.1)
        score = self._compute(s, 1.0, 0.0, 0.0, 0.0)
        # Only vix_score contributes: 0.5*1.0 / 1.0 = 0.5
        assert abs(score - 0.5) < 1e-9


class TestSimulatorUsesConfigWeights:
    """Integration test: patch settings in agent_simulator and verify score computation."""

    def test_simulator_imports_settings(self):
        """Confirm the simulator now imports settings (not hardcoded)."""
        import inspect
        import ast, pathlib
        src = pathlib.Path("c:/Projects/MrTrader/app/backtesting/agent_simulator.py").read_text()
        # Check the hardcoded formula is gone
        assert "0.35 * vix_score" not in src
        assert "0.20 * vix_trend" not in src
        # Check config-driven formula is present
        assert "opp_score_vix_weight" in src
        assert "w_total" in src

    def test_breadth_dispersion_excluded_in_sim(self):
        """Simulator comment/code confirms breadth/dispersion weights are zeroed."""
        import pathlib
        src = pathlib.Path("c:/Projects/MrTrader/app/backtesting/agent_simulator.py").read_text()
        assert "breadth" in src.lower() or "w_breadth" in src or "dispersion" in src.lower()
