"""
Regression test: LambdaRank models must be gated in rebalance mode, not scan mode.

Root cause of v216/v218 misdiagnosis: run_v201_lambdarank_plus.py was calling
run_swing_walkforward() without rebalance_mode=True. LambdaRank outputs are
relative ranks within a query group, not calibrated trade-success probabilities.
Scan-mode gate on a ranker = noise. This test prevents recurrence.
"""
import ast
import textwrap
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "run_v201_lambdarank_plus.py"


def _get_run_gate_source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


class TestLambdarankGateMode:
    def test_gate_uses_rebalance_mode(self):
        """_run_gate must pass rebalance_mode=True when model_type is lambdarank."""
        src = _get_run_gate_source()
        assert "rebalance_mode=use_rebalance" in src or "rebalance_mode=True" in src, (
            "run_v201_lambdarank_plus._run_gate must enable rebalance_mode for LambdaRank. "
            "Scan-mode gate on a ranker is invalid (v218 post-mortem, 2026-05-25)."
        )

    def test_gate_disables_opportunity_score(self):
        """use_opportunity_score must be False for LambdaRank gates."""
        src = _get_run_gate_source()
        assert "use_opportunity_score=False" in src, (
            "LambdaRank gate must set use_opportunity_score=False. "
            "Ranker margins are not calibrated probabilities."
        )

    def test_gate_enables_regime_gate(self):
        """rebalance_regime_gate must be True in the gate call."""
        src = _get_run_gate_source()
        assert "rebalance_regime_gate=use_rebalance" in src or "rebalance_regime_gate=True" in src, (
            "Gate must enable regime filtering (rebalance_regime_gate=True)."
        )

    def test_gate_only_flag_exists(self):
        """--gate-only flag must exist so we can re-gate without retraining."""
        src = _get_run_gate_source()
        assert "--gate-only" in src, (
            "Script must support --gate-only <version> to re-run gate without retraining."
        )
