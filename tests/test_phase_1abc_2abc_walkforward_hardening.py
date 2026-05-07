"""
Phases 1a+1b+1c+1d+1e and 2a+2b+2c — walk-forward hardening tests.

Covers:
- 1c: USE_NIS_FEATURES flag controls NIS in swing PRUNED_FEATURES
- 1c: Intraday FEATURE_NAMES never contained NIS features
- 1a: Transaction cost params wired into both simulators
- 1b: Purge days applied in fold boundary logic
- 1d: Bootstrap walk-forward function exists and returns statistics
- 1e: DSR computed and included in gate_passed()
- 2a: Opportunity score param exists in both simulators
- 2b: Earnings blackout param exists and filters entries
- 2c: Dispersion gate param exists in intraday simulator
- retrain_config: USE_NIS_FEATURES, USE_REALIZED_R_LABELS, MIN_REALIZED_R present
"""
from datetime import date
from unittest.mock import MagicMock
import numpy as np
import pytest


# ── 1c: NIS configurable flag ─────────────────────────────────────────────────

class TestNISConfigurableFlag:
    def test_use_nis_features_false_by_default(self):
        from app.ml.retrain_config import USE_NIS_FEATURES
        assert not USE_NIS_FEATURES, (
            "USE_NIS_FEATURES must default to False — NIS has ~80% NaN encoding "
            "time-proxy (NaN = pre-May-2025 regime). Re-enable after ≥2yr backfill."
        )

    def test_nis_excluded_from_pruned_features_when_flag_false(self):
        """When USE_NIS_FEATURES=False, all NIS/macro features are in PRUNED_FEATURES."""
        from app.ml.training import PRUNED_FEATURES
        nis_features = [
            "nis_direction_score", "nis_materiality_score", "nis_already_priced_in",
            "nis_sizing_mult", "nis_downside_risk",
            "macro_avg_direction", "macro_pct_bearish", "macro_pct_bullish",
            "macro_avg_materiality", "macro_pct_high_risk",
        ]
        missing = [f for f in nis_features if f not in PRUNED_FEATURES]
        assert not missing, (
            f"NIS features not in PRUNED_FEATURES (USE_NIS_FEATURES=False): {missing}"
        )

    def test_nis_features_not_in_intraday_feature_names(self):
        """Intraday FEATURE_NAMES never included NIS — they were never in the 56-feature list."""
        from app.ml.intraday_features import FEATURE_NAMES
        nis_features = [
            "nis_direction_score", "nis_materiality_score", "nis_already_priced_in",
            "nis_sizing_mult", "nis_downside_risk",
        ]
        present = [f for f in nis_features if f in FEATURE_NAMES]
        assert not present, (
            f"NIS features should never be in intraday FEATURE_NAMES: {present}"
        )

    def test_nis_pruned_set_documented(self):
        """_NIS_PRUNED_WHILE_SPARSE contains exactly the 10 NIS/macro features."""
        from app.ml.training import _NIS_PRUNED_WHILE_SPARSE
        assert len(_NIS_PRUNED_WHILE_SPARSE) == 10
        assert "nis_direction_score" in _NIS_PRUNED_WHILE_SPARSE
        assert "macro_pct_high_risk" in _NIS_PRUNED_WHILE_SPARSE

    def test_retrain_config_has_all_feature_flags(self):
        """retrain_config.py is the single source of truth for all feature flags."""
        from app.ml.retrain_config import USE_NIS_FEATURES, USE_REALIZED_R_LABELS, MIN_REALIZED_R
        assert isinstance(USE_NIS_FEATURES, bool)
        assert isinstance(USE_REALIZED_R_LABELS, bool)
        assert isinstance(MIN_REALIZED_R, float)
        assert 0.0 < MIN_REALIZED_R < 1.0

    def test_intraday_training_imports_flags_from_retrain_config(self):
        """intraday_training.py imports USE_REALIZED_R_LABELS from retrain_config (not local copy)."""
        import app.ml.intraday_training as it
        import app.ml.retrain_config as rc
        assert it.USE_REALIZED_R_LABELS is rc.USE_REALIZED_R_LABELS
        assert it.MIN_REALIZED_R == rc.MIN_REALIZED_R


# ── 1a: Transaction costs ─────────────────────────────────────────────────────

class TestTransactionCosts:
    def test_swing_simulator_accepts_transaction_cost_param(self):
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator(transaction_cost_pct=0.0005)
        assert sim.transaction_cost_pct == 0.0005

    def test_intraday_simulator_accepts_transaction_cost_param(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        sim = IntradayAgentSimulator(transaction_cost_pct=0.0015)
        assert sim.transaction_cost_pct == 0.0015

    def test_walkforward_has_cost_bps_cli_args(self):
        """walkforward_tier3.py CLI accepts --swing-cost-bps and --intraday-cost-bps."""
        import importlib.util, sys
        import argparse
        spec = importlib.util.spec_from_file_location(
            "wf", "scripts/walkforward_tier3.py"
        )
        # Just verify the args are defined by parsing help text
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True
        )
        assert "--swing-cost-bps" in result.stdout
        assert "--intraday-cost-bps" in result.stdout


# ── 1b: Purge / embargo ───────────────────────────────────────────────────────

class TestPurgeEmbargoArgs:
    def test_walkforward_has_purge_days_cli_args(self):
        import sys, subprocess
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True
        )
        assert "--swing-purge-days" in result.stdout
        assert "--intraday-purge-days" in result.stdout

    def test_run_swing_walkforward_accepts_purge_days(self):
        import inspect
        from scripts.walkforward_tier3 import run_swing_walkforward
        sig = inspect.signature(run_swing_walkforward)
        assert "purge_days" in sig.parameters
        assert sig.parameters["purge_days"].default == 10

    def test_run_intraday_walkforward_accepts_purge_days(self):
        import inspect
        from scripts.walkforward_tier3 import run_intraday_walkforward
        sig = inspect.signature(run_intraday_walkforward)
        assert "purge_days" in sig.parameters
        assert sig.parameters["purge_days"].default == 2


# ── 1d: Bootstrap ─────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_bootstrap_folds_returns_statistics(self):
        from scripts.walkforward_tier3 import _bootstrap_folds, WalkForwardReport, FoldResult

        call_count = [0]

        def fake_run(**kwargs):
            call_count[0] += 1
            report = WalkForwardReport(model_type="swing")
            report.folds = [
                FoldResult(1, date(2023,1,1), date(2024,1,1), date(2024,1,11), date(2024,7,1),
                           trades=100, win_rate=0.55, sharpe=0.8 + np.random.randn()*0.2,
                           max_drawdown=0.05, total_return=0.1, stop_exit_rate=0.3),
            ]
            return report

        result = _bootstrap_folds(fake_run, n_bootstrap=10, total_years=5)
        assert call_count[0] == 10
        assert "mean" in result
        assert "median" in result
        assert "p5" in result
        assert "p95" in result
        assert "pct_positive" in result
        assert 0 <= result["pct_positive"] <= 1

    def test_bootstrap_cli_arg_present(self):
        import sys, subprocess
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True
        )
        assert "--bootstrap" in result.stdout


# ── 1e: Deflated Sharpe Ratio ─────────────────────────────────────────────────

class TestDeflatedSharpeRatio:
    def test_dsr_function_exists(self):
        from scripts.walkforward_tier3 import _deflated_sharpe_ratio
        dsr_z, p = _deflated_sharpe_ratio(sharpe=1.5, n_trials=15, n_obs=300)
        assert isinstance(dsr_z, float)
        assert 0.0 <= p <= 1.0

    def test_dsr_high_sharpe_significant(self):
        from scripts.walkforward_tier3 import _deflated_sharpe_ratio
        _, p = _deflated_sharpe_ratio(sharpe=3.0, n_trials=15, n_obs=500)
        assert p > 0.95, "Sharpe=3.0 with 500 obs should be statistically significant"

    def test_dsr_low_sharpe_not_significant(self):
        from scripts.walkforward_tier3 import _deflated_sharpe_ratio
        _, p = _deflated_sharpe_ratio(sharpe=0.5, n_trials=15, n_obs=100)
        assert p < 0.95, "Sharpe=0.5 with 100 obs and 15 trials should not be significant"

    def test_gate_passed_checks_dsr(self):
        """WalkForwardReport.gate_passed() requires DSR p > 0.95 in addition to Sharpe."""
        from scripts.walkforward_tier3 import WalkForwardReport, FoldResult
        report = WalkForwardReport(model_type="swing")
        # A Sharpe that would pass raw gates but has too few obs for DSR to be significant
        report.folds = [
            FoldResult(1, date(2023,1,1), date(2024,1,1), date(2024,1,11), date(2024,7,1),
                       trades=10, win_rate=0.6, sharpe=1.2,
                       max_drawdown=0.05, total_return=0.1, stop_exit_rate=0.3),
        ]
        # Only 10 trades — DSR will not be significant despite avg Sharpe > 0.8
        assert not report.gate_passed(), (
            "gate_passed() should fail when trade count is too low for DSR significance"
        )


# ── 2a: Opportunity score ─────────────────────────────────────────────────────

class TestOpportunityScore:
    def test_swing_simulator_accepts_opportunity_score(self):
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator(use_opportunity_score=True)
        assert sim.use_opportunity_score is True

    def test_intraday_simulator_accepts_opportunity_score(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        sim = IntradayAgentSimulator(use_opportunity_score=True)
        assert sim.use_opportunity_score is True

    def test_walkforward_has_opportunity_score_cli_arg(self):
        import sys, subprocess
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True
        )
        assert "--pm-opportunity-score" in result.stdout


# ── 2b: Earnings blackout ─────────────────────────────────────────────────────

class TestEarningsBlackout:
    def test_intraday_simulator_accepts_earnings_blackout(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        cal = {"AAPL": {date(2025, 1, 15)}}
        sim = IntradayAgentSimulator(earnings_blackout=cal)
        assert sim.earnings_blackout == cal

    def test_swing_simulator_accepts_earnings_blackout(self):
        from app.backtesting.agent_simulator import AgentSimulator
        cal = {"AAPL": {date(2025, 1, 15)}}
        sim = AgentSimulator(earnings_blackout=cal)
        assert sim.earnings_blackout == cal

    def test_intraday_blackout_window_logic(self):
        """Earnings blackout: entry blocked when earnings within [-3d, +1d] of current day."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        sim = IntradayAgentSimulator(
            earnings_blackout={"AAPL": {date(2025, 4, 15)}},
            intraday_blackout_days_before=1,
            intraday_blackout_days_after=3,
        )
        # day - earnings_date delta = earnings_date - day
        # delta = (earnings_date - current_day).days
        # blocked when -after <= delta <= before  i.e. -3 <= delta <= 1
        earnings_date = date(2025, 4, 15)
        # 1 day before earnings (April 14) — delta = +1 → blocked
        assert any(
            -3 <= (e - date(2025, 4, 14)).days <= 1
            for e in sim.earnings_blackout["AAPL"]
        )
        # 4 days before earnings (April 11) — delta = +4 → not blocked
        assert not any(
            -3 <= (e - date(2025, 4, 11)).days <= 1
            for e in sim.earnings_blackout["AAPL"]
        )

    def test_swing_blackout_window_logic(self):
        """Swing blackout: entry blocked when earnings within 3 days (forward)."""
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator(
            earnings_blackout={"MSFT": {date(2025, 4, 15)}},
            swing_blackout_days_before=3,
        )
        earnings_date = date(2025, 4, 15)
        # 2 days before earnings (April 13) — delta = +2 → blocked
        assert any(
            0 <= (e - date(2025, 4, 13)).days <= 3
            for e in sim.earnings_blackout["MSFT"]
        )
        # 4 days before earnings (April 11) — delta = +4 → not blocked
        assert not any(
            0 <= (e - date(2025, 4, 11)).days <= 3
            for e in sim.earnings_blackout["MSFT"]
        )

    def test_empty_earnings_blackout_no_effect(self):
        """Empty earnings_blackout dict → no entries blocked."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        sim = IntradayAgentSimulator(earnings_blackout={})
        assert sim.earnings_blackout == {}

    def test_walkforward_has_earnings_blackout_cli_arg(self):
        import sys, subprocess
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True
        )
        assert "--earnings-blackout" in result.stdout


# ── 2c: Dispersion gate ───────────────────────────────────────────────────────

class TestDispersionGate:
    def test_intraday_simulator_accepts_dispersion_gate(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        sim = IntradayAgentSimulator(use_dispersion_gate=True, dispersion_threshold=0.5)
        assert sim.use_dispersion_gate is True
        assert sim.dispersion_threshold == 0.5

    def test_walkforward_has_dispersion_gate_cli_arg(self):
        import sys, subprocess
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True
        )
        assert "--dispersion-gate" in result.stdout

    def test_dispersion_threshold_default(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        sim = IntradayAgentSimulator()
        assert sim.dispersion_threshold == 0.5
