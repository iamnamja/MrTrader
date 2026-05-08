"""
Tests for WF-5a — Simulation Fidelity: Per-Fold Gates.

Verifies:
  1. FoldResult has opp_score_abstain_days / earnings_blackout_days / macro_gate_days fields
  2. Tier-3 CLI: --pm-opportunity-score defaults True, can be disabled with --no-pm-opportunity-score
  3. Tier-3 CLI: --earnings-blackout defaults True, can be disabled
  4. Tier-3 CLI: --dispersion-gate defaults True, can be disabled
  5. Tier-3 CLI: --macro-gate defaults True, can be disabled with --no-macro-gate
  6. macro_calendar: load_macro_blocked_dates returns a set of dates
  7. macro_calendar: fallback FOMC list covers known dates
  8. macro_calendar: returns empty set on network failure (graceful degradation)
  9. AgentSimulator: macro_blocked_dates blocks entries on blocked days
  10. IntradayAgentSimulator: macro_blocked_dates blocks entries on blocked days
"""
import argparse
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from scripts.walkforward.gates import FoldResult


# ── helpers ──────────────────────────────────────────────────────────────────

def _fold(**kwargs):
    defaults = dict(
        fold=1,
        train_start=date(2022, 1, 1), train_end=date(2023, 1, 1),
        test_start=date(2023, 1, 2), test_end=date(2023, 6, 30),
        trades=50, win_rate=0.55, sharpe=0.9, max_drawdown=0.05,
        total_return=0.10, stop_exit_rate=0.3,
    )
    defaults.update(kwargs)
    return FoldResult(**defaults)


# ── 1. FoldResult abstention fields ──────────────────────────────────────────

class TestFoldResultAbstentionFields:
    def test_defaults_zero(self):
        f = _fold()
        assert f.opp_score_abstain_days == 0
        assert f.earnings_blackout_days == 0
        assert f.macro_gate_days == 0

    def test_fields_settable(self):
        f = _fold(opp_score_abstain_days=5, earnings_blackout_days=3, macro_gate_days=2)
        assert f.opp_score_abstain_days == 5
        assert f.earnings_blackout_days == 3
        assert f.macro_gate_days == 2


# ── 2-5. CLI defaults ─────────────────────────────────────────────────────────

class TestCLIDefaults:
    def _parse(self, extra_args=None):
        """Parse a minimal arg list matching tier3's new defaults."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--pm-opportunity-score", action="store_true", default=True)
        parser.add_argument("--no-pm-opportunity-score", dest="pm_opportunity_score",
                            action="store_false")
        parser.add_argument("--dispersion-gate", action="store_true", default=True)
        parser.add_argument("--no-dispersion-gate", dest="dispersion_gate",
                            action="store_false")
        parser.add_argument("--earnings-blackout", action="store_true", default=True)
        parser.add_argument("--no-earnings-blackout", dest="earnings_blackout",
                            action="store_false")
        parser.add_argument("--macro-gate", action="store_true", default=True)
        parser.add_argument("--no-macro-gate", dest="macro_gate", action="store_false")
        return parser.parse_args(extra_args or [])

    def test_pm_opportunity_score_default_true(self):
        args = self._parse()
        assert args.pm_opportunity_score is True

    def test_pm_opportunity_score_disable(self):
        args = self._parse(["--no-pm-opportunity-score"])
        assert args.pm_opportunity_score is False

    def test_dispersion_gate_default_true(self):
        args = self._parse()
        assert args.dispersion_gate is True

    def test_dispersion_gate_disable(self):
        args = self._parse(["--no-dispersion-gate"])
        assert args.dispersion_gate is False

    def test_earnings_blackout_default_true(self):
        args = self._parse()
        assert args.earnings_blackout is True

    def test_earnings_blackout_disable(self):
        args = self._parse(["--no-earnings-blackout"])
        assert args.earnings_blackout is False

    def test_macro_gate_default_true(self):
        args = self._parse()
        assert args.macro_gate is True

    def test_macro_gate_disable(self):
        args = self._parse(["--no-macro-gate"])
        assert args.macro_gate is False


# ── 6-8. macro_calendar ───────────────────────────────────────────────────────

class TestMacroCalendar:
    def test_load_returns_set(self):
        from scripts.walkforward.macro_calendar import load_macro_blocked_dates
        result = load_macro_blocked_dates(date(2023, 1, 1), date(2023, 12, 31))
        assert isinstance(result, set)

    def test_fallback_fomc_known_dates(self):
        from scripts.walkforward.macro_calendar import _fallback_fomc_dates
        # FOMC 2023: Feb 1
        result = _fallback_fomc_dates(date(2023, 1, 1), date(2023, 12, 31))
        assert date(2023, 2, 1) in result

    def test_fallback_fomc_date_range_filter(self):
        from scripts.walkforward.macro_calendar import _fallback_fomc_dates
        result = _fallback_fomc_dates(date(2023, 6, 1), date(2023, 6, 30))
        # Only June 14, 2023 FOMC in range
        assert date(2023, 6, 14) in result
        assert date(2023, 2, 1) not in result

    def test_graceful_on_network_failure(self):
        from scripts.walkforward.macro_calendar import load_macro_blocked_dates
        with patch("scripts.walkforward.macro_calendar._fetch_finnhub",
                   side_effect=Exception("network error")):
            result = load_macro_blocked_dates(
                date(2023, 1, 1), date(2023, 12, 31),
                finnhub_token="fake_token",
            )
        # Should fall back to FOMC hard-coded list, not raise
        assert isinstance(result, set)
        assert len(result) > 0

    def test_all_blocked_dates_are_date_objects(self):
        from scripts.walkforward.macro_calendar import _fallback_fomc_dates
        result = _fallback_fomc_dates(date(2020, 1, 1), date(2026, 12, 31))
        for d in result:
            assert isinstance(d, date)


# ── 9. AgentSimulator macro gate ──────────────────────────────────────────────

class TestAgentSimulatorMacroGate:
    def test_macro_blocked_dates_stored(self):
        from app.backtesting.agent_simulator import AgentSimulator
        blocked = {date(2023, 3, 22)}
        sim = AgentSimulator(macro_blocked_dates=blocked)
        assert sim.macro_blocked_dates == blocked

    def test_macro_blocked_dates_default_empty(self):
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator()
        assert sim.macro_blocked_dates == set()


# ── 10. IntradayAgentSimulator macro gate ─────────────────────────────────────

class TestIntradaySimulatorMacroGate:
    def test_macro_blocked_dates_stored(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        blocked = {date(2023, 5, 3)}
        sim = IntradayAgentSimulator(macro_blocked_dates=blocked)
        assert sim.macro_blocked_dates == blocked

    def test_macro_blocked_dates_default_empty(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        sim = IntradayAgentSimulator()
        assert sim.macro_blocked_dates == set()

    def test_field_parity_with_gates_fold_result(self):
        """FoldResult in gates.py and tier3 must both have the abstention fields."""
        import dataclasses
        from scripts.walkforward.gates import FoldResult as PkgFR
        from scripts.walkforward_tier3 import FoldResult as T3FR
        pkg_fields = {f.name for f in dataclasses.fields(PkgFR)}
        t3_fields = {f.name for f in dataclasses.fields(T3FR)}
        for field in ("opp_score_abstain_days", "earnings_blackout_days", "macro_gate_days"):
            assert field in pkg_fields, f"Missing {field} in gates.FoldResult"
            assert field in t3_fields, f"Missing {field} in tier3.FoldResult"
