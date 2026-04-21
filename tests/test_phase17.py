"""
Unit tests for Phase 17: Core Workflow Overhaul.

Covers:
- Conviction-based position sizing (position_sizer.py)
- Strategy-level circuit breakers (circuit_breaker.py)
- check_exit dynamic max_hold (signals.py)
- days_until_earnings helper (earnings_filter.py)
- Agent config new keys (agent_config.py)

All tests are pure-Python — no database, Redis, or Alpaca connections.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import date, timedelta


# ─── Position Sizer: conviction multiplier ───────────────────────────────────

class TestConvictionMultiplier:
    from app.strategy.position_sizer import conviction_multiplier

    def test_low_score_returns_floor(self):
        from app.strategy.position_sizer import conviction_multiplier
        assert conviction_multiplier(0.50) == 0.75
        assert conviction_multiplier(0.55) == 0.75

    def test_high_score_returns_ceiling(self):
        from app.strategy.position_sizer import conviction_multiplier
        assert conviction_multiplier(0.75) == 1.25
        assert conviction_multiplier(0.90) == 1.25

    def test_mid_score_is_interpolated(self):
        from app.strategy.position_sizer import conviction_multiplier
        mid = conviction_multiplier(0.65)
        assert 0.75 < mid < 1.25

    def test_exactly_midpoint(self):
        from app.strategy.position_sizer import conviction_multiplier
        val = conviction_multiplier(0.65)
        assert abs(val - 1.00) < 0.01  # midpoint between 0.55 and 0.75 → 1.0×


class TestSizePositionWithConviction:
    def test_no_ml_score_unchanged(self):
        from app.strategy.position_sizer import size_position
        baseline = size_position(100_000, 50_000, 100.0, 97.0, risk_fraction=0.02)
        same = size_position(100_000, 50_000, 100.0, 97.0, risk_fraction=0.02, ml_score=0.0)
        assert baseline == same

    def test_high_conviction_gives_more_shares(self):
        from app.strategy.position_sizer import size_position
        # Use large cash so the cash cap doesn't bind (conviction multiplier is the constraint)
        low = size_position(100_000, 500_000, 10.0, 9.7, ml_score=0.55)
        high = size_position(100_000, 500_000, 10.0, 9.7, ml_score=0.80)
        assert high > low

    def test_conviction_multiplier_applied(self):
        from app.strategy.position_sizer import size_position, conviction_multiplier, RISK_FRACTION
        # Large cash so cash cap doesn't bind
        equity, cash, entry, stop = 100_000.0, 500_000.0, 10.0, 9.7
        ml_score = 0.75

        risk_per_share = entry - stop
        expected_risk = equity * RISK_FRACTION * conviction_multiplier(ml_score)
        expected_shares = int(expected_risk / risk_per_share)

        result = size_position(equity, cash, entry, stop, ml_score=ml_score)
        assert result == expected_shares

    def test_cash_cap_still_respected(self):
        from app.strategy.position_sizer import size_position
        # Very small cash but high conviction — should still be capped
        result = size_position(100_000, 100.0, 50.0, 48.0, ml_score=0.80)
        assert result <= int(100.0 * 0.90 / 50.0)

    def test_invalid_inputs_return_zero(self):
        from app.strategy.position_sizer import size_position
        assert size_position(0, 50_000, 100.0, 97.0, ml_score=0.70) == 0
        assert size_position(100_000, 50_000, 0.0, 97.0, ml_score=0.70) == 0
        assert size_position(100_000, 50_000, 97.0, 100.0, ml_score=0.70) == 0  # stop > entry


# ─── Circuit Breaker: strategy-level ─────────────────────────────────────────

class TestStrategyCircuitBreaker:
    def setup_method(self):
        from app.agents.circuit_breaker import CircuitBreaker
        self.cb = CircuitBreaker()

    def test_strategy_not_paused_initially(self):
        assert not self.cb.is_strategy_paused("swing")
        assert not self.cb.is_strategy_paused("intraday")

    def test_manual_pause_and_resume(self):
        self.cb.pause_strategy("intraday", "testing")
        assert self.cb.is_strategy_paused("intraday")
        assert not self.cb.is_strategy_paused("swing")

        self.cb.resume_strategy("intraday")
        assert not self.cb.is_strategy_paused("intraday")

    def test_auto_pause_on_low_win_rate(self):
        # Record enough losses to trigger auto-pause
        for _ in range(20):
            self.cb.record_trade_result(won=False, strategy="intraday")
        assert self.cb.is_strategy_paused("intraday")
        assert not self.cb.is_strategy_paused("swing")  # global unaffected

    def test_no_pause_with_high_win_rate(self):
        for i in range(20):
            self.cb.record_trade_result(won=(i % 2 == 0), strategy="swing")  # 50% win rate
        assert not self.cb.is_strategy_paused("swing")

    def test_global_breaker_independent_of_strategy(self):
        # Trip global via consecutive losses on swing
        for _ in range(3):
            self.cb.record_trade_result(won=False, strategy="swing")
        assert self.cb.is_open  # global tripped
        # But intraday strategy itself not auto-paused (not enough history)
        assert not self.cb.is_strategy_paused("intraday")

    def test_strategy_win_rate_none_below_threshold(self):
        # Not enough data → win rate is None → no auto-pause
        for _ in range(4):  # less than 5 trades
            self.cb.record_trade_result(won=False, strategy="swing")
        assert not self.cb.is_strategy_paused("swing")

    def test_status_includes_strategy_info(self):
        status = self.cb.status()
        assert "strategies" in status
        assert "swing" in status["strategies"]
        assert "intraday" in status["strategies"]
        for s in status["strategies"].values():
            assert "paused" in s
            assert "paused_reason" in s

    def test_resume_clears_history(self):
        for _ in range(20):
            self.cb.record_trade_result(won=False, strategy="swing")
        assert self.cb.is_strategy_paused("swing")
        self.cb.resume_strategy("swing")
        assert not self.cb.is_strategy_paused("swing")
        # After resume, history cleared so no immediate re-pause on one loss
        self.cb.record_trade_result(won=False, strategy="swing")
        assert not self.cb.is_strategy_paused("swing")


# ─── signals.check_exit: dynamic max_hold ────────────────────────────────────

class TestCheckExitMaxHold:
    def test_default_max_hold_is_20(self):
        from app.strategy.signals import check_exit
        # bars_held=10 should NOT trigger with default max_hold=20
        should_exit, reason, _ = check_exit(
            symbol="TEST",
            current_price=100.0,
            entry_price=100.0,
            stop_price=95.0,
            target_price=120.0,
            highest_price=100.0,
            bars_held=10,
        )
        assert not should_exit

    def test_max_hold_20_triggers_at_20(self):
        from app.strategy.signals import check_exit
        should_exit, reason, _ = check_exit(
            symbol="TEST",
            current_price=100.0,
            entry_price=100.0,
            stop_price=95.0,
            target_price=120.0,
            highest_price=100.0,
            bars_held=20,
        )
        assert should_exit
        assert "max_hold_20d" in reason

    def test_custom_max_hold_respected(self):
        from app.strategy.signals import check_exit
        should_exit, reason, _ = check_exit(
            symbol="TEST",
            current_price=100.0,
            entry_price=100.0,
            stop_price=95.0,
            target_price=120.0,
            highest_price=100.0,
            bars_held=10,
            max_hold_bars=10,
        )
        assert should_exit
        assert "max_hold_10d" in reason

    def test_min_hold_still_blocks_early_exit(self):
        from app.strategy.signals import check_exit
        # Price below stop but within min_hold_bars — should not exit
        should_exit, _, _ = check_exit(
            symbol="TEST",
            current_price=94.0,  # below stop of 95
            entry_price=100.0,
            stop_price=95.0,
            target_price=120.0,
            highest_price=100.0,
            bars_held=2,
            min_hold_bars=3,
        )
        assert not should_exit


# ─── earnings_filter: days_until_earnings ────────────────────────────────────

class TestDaysUntilEarnings:
    def test_returns_correct_days(self):
        from app.strategy.earnings_filter import days_until_earnings, earnings_filter
        future_date = date.today() + timedelta(days=5)
        earnings_filter._cache["AAPL"] = (future_date, float("inf"))
        result = days_until_earnings("AAPL")
        assert result == 5

    def test_returns_none_when_no_date(self):
        from app.strategy.earnings_filter import days_until_earnings, earnings_filter
        earnings_filter._cache["NODATE"] = (None, float("inf"))
        result = days_until_earnings("NODATE")
        assert result is None

    def test_past_earnings_returns_negative(self):
        from app.strategy.earnings_filter import days_until_earnings, earnings_filter
        past_date = date.today() - timedelta(days=3)
        earnings_filter._cache["PAST"] = (past_date, float("inf"))
        result = days_until_earnings("PAST")
        assert result == -3


# ─── agent_config: new keys have correct defaults ────────────────────────────

class TestAgentConfigNewKeys:
    def test_new_keys_have_defaults(self):
        from app.database.agent_config import _DEFAULTS
        assert "pm.exit_threshold" in _DEFAULTS
        assert "pm.top_n_stocks" in _DEFAULTS
        assert "strategy.partial_exit_pct" in _DEFAULTS
        assert "strategy.max_hold_bars" in _DEFAULTS

    def test_default_values_are_sensible(self):
        from app.database.agent_config import _DEFAULTS
        assert _DEFAULTS["pm.exit_threshold"] == 0.35
        assert _DEFAULTS["pm.top_n_stocks"] == 10
        assert _DEFAULTS["strategy.partial_exit_pct"] == 0.50
        assert _DEFAULTS["strategy.max_hold_bars"] == 20
