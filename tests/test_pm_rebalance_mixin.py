"""Tests for Phase RA: RebalanceMixin integrated into PortfolioManager.

Verifies that:
1. PM initialises with execution_mode from settings.
2. _should_rebalance() respects the day counter.
3. Config additions are importable and have correct defaults.
4. gross_exposure_multiplier() returns correct values per regime.
"""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_execution_mode_default(self):
        from app.config import settings
        # Default is SIGNAL (non-breaking for existing runs)
        assert settings.execution_mode == "SIGNAL"

    def test_rebalance_days_default(self):
        from app.config import settings
        assert settings.rebalance_days == 20

    def test_target_positions_default(self):
        from app.config import settings
        assert settings.target_positions == 30

    def test_regime_exposure_defaults(self):
        from app.config import settings
        assert settings.regime_exposure_bull == 1.0
        assert settings.regime_exposure_neutral == 0.7
        assert settings.regime_exposure_bear == 0.3


# ---------------------------------------------------------------------------
# RebalanceMixin _should_rebalance
# ---------------------------------------------------------------------------

class TestShouldRebalance:
    def _make_mixin(self):
        from app.agents.portfolio_manager_rebalance import RebalanceMixin
        m = RebalanceMixin()
        m._init_rebalance_state()
        return m

    def test_first_call_always_rebalances(self):
        m = self._make_mixin()
        assert m._should_rebalance(date(2024, 1, 2)) is True

    def test_same_day_does_not_rebalance_twice(self):
        m = self._make_mixin()
        today = date(2024, 1, 2)
        m._last_rebalance_date = today
        assert m._should_rebalance(today) is False

    def test_rebalances_after_20_trading_days(self):
        m = self._make_mixin()
        # 20 trading days ≈ 28 calendar days
        last = date(2024, 1, 2)
        m._last_rebalance_date = last
        # 29 calendar days later → should rebalance
        assert m._should_rebalance(last + timedelta(days=29)) is True

    def test_does_not_rebalance_too_early(self):
        m = self._make_mixin()
        last = date(2024, 1, 2)
        m._last_rebalance_date = last
        # Only 10 calendar days later → ~7 trading days → too early
        assert m._should_rebalance(last + timedelta(days=10)) is False


# ---------------------------------------------------------------------------
# RegimeDetector.gross_exposure_multiplier
# ---------------------------------------------------------------------------

class TestGrossExposureMultiplier:
    def test_bull_regime(self):
        from app.strategy.regime_detector import RegimeDetector, REGIME_LOW
        rd = RegimeDetector()
        with patch.object(rd, "get_regime", return_value=REGIME_LOW):
            mult = rd.gross_exposure_multiplier()
        assert mult == 1.0

    def test_neutral_regime(self):
        from app.strategy.regime_detector import RegimeDetector, REGIME_MEDIUM
        rd = RegimeDetector()
        with patch.object(rd, "get_regime", return_value=REGIME_MEDIUM):
            mult = rd.gross_exposure_multiplier()
        assert mult == 0.7

    def test_bear_regime(self):
        from app.strategy.regime_detector import RegimeDetector, REGIME_HIGH
        rd = RegimeDetector()
        with patch.object(rd, "get_regime", return_value=REGIME_HIGH):
            mult = rd.gross_exposure_multiplier()
        assert mult == 0.3
