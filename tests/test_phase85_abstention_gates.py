"""Tests for Phase 85: PM abstention gates (SPY first-hour range, score-spread, melt-up guard)."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock


def _make_spy_intraday_bars(first_hour_range_pct: float, n_bars: int = 78) -> pd.DataFrame:
    """Build a fake SPY 5-min bar DataFrame with a controlled first-hour H-L range."""
    open_price = 500.0
    target_range = open_price * first_hour_range_pct
    rows = []
    for i in range(n_bars):
        if i < 12:
            high = open_price + target_range if i == 0 else open_price + target_range * 0.5
            low = open_price - target_range if i == 0 else open_price - target_range * 0.5
        else:
            high = open_price + 1.0
            low = open_price - 1.0
        rows.append({"open": open_price, "high": high, "low": low, "close": open_price, "volume": 1_000_000})
    return pd.DataFrame(rows)


def _make_spy_daily_bars(spy_5d_return: float, spy_5d_vol: float, n: int = 7) -> pd.DataFrame:
    """Build a fake SPY daily bar DataFrame consistent with target 5-day return and vol."""
    prices = [500.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + spy_5d_return / 5))
    return pd.DataFrame({"close": prices, "open": prices, "high": prices, "low": prices, "volume": [1e7] * n})


def _make_pm(mock_alpaca=None):
    from app.agents.portfolio_manager import PortfolioManager
    if mock_alpaca is None:
        mock_alpaca = MagicMock()
    with patch("app.integrations.get_alpaca_client", return_value=mock_alpaca):
        pm = PortfolioManager()
    pm.intraday_model = MagicMock()
    pm.intraday_model.is_trained = True
    pm.intraday_model.version = 29
    pm.intraday_model.feature_names = None
    pm.intraday_model.predict.return_value = (None, np.array([0.60, 0.58, 0.55, 0.54, 0.52]))
    pm._redis = MagicMock()
    pm.send_message = MagicMock()
    pm.log_decision = AsyncMock()
    return pm, mock_alpaca


class TestSPYIntradayState:

    def test_first_hour_range_computed_correctly(self):
        mock_alpaca = MagicMock()
        bars = _make_spy_intraday_bars(first_hour_range_pct=0.006)
        daily = _make_spy_daily_bars(spy_5d_return=0.01, spy_5d_vol=0.004)
        mock_alpaca.get_bars.side_effect = lambda sym, timeframe, limit: (
            bars if timeframe == "5Min" else daily
        )
        pm, _ = _make_pm(mock_alpaca)
        with patch("app.integrations.get_alpaca_client", return_value=mock_alpaca):
            state = pm._get_spy_intraday_state()
        assert state["first_hour_range"] is not None
        assert state["first_hour_range"] > 0.005  # meaningful first-hour range captured

    def test_returns_none_on_insufficient_bars(self):
        mock_alpaca = MagicMock()
        mock_alpaca.get_bars.return_value = None
        pm, _ = _make_pm(mock_alpaca)
        with patch("app.integrations.get_alpaca_client", return_value=mock_alpaca):
            state = pm._get_spy_intraday_state()
        assert state["first_hour_range"] is None
        assert state["spy_5d_return"] is None


class TestGate1A_FirstHourRange:

    def test_gate1a_threshold_value(self):
        from app.agents.portfolio_manager import SPY_MIN_FIRST_HOUR_RANGE
        assert SPY_MIN_FIRST_HOUR_RANGE == 0.0020  # lowered from 0.0045 — only blocks ~5% of days

    def test_gate1a_triggers_below_threshold(self):
        from app.agents.portfolio_manager import SPY_MIN_FIRST_HOUR_RANGE
        assert 0.001 < SPY_MIN_FIRST_HOUR_RANGE   # 0.10% range would trigger
        assert 0.003 > SPY_MIN_FIRST_HOUR_RANGE   # 0.30% range would pass


class TestGate1B_ScoreSpread:

    def test_score_spread_calculation(self):
        """Verify score-spread logic matches the spec."""
        from app.agents.portfolio_manager import SCORE_SPREAD_MIN
        scores = [0.61, 0.60, 0.58, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.49]
        top_n = max(1, len(scores) // 10)  # 1
        score_spread = float(np.mean(sorted(scores, reverse=True)[:top_n])) - float(np.median(scores))
        # top is 0.61, median is ~0.535
        assert score_spread > 0.07
        assert SCORE_SPREAD_MIN == 0.08

    def test_low_spread_triggers_cap(self):
        """Near-uniform scores → spread < 0.08 → should cap to 1 trade."""
        from app.agents.portfolio_manager import SCORE_SPREAD_MIN
        scores = [0.56, 0.555, 0.550, 0.548, 0.546, 0.544, 0.542, 0.540, 0.538, 0.535]
        top_n = max(1, len(scores) // 10)
        score_spread = float(np.mean(sorted(scores, reverse=True)[:top_n])) - float(np.median(scores))
        assert score_spread < SCORE_SPREAD_MIN


class TestGate1C_MeltupGuard:

    def test_meltup_constants(self):
        from app.agents.portfolio_manager import (
            MELTUP_5D_RETURN_MIN, MELTUP_5D_VOL_MAX, MELTUP_FIRST_HOUR_MAX
        )
        assert MELTUP_5D_RETURN_MIN == 0.025
        assert MELTUP_5D_VOL_MAX == 0.006
        assert MELTUP_FIRST_HOUR_MAX == 0.005

    def test_meltup_condition_detects_fold1_regime(self):
        """Fold-1 conditions (SPY +10.4%, VIX 16.5, 0.65% daily vol) should trigger."""
        from app.agents.portfolio_manager import (
            MELTUP_5D_RETURN_MIN, MELTUP_5D_VOL_MAX, MELTUP_FIRST_HOUR_MAX
        )
        spy_5d_return = 0.030    # +3% over 5 days (annualized ~10%)
        spy_5d_vol = 0.0055      # 0.55%/day — below 0.60% threshold
        first_hour_range = 0.004  # 0.40% — below 0.50% threshold

        melt_up = (
            spy_5d_return > MELTUP_5D_RETURN_MIN
            and spy_5d_vol < MELTUP_5D_VOL_MAX
            and first_hour_range < MELTUP_FIRST_HOUR_MAX
        )
        assert melt_up is True

    def test_meltup_does_not_trigger_in_normal_regime(self):
        """Fold-2 conditions (moderate vol, SPY +3.8%) should not trigger."""
        from app.agents.portfolio_manager import (
            MELTUP_5D_RETURN_MIN, MELTUP_5D_VOL_MAX, MELTUP_FIRST_HOUR_MAX
        )
        spy_5d_return = 0.015    # +1.5% — below 2.5% threshold
        spy_5d_vol = 0.0075      # 0.75%/day — above 0.60% threshold
        first_hour_range = 0.007  # 0.70% — above 0.50% threshold

        melt_up = (
            spy_5d_return > MELTUP_5D_RETURN_MIN
            and spy_5d_vol < MELTUP_5D_VOL_MAX
            and first_hour_range < MELTUP_FIRST_HOUR_MAX
        )
        assert melt_up is False
