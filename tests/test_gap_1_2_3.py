"""
Tests for Gaps 1-3 fixes:
  Gap 1 — entry_quality.check_entry_quality()
  Gap 2 — premarket_intel.is_swing_blocked() + _get_spy_intraday_drawdown()
  Gap 3 — signals.check_dynamic_adjustments()
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ─── Gap 1: Entry Quality ─────────────────────────────────────────────────────

class TestEntryQuality:
    def _bars(self, n=78, slope=0.0):
        """Generate synthetic 5-min bars."""
        closes = 100.0 + slope * np.arange(n) + np.random.default_rng(0).normal(0, 0.1, n)
        return pd.DataFrame({
            "open": closes, "high": closes * 1.001, "low": closes * 0.999,
            "close": closes, "volume": np.full(n, 100_000.0),
        })

    def test_approved_when_all_conditions_met(self):
        from app.strategy.entry_quality import check_entry_quality
        bars = self._bars(78, slope=0.01)
        result = check_entry_quality(
            symbol="TEST", signal_price=100.0, current_price=100.5,
            trade_type="swing", quote={"bid": 100.4, "ask": 100.6},
            intraday_bars=bars,
        )
        assert result.approved

    def test_rejected_on_price_run(self):
        from app.strategy.entry_quality import check_entry_quality
        # Price has run 2% above signal
        result = check_entry_quality(
            symbol="TEST", signal_price=100.0, current_price=102.1, trade_type="swing",
        )
        assert not result.approved
        assert "price_run" in result.reason

    def test_rejected_on_wide_spread(self):
        from app.strategy.entry_quality import check_entry_quality
        # Spread = 1% of mid
        result = check_entry_quality(
            symbol="TEST", signal_price=100.0, current_price=100.0, trade_type="swing",
            quote={"bid": 99.5, "ask": 100.5},
        )
        assert not result.approved
        assert "spread" in result.reason

    def test_rejected_on_negative_intraday_momentum_intraday(self):
        from app.strategy.entry_quality import check_entry_quality, INTRADAY_MOMENTUM_LOOKBACK
        # Build bars where the last LOOKBACK closes drop by >0.8% to trigger intraday threshold
        n = 78
        closes = np.ones(n) * 100.0
        # Make the final LOOKBACK bars drop 1% total (well over the -0.8% threshold)
        for i in range(INTRADAY_MOMENTUM_LOOKBACK):
            closes[n - INTRADAY_MOMENTUM_LOOKBACK + i] = 100.0 * (1 - 0.01 * i / INTRADAY_MOMENTUM_LOOKBACK)
        closes[-1] = 99.0  # final close clearly down
        bars = pd.DataFrame({
            "open": closes, "high": closes * 1.001, "low": closes * 0.999,
            "close": closes, "volume": np.full(n, 100_000.0),
        })
        result = check_entry_quality(
            symbol="TEST", signal_price=100.0, current_price=100.0,
            trade_type="intraday", intraday_bars=bars,
        )
        assert not result.approved
        assert "momentum" in result.reason

    def test_swing_more_forgiving_on_momentum(self):
        from app.strategy.entry_quality import check_entry_quality, INTRADAY_MOMENTUM_LOOKBACK
        # Slope that exceeds intraday -0.8% but stays within swing -1.5%
        n = 78
        closes = np.ones(n) * 100.0
        # Drop ~1.0% over last lookback bars — fails intraday, passes swing
        for i in range(INTRADAY_MOMENTUM_LOOKBACK):
            closes[n - INTRADAY_MOMENTUM_LOOKBACK + i] = 100.0 - 1.0 * (i / INTRADAY_MOMENTUM_LOOKBACK)
        closes[-1] = 99.1
        bars = pd.DataFrame({
            "open": closes, "high": closes * 1.001, "low": closes * 0.999,
            "close": closes, "volume": np.full(n, 100_000.0),
        })
        intraday_result = check_entry_quality(
            symbol="TEST", signal_price=100.0, current_price=100.0,
            trade_type="intraday", intraday_bars=bars,
        )
        swing_result = check_entry_quality(
            symbol="TEST", signal_price=100.0, current_price=100.0,
            trade_type="swing", intraday_bars=bars,
        )
        assert not intraday_result.approved
        assert swing_result.approved

    def test_price_run_pct_populated(self):
        from app.strategy.entry_quality import check_entry_quality
        result = check_entry_quality(
            symbol="TEST", signal_price=100.0, current_price=101.0, trade_type="swing",
        )
        assert abs(result.price_run_pct - 0.01) < 0.001

    def test_invalid_prices_rejected(self):
        from app.strategy.entry_quality import check_entry_quality
        result = check_entry_quality(
            symbol="TEST", signal_price=0.0, current_price=100.0, trade_type="swing",
        )
        assert not result.approved
        assert result.reason == "invalid_prices"


# ─── Gap 2: Macro Gate ────────────────────────────────────────────────────────

class TestPremarketMacroGate:
    def setup_method(self):
        from app.agents.premarket import PremarketIntelligence
        self.pi = PremarketIntelligence()

    def test_swing_not_blocked_by_default(self):
        # Fresh instance with no flags set — should not block
        assert not self.pi.is_swing_blocked()

    def test_swing_blocked_on_fomc(self):
        self.pi._macro_flags = {"FOMC": {"type": "FOMC"}}
        assert self.pi.is_swing_blocked()

    def test_swing_blocked_on_severe_spy_gap(self):
        self.pi._spy_premarket_pct = -0.03  # -3%
        assert self.pi.is_swing_blocked()

    def test_swing_not_blocked_on_moderate_spy_gap(self):
        self.pi._spy_premarket_pct = -0.01  # -1% — not severe
        assert not self.pi.is_swing_blocked()

    def test_intraday_blocked_on_fomc(self):
        self.pi._macro_flags = {"FOMC": {"type": "FOMC"}}
        assert self.pi.is_intraday_blocked()

    def test_intraday_blocked_on_severe_spy_gap(self):
        self.pi._spy_premarket_pct = -0.03
        assert self.pi.is_intraday_blocked()

    def test_get_market_context_keys(self):
        ctx = self.pi.get_market_context()
        assert "spy_premarket_pct" in ctx
        assert "spy_intraday_drawdown_pct" in ctx
        assert "intraday_blocked" in ctx
        assert "swing_blocked" in ctx
        assert "sizing_factor" in ctx


# ─── Gap 3: Dynamic Adjustments ──────────────────────────────────────────────

class TestDynamicAdjustments:
    def test_no_adjustment_needed_at_entry(self):
        from app.strategy.signals import check_dynamic_adjustments
        adj = check_dynamic_adjustments(
            symbol="TEST", current_price=100.0, entry_price=100.0,
            stop_price=97.0, target_price=110.0, highest_price=100.0,
            shares=100, atr=1.5, trade_type="swing", vix=15.0,
        )
        assert not adj.partial_exit
        assert not adj.stop_tightened
        assert not adj.target_extended

    def test_partial_exit_triggers_at_t1(self):
        from app.strategy.signals import check_dynamic_adjustments
        # T1 = entry + 2×ATR = 100 + 3 = 103
        adj = check_dynamic_adjustments(
            symbol="TEST", current_price=103.5, entry_price=100.0,
            stop_price=97.0, target_price=110.0, highest_price=103.5,
            shares=100, atr=1.5, trade_type="swing", vix=15.0,
        )
        assert adj.partial_exit
        assert adj.partial_exit_qty > 0
        assert "t1_hit" in adj.partial_exit_reason

    def test_stop_tightened_on_high_vix(self):
        from app.strategy.signals import check_dynamic_adjustments
        # Position profitable (up 5%), VIX = 25 → should tighten stop
        adj = check_dynamic_adjustments(
            symbol="TEST", current_price=105.0, entry_price=100.0,
            stop_price=97.0, target_price=115.0, highest_price=105.0,
            shares=100, atr=1.5, trade_type="swing", vix=25.0,
        )
        assert adj.stop_tightened
        assert adj.new_stop > 97.0  # stop moved up

    def test_stop_not_moved_down(self):
        from app.strategy.signals import check_dynamic_adjustments
        # new_stop must never be < current stop_price
        adj = check_dynamic_adjustments(
            symbol="TEST", current_price=103.0, entry_price=100.0,
            stop_price=99.0, target_price=110.0, highest_price=103.0,
            shares=100, atr=1.5, trade_type="swing", vix=25.0,
        )
        assert adj.new_stop >= 99.0

    def test_target_extended_on_strong_momentum(self):
        from app.strategy.signals import check_dynamic_adjustments
        # Within 0.5×ATR of target AND price > entry + 3×ATR
        # entry=100, ATR=2.0 → T_full=112, threshold=entry+3×ATR=106
        # current=111.5 (within 0.5×ATR of target 112), price > 106
        adj = check_dynamic_adjustments(
            symbol="TEST", current_price=111.5, entry_price=100.0,
            stop_price=97.0, target_price=112.0, highest_price=111.5,
            shares=100, atr=2.0, trade_type="swing", vix=15.0,
        )
        assert adj.target_extended
        assert adj.new_target > 112.0

    def test_no_partial_exit_with_zero_atr(self):
        from app.strategy.signals import check_dynamic_adjustments
        adj = check_dynamic_adjustments(
            symbol="TEST", current_price=110.0, entry_price=100.0,
            stop_price=97.0, target_price=115.0, highest_price=110.0,
            shares=100, atr=0.0, trade_type="swing", vix=15.0,
        )
        assert not adj.partial_exit
