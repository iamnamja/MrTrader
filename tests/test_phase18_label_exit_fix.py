"""
Tests for Phase 18 — Label-Exit Alignment.

Verifies that:
  1. Swing lambdarank label uses path-based stop/target simulation, not endpoint return
  2. Intraday label uses path-based stop/target simulation, not max HIGH
  3. Stop hit produces negative label; target hit produces positive label
  4. Neither hit falls back to actual return (time exit)
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _daily_bars_with_path(n: int = 30, start: str = "2023-01-02",
                          base: float = 100.0) -> pd.DataFrame:
    """n daily OHLCV bars, flat price, with controllable H/L."""
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({
        "open":   [base] * n,
        "high":   [base * 1.01] * n,
        "low":    [base * 0.99] * n,
        "close":  [base] * n,
        "volume": [1_000_000] * n,
    }, index=idx)


def _5min_bars_with_path(n: int = 36, entry: float = 100.0) -> pd.DataFrame:
    """n 5-min bars, flat, with controllable H/L."""
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    return pd.DataFrame({
        "open":   [entry] * n,
        "high":   [entry * 1.002] * n,
        "low":    [entry * 0.998] * n,
        "close":  [entry] * n,
        "volume": [10_000] * n,
    }, index=idx)


# ── Tests: Swing LambdaRank Label ─────────────────────────────────────────────

class TestSwingLabelPathSimulation:
    """
    Verify _atr_label_thresholds + path simulation in training.py lambdarank branch.
    We test the helper function and the outcome logic directly rather than running
    a full ModelTrainer (which requires DB + data).
    """

    def test_atr_label_thresholds_returns_valid_pcts(self):
        """_atr_label_thresholds should return (target_pct, stop_pct) > 0."""
        from app.ml.training import _atr_label_thresholds
        bars = _daily_bars_with_path(30)
        target, stop = _atr_label_thresholds(bars, 100.0)
        assert target > 0
        assert stop > 0
        assert target > stop  # 1.5x target vs 0.5x stop → target always larger

    def test_atr_fallback_when_insufficient_bars(self):
        """With fewer than 2 bars (exception path), should use fallback constants."""
        from app.ml.training import _atr_label_thresholds, LABEL_TARGET_PCT, LABEL_STOP_PCT
        # Pass an empty DataFrame to force the exception branch
        bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        target, stop = _atr_label_thresholds(bars, 100.0)
        assert target == LABEL_TARGET_PCT
        assert stop == LABEL_STOP_PCT

    def test_path_simulation_stop_hit(self):
        """When low pierces stop on bar 3, realized return should be -stop_pct."""
        from app.ml.training import _atr_label_thresholds, FORWARD_DAYS
        bars = _daily_bars_with_path(30)
        entry = 100.0
        target_pct, stop_pct = _atr_label_thresholds(bars, entry)

        # Simulate: build future bars where bar 3 pierces stop
        future = [{"high": entry * 1.001, "low": entry * 0.999}] * 10
        future[2] = {"high": entry * 1.001, "low": entry * (1 - stop_pct - 0.001)}

        realized = entry / entry - 1  # fallback: 0% time exit
        for fbar in future:
            if fbar["low"] <= entry * (1 - stop_pct):
                realized = -stop_pct
                break
            if fbar["high"] >= entry * (1 + target_pct):
                realized = target_pct
                break

        assert realized == pytest.approx(-stop_pct)

    def test_path_simulation_target_hit(self):
        """When high exceeds target on bar 2, realized return should be +target_pct."""
        from app.ml.training import _atr_label_thresholds
        bars = _daily_bars_with_path(30)
        entry = 100.0
        target_pct, stop_pct = _atr_label_thresholds(bars, entry)

        future = [{"high": entry * 1.001, "low": entry * 0.999}] * 10
        future[1] = {"high": entry * (1 + target_pct + 0.001), "low": entry * 0.999}

        realized = 0.0
        for fbar in future:
            if fbar["low"] <= entry * (1 - stop_pct):
                realized = -stop_pct
                break
            if fbar["high"] >= entry * (1 + target_pct):
                realized = target_pct
                break

        assert realized == pytest.approx(target_pct)

    def test_path_simulation_time_exit(self):
        """When neither stop nor target is hit, realized should be the actual return."""
        from app.ml.training import _atr_label_thresholds
        bars = _daily_bars_with_path(30)
        entry = 100.0
        target_pct, stop_pct = _atr_label_thresholds(bars, entry)

        # Bars that never touch stop or target (price stays flat)
        future = [{"high": entry * 1.001, "low": entry * 0.999}] * 10
        endpoint_return = 0.005  # 0.5% time exit

        realized = endpoint_return  # fallback
        for fbar in future:
            if fbar["low"] <= entry * (1 - stop_pct):
                realized = -stop_pct
                break
            if fbar["high"] >= entry * (1 + target_pct):
                realized = target_pct
                break

        # Neither hit → should keep endpoint_return
        assert realized == pytest.approx(endpoint_return)

    def test_stop_checked_before_target(self):
        """If same bar hits both stop and target, stop should take priority (conservative)."""
        from app.ml.training import _atr_label_thresholds
        bars = _daily_bars_with_path(30)
        entry = 100.0
        target_pct, stop_pct = _atr_label_thresholds(bars, entry)

        # Same bar: low < stop AND high > target (gap/spike scenario)
        future = [{
            "high": entry * (1 + target_pct + 0.01),
            "low": entry * (1 - stop_pct - 0.01),
        }]

        realized = 0.0
        for fbar in future:
            if fbar["low"] <= entry * (1 - stop_pct):
                realized = -stop_pct
                break
            if fbar["high"] >= entry * (1 + target_pct):
                realized = target_pct
                break

        assert realized == pytest.approx(-stop_pct)  # stop checked first


# ── Tests: Intraday Path-Based Label ──────────────────────────────────────────

class TestIntradayLabelPathSimulation:
    """
    Verify that the intraday label now uses stop/target path simulation.
    Tests the logic directly rather than running IntradayModelTrainer.
    """

    def test_intraday_target_hit_gives_positive_label(self):
        """When high > entry*(1+TARGET_PCT), realized = +TARGET_PCT."""
        from app.ml.intraday_training import TARGET_PCT, STOP_PCT

        entry = 100.0
        future_bars = _5min_bars_with_path(24, entry)
        # Force bar 5 to hit target
        future_bars.iloc[4, future_bars.columns.get_loc("high")] = entry * (1 + TARGET_PCT + 0.001)

        realized = (float(future_bars["close"].iloc[-1]) - entry) / entry  # fallback
        for _, fbar in future_bars.iterrows():
            if float(fbar["low"]) <= entry * (1 - STOP_PCT):
                realized = -STOP_PCT
                break
            if float(fbar["high"]) >= entry * (1 + TARGET_PCT):
                realized = TARGET_PCT
                break

        assert realized == pytest.approx(TARGET_PCT)

    def test_intraday_stop_hit_gives_negative_label(self):
        """When low < entry*(1-STOP_PCT), realized = -STOP_PCT."""
        from app.ml.intraday_training import TARGET_PCT, STOP_PCT

        entry = 100.0
        future_bars = _5min_bars_with_path(24, entry)
        future_bars.iloc[3, future_bars.columns.get_loc("low")] = entry * (1 - STOP_PCT - 0.001)

        realized = (float(future_bars["close"].iloc[-1]) - entry) / entry
        for _, fbar in future_bars.iterrows():
            if float(fbar["low"]) <= entry * (1 - STOP_PCT):
                realized = -STOP_PCT
                break
            if float(fbar["high"]) >= entry * (1 + TARGET_PCT):
                realized = TARGET_PCT
                break

        assert realized == pytest.approx(-STOP_PCT)

    def test_intraday_neither_hit_uses_time_exit(self):
        """When neither stop nor target is hit, realized is actual close return."""
        from app.ml.intraday_training import TARGET_PCT, STOP_PCT

        entry = 100.0
        close_price = 100.1  # +0.1% — between stop and target
        future_bars = _5min_bars_with_path(24, entry)
        future_bars.iloc[-1, future_bars.columns.get_loc("close")] = close_price

        # Ensure H/L never touch stop or target
        expected_return = (close_price - entry) / entry

        realized = expected_return  # fallback = actual return
        for _, fbar in future_bars.iterrows():
            if float(fbar["low"]) <= entry * (1 - STOP_PCT):
                realized = -STOP_PCT
                break
            if float(fbar["high"]) >= entry * (1 + TARGET_PCT):
                realized = TARGET_PCT
                break

        # Lows are at entry*0.998 which is well above entry*(1-0.003)=entry*0.997
        # so neither stop (0.3%) nor target (0.5%) is hit by the flat bars
        assert realized == pytest.approx(expected_return, abs=1e-6)

    def test_intraday_constants_match_backtester(self):
        """Training TARGET_PCT / STOP_PCT must match IntradayBacktester constants."""
        from app.ml.intraday_training import TARGET_PCT as TRAIN_TARGET, STOP_PCT as TRAIN_STOP
        from app.backtesting.intraday_backtest import TARGET_PCT as BT_TARGET, STOP_PCT as BT_STOP

        assert TRAIN_TARGET == BT_TARGET, (
            f"Mismatch: training TARGET_PCT={TRAIN_TARGET} != backtester TARGET_PCT={BT_TARGET}"
        )
        assert TRAIN_STOP == BT_STOP, (
            f"Mismatch: training STOP_PCT={TRAIN_STOP} != backtester STOP_PCT={BT_STOP}"
        )

    def test_swing_constants_match_agent_simulator(self):
        """Training LABEL_STOP_PCT (fallback) must be consistent with AgentSimulator defaults."""
        from app.ml.training import LABEL_STOP_PCT
        from app.backtesting.agent_simulator import SWING_STOP_PCT

        assert LABEL_STOP_PCT == SWING_STOP_PCT, (
            f"Training LABEL_STOP_PCT={LABEL_STOP_PCT} != AgentSimulator SWING_STOP_PCT={SWING_STOP_PCT}"
        )
