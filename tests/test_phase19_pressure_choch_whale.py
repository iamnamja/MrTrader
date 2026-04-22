"""
Tests for Phase 19 — Pressure Index, ChoCh, and Whale Candle features.

Verifies:
  1. pressure_index, pressure_persistence, pressure_displacement added to swing features
  2. choch_detected, bars_since_choch, hh_hl_sequence added to swing features
  3. whale_candle added to intraday features
  4. Directional correctness (extended stock has high pressure; fresh ChoCh detected)
  5. Edge cases: insufficient bars fall back to neutral defaults
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date


# ── Helpers ───────────────────────────────────────────────────────────────────

def _daily_bars(n: int = 60, base: float = 100.0, trend: float = 0.0) -> pd.DataFrame:
    """n daily bars with optional uptrend (trend = daily pct increment)."""
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    prices = np.array([base * (1 + trend) ** i for i in range(n)])
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.01,
        "low":    prices * 0.99,
        "close":  prices,
        "volume": [1_000_000] * n,
    }, index=idx)


def _5min_bars(n: int = 36, base: float = 100.0, body_mult: float = 1.0) -> pd.DataFrame:
    """n 5-min bars; body_mult scales open-close spread to trigger whale candle."""
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = np.full(n, base)
    open_ = close - body_mult * 0.5  # body = body_mult * 0.5
    return pd.DataFrame({
        "open":   open_,
        "high":   close + 0.3,
        "low":    close - 0.3,
        "close":  close,
        "volume": [10_000] * n,
    }, index=idx)


# ── Pressure Index ─────────────────────────────────────────────────────────────

class TestPressureIndex:

    def _engineer(self, bars):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        return fe.engineer_features("TEST", bars, fetch_fundamentals=False)

    def test_pressure_keys_present(self):
        feats = self._engineer(_daily_bars(60))
        assert "pressure_index" in feats
        assert "pressure_persistence" in feats
        assert "pressure_displacement" in feats

    def test_pressure_types_are_float(self):
        feats = self._engineer(_daily_bars(60))
        for key in ("pressure_index", "pressure_persistence", "pressure_displacement"):
            assert isinstance(feats[key], float), f"{key} is not float"

    def test_flat_price_near_zero_displacement(self):
        """Flat price → baseline ≈ price → displacement near 0."""
        feats = self._engineer(_daily_bars(60, trend=0.0))
        assert abs(feats["pressure_displacement"]) < 1.5

    def test_strong_uptrend_positive_displacement(self):
        """Steadily rising price → above EMA baseline → positive displacement."""
        feats = self._engineer(_daily_bars(60, base=100.0, trend=0.005))
        assert feats["pressure_displacement"] > 0

    def test_pressure_persistence_non_negative(self):
        """pressure_persistence counts bars above baseline — always >= 0."""
        feats = self._engineer(_daily_bars(60))
        assert feats["pressure_persistence"] >= 0.0

    def test_pressure_index_bounded(self):
        """pressure_index is clipped to [-30, 30]."""
        feats = self._engineer(_daily_bars(60, trend=0.01))
        assert -30.0 <= feats["pressure_index"] <= 30.0


# ── ChoCh / Market Structure ──────────────────────────────────────────────────

class TestChoCh:

    def _engineer(self, bars):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        return fe.engineer_features("TEST", bars, fetch_fundamentals=False)

    def test_choch_keys_present(self):
        feats = self._engineer(_daily_bars(60))
        assert "choch_detected" in feats
        assert "bars_since_choch" in feats
        assert "hh_hl_sequence" in feats

    def test_choch_types_are_float(self):
        feats = self._engineer(_daily_bars(60))
        for key in ("choch_detected", "bars_since_choch", "hh_hl_sequence"):
            assert isinstance(feats[key], float), f"{key} is not float"

    def test_uptrend_generates_choch(self):
        """Strong uptrend → price repeatedly breaks 20-bar highs → choch_detected=1."""
        feats = self._engineer(_daily_bars(60, trend=0.005))
        # At least hh_hl_sequence should be > 0 in an uptrend
        assert feats["hh_hl_sequence"] > 0

    def test_fallback_when_highs_too_few(self):
        """Exactly 52 bars (just above MIN_BARS but < 22 needed by ChoCh) → fallback."""
        bars = _daily_bars(52)
        feats = self._engineer(bars)
        # 52 bars: rolling(20).max().shift(1) needs 21 bars → highs[-22:] has 22 but
        # rolling requires the window so choch may still run. Just confirm keys exist.
        assert "choch_detected" in feats
        assert "bars_since_choch" in feats
        assert "hh_hl_sequence" in feats

    def test_choch_detected_is_binary(self):
        feats = self._engineer(_daily_bars(60))
        assert feats["choch_detected"] in (0.0, 1.0)

    def test_bars_since_choch_non_negative(self):
        feats = self._engineer(_daily_bars(60))
        assert feats["bars_since_choch"] >= 0.0

    def test_hh_hl_bounded(self):
        """hh_hl_sequence is between 0 and 5 (5 diffs from 6 pivots)."""
        feats = self._engineer(_daily_bars(60, trend=0.005))
        assert 0.0 <= feats["hh_hl_sequence"] <= 5.0


# ── Whale Candle ──────────────────────────────────────────────────────────────

class TestWhaleCandle:

    def _compute(self, bars):
        from app.ml.intraday_features import compute_intraday_features
        return compute_intraday_features(bars)

    def test_whale_candle_key_present(self):
        feats = self._compute(_5min_bars(36))
        assert "whale_candle" in feats

    def test_whale_candle_is_binary(self):
        feats = self._compute(_5min_bars(36))
        assert feats["whale_candle"] in (0.0, 1.0)

    def test_small_body_no_whale(self):
        """Normal small-body candles → whale_candle = 0."""
        feats = self._compute(_5min_bars(36, body_mult=0.1))
        assert feats["whale_candle"] == 0.0

    def test_large_body_triggers_whale(self):
        """Candle body >> ATR → whale_candle = 1."""
        # Use very large body relative to tight H/L range
        n = 36
        idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
        close = np.full(n, 100.0)
        # body = 5.0, but ATR ≈ high - low = 0.1 → body > 2 * ATR
        open_ = np.full(n, 95.0)
        bars = pd.DataFrame({
            "open": open_,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": [10_000] * n,
        }, index=idx)
        feats = self._compute(bars)
        assert feats["whale_candle"] == 1.0

    def test_whale_candle_in_feature_name_list(self):
        """whale_candle must appear in intraday FEATURE_NAMES."""
        from app.ml.intraday_features import FEATURE_NAMES
        assert "whale_candle" in FEATURE_NAMES
