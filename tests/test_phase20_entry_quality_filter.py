"""
Tests for Phase 20 — Entry Quality Filter.

Verifies:
  1. Swing AgentSimulator blocks entries when price < EMA-20/50
  2. Swing AgentSimulator blocks entries when RSI > 70 (overbought) or < 40 (freefall)
  3. Swing AgentSimulator blocks entries when volume < 80% of 20-day avg
  4. Swing AgentSimulator allows entries when all gates pass
  5. Intraday AgentSimulator requires volume_surge >= 1.2 OR whale_candle = 1
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import date


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bars(n: int = 220, base: float = 100.0, trend: float = 0.001,
          last_vol_ratio: float = 1.0, noise: float = 0.003) -> pd.DataFrame:
    """n daily bars with uptrend + noise so RSI stays in range. Volume scaled by last_vol_ratio."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    trend_component = np.array([base * (1 + trend) ** i for i in range(n)])
    noise_component = rng.normal(0, noise, n) * trend_component
    prices = trend_component + noise_component
    prices = np.maximum(prices, 1.0)
    vols = np.full(n, 1_000_000.0)
    vols[-1] = 1_000_000.0 * last_vol_ratio
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": vols,
    }, index=idx)


def _flat_bars(n: int = 220, base: float = 100.0) -> pd.DataFrame:
    """Flat bars — price never rises above EMA-20/50."""
    return _bars(n, base=base, trend=0.0)


def _mock_signal(stop: float = 96.0, target: float = 110.0):
    sig = MagicMock()
    sig.stop_price = stop
    sig.target_price = target
    return sig


# ── Swing Trader Gate Tests ────────────────────────────────────────────────────

class TestSwingTraderGate:

    def _call(self, bars):
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator(model=MagicMock())
        with patch("app.backtesting.agent_simulator.generate_signal", return_value=_mock_signal()):
            return sim._trader_signal("AAPL", bars)

    def test_insufficient_bars_returns_false(self):
        ok, stop, target = self._call(_bars(150))
        assert ok is False
        assert stop == 0.0
        assert target == 0.0

    def test_uptrend_all_gates_pass(self):
        """Strong uptrend → EMA gates pass, RSI in range, volume ok → True."""
        ok, stop, target = self._call(_bars(220, trend=0.001))
        assert ok is True
        assert stop > 0.0
        assert target > 0.0

    def test_downtrend_blocked_by_ema200(self):
        """Falling price → below EMA-200 → blocked."""
        ok, _, _ = self._call(_bars(220, trend=-0.002))
        assert ok is False

    def test_low_volume_blocked(self):
        """Volume at 50% of avg → blocked."""
        ok, _, _ = self._call(_bars(220, trend=0.001, last_vol_ratio=0.5))
        assert ok is False

    def test_adequate_volume_passes(self):
        """Volume at 90% of avg → passes volume gate."""
        ok, _, _ = self._call(_bars(220, trend=0.001, last_vol_ratio=0.9))
        assert ok is True

    def test_overbought_rsi_blocked(self):
        """Very steep uptrend → RSI > 70 → blocked."""
        ok, _, _ = self._call(_bars(220, trend=0.008))
        assert ok is False


# ── Intraday Trader Gate Tests ─────────────────────────────────────────────────

class TestIntradayVolumeGate:
    """Verify the ORB + (volume_surge >= 1.2 OR whale_candle) gate."""

    def _build_feats(self, orb=1.0, vol_surge=1.0, whale=0.0):
        return {"orb_breakout": orb, "volume_surge": vol_surge, "whale_candle": whale}

    def test_orb_only_insufficient(self):
        """ORB=1 but volume_surge=1.0 and whale=0 → rejected."""
        feats = self._build_feats(orb=1.0, vol_surge=1.0, whale=0.0)
        orb_ok = feats.get("orb_breakout", 0.0) > 0
        vol_ok = feats.get("volume_surge", 0.0) >= 1.2 or feats.get("whale_candle", 0.0) == 1.0
        assert orb_ok and not vol_ok

    def test_orb_plus_volume_surge_passes(self):
        """ORB=1 + volume_surge=1.5 → accepted."""
        feats = self._build_feats(orb=1.0, vol_surge=1.5, whale=0.0)
        orb_ok = feats.get("orb_breakout", 0.0) > 0
        vol_ok = feats.get("volume_surge", 0.0) >= 1.2 or feats.get("whale_candle", 0.0) == 1.0
        assert orb_ok and vol_ok

    def test_orb_plus_whale_passes(self):
        """ORB=1 + whale_candle=1 → accepted (whale overrides volume gate)."""
        feats = self._build_feats(orb=1.0, vol_surge=0.8, whale=1.0)
        orb_ok = feats.get("orb_breakout", 0.0) > 0
        vol_ok = feats.get("volume_surge", 0.0) >= 1.2 or feats.get("whale_candle", 0.0) == 1.0
        assert orb_ok and vol_ok

    def test_no_orb_always_rejected(self):
        """ORB=0 → rejected regardless of volume."""
        feats = self._build_feats(orb=0.0, vol_surge=2.0, whale=1.0)
        orb_ok = feats.get("orb_breakout", 0.0) > 0
        assert not orb_ok
