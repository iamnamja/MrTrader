"""
Unit tests for Phase 3: Technical indicators and Trader Agent signal logic.

All tests are pure-Python — no database, Redis, or Alpaca connections needed.
"""

import pytest

from app.indicators.technical import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    is_overbought,
    is_oversold,
    price_near_band,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _downtrend(n: int = 30) -> list:
    """Steadily falling prices: 100 → (100-n)."""
    return [100 - i for i in range(n)]


def _uptrend(n: int = 30) -> list:
    """Steadily rising prices: 70 → (70+n)."""
    return [70 + i for i in range(n)]


def _flat(value: float = 100.0, n: int = 30) -> list:
    return [value] * n


# ─── RSI ─────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_downtrend_is_oversold(self):
        rsi = calculate_rsi(_downtrend(30))
        assert rsi is not None
        assert rsi < 30

    def test_uptrend_is_overbought(self):
        rsi = calculate_rsi(_uptrend(30))
        assert rsi is not None
        assert rsi > 70

    def test_flat_prices_near_50(self):
        # After a flat period RSI should be near 50
        # Start with a balanced mix then flatten
        prices = _uptrend(15) + _downtrend(15)
        rsi = calculate_rsi(prices)
        assert rsi is not None
        assert 20 < rsi < 80  # broad range — just not extreme

    def test_returns_none_when_insufficient_data(self):
        assert calculate_rsi([100, 99, 98], period=14) is None

    def test_exactly_at_min_data(self):
        # period+1 prices is the minimum
        prices = list(range(100, 85, -1))  # 15 prices, period=14
        rsi = calculate_rsi(prices, period=14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_range_0_to_100(self):
        for prices in [_uptrend(30), _downtrend(30)]:
            rsi = calculate_rsi(prices)
            assert 0 <= rsi <= 100

    def test_custom_period(self):
        rsi = calculate_rsi(_downtrend(20), period=7)
        assert rsi is not None
        assert rsi < 30


# ─── EMA ─────────────────────────────────────────────────────────────────────

class TestEMA:
    def test_uptrend_ema_below_last_price(self):
        prices = _uptrend(30)
        ema = calculate_ema(prices, period=10)
        assert ema is not None
        assert ema < prices[-1]

    def test_downtrend_ema_above_last_price(self):
        prices = _downtrend(30)
        ema = calculate_ema(prices, period=10)
        assert ema is not None
        assert ema > prices[-1]

    def test_flat_ema_equals_price(self):
        prices = _flat(100.0, 30)
        ema = calculate_ema(prices, period=10)
        assert ema == pytest.approx(100.0, rel=1e-3)

    def test_returns_none_when_insufficient(self):
        assert calculate_ema([100, 101, 102], period=20) is None

    def test_value_in_price_range(self):
        prices = _uptrend(30)
        ema = calculate_ema(prices, period=5)
        assert min(prices) <= ema <= max(prices)


# ─── MACD ────────────────────────────────────────────────────────────────────

class TestMACD:
    def test_returns_three_values(self):
        result = calculate_macd(_uptrend(50))
        assert result is not None
        assert len(result) == 3

    def test_uptrend_macd_positive(self):
        macd_line, signal_line, histogram = calculate_macd(_uptrend(50))
        # In a sustained uptrend fast EMA > slow EMA
        assert macd_line > 0

    def test_downtrend_macd_negative(self):
        macd_line, signal_line, histogram = calculate_macd(_downtrend(50))
        assert macd_line < 0

    def test_returns_none_when_insufficient(self):
        assert calculate_macd([100] * 20) is None  # need slow+signal = 35

    def test_histogram_equals_macd_minus_signal(self):
        macd_line, signal_line, histogram = calculate_macd(_uptrend(50))
        assert histogram == pytest.approx(macd_line - signal_line, rel=1e-6)


# ─── Bollinger Bands ──────────────────────────────────────────────────────────

class TestBollingerBands:
    def test_returns_three_values(self):
        result = calculate_bollinger_bands(_uptrend(30))
        assert result is not None
        upper, middle, lower = result
        assert upper > middle > lower

    def test_middle_band_is_sma(self):
        prices = _flat(100.0, 30)
        upper, middle, lower = calculate_bollinger_bands(prices, period=20)
        assert middle == pytest.approx(100.0, rel=1e-3)

    def test_flat_prices_narrow_bands(self):
        prices = _flat(100.0, 30)
        upper, middle, lower = calculate_bollinger_bands(prices, period=20)
        # Perfectly flat → std=0 → bands equal middle
        assert upper == pytest.approx(middle, abs=0.01)
        assert lower == pytest.approx(middle, abs=0.01)

    def test_volatile_prices_wider_bands(self):
        volatile = [100 + (i % 2) * 10 for i in range(30)]  # alternates 100/110
        upper, middle, lower = calculate_bollinger_bands(volatile, period=20)
        assert (upper - lower) > 5

    def test_returns_none_when_insufficient(self):
        assert calculate_bollinger_bands([100] * 10, period=20) is None


# ─── ATR ─────────────────────────────────────────────────────────────────────

class TestATR:
    def _ohlc(self, n: int = 20):
        closes = _uptrend(n)
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        return highs, lows, closes

    def test_returns_positive_value(self):
        h, l, c = self._ohlc(25)
        atr = calculate_atr(h, l, c)
        assert atr is not None
        assert atr > 0

    def test_high_volatility_larger_atr(self):
        n = 25
        closes = _flat(100.0, n)
        # Narrow range
        narrow_atr = calculate_atr(
            [c + 0.5 for c in closes], [c - 0.5 for c in closes], closes
        )
        # Wide range
        wide_atr = calculate_atr(
            [c + 5 for c in closes], [c - 5 for c in closes], closes
        )
        assert wide_atr > narrow_atr

    def test_returns_none_when_insufficient(self):
        h, l, c = self._ohlc(10)
        assert calculate_atr(h, l, c, period=14) is None


# ─── SMA ─────────────────────────────────────────────────────────────────────

class TestSMA:
    def test_simple_average(self):
        prices = [10, 20, 30, 40, 50]
        sma = calculate_sma(prices, period=5)
        assert sma == pytest.approx(30.0)

    def test_returns_none_when_insufficient(self):
        assert calculate_sma([10, 20], period=5) is None


# ─── Signal Helpers ───────────────────────────────────────────────────────────

class TestSignalHelpers:
    def test_is_oversold_downtrend(self):
        assert is_oversold(_downtrend(30)) is True

    def test_is_oversold_uptrend(self):
        assert is_oversold(_uptrend(30)) is False

    def test_is_overbought_uptrend(self):
        assert is_overbought(_uptrend(30)) is True

    def test_is_overbought_downtrend(self):
        assert is_overbought(_downtrend(30)) is False

    def test_is_oversold_custom_threshold(self):
        # RSI of downtrend is well below 30; custom threshold of 50 should also fire
        assert is_oversold(_downtrend(30), rsi_threshold=50) is True

    def test_price_near_band_within_tolerance(self):
        assert price_near_band(100.0, 100.3, tolerance_pct=0.005) is True

    def test_price_near_band_outside_tolerance(self):
        assert price_near_band(100.0, 102.0, tolerance_pct=0.005) is False

    def test_price_near_band_zero_band(self):
        assert price_near_band(100.0, 0.0) is False

    def test_price_exact_match(self):
        assert price_near_band(100.0, 100.0) is True


# ─── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_price_rsi_returns_none(self):
        assert calculate_rsi([100]) is None

    def test_empty_list_rsi_returns_none(self):
        assert calculate_rsi([]) is None

    def test_all_same_price_rsi(self):
        # All gains/losses = 0 → RSI should return a value, not crash
        prices = _flat(100.0, 20)
        rsi = calculate_rsi(prices)
        # With no movement avg_loss = 0, RS = inf, RSI = 100
        assert rsi is None or 0 <= rsi <= 100

    def test_rsi_with_nan_values(self):
        import math
        prices = [100, float("nan"), 98, 97, 96, 95, 94, 93, 92, 91,
                  90, 89, 88, 87, 86, 85]
        rsi = calculate_rsi(prices)
        # Should not raise; NaN is forward-filled
        assert rsi is None or (0 <= rsi <= 100)
