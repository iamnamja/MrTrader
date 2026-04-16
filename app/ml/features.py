"""
Feature engineering for the ML portfolio selection model.

Accepts OHLCV DataFrames from either Alpaca (lowercase columns) or
yfinance (capitalized columns) — both are normalised internally.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.indicators.technical import (
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case all column names so Alpaca and yfinance DataFrames work identically."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    # yfinance uses "adj close"; rename to "close" if "close" is absent
    if "close" not in df.columns and "adj close" in df.columns:
        df = df.rename(columns={"adj close": "close"})
    return df


class FeatureEngineer:
    """Extract a fixed feature vector for any stock given its OHLCV history."""

    # Minimum bars needed to compute all features
    MIN_BARS = 52

    def engineer_features(
        self,
        symbol: str,
        bars: pd.DataFrame,
        sentiment: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Compute features for a single stock.

        Args:
            symbol:    Stock ticker (used only for logging).
            bars:      OHLCV DataFrame (Alpaca or yfinance format).
            sentiment: Optional news sentiment score in [-1, 1].

        Returns:
            Dict of feature_name → float, or None if insufficient data.
        """
        bars = _normalise_columns(bars)

        required = {"close", "high", "low", "volume"}
        if not required.issubset(set(bars.columns)):
            return None

        if len(bars) < self.MIN_BARS:
            return None

        prices = bars["close"].tolist()
        highs = bars["high"].tolist()
        lows = bars["low"].tolist()
        volumes = bars["volume"].tolist()
        current_price = prices[-1]

        features: Dict[str, float] = {}

        # ── 1. RSI ────────────────────────────────────────────────────────────
        rsi_14 = calculate_rsi(prices, period=14)
        rsi_7 = calculate_rsi(prices, period=7)
        features["rsi_14"] = rsi_14 if rsi_14 is not None else 50.0
        features["rsi_7"] = rsi_7 if rsi_7 is not None else 50.0

        # ── 2. MACD ───────────────────────────────────────────────────────────
        macd_result = calculate_macd(prices)
        if macd_result is not None:
            macd_line, signal_line, histogram = macd_result
            features["macd"] = macd_line
            features["macd_signal"] = signal_line
            features["macd_histogram"] = histogram
        else:
            features["macd"] = 0.0
            features["macd_signal"] = 0.0
            features["macd_histogram"] = 0.0

        # ── 3. EMAs & price position ──────────────────────────────────────────
        ema_20 = calculate_ema(prices, period=20) or current_price
        ema_50 = calculate_ema(prices, period=50) or current_price
        features["ema_20"] = ema_20
        features["ema_50"] = ema_50
        features["price_above_ema20"] = 1.0 if current_price > ema_20 else 0.0
        features["price_above_ema50"] = 1.0 if current_price > ema_50 else 0.0

        # ── 4. Price change ───────────────────────────────────────────────────
        prev_close = prices[-2] if len(prices) > 1 else current_price
        features["price_change_pct"] = (current_price - prev_close) / prev_close if prev_close else 0.0

        # ── 5. 52-week high / low ratio ───────────────────────────────────────
        lookback = min(252, len(prices))
        high_52w = max(prices[-lookback:])
        low_52w = min(prices[-lookback:])
        features["price_to_52w_high"] = current_price / high_52w if high_52w else 1.0
        features["price_to_52w_low"] = current_price / low_52w if low_52w else 1.0

        # ── 6. Volume ratio ───────────────────────────────────────────────────
        avg_volume = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        features["volume_ratio"] = (volumes[-1] / avg_volume) if avg_volume > 0 else 1.0

        # ── 7. Trend flags ────────────────────────────────────────────────────
        lookback_5 = min(5, len(prices) - 1)
        features["uptrend"] = 1.0 if prices[-1] > prices[-(lookback_5 + 1)] else 0.0
        features["downtrend"] = 1.0 if prices[-1] < prices[-(lookback_5 + 1)] else 0.0

        # ── 8. Volatility (annualised) ────────────────────────────────────────
        returns = np.diff(prices) / np.array(prices[:-1])
        features["volatility"] = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0

        # ── 9. Momentum ───────────────────────────────────────────────────────
        lookback_20 = min(20, len(prices) - 1)
        features["momentum_5d"] = (prices[-1] - prices[-(lookback_5 + 1)]) / prices[-(lookback_5 + 1)] if prices[-(lookback_5 + 1)] else 0.0
        features["momentum_20d"] = (prices[-1] - prices[-(lookback_20 + 1)]) / prices[-(lookback_20 + 1)] if prices[-(lookback_20 + 1)] else 0.0

        # ── 10. Sentiment ─────────────────────────────────────────────────────
        features["sentiment"] = float(sentiment) if sentiment is not None else 0.0

        return features
