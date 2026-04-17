"""
Market regime detection based on VIX.

Regimes:
  LOW    : VIX < 15  — trend-following favoured
  MEDIUM : VIX 15-25 — both strategies active
  HIGH   : VIX > 25  — mean-reversion only, reduced sizing

Reuses the circuit breaker's cached VIX value to avoid extra yfinance calls.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

VIX_LOW_THRESHOLD = 15.0
VIX_HIGH_THRESHOLD = 25.0
CACHE_TTL_SECONDS = 300   # 5-minute cache

REGIME_LOW = "LOW"
REGIME_MEDIUM = "MEDIUM"
REGIME_HIGH = "HIGH"


class RegimeDetector:
    def __init__(self):
        self._cached_vix: Optional[float] = None
        self._cache_ts: float = 0.0

    def get_vix(self) -> Optional[float]:
        now = time.monotonic()
        if now - self._cache_ts < CACHE_TTL_SECONDS and self._cached_vix is not None:
            return self._cached_vix
        try:
            import yfinance as yf
            df = yf.download("^VIX", period="1d", progress=False, auto_adjust=True)
            if not df.empty:
                vix = float(df["Close"].iloc[-1])
                self._cached_vix = vix
                self._cache_ts = now
                return vix
        except Exception as exc:
            logger.debug("Could not fetch VIX: %s", exc)
        return self._cached_vix  # stale or None

    def get_regime(self) -> str:
        """Return REGIME_LOW / REGIME_MEDIUM / REGIME_HIGH. Defaults to MEDIUM on error."""
        vix = self.get_vix()
        if vix is None:
            return REGIME_MEDIUM
        if vix < VIX_LOW_THRESHOLD:
            return REGIME_LOW
        if vix > VIX_HIGH_THRESHOLD:
            return REGIME_HIGH
        return REGIME_MEDIUM

    def trend_following_active(self) -> bool:
        return self.get_regime() in (REGIME_LOW, REGIME_MEDIUM)

    def mean_reversion_active(self) -> bool:
        return self.get_regime() in (REGIME_MEDIUM, REGIME_HIGH)

    def position_size_multiplier(self) -> float:
        """Scale down sizing in high-volatility regime."""
        regime = self.get_regime()
        if regime == REGIME_HIGH:
            return 0.5
        if regime == REGIME_MEDIUM:
            return 0.75
        return 1.0


regime_detector = RegimeDetector()
