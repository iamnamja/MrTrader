"""
Market regime detection using a composite VIX + FRED macro score.

Composite = VIX_WEIGHT * vix_score + MACRO_WEIGHT * macro_score  (both 0–1)

Regimes:
  LOW    : composite < 0.25 — trend-following favoured
  MEDIUM : composite 0.25–0.60 — both strategies active
  HIGH   : composite > 0.60 — mean-reversion only, reduced sizing
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VIX_LOW_THRESHOLD = 15.0
VIX_HIGH_THRESHOLD = 25.0
CACHE_TTL_SECONDS = 300   # 5-minute cache

VIX_WEIGHT = 0.70
MACRO_WEIGHT = 0.30
COMPOSITE_LOW = 0.25
COMPOSITE_HIGH = 0.60

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
            df = yf.download("^VIX", period="1d", progress=False, auto_adjust=True, timeout=10)
            if not df.empty:
                vix = float(df["Close"].iat[-1])
                self._cached_vix = vix
                self._cache_ts = now
                return vix
        except Exception as exc:
            logger.debug("Could not fetch VIX: %s", exc)
        return self._cached_vix  # stale or None

    def _vix_score(self) -> float:
        """Normalize VIX to 0–1 (clamped)."""
        vix = self.get_vix()
        if vix is None:
            return 0.5
        if vix <= VIX_LOW_THRESHOLD:
            return 0.0
        if vix >= VIX_HIGH_THRESHOLD:
            return 1.0
        return (vix - VIX_LOW_THRESHOLD) / (VIX_HIGH_THRESHOLD - VIX_LOW_THRESHOLD)

    def _macro_score(self) -> float:
        """0–1 macro risk score from FRED indicators. Returns 0.5 on failure."""
        try:
            from app.macro.fred_client import fred_client
            return fred_client.macro_risk_score()
        except Exception as exc:
            logger.debug("Macro score unavailable: %s", exc)
            return 0.5

    def composite_score(self) -> float:
        """Weighted blend of VIX + macro scores."""
        return round(VIX_WEIGHT * self._vix_score() + MACRO_WEIGHT * self._macro_score(), 3)

    def get_regime(self) -> str:
        """Return REGIME_LOW / REGIME_MEDIUM / REGIME_HIGH based on composite score."""
        score = self.composite_score()
        if score < COMPOSITE_LOW:
            return REGIME_LOW
        if score > COMPOSITE_HIGH:
            return REGIME_HIGH
        return REGIME_MEDIUM

    def get_regime_detail(self) -> Dict[str, Any]:
        """Full breakdown for dashboard display."""
        vix = self.get_vix()
        vix_s = self._vix_score()
        macro_s = self._macro_score()
        composite = self.composite_score()
        regime = self.get_regime()

        try:
            from app.macro.fred_client import fred_client
            macro_indicators = fred_client.get_all()
        except Exception:
            macro_indicators = {}

        return {
            "regime": regime,
            "composite_score": composite,
            "vix": vix,
            "vix_score": round(vix_s, 3),
            "macro_score": round(macro_s, 3),
            "vix_weight": VIX_WEIGHT,
            "macro_weight": MACRO_WEIGHT,
            "macro_indicators": macro_indicators,
        }

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
