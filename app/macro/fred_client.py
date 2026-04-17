"""
FRED (Federal Reserve Economic Data) API client.

Free API — get a key at https://fred.stlouisfed.org/docs/api/api_key.html
Key is optional: without it we fall back to the public JSON API (rate-limited).

Series used:
  FEDFUNDS   — Federal Funds Effective Rate (monthly)
  DGS10      — 10-Year Treasury Constant Maturity Rate (daily)
  CPIAUCSL   — CPI All Urban Consumers (monthly, used for YoY calc)
  UNRATE     — Unemployment Rate (monthly)
  T10Y2Y     — 10Y-2Y Treasury Spread (yield curve, daily)

All values cached for CACHE_TTL seconds to avoid repeated API calls.
"""
import logging
import time
import urllib.request
import urllib.parse
import json
from typing import Optional, Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)

CACHE_TTL = 3600  # 1 hour — macro data doesn't move intraday

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.json"
_FRED_API = "https://api.stlouisfed.org/fred/series/observations"


class FredClient:
    """Fetches key macro indicators from FRED with caching."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_fed_funds_rate(self) -> Optional[float]:
        """Federal Funds Effective Rate (%)."""
        return self._latest("FEDFUNDS")

    def get_10y_yield(self) -> Optional[float]:
        """10-Year Treasury yield (%)."""
        return self._latest("DGS10")

    def get_yield_spread(self) -> Optional[float]:
        """10Y-2Y Treasury spread (%). Negative = inverted curve."""
        return self._latest("T10Y2Y")

    def get_cpi_yoy(self) -> Optional[float]:
        """CPI year-over-year change (%). Uses last two readings."""
        return self._cpi_yoy()

    def get_unemployment_rate(self) -> Optional[float]:
        """US Unemployment Rate (%)."""
        return self._latest("UNRATE")

    def get_all(self) -> Dict[str, Optional[float]]:
        """Fetch all indicators in one call."""
        return {
            "fed_funds_rate":   self.get_fed_funds_rate(),
            "yield_10y":        self.get_10y_yield(),
            "yield_spread_10y2y": self.get_yield_spread(),
            "cpi_yoy":          self.get_cpi_yoy(),
            "unemployment_rate": self.get_unemployment_rate(),
        }

    # ── Macro regime signal ───────────────────────────────────────────────────

    def macro_risk_score(self) -> float:
        """
        Return a 0–1 risk score combining macro indicators.
        0 = very low risk (growth environment), 1 = high risk (recession signals).
        Used alongside VIX to adjust regime detection.
        """
        score = 0.0
        factors = 0

        spread = self.get_yield_spread()
        if spread is not None:
            # Inverted curve (spread < 0) is bearish
            if spread < -0.5:
                score += 1.0
            elif spread < 0:
                score += 0.7
            elif spread < 0.5:
                score += 0.3
            factors += 1

        fed = self.get_fed_funds_rate()
        if fed is not None:
            # High rates → tighter financial conditions
            if fed > 5.0:
                score += 0.6
            elif fed > 3.0:
                score += 0.3
            factors += 1

        cpi = self.get_cpi_yoy()
        if cpi is not None:
            # High inflation → Fed likely to keep hiking
            if cpi > 5.0:
                score += 0.5
            elif cpi > 3.0:
                score += 0.2
            factors += 1

        unemployment = self.get_unemployment_rate()
        if unemployment is not None:
            if unemployment > 5.0:
                score += 0.4
            elif unemployment > 4.0:
                score += 0.1
            factors += 1

        return round(score / max(factors, 1), 3)

    # ── Internal fetch logic ──────────────────────────────────────────────────

    def _latest(self, series_id: str) -> Optional[float]:
        data = self._fetch(series_id)
        if data is None:
            return None
        try:
            # Find last non-null observation
            for obs in reversed(data):
                val = obs.get("value", ".")
                if val not in (".", "", None):
                    return float(val)
        except Exception:
            pass
        return None

    def _cpi_yoy(self) -> Optional[float]:
        data = self._fetch("CPIAUCSL")
        if data is None or len(data) < 13:
            return None
        try:
            def last_val(obs_list, offset=0):
                for obs in reversed(obs_list[:-offset] if offset else obs_list):
                    v = obs.get("value", ".")
                    if v not in (".", "", None):
                        return float(v)
                return None

            recent = last_val(data)
            year_ago = last_val(data[:-12])
            if recent and year_ago and year_ago != 0:
                return round((recent - year_ago) / year_ago * 100, 2)
        except Exception:
            pass
        return None

    def _fetch(self, series_id: str) -> Optional[list]:
        now = time.monotonic()
        cached = self._cache.get(series_id)
        if cached and now - cached["ts"] < CACHE_TTL:
            return cached["data"]

        data = self._fetch_api(series_id) or self._fetch_graph(series_id)
        if data is not None:
            self._cache[series_id] = {"data": data, "ts": now}
            logger.debug("FRED %s: fetched %d observations", series_id, len(data))
        return data

    def _fetch_api(self, series_id: str) -> Optional[list]:
        """Use official FRED API if key is configured."""
        api_key = getattr(settings, "fred_api_key", None)
        if not api_key:
            return None
        try:
            params = urllib.parse.urlencode({
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "asc",
                "limit": 24,  # last 24 observations
            })
            url = f"{_FRED_API}?{params}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                body = json.loads(resp.read())
                return body.get("observations", [])
        except Exception as exc:
            logger.debug("FRED API error for %s: %s", series_id, exc)
            return None

    def _fetch_graph(self, series_id: str) -> Optional[list]:
        """Fallback: FRED public graph JSON endpoint (no key required)."""
        try:
            url = f"{_FRED_BASE}?id={series_id}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                body = json.loads(resp.read())
                # Graph API returns [[timestamp_ms, value], ...]
                rows = []
                for point in body:
                    if isinstance(point, list) and len(point) == 2:
                        rows.append({"value": str(point[1]) if point[1] is not None else "."})
                return rows if rows else None
        except Exception as exc:
            logger.debug("FRED graph API error for %s: %s", series_id, exc)
            return None


# Module-level singleton
fred_client = FredClient()
