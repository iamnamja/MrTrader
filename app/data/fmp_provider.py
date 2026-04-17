"""
Financial Modeling Prep (FMP) data provider.

Supplies point-in-time fundamental features for the swing ML model:

  - Historical earnings surprises (EPS actual vs estimate, per quarter)
  - Analyst upgrade/downgrade history
  - Institutional holdings changes (13F)

Point-in-time correctness
-------------------------
For a training window ending on date D, we return only the data that
was publicly known on or before D.  This eliminates the look-ahead bias
that caused the swing model AUC to be stuck at ~0.51 when using current
fundamentals for all historical windows.

All endpoints use FMP's /stable/ API (v4 — post-Aug 2025 plan).
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com/stable"
_CACHE_TTL = 86_400  # 24 h in-process cache

# In-process cache: symbol → (data, fetched_at)
_earnings_cache: Dict[str, tuple] = {}
_grades_cache: Dict[str, tuple] = {}
_institutional_cache: Dict[str, tuple] = {}


def _api_key() -> str:
    from app.config import settings
    return settings.fmp_api_key or ""


# ── Earnings history ──────────────────────────────────────────────────────────

def get_earnings_history_fmp(symbol: str) -> List[Dict]:
    """
    Return all available quarterly earnings records for *symbol*.

    Each record:
      date         — report date (YYYY-MM-DD string)
      epsActual    — reported EPS (float or None)
      epsEstimated — consensus estimate (float or None)
      surprise_pct — (actual - est) / |est|, clipped to [-1, 1]
    """
    now = time.time()
    cached = _earnings_cache.get(symbol)
    if cached and now - cached[1] < _CACHE_TTL:
        return cached[0]

    records = []
    try:
        resp = requests.get(
            f"{_BASE}/earnings",
            params={"symbol": symbol, "limit": 20, "apikey": _api_key()},
            timeout=10,
        )
        if resp.status_code == 200:
            for row in resp.json():
                actual = row.get("epsActual")
                est = row.get("epsEstimated")
                surprise = None
                if actual is not None and est is not None:
                    try:
                        a, e = float(actual), float(est)
                        if abs(e) > 0.001:
                            surprise = float(max(-1.0, min(1.0, (a - e) / abs(e))))
                    except (TypeError, ValueError):
                        pass
                records.append({
                    "date": row.get("date", ""),
                    "epsActual": actual,
                    "epsEstimated": est,
                    "surprise_pct": surprise,
                })
    except Exception as exc:
        logger.debug("FMP earnings fetch failed for %s: %s", symbol, exc)

    _earnings_cache[symbol] = (records, now)
    return records


def get_earnings_features_at(symbol: str, as_of: date) -> Dict[str, float]:
    """
    Return earnings features using only data known on or before *as_of*.

    Features:
      fmp_surprise_1q       — most recent quarterly EPS surprise before as_of
      fmp_surprise_2q_avg   — mean of last 2 surprises before as_of
      fmp_days_since_earnings — calendar days since last report before as_of
    """
    result = {
        "fmp_surprise_1q": 0.0,
        "fmp_surprise_2q_avg": 0.0,
        "fmp_days_since_earnings": 90.0,
    }

    try:
        records = get_earnings_history_fmp(symbol)
        # Filter to reports on or before as_of
        past = [
            r for r in records
            if r["date"] and r["surprise_pct"] is not None
            and datetime.strptime(r["date"], "%Y-%m-%d").date() <= as_of
        ]
        if not past:
            return result

        result["fmp_surprise_1q"] = float(past[0]["surprise_pct"])
        result["fmp_surprise_2q_avg"] = float(
            sum(r["surprise_pct"] for r in past[:2]) / min(len(past), 2)
        )
        last_date = datetime.strptime(past[0]["date"], "%Y-%m-%d").date()
        result["fmp_days_since_earnings"] = float(
            max(0, min(365, (as_of - last_date).days))
        )
    except Exception as exc:
        logger.debug("FMP earnings features failed for %s: %s", symbol, exc)

    return result


# ── Analyst grades ────────────────────────────────────────────────────────────

def get_analyst_grades_fmp(symbol: str) -> List[Dict]:
    """
    Return all available analyst upgrade/downgrade records for *symbol*.

    Each record:
      date           — grade date (YYYY-MM-DD)
      gradingCompany — analyst firm
      previousGrade  — prior rating
      newGrade       — new rating
      action         — 'upgrade', 'downgrade', 'maintain', 'init'
    """
    now = time.time()
    cached = _grades_cache.get(symbol)
    if cached and now - cached[1] < _CACHE_TTL:
        return cached[0]

    records = []
    try:
        resp = requests.get(
            f"{_BASE}/grades",
            params={"symbol": symbol, "limit": 50, "apikey": _api_key()},
            timeout=10,
        )
        if resp.status_code == 200:
            for row in resp.json():
                records.append({
                    "date": row.get("date", ""),
                    "gradingCompany": row.get("gradingCompany", ""),
                    "previousGrade": row.get("previousGrade", ""),
                    "newGrade": row.get("newGrade", ""),
                    "action": _classify_action(
                        row.get("previousGrade", ""),
                        row.get("newGrade", ""),
                        row.get("action", ""),
                    ),
                })
    except Exception as exc:
        logger.debug("FMP grades fetch failed for %s: %s", symbol, exc)

    _grades_cache[symbol] = (records, now)
    return records


def get_analyst_features_at(symbol: str, as_of: date, lookback_days: int = 30) -> Dict[str, float]:
    """
    Return analyst momentum features using only data known on or before *as_of*.

    Features:
      fmp_analyst_upgrades_30d   — upgrades in last *lookback_days* days
      fmp_analyst_downgrades_30d — downgrades in last *lookback_days* days
      fmp_analyst_momentum_30d   — net (upgrades - downgrades), clipped [-5, 5]
    """
    result = {
        "fmp_analyst_upgrades_30d": 0.0,
        "fmp_analyst_downgrades_30d": 0.0,
        "fmp_analyst_momentum_30d": 0.0,
    }

    try:
        records = get_analyst_grades_fmp(symbol)
        cutoff = as_of - timedelta(days=lookback_days)
        window = [
            r for r in records
            if r["date"]
            and cutoff <= datetime.strptime(r["date"], "%Y-%m-%d").date() <= as_of
        ]
        upgrades = sum(1 for r in window if r["action"] == "upgrade")
        downgrades = sum(1 for r in window if r["action"] == "downgrade")
        result["fmp_analyst_upgrades_30d"] = float(upgrades)
        result["fmp_analyst_downgrades_30d"] = float(downgrades)
        result["fmp_analyst_momentum_30d"] = float(
            max(-5, min(5, upgrades - downgrades))
        )
    except Exception as exc:
        logger.debug("FMP analyst features failed for %s: %s", symbol, exc)

    return result


# ── Institutional holdings ────────────────────────────────────────────────────

def get_institutional_features_at(symbol: str, as_of: date) -> Dict[str, float]:
    """
    Return institutional ownership change features as of *as_of*.

    Features:
      fmp_inst_ownership_pct    — % of float held by institutions (latest before as_of)
      fmp_inst_change_pct       — change in ownership vs prior quarter
    """
    result = {
        "fmp_inst_ownership_pct": 0.0,
        "fmp_inst_change_pct": 0.0,
    }

    now = time.time()
    cached = _institutional_cache.get(symbol)
    if not cached or now - cached[1] >= _CACHE_TTL:
        try:
            resp = requests.get(
                f"{_BASE}/institutional-ownership/symbol",
                params={"symbol": symbol, "limit": 8, "apikey": _api_key()},
                timeout=10,
            )
            if resp.status_code == 200:
                _institutional_cache[symbol] = (resp.json(), now)
            else:
                _institutional_cache[symbol] = ([], now)
        except Exception as exc:
            logger.debug("FMP institutional fetch failed for %s: %s", symbol, exc)
            _institutional_cache[symbol] = ([], now)

    records = _institutional_cache[symbol][0]

    try:
        # Filter to filings on or before as_of
        past = [
            r for r in records
            if r.get("dateReported")
            and datetime.strptime(r["dateReported"][:10], "%Y-%m-%d").date() <= as_of
        ]
        if past:
            latest = past[0]
            pct = latest.get("ownershipPercent") or latest.get("percentOwned") or 0.0
            result["fmp_inst_ownership_pct"] = float(min(pct, 100.0))
            if len(past) >= 2:
                prev_pct = past[1].get("ownershipPercent") or past[1].get("percentOwned") or 0.0
                result["fmp_inst_change_pct"] = float(
                    max(-20, min(20, float(pct) - float(prev_pct)))
                )
    except Exception as exc:
        logger.debug("FMP inst ownership features failed for %s: %s", symbol, exc)

    return result


# ── Convenience: all FMP features at once ────────────────────────────────────

def get_fmp_features_at(symbol: str, as_of: date) -> Dict[str, float]:
    """
    Return all FMP features for *symbol* as of *as_of* date.
    Safe — always returns a complete dict of defaults on any error.
    """
    features: Dict[str, float] = {}
    features.update(get_earnings_features_at(symbol, as_of))
    features.update(get_analyst_features_at(symbol, as_of))
    features.update(get_institutional_features_at(symbol, as_of))
    return features


def prefetch_fmp(symbols: List[str]) -> None:
    """
    Pre-warm all FMP caches for *symbols* in one pass.
    Call this once before a training run to avoid per-window API calls.
    """
    logger.info("Pre-fetching FMP data for %d symbols...", len(symbols))
    for sym in symbols:
        try:
            get_earnings_history_fmp(sym)
        except Exception:
            pass
        try:
            get_analyst_grades_fmp(sym)
        except Exception:
            pass
    logger.info("FMP prefetch complete")


# ── Helpers ───────────────────────────────────────────────────────────────────

_BULLISH_GRADES = {"buy", "outperform", "overweight", "strong buy", "accumulate", "positive"}
_BEARISH_GRADES = {"sell", "underperform", "underweight", "reduce", "negative", "strong sell"}


def _classify_action(prev: str, new: str, action_hint: str) -> str:
    """Map grade change to upgrade / downgrade / maintain / init."""
    hint = action_hint.lower()
    if "upgrade" in hint:
        return "upgrade"
    if "downgrade" in hint:
        return "downgrade"
    if "init" in hint or "initiate" in hint:
        return "init"
    p, n = prev.lower(), new.lower()
    if not p:
        return "init"
    p_bull = any(g in p for g in _BULLISH_GRADES)
    n_bull = any(g in n for g in _BULLISH_GRADES)
    p_bear = any(g in p for g in _BEARISH_GRADES)
    n_bear = any(g in n for g in _BEARISH_GRADES)
    if n_bull and not p_bull:
        return "upgrade"
    if n_bear and not p_bear:
        return "downgrade"
    return "maintain"
