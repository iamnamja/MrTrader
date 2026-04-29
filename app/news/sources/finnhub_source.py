"""
Finnhub data adapter — economic calendar, earnings calendar, company news.

All methods fail open: return empty lists/None on API error so a Finnhub
outage never blocks trading.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://finnhub.io/api/v1"
_TIMEOUT = 8  # seconds

# High-impact US macro event keywords (subset of what Finnhub returns)
_HIGH_IMPACT_KEYWORDS = {
    "nonfarm": "NFP",
    "non-farm": "NFP",
    "federal funds": "FOMC",
    "fomc": "FOMC",
    "fed interest": "FOMC",
    "interest rate decision": "FOMC",
    "consumer price": "CPI",
    "cpi": "CPI",
    "producer price": "PPI",
    "ppi": "PPI",
    "gdp": "GDP",
    "gross domestic": "GDP",
    "core pce": "PCE",
    "pce": "PCE",
    "retail sales": "RETAIL_SALES",
    "unemployment": "UNEMPLOYMENT",
    "initial jobless": "JOBLESS_CLAIMS",
    "ism manufacturing": "ISM_MFG",
    "ism services": "ISM_SVC",
    "core inflation": "CPI",
}


def _key() -> Optional[str]:
    """Return Finnhub API key from env, preferring the correctly-spelled var."""
    from app.config import settings
    return (
        settings.finnhub_api_key
        or settings.finhub_api_key
        or os.environ.get("FINNHUB_API_KEY")
        or os.environ.get("FINHUB_API_KEY")
    )


def _get(endpoint: str, params: dict) -> Optional[dict]:
    key = _key()
    if not key:
        logger.debug("Finnhub key not configured — skipping %s", endpoint)
        return None
    params["token"] = key
    try:
        r = requests.get(f"{_BASE}/{endpoint}", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("Finnhub %s failed: %s", endpoint, exc)
        return None


# ── Economic calendar ─────────────────────────────────────────────────────────

def _classify_event(event_name: str) -> Optional[str]:
    """Map a Finnhub event name to our canonical event type, or None."""
    name_lower = event_name.lower()
    for keyword, event_type in _HIGH_IMPACT_KEYWORDS.items():
        if keyword in name_lower:
            return event_type
    return None


def fetch_economic_calendar(
    days_ahead: int = 3,
    min_impact: str = "medium",
) -> list[dict]:
    """
    Return today's (and near-term) economic events from Finnhub.

    Each dict has:
        event_type, event_name, event_time (datetime UTC), importance,
        estimate, prior, actual, country, currency, id
    """
    data = _get("calendar/economic", {})
    if not data:
        return []

    events = data.get("economicCalendar", [])
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc + timedelta(days=days_ahead)
    impact_rank = {"low": 0, "medium": 1, "high": 2}
    min_rank = impact_rank.get(min_impact, 1)

    results = []
    for e in events:
        # Parse event time
        raw_time = e.get("time", "")
        if not raw_time:
            continue
        try:
            # Finnhub returns "2026-04-29 14:00:00" in UTC
            evt_dt = datetime.strptime(raw_time[:19], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        if evt_dt < now_utc - timedelta(hours=1) or evt_dt > cutoff:
            continue

        impact = (e.get("impact") or "low").lower()
        if impact_rank.get(impact, 0) < min_rank:
            continue

        # Classify to canonical type
        event_name = e.get("event", "")
        event_type = _classify_event(event_name)
        if not event_type:
            # Keep high-impact events even without a match
            if impact != "high":
                continue
            event_type = "OTHER_HIGH"

        # Only include US / global events (skip individual country-specific data)
        country = (e.get("country") or "").upper()
        if country and country not in ("US", "UNITED STATES", ""):
            # Keep global events (no country) + US events
            if country not in ("", "US"):
                continue

        uid = hashlib.sha256(f"{event_type}{raw_time}".encode()).hexdigest()[:16]
        results.append({
            "id": uid,
            "event_type": event_type,
            "event_name": event_name,
            "event_time": evt_dt,
            "importance": impact,
            "estimate": e.get("estimate"),
            "prior": e.get("prev"),
            "actual": e.get("actual"),
            "country": country,
            "currency": e.get("unit", ""),
            "source": "finnhub",
        })

    return results


# ── Earnings calendar ─────────────────────────────────────────────────────────

def fetch_earnings_calendar(
    symbols: list[str],
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
) -> dict[str, dict]:
    """
    Return upcoming earnings dates for the given symbols.

    Returns: { symbol -> {date, eps_estimate, revenue_estimate, hour} }
    hour: 'bmo' (before market open) | 'amc' (after market close) | None
    """
    if not symbols:
        return {}

    from_date = from_date or date.today()
    to_date = to_date or (from_date + timedelta(days=14))

    data = _get("calendar/earnings", {
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
    })
    if not data:
        return {}

    symbol_set = {s.upper() for s in symbols}
    results: dict[str, dict] = {}

    for e in data.get("earningsCalendar", []):
        sym = (e.get("symbol") or "").upper()
        if sym not in symbol_set:
            continue
        results[sym] = {
            "date": e.get("date"),
            "eps_estimate": e.get("epsEstimate"),
            "revenue_estimate": e.get("revenueEstimate"),
            "hour": e.get("hour"),   # bmo / amc / dmh (during market hours)
        }

    return results


# ── Company news ──────────────────────────────────────────────────────────────

def fetch_company_news(
    symbol: str,
    lookback_hours: int = 24,
) -> list[dict]:
    """
    Return recent news articles for a symbol from Finnhub.

    Each dict has: headline, summary, source, url, datetime (UTC), sentiment
    Sorted newest-first. Max 20 articles returned to control token budget.
    """
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(hours=lookback_hours)

    data = _get("company-news", {
        "symbol": symbol.upper(),
        "from": from_dt.date().isoformat(),
        "to": to_dt.date().isoformat(),
    })
    if not data or not isinstance(data, list):
        return []

    articles = []
    cutoff_ts = from_dt.timestamp()

    for a in data:
        ts = a.get("datetime", 0)
        if ts < cutoff_ts:
            continue
        articles.append({
            "headline": a.get("headline", ""),
            "summary": a.get("summary", ""),
            "source": a.get("source", ""),
            "url": a.get("url", ""),
            "published_at": datetime.fromtimestamp(ts, tz=timezone.utc),
            "sentiment": a.get("sentiment", ""),   # Finnhub pre-label (positive/negative/neutral)
        })

    # Newest first, cap at 20
    articles.sort(key=lambda x: x["published_at"], reverse=True)
    return articles[:20]
