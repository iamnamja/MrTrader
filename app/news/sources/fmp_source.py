"""Financial Modeling Prep (FMP) data adapter — economic calendar.

FMP's economic calendar is available on the free tier (unlike Finnhub's, which is
premium-only), so it is the primary source for macro-event awareness. Returns the SAME
normalized event-dict schema as ``finnhub_source.fetch_economic_calendar`` so callers are
source-agnostic.

Fails open: returns ``None`` when FMP is unavailable (no key / API error) so a dispatcher
can fall back to another provider; returns ``[]`` on a successful call with no matching
events. Never logs the API key (it rides in the URL query string).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

from app.news.sources.finnhub_source import _classify_event, event_uid

logger = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 8  # seconds
_IMPACT_RANK = {"low": 0, "medium": 1, "high": 2}


# Uses FMP's /stable/economic-calendar (covered by the Starter plan). NOTE: FMP deprecated
# the legacy /api/v3 API on 2025-08-31 — it returns 403 "Legacy Endpoint" for non-legacy
# keys, which was the real cause of the old econ-calendar 403 spam (NOT a missing paid tier);
# fixed by moving to /stable/. We still disable + log ONCE on any 401/403 (e.g. a key that
# genuinely lacks the endpoint) and skip thereafter (no per-poll spam); reset by a process
# restart. Same discipline as finnhub_source._get.
_ECON_CAL_DISABLED = False


def _key() -> Optional[str]:
    from app.config import settings
    return getattr(settings, "fmp_api_key", None)


def fetch_economic_calendar(
    days_ahead: int = 3,
    min_impact: str = "medium",
    *,
    include_past_today: bool = False,
) -> Optional[list[dict]]:
    """Return today's (and near-term) economic events from FMP, normalized.

    Each dict: event_type, event_name, event_time (UTC datetime), importance, estimate,
    prior, actual, country, currency, id, source. Returns ``None`` if FMP is unavailable
    (so a caller may fall back to another provider).

    ``include_past_today`` (default False): normally an event more than 1h in the past is
    dropped (the trading gate wants a forward-looking view — do NOT change that default).
    When True, events from earlier TODAY (already released) are KEPT so their now-available
    ``actual`` can be read — used by the read-only Macro-Intel display back-fill, NOT the gate.
    """
    global _ECON_CAL_DISABLED
    key = _key()
    if not key:
        logger.debug("FMP key not configured — skipping economic calendar")
        return None
    if _ECON_CAL_DISABLED:
        return None

    today = date.today()
    to_date = today + timedelta(days=max(0, days_ahead))
    try:
        r = requests.get(
            f"{_BASE}/economic-calendar",
            params={"from": today.isoformat(), "to": to_date.isoformat(), "apikey": key},
            timeout=_TIMEOUT,
        )
        # Log status only — the URL/exception carries ?apikey=...
        if r.status_code in (401, 403):
            _ECON_CAL_DISABLED = True
            logger.warning(
                "FMP economic-calendar returned %d — the configured key lacks access to this "
                "/stable endpoint. Disabling further calls this session; restart after fixing "
                "the key/plan to re-enable.",
                r.status_code,
            )
            return None
        if r.status_code != 200:
            logger.warning("FMP economic_calendar failed: HTTP %s (transient)", r.status_code)
            return None
        data = r.json()
    except Exception as exc:
        logger.warning("FMP economic_calendar failed: %s", type(exc).__name__)
        return None

    if not isinstance(data, list):
        return None

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc + timedelta(days=days_ahead)
    # Lower bound: normally "1h ago" (forward-looking gate view); with include_past_today, midnight
    # UTC today so already-released same-day events (and their actuals) are kept for the display.
    lower_bound = (now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                   if include_past_today else now_utc - timedelta(hours=1))
    min_rank = _IMPACT_RANK.get(min_impact, 1)

    results: list[dict] = []
    for e in data:
        raw_time = e.get("date", "")
        if not raw_time:
            continue
        try:
            # FMP dates come as "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD" (UTC).
            fmt = "%Y-%m-%d %H:%M:%S" if len(raw_time) > 10 else "%Y-%m-%d"
            evt_dt = datetime.strptime(raw_time[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if evt_dt < lower_bound or evt_dt > cutoff:
            continue

        impact = (e.get("impact") or "low").lower()
        if _IMPACT_RANK.get(impact, 0) < min_rank:
            continue

        event_name = e.get("event", "")
        event_type = _classify_event(event_name)
        if not event_type:
            if impact != "high":
                continue
            event_type = "OTHER_HIGH"

        country = (e.get("country") or "").upper()
        if country and country not in ("US", "UNITED STATES", ""):
            continue

        uid = event_uid(event_type, event_name, raw_time)
        results.append({
            "id": uid,
            "event_type": event_type,
            "event_name": event_name,
            "event_time": evt_dt,
            "importance": impact,
            "estimate": e.get("estimate"),
            "prior": e.get("previous"),   # FMP uses "previous" (Finnhub uses "prev")
            "actual": e.get("actual"),
            "country": country,
            "currency": e.get("currency", ""),
            "source": "fmp",
        })

    return results
