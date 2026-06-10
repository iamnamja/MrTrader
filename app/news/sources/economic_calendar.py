"""Provider-agnostic economic-calendar lookup.

FMP is the PRIMARY source (its economic calendar works on the free tier); Finnhub is a
FALLBACK for deployments that have a premium Finnhub token. Both providers return the same
normalized event-dict schema, so callers import only this function and stay source-agnostic.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def fetch_economic_calendar(
    days_ahead: int = 1,
    min_impact: str = "medium",
) -> list[dict]:
    """Return near-term economic events, preferring FMP and falling back to Finnhub.

    FMP returns ``None`` only when it is unavailable (no key / API error) — a legitimate
    empty day returns ``[]`` and is used as-is (no needless fallback call).
    """
    from app.news.sources import fmp_source, finnhub_source

    events = fmp_source.fetch_economic_calendar(days_ahead=days_ahead, min_impact=min_impact)
    if events is not None:
        return events

    # FMP unavailable — fall back to Finnhub (only yields data with a premium token).
    logger.debug("Economic calendar: FMP unavailable, falling back to Finnhub")
    return finnhub_source.fetch_economic_calendar(days_ahead=days_ahead, min_impact=min_impact)
