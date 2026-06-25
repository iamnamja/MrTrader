"""Provider-agnostic economic-calendar lookup.

FMP is the PRIMARY source (its economic calendar works on the free tier); Finnhub is a
FALLBACK for deployments that have a premium Finnhub token. Both providers return the same
normalized event-dict schema, so callers import only this function and stay source-agnostic.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_economic_calendar(
    days_ahead: int = 1,
    min_impact: str = "medium",
) -> Optional[list[dict]]:
    """Return near-term economic events, preferring FMP and falling back to Finnhub.

    Returns:
      - ``list[dict]`` (possibly ``[]``) when a source AUTHORITATIVELY reports the calendar —
        FMP is primary and ``[]`` from a reachable FMP means a genuine no-events day.
      - ``None`` when the calendar is UNAVAILABLE (FMP unreachable AND Finnhub yields nothing).
        Callers MUST treat None as "macro state unknown" and apply a conservative stance — NOT as
        "no high-impact events" (that conflation is a fail-OPEN: a transient outage would let the
        book trade freely into an unverified macro event).
    """
    from app.news.sources import fmp_source, finnhub_source

    events = fmp_source.fetch_economic_calendar(days_ahead=days_ahead, min_impact=min_impact)
    if events is not None:
        return events                       # FMP reachable: [] = genuine no-events day

    # FMP unavailable — try Finnhub (only yields data with a premium token; free tier returns []).
    logger.debug("Economic calendar: FMP unavailable, falling back to Finnhub")
    fallback = finnhub_source.fetch_economic_calendar(days_ahead=days_ahead, min_impact=min_impact)
    if fallback:
        return fallback                     # premium Finnhub returned real events
    # BOTH sources unavailable — the calendar is genuinely UNKNOWN (Finnhub [] on the free tier is
    # indistinguishable from "unavailable", so we must NOT report it as a confirmed no-events day).
    return None
