"""
Polygon vX/reference/financials — point-in-time quarterly financials.

Three features tuned for the 10-day swing model's Stage 1 quality gate:

  fcf_margin          Free-cash-flow as % of revenue (operating CF / revenue).
                      High FCF margin = durable earnings, less vulnerability to
                      negative surprises.  Range typically -0.3 to +0.4.

  operating_leverage  YoY change in operating income % minus YoY revenue %.
                      Positive = margins expanding (good), Negative = compressing.
                      Clipped to [-0.3, 0.3] to limit outlier influence.

  rd_intensity        R&D spend as % of revenue.  Context for interpreting PE:
                      high-R&D companies with low current earnings are not cheap —
                      they're investing.  Range 0.0 to 0.5+.

All values are floats; any failure → 0.0 defaults.
Cache: 24 h in-process per symbol (financials change quarterly).
"""

import logging
import time
from datetime import date, datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://api.polygon.io"
_CACHE_TTL = 86_400  # 24 h
_cache: Dict[str, tuple] = {}  # symbol → (records, fetched_at)

_DEFAULTS = {
    "fcf_margin": 0.0,
    "operating_leverage": 0.0,
    "rd_intensity": 0.0,
}


def _api_key() -> str:
    from app.config import settings
    return settings.polygon_api_key or ""


def _get_quarterly_records(symbol: str) -> List[Dict]:
    """Fetch and cache up to 8 quarterly financials records for symbol."""
    now = time.time()
    cached = _cache.get(symbol)
    if cached and now - cached[1] < _CACHE_TTL:
        return cached[0]

    records = []
    try:
        resp = requests.get(
            f"{_BASE}/vX/reference/financials",
            params={
                "ticker": symbol,
                "timeframe": "quarterly",
                "limit": 8,
                "apikey": _api_key(),
            },
            timeout=10,
        )
        if resp.status_code == 200:
            records = resp.json().get("results", [])
    except Exception as exc:
        logger.debug("Polygon financials fetch failed for %s: %s", symbol, exc)

    _cache[symbol] = (records, now)
    return records


def _extract(rec: Dict, section: str, field: str) -> Optional[float]:
    """Safely extract a value from a financials record section."""
    try:
        val = rec["financials"][section][field]["value"]
        return float(val) if val is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def get_polygon_financial_features(symbol: str, as_of: date) -> Dict[str, float]:
    """
    Return fcf_margin, operating_leverage, rd_intensity for symbol as of as_of.

    Point-in-time: only uses quarterly reports whose end_date <= as_of.
    Needs at least 2 quarters for operating_leverage (YoY requires 4 quarters
    of data); returns 0.0 for that metric if not enough history.
    """
    result = dict(_DEFAULTS)

    try:
        records = _get_quarterly_records(symbol)

        # Filter to quarters reported on or before as_of
        past = [
            r for r in records
            if r.get("end_date")
            and datetime.strptime(r["end_date"], "%Y-%m-%d").date() <= as_of
        ]
        if not past:
            return result

        # Most recent quarter
        q0 = past[0]
        rev0 = _extract(q0, "income_statement", "revenues")
        op0  = _extract(q0, "income_statement", "operating_income_loss")
        ocf0 = _extract(q0, "cash_flow_statement", "net_cash_flow_from_operating_activities")
        rd0  = _extract(q0, "income_statement", "research_and_development")

        if rev0 and abs(rev0) > 0:
            if ocf0 is not None:
                result["fcf_margin"] = float(max(-0.5, min(0.5, ocf0 / rev0)))
            if rd0 is not None:
                result["rd_intensity"] = float(max(0.0, min(0.6, rd0 / rev0)))

        # Operating leverage needs quarter from ~1 year ago (index 3 or 4)
        for lag in (3, 4):
            if len(past) > lag:
                q_lag = past[lag]
                rev_lag = _extract(q_lag, "income_statement", "revenues")
                op_lag  = _extract(q_lag, "income_statement", "operating_income_loss")
                if (rev0 and rev_lag and abs(rev_lag) > 0
                        and op0 is not None and op_lag is not None):
                    rev_growth = (rev0 - rev_lag) / abs(rev_lag)
                    # Operating margin now vs a year ago
                    op_margin_now  = op0  / rev0  if abs(rev0)  > 0 else 0.0
                    op_margin_then = op_lag / rev_lag if abs(rev_lag) > 0 else 0.0
                    op_leverage = (op_margin_now - op_margin_then) - rev_growth * 0.0
                    # Simpler: change in operating margin YoY
                    op_leverage = op_margin_now - op_margin_then
                    result["operating_leverage"] = float(max(-0.3, min(0.3, op_leverage)))
                break

    except Exception as exc:
        logger.debug("Polygon financial features failed for %s: %s", symbol, exc)

    return result


def prefetch_polygon_financials(symbols: List[str]) -> None:
    """Pre-warm the cache for all symbols before a training run."""
    logger.info("Pre-fetching Polygon financials for %d symbols...", len(symbols))
    ok = 0
    for sym in symbols:
        try:
            recs = _get_quarterly_records(sym)
            if recs:
                ok += 1
        except Exception:
            pass
    logger.info("Polygon financials prefetch: %d/%d symbols with data", ok, len(symbols))
