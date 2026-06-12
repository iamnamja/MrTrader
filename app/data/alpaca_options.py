"""
Alpaca options market-data client (FUSE A — Alpha-v6 P1c slow fuse).

Why Alpaca and not Polygon: the Polygon $79 plan does NOT serve options NBBO —
``/v3/snapshot/options/{u}`` returns ``last_quote: null`` and ``/v3/quotes/{ticker}``
is 403 NOT_AUTHORIZED. Alpaca's ``/v1beta1/options/snapshots/{underlying}`` DOES
return real bid/ask on the free ``indicative`` feed (possibly delayed/synthesized —
fine for nightly spread-STRUCTURE calibration, not for execution). The feed name is
recorded in every observation row so the quality caveat stays auditable.

This module is a thin requests wrapper: pagination + field passthrough only. Row
flattening / schema decisions live in scripts/log_options_nbbo.py.

Snapshot payload shape (per OCC ticker, keys WITHOUT the ``O:`` prefix):
    {"latestQuote": {"bp": bid, "ap": ask, "bs": bid_size, "as": ask_size,
                     "bx": bid_exch, "ax": ask_exch, "t": ts},
     "latestTrade": {...}, "dailyBar": {"c":..,"v":..,...},
     "greeks": {...}?, "impliedVolatility": ..?}
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_DATA_BASE = "https://data.alpaca.markets"
_SNAPSHOT_PATH = "/v1beta1/options/snapshots/{underlying}"
_STOCK_TRADES_LATEST_PATH = "/v2/stocks/trades/latest"
_PAGE_LIMIT = 1000


def _headers() -> Dict[str, str]:
    from app.config import settings
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def fetch_option_snapshots(
    underlying: str,
    *,
    feed: str = "indicative",
    exp_lo: Optional[date] = None,
    exp_hi: Optional[date] = None,
    max_pages: int = 40,
    limit: int = _PAGE_LIMIT,
    session=None,
    timeout: int = 30,
) -> Dict[str, dict]:
    """Full option-chain snapshot for one underlying -> {occ_ticker: snapshot_dict}.

    ``feed="indicative"`` is the free-tier feed (the paid one is ``opra``).
    ``exp_lo``/``exp_hi`` bound expiration server-side (caps pagination — a full SPY
    chain is tens of thousands of contracts). Paginates via ``next_page_token`` up to
    ``max_pages`` (logged if the cap truncates, so a silent partial chain can't hide).
    Raises on HTTP error (caller decides whether one underlying failing is fatal).
    """
    sess = session or requests
    url = _DATA_BASE + _SNAPSHOT_PATH.format(underlying=underlying.upper())
    params: Dict[str, object] = {"feed": feed, "limit": limit}
    if exp_lo is not None:
        params["expiration_date_gte"] = exp_lo.isoformat()
    if exp_hi is not None:
        params["expiration_date_lte"] = exp_hi.isoformat()

    out: Dict[str, dict] = {}
    for page in range(max_pages):
        r = sess.get(url, headers=_headers(), params=params, timeout=timeout)
        r.raise_for_status()
        body = r.json() or {}
        out.update(body.get("snapshots") or {})
        token = body.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    else:
        logger.warning("fetch_option_snapshots(%s): hit max_pages=%d — chain truncated",
                       underlying, max_pages)
    return out


def fetch_latest_underlying_prices(
    symbols: List[str], *, feed: str = "iex", session=None, timeout: int = 20,
) -> Dict[str, float]:
    """Latest trade price per underlying (one batched request; IEX feed = free tier).

    Returns only the symbols that came back with a positive price — callers must
    tolerate missing keys (moneyness then records as NaN rather than a fake spot).
    """
    sess = session or requests
    r = sess.get(_DATA_BASE + _STOCK_TRADES_LATEST_PATH, headers=_headers(),
                 params={"symbols": ",".join(s.upper() for s in symbols), "feed": feed},
                 timeout=timeout)
    r.raise_for_status()
    trades = (r.json() or {}).get("trades") or {}
    out: Dict[str, float] = {}
    for sym, t in trades.items():
        try:
            px = float(t.get("p") or 0.0)
        except (TypeError, ValueError):
            continue
        if px > 0:
            out[sym] = px
    return out
