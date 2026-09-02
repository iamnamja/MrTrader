"""Uncredited-dividend accrual for the paper book (2026-09-02).

WHY
---
Alpaca paper credits NO dividends (verified: the account has zero DIV activities). Marking is
price-only, so every distribution shows up as an ex-day price drop with no offsetting cash. The
paper record therefore understates the book by the full distribution stream.

It is NOT only the cash sleeve. Over 2026-06-17..2026-09-02 the held book distributed:

    SGOV  0.9100/share  (3 payments)   <- cash sleeve; its entire return IS the distribution
    SPY   1.9040/share  (1 payment)    <- trend sleeve
    QQQ   0.8130/share  (1 payment)    <- trend sleeve

WHAT THIS DOES — AND DELIBERATELY DOES NOT DO
---------------------------------------------
It records what a live account WOULD have received, as a SEPARATE figure. It is never folded into
`daily_pnl`, because that series has to keep reconciling against the broker's own NAV — and that
reconciliation is the only thing that caught the three construction errors documented in
`docs/reference/PNL_TRACKING_SCOPE_2026-09-01.md`. A scorecard that quietly disagrees with the
broker is worth less than one that is visibly distorted in a known, quantified way.

So the scorecard can report two honest numbers:

    paper     = what the broker shows        (reconciles, understated)
    economic  = paper + uncredited dividends (what live capital would have earned)

The gap closes by itself the moment real money is used.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

log = logging.getLogger(__name__)

# Cache, keyed by (symbols, start, TODAY). The date component is load-bearing, not cosmetic: the
# app process runs for days or weeks, so a cache keyed only on (symbols, start) would pin the
# dividend history at whatever was known when the process booted — and a newly-declared ex-div
# would never appear. Including today's date expires it naturally once per day, which matches how
# often the data can change, while still collapsing the backfill's per-sleeve re-fetches.
_CACHE: Dict[tuple, Dict[str, Dict[str, float]]] = {}


def clear_cache() -> None:
    """Drop the memo. Used by tests; also a manual escape hatch if a vendor correction lands."""
    _CACHE.clear()


def fetch_dividends(symbols: Iterable[str], start: str) -> Dict[str, Dict[str, float]]:
    """``{symbol: {ex_date: per_share_amount}}`` from yfinance, on or after `start`.

    Never raises — a dividend lookup failing must not break P&L reconstruction, which is the
    series that actually reconciles. A missing symbol simply accrues nothing and is logged.
    """
    from datetime import date as _date
    syms = tuple(sorted({s for s in symbols if s}))
    key = (syms, str(start)[:10], _date.today().isoformat())
    if key in _CACHE:
        return _CACHE[key]

    out: Dict[str, Dict[str, float]] = {}
    try:
        import yfinance as yf
        for sym in syms:
            try:
                ser = yf.Ticker(sym).dividends
                if ser is None or len(ser) == 0:
                    continue
                # Depending on the yfinance version this is a Series OR a single-column
                # DataFrame; .items() on the latter yields (column_name, Series), which then
                # blows up on a truthiness test. Normalize to a Series first.
                if hasattr(ser, "columns"):
                    col = "Dividends" if "Dividends" in ser.columns else ser.columns[0]
                    ser = ser[col]
                ser = ser[ser.index >= start]
                per_date = {}
                for idx, val in ser.items():
                    try:
                        amt = float(val)
                    except (TypeError, ValueError):
                        continue
                    if amt > 0:
                        per_date[str(idx)[:10]] = amt
                if per_date:
                    out[sym] = per_date
            except Exception as exc:  # noqa: BLE001
                log.warning("dividends: lookup failed for %s: %s", sym, exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("dividends: yfinance unavailable (%s) — accrual will be 0", exc)

    _CACHE[key] = out
    return out


def accrual_for(positions: Dict[str, float], date: str,
                divs: Optional[Dict[str, Dict[str, float]]]) -> float:
    """Dollars a live account would have received on `date` for this book.

    Uses the position held ON the ex-date. That is the correct convention (the holder of record on
    the ex-date receives the distribution) and it is also what makes the accrual consistent with
    the price drop the paper account DID take on that same day.
    """
    if not divs or not positions:
        return 0.0
    d = str(date)[:10]
    total = 0.0
    for sym, qty in positions.items():
        per_share = (divs.get(sym) or {}).get(d)
        if per_share:
            total += float(qty) * float(per_share)
    return total
