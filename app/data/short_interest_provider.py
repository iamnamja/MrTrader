"""
Short interest & short volume provider (FINRA-sourced, via Polygon).

Design + point-in-time contract: docs/reference/SHORT_INTEREST_DATA.md

Point-in-time correctness (THE leak-killer)
-------------------------------------------
Short interest is *settlement-dated* but only disseminated by FINRA ~8 business
days later -- the value does not exist publicly before then. We store an explicit
``knowable_date = settlement_date + SI_PUBLICATION_LAG_BDAYS`` (conservative; the
actual lag is ~8 bdays, +10 is always >= actual so we never act early) and EVERY
accessor filters ``knowable_date <= as_of``. Trading a settlement-dated value
before dissemination would be a look-ahead leak.

Short *volume* (Reg SHO daily) publishes the next business day, so its lag is +1.

Source: Polygon ``/stocks/v1/short-interest`` and ``/stocks/v1/short-volume``
(both on our plan; FINRA-originated). FMP does not offer these on our plan.
"""

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_REST_BASE = "https://api.polygon.io"
_SI_PATH = "/stocks/v1/short-interest"
_SV_PATH = "/stocks/v1/short-volume"
_PAGE_LIMIT = 1000

# Conservative publication lags (business days). FINRA disseminates short interest
# ~8 bdays after settlement; +10 carries a 2-bday buffer that absorbs the (at most
# one) market holiday in a ~2-week window, so knowable_date is never optimistic.
# Reg SHO daily short-volume files publish next business day -> +1.
SI_PUBLICATION_LAG_BDAYS = 10
SV_PUBLICATION_LAG_BDAYS = 1

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SI_PARQUET = _DATA_DIR / "short_interest.parquet"
SV_PARQUET = _DATA_DIR / "short_volume.parquet"

_SI_COLS = ["ticker", "settlement_date", "knowable_date",
            "short_interest", "avg_daily_volume", "days_to_cover"]
_SV_COLS = ["ticker", "date", "knowable_date",
            "short_volume", "total_volume", "exempt_volume", "short_volume_ratio"]

# In-process store caches (parquet loaded once per process).
_si_store: Optional[pd.DataFrame] = None
_sv_store: Optional[pd.DataFrame] = None


def _api_key() -> str:
    from app.config import settings
    return settings.polygon_api_key or ""


def knowable_date(d: date, lag_bdays: int) -> date:
    """First date a settlement/trade-dated value is publicly knowable.

    Uses weekday business days (Mon-Fri). The +2 buffer baked into the SI lag
    (10 vs the ~8 actual) absorbs holidays, keeping the result conservative.
    """
    return (pd.Timestamp(d) + pd.offsets.BDay(lag_bdays)).date()


# ── Fetch (Polygon) ───────────────────────────────────────────────────────────

def _fetch_paginated(path: str, ticker: str, extra: Optional[dict] = None,
                     max_pages: int = 500) -> List[Dict]:
    key = _api_key()
    if not key:
        logger.warning("POLYGON_API_KEY missing; cannot fetch %s for %s", path, ticker)
        return []
    url = f"{_REST_BASE}{path}"
    params = {"ticker": ticker, "limit": _PAGE_LIMIT, "apiKey": key}
    if extra:
        params.update(extra)
    out: List[Dict] = []
    pages = 0
    retries = 0
    _MAX_RETRIES = 6
    while url and pages < max_pages:
        resp = requests.get(url, params=params, timeout=25)
        if resp.status_code == 429:
            if retries >= _MAX_RETRIES:
                logger.warning("rate-limited %d× on %s for %s; returning partial",
                               retries, path, ticker)
                break
            retries += 1
            time.sleep(min(1.5 * retries, 10.0))  # linear backoff, capped
            continue
        resp.raise_for_status()
        retries = 0
        data = resp.json()
        out.extend(data.get("results", []) or [])
        url = data.get("next_url")
        params = {"apiKey": key}  # next_url already carries ticker/sort/cursor
        pages += 1
    return out


def fetch_short_interest(ticker: str) -> List[Dict]:
    """Full bi-monthly short-interest history for *ticker* (raw Polygon rows)."""
    return _fetch_paginated(_SI_PATH, ticker, {"sort": "settlement_date.asc"})


def fetch_short_volume(ticker: str, start: Optional[date] = None) -> List[Dict]:
    """Daily Reg SHO short-volume history for *ticker* (raw Polygon rows)."""
    extra = {"sort": "date.asc"}
    if start:
        extra["date.gte"] = start.isoformat()
    return _fetch_paginated(_SV_PATH, ticker, extra)


# ── Raw rows -> normalized, knowable-stamped DataFrames ────────────────────────

def short_interest_to_df(ticker: str, rows: List[Dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        sd = r.get("settlement_date")
        if not sd:
            continue
        sdt = datetime.strptime(sd, "%Y-%m-%d").date()
        recs.append({
            "ticker": ticker,
            "settlement_date": pd.Timestamp(sdt),
            "knowable_date": pd.Timestamp(knowable_date(sdt, SI_PUBLICATION_LAG_BDAYS)),
            "short_interest": _f(r.get("short_interest")),
            "avg_daily_volume": _f(r.get("avg_daily_volume")),
            "days_to_cover": _f(r.get("days_to_cover")),
        })
    df = pd.DataFrame(recs, columns=_SI_COLS)
    if not df.empty:
        df = df.sort_values("settlement_date").reset_index(drop=True)
    return df


def short_volume_to_df(ticker: str, rows: List[Dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        d = r.get("date")
        if not d:
            continue
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        recs.append({
            "ticker": ticker,
            "date": pd.Timestamp(dt),
            "knowable_date": pd.Timestamp(knowable_date(dt, SV_PUBLICATION_LAG_BDAYS)),
            "short_volume": _f(r.get("short_volume")),
            "total_volume": _f(r.get("total_volume")),
            "exempt_volume": _f(r.get("exempt_volume")),
            "short_volume_ratio": _f(r.get("short_volume_ratio")),  # percent (e.g. 46.7)
        })
    df = pd.DataFrame(recs, columns=_SV_COLS)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ── Store load (parquet, in-process cached) ───────────────────────────────────

def _empty(cols) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


def load_short_interest(refresh: bool = False) -> pd.DataFrame:
    global _si_store
    if _si_store is None or refresh:
        if SI_PARQUET.exists():
            _si_store = pd.read_parquet(SI_PARQUET)
        else:
            _si_store = _empty(_SI_COLS)
    return _si_store


def load_short_volume(refresh: bool = False) -> pd.DataFrame:
    global _sv_store
    if _sv_store is None or refresh:
        if SV_PARQUET.exists():
            _sv_store = pd.read_parquet(SV_PARQUET)
        else:
            _sv_store = _empty(_SV_COLS)
    return _sv_store


# ── Point-in-time accessors (mirror fmp_provider.get_*_at) ─────────────────────

def get_short_interest_at(symbol: str, as_of: date,
                          store: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """Most recent short-interest record publicly knowable on or before *as_of*.

    Returns None if nothing is knowable yet. ``si_change_pct`` is the change vs the
    prior (also-knowable) settlement -- the SI build/cover signal.
    """
    df = store if store is not None else load_short_interest()
    if df is None or df.empty:
        return None
    as_of_ts = pd.Timestamp(as_of)
    sub = df[(df["ticker"] == symbol) & (df["knowable_date"] <= as_of_ts)]
    if sub.empty:
        return None
    sub = sub.sort_values("settlement_date")
    cur = sub.iloc[-1]
    si_change_pct = None
    if len(sub) >= 2:
        prev = float(sub.iloc[-2]["short_interest"])
        if prev and not np.isnan(prev):
            si_change_pct = float((float(cur["short_interest"]) - prev) / prev)
    return {
        "short_interest": float(cur["short_interest"]),
        "avg_daily_volume": float(cur["avg_daily_volume"]),
        "days_to_cover": float(cur["days_to_cover"]),
        "si_change_pct": si_change_pct,
        "settlement_date": pd.Timestamp(cur["settlement_date"]).date(),
        "knowable_date": pd.Timestamp(cur["knowable_date"]).date(),
    }


def get_short_volume_features_at(symbol: str, as_of: date, lookback_days: int = 20,
                                 store: Optional[pd.DataFrame] = None) -> Dict:
    """Trailing short-volume-ratio features over rows knowable on/before *as_of*.

    short_volume_ratio is a percent (0-100). Returns last / mean / z-score over the
    trailing ``lookback_days`` knowable trading days; None values if no history.
    """
    out = {"sv_ratio_last": None, "sv_ratio_mean": None, "sv_ratio_z": None, "n_obs": 0}
    df = store if store is not None else load_short_volume()
    if df is None or df.empty:
        return out
    as_of_ts = pd.Timestamp(as_of)
    # calendar buffer so we still net ~lookback_days trading rows
    start_ts = as_of_ts - pd.Timedelta(days=lookback_days * 2 + 10)
    sub = df[(df["ticker"] == symbol)
             & (df["knowable_date"] <= as_of_ts)
             & (df["date"] >= start_ts)]
    if sub.empty:
        return out
    sub = sub.sort_values("date")
    ratios = sub["short_volume_ratio"].astype(float).to_numpy()
    ratios = ratios[~np.isnan(ratios)][-lookback_days:]
    if ratios.size == 0:
        return out
    last = float(ratios[-1])
    mean = float(np.mean(ratios))
    std = float(np.std(ratios))
    out.update(
        sv_ratio_last=last,
        sv_ratio_mean=mean,
        sv_ratio_z=float((last - mean) / std) if std > 1e-9 else 0.0,
        n_obs=int(ratios.size),
    )
    return out


def latest_known_si(as_of: date, store: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Cross-section of the latest short-interest record knowable on/before *as_of*,
    one row per ticker. Efficient (one filter + groupby) for factor-ranking scorers
    that need the whole universe each scoring day, rather than per-symbol calls.

    PIT-safe: filters knowable_date <= as_of, then takes each ticker's latest
    settlement. Columns are _SI_COLS (ticker, settlement_date, knowable_date,
    short_interest, avg_daily_volume, days_to_cover).
    """
    df = store if store is not None else load_short_interest()
    if df is None or df.empty:
        return _empty(_SI_COLS)
    sub = df[df["knowable_date"] <= pd.Timestamp(as_of)]
    if sub.empty:
        return _empty(_SI_COLS)
    idx = sub.groupby("ticker")["settlement_date"].idxmax()
    return sub.loc[idx].reset_index(drop=True)
