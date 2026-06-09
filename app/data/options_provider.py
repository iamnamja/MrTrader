"""
Options data provider (Polygon Developer) — OPT-1b.

Design + point-in-time contract: docs/reference/OPTIONS_DATA.md
Implements the OPT-0 ``OptionsDataProvider`` Protocol (app/options/contracts.py).

Point-in-time + survivorship correctness (the two leak-killers)
---------------------------------------------------------------
1. **PIT.** An option's EOD bar for trade date D is published after the close, so it is
   only *knowable* the next business day. We stamp ``knowable_date = D + 1 bday`` and EVERY
   historical accessor filters ``knowable_date <= as_of``. Using a bar before it printed
   (or a contract before it listed) would be look-ahead.
2. **Survivorship.** The universe is built FROM the OPRA daily flat files (every contract
   that *actually traded* that day, expired ones included), not from the current active
   chain. A backtest as-of date therefore sees exactly the contracts that existed then —
   including those that have since expired worthless. (The current REST chain is used ONLY
   for the live snapshot / engine validation, never for backtests.)

Polygon Developer serves NO historical IV/greeks/OI — those are computed by
app/options/pricing_engine.py. This layer carries only PIT OHLCV + the contract universe.

Data sources:
  * Historical OHLCV + universe: S3 flat files ``us_options_opra/day_aggs_v1`` (via
    app/data/polygon_s3.PolygonS3.get_options_day_file), assembled by scripts/backfill_options.py.
  * Current chain snapshot (IV/greeks/OI): REST ``/v3/snapshot/options/{underlying}``.
  * Current/active + expired contract reference: REST ``/v3/reference/options/contracts``.
"""

import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, GoodFriday,
                                    USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
                                    USLaborDay, USThanksgivingDay, nearest_workday)
from pandas.tseries.offsets import CustomBusinessDay

logger = logging.getLogger(__name__)

_REST_BASE = "https://api.polygon.io"
_CONTRACTS_PATH = "/v3/reference/options/contracts"
_PAGE_LIMIT = 1000

# An option's EOD bar prints after the close -> knowable the next TRADING day.
OPT_BAR_LAG_BDAYS = 1


class _NYSECalendar(AbstractHolidayCalendar):
    """NYSE holiday rules (differs from the federal calendar: observes Good Friday;
    does NOT observe Columbus/Veterans Day). Enough to land knowable_date on a real
    trading day rather than a closed session."""
    rules = [
        Holiday("NewYear", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2021-06-19",
                observance=nearest_workday),
        Holiday("Independence", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_NYSE_BDAY = CustomBusinessDay(calendar=_NYSECalendar())


def knowable_date(d: date, lag_bdays: int = OPT_BAR_LAG_BDAYS) -> date:
    """First date an option bar trade-dated *d* is publicly knowable: *lag_bdays* NYSE
    trading days after *d* (holiday-aware, so it never lands on a closed session). The
    +1 default is exact for an EOD bar that prints after the close. This is the
    leak-killer the whole PIT contract rests on (docs/reference/OPTIONS_DATA.md §4)."""
    return (pd.Timestamp(d) + lag_bdays * _NYSE_BDAY).date()


_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OPTIONS_BARS_PARQUET = _DATA_DIR / "options_bars.parquet"
OPTIONS_CONTRACTS_PARQUET = _DATA_DIR / "options_contracts.parquet"

BARS_COLS = ["underlying", "contract", "date", "open", "high", "low", "close",
             "volume", "knowable_date"]
CONTRACT_COLS = ["underlying", "contract", "contract_type", "strike", "expiration",
                 "first_date", "knowable_date"]

# OCC option symbol: O:{ROOT}{YYMMDD}{C|P}{strike*1000, 8 digits}
_OCC_RE = re.compile(r"^O:([A-Z]+)(\d{6})([CP])(\d{8})$")

# In-process parquet caches.
_bars_store: Optional[pd.DataFrame] = None
_contracts_store: Optional[pd.DataFrame] = None


def _api_key() -> str:
    from app.config import settings
    return settings.polygon_api_key or ""


# ── OCC parsing ────────────────────────────────────────────────────────────────

def parse_occ(contract: str) -> Optional[Dict[str, Any]]:
    """Decode an OCC option ticker into its parts, or None if malformed.

    >>> parse_occ("O:SPY260116C00500000")
    {'underlying': 'SPY', 'contract_type': 'call', 'strike': 500.0,
     'expiration': datetime.date(2026, 1, 16)}
    """
    m = _OCC_RE.match(contract or "")
    if not m:
        return None
    root, ymd, cp, strike8 = m.groups()
    try:
        # %y maps 00-68 -> 2000-2068, 69-99 -> 1969-1999. OPRA's O: 2-digit-year format
        # has only existed since ~2010 and no listed expiration reaches 2069, so there is
        # no collision today; revisit if expirations ever exceed 2068.
        exp = datetime.strptime(ymd, "%y%m%d").date()
    except ValueError:
        return None
    return {
        "underlying": root,
        "contract_type": "call" if cp == "C" else "put",
        "strike": int(strike8) / 1000.0,
        "expiration": exp,
    }


def occ_root_pattern(underlyings: List[str]) -> re.Pattern:
    """Regex matching OCC tickers for the given underlying roots. The trailing
    ``\\d{6}`` disambiguates roots that are prefixes of one another (SPY vs SPYG):
    only SPY *followed by the 6-digit date* matches. Group 1 captures the root."""
    roots = "|".join(re.escape(u.upper()) for u in underlyings)
    return re.compile(rf"^O:({roots})(\d{{6}})([CP])(\d{{8}})$")


# ── REST fetch (current chain / reference) ─────────────────────────────────────

def _fetch_paginated(path: str, params: Dict[str, Any], max_pages: int = 200) -> List[Dict]:
    key = _api_key()
    if not key:
        logger.warning("POLYGON_API_KEY missing; cannot fetch %s", path)
        return []
    url = f"{_REST_BASE}{path}"
    params = {**params, "limit": _PAGE_LIMIT, "apiKey": key}
    out: List[Dict] = []
    pages = retries = 0
    _MAX_RETRIES = 6
    while url and pages < max_pages:
        resp = requests.get(url, params=params, timeout=25)
        if resp.status_code == 429:
            if retries >= _MAX_RETRIES:
                logger.warning("rate-limited %d× on %s; returning partial", retries, path)
                break
            retries += 1
            time.sleep(min(1.5 * retries, 10.0))
            continue
        resp.raise_for_status()
        retries = 0
        data = resp.json()
        out.extend(data.get("results", []) or [])
        url = data.get("next_url")
        params = {"apiKey": key}  # next_url carries the cursor + original filters
        pages += 1
    return out


def fetch_contracts(underlying: str, as_of: Optional[date] = None,
                    expired: bool = True) -> List[Dict]:
    """Raw Polygon contract-reference rows for *underlying*. as_of returns the chain as
    listed on that date; expired=True includes already-expired contracts (survivorship)."""
    params: Dict[str, Any] = {"underlying_ticker": underlying.upper(),
                              "expired": str(expired).lower(), "sort": "expiration_date"}
    if as_of:
        params["as_of"] = as_of.isoformat()
    return _fetch_paginated(_CONTRACTS_PATH, params)


def fetch_current_snapshot(underlying: str,
                           exp_lo: Optional[date] = None,
                           exp_hi: Optional[date] = None) -> Dict[str, Dict[str, Any]]:
    """CURRENT chain snapshot {contract: {implied_volatility, greeks, open_interest,
    day_close, underlying_price, ...}}. Validation / live ONLY — carries Polygon-served
    IV/greeks/OI that do NOT exist historically. Never use in a backtest."""
    key = _api_key()
    out: Dict[str, Dict[str, Any]] = {}
    if not key:
        return out
    params: Dict[str, Any] = {"apiKey": key, "limit": 250}
    if exp_lo:
        params["expiration_date.gte"] = exp_lo.isoformat()
    if exp_hi:
        params["expiration_date.lte"] = exp_hi.isoformat()
    url = f"{_REST_BASE}/v3/snapshot/options/{underlying.upper()}"
    pages = 0
    while url and pages < 50:
        resp = requests.get(url, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        for c in (data.get("results") or []):
            d = c.get("details") or {}
            ticker = d.get("ticker")
            if not ticker:
                continue
            day = c.get("day") or {}
            out[ticker] = {
                "implied_volatility": c.get("implied_volatility"),
                "greeks": c.get("greeks") or {},
                "open_interest": c.get("open_interest"),
                "day_close": day.get("close"),
                "day_volume": day.get("volume"),
                "underlying_price": (c.get("underlying_asset") or {}).get("price"),
                "strike": d.get("strike_price"),
                "expiration": d.get("expiration_date"),
                "contract_type": d.get("contract_type"),
            }
        url = data.get("next_url")
        params = {"apiKey": key}
        pages += 1
    return out


# ── Store load (parquet, in-process cached) ────────────────────────────────────

def _empty(cols) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


def _coerce_dt(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Force date columns to datetime64[ns] so the PIT `<=` filter can never silently
    depend on (or crash on) an upstream object/string dtype, and so concat with
    freshly-stamped (ns) bars doesn't hit a resolution mismatch (parquet round-trips
    datetimes as ms)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    return df


def load_options_bars(refresh: bool = False) -> pd.DataFrame:
    global _bars_store
    if _bars_store is None or refresh:
        if OPTIONS_BARS_PARQUET.exists():
            _bars_store = _coerce_dt(pd.read_parquet(OPTIONS_BARS_PARQUET),
                                     ["date", "knowable_date"])
        else:
            _bars_store = _empty(BARS_COLS)
    return _bars_store


def load_options_contracts(refresh: bool = False) -> pd.DataFrame:
    global _contracts_store
    if _contracts_store is None or refresh:
        if OPTIONS_CONTRACTS_PARQUET.exists():
            _contracts_store = _coerce_dt(pd.read_parquet(OPTIONS_CONTRACTS_PARQUET),
                                          ["first_date", "knowable_date", "expiration"])
        else:
            _contracts_store = _empty(CONTRACT_COLS)
    return _contracts_store


def contracts_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Derive the contract-metadata table from observed bars (survivorship-safe: a contract
    enters the universe only once it has actually traded). knowable_date = first traded
    bar's knowable_date; strike/expiration/type parsed from the OCC ticker."""
    if bars is None or bars.empty:
        return _empty(CONTRACT_COLS)
    g = bars.groupby("contract").agg(
        underlying=("underlying", "first"),
        first_date=("date", "min"),
        knowable_date=("knowable_date", "min"),
    ).reset_index()
    meta = g["contract"].map(parse_occ)
    g["contract_type"] = meta.map(lambda m: m["contract_type"] if m else None)
    g["strike"] = meta.map(lambda m: m["strike"] if m else None)
    g["expiration"] = meta.map(
        lambda m: pd.Timestamp(m["expiration"]) if m else pd.NaT).astype("datetime64[ns]")
    return g.dropna(subset=["contract_type"])[CONTRACT_COLS].reset_index(drop=True)


# ── Provider (implements app/options/contracts.OptionsDataProvider) ────────────

class PolygonOptionsProvider:
    """PIT, survivorship-safe options data provider backed by the backfilled parquet
    (historical) + Polygon REST (current snapshot only)."""

    def __init__(self, bars: Optional[pd.DataFrame] = None,
                 contracts: Optional[pd.DataFrame] = None):
        # Allow injecting frames (tests); otherwise lazy-load the parquet stores.
        self._bars = bars
        self._contracts = contracts

    def _bars_df(self) -> pd.DataFrame:
        return self._bars if self._bars is not None else load_options_bars()

    def _contracts_df(self) -> pd.DataFrame:
        if self._contracts is not None:
            return self._contracts
        if self._bars is not None:
            # Bars were injected (tests / ad-hoc) -> derive the universe from THEM, never
            # from the on-disk store (which would mix unrelated data into the result).
            return contracts_from_bars(self._bars)
        c = load_options_contracts()
        if c.empty:  # fall back to deriving from bars if the metadata table is absent
            c = contracts_from_bars(self._bars_df())
        return c

    def coverage_start(self, underlying: str) -> Optional[date]:
        """First date for which *underlying* has any data (min knowable_date). Lets
        callers tell 'no contracts existed then' apart from 'we have no data that far
        back' — a coverage gap must not masquerade as an empty (truthful) universe."""
        df = self._contracts_df()
        if df is None or df.empty:
            return None
        sub = df[df["underlying"] == underlying.upper()]
        if sub.empty:
            return None
        return pd.to_datetime(sub["knowable_date"]).min().date()

    def get_universe(self, underlying: str, as_of: date,
                     include_expired: bool = True) -> List[str]:
        df = self._contracts_df()
        if df is None or df.empty:
            return []
        as_of_ts = pd.Timestamp(as_of)
        cov = self.coverage_start(underlying)
        if cov is not None and as_of < cov:
            logger.warning("options coverage for %s starts %s; as_of=%s precedes it — "
                           "empty universe is a DATA GAP, not 'no contracts existed'",
                           underlying.upper(), cov, as_of)
        sub = df[(df["underlying"] == underlying.upper())
                 & (df["knowable_date"] <= as_of_ts)]
        if not include_expired:
            sub = sub[pd.to_datetime(sub["expiration"]) >= as_of_ts]
        return sorted(sub["contract"].tolist())

    def get_contract_bars(self, underlying: str, as_of: date) -> pd.DataFrame:
        df = self._bars_df()
        if df is None or df.empty:
            return _empty(BARS_COLS)
        sub = df[(df["underlying"] == underlying.upper())
                 & (df["knowable_date"] <= pd.Timestamp(as_of))]
        return sub.sort_values(["contract", "date"]).reset_index(drop=True)

    def get_current_snapshot(self, underlying: str) -> Dict[str, Dict[str, Any]]:
        return fetch_current_snapshot(underlying)
