"""
norgate_provider.py — Alpha-v9 P4: local parquet mirror of Norgate futures data.

Norgate data is LICENSE-GATED (readable only via NDU + the norgatedata package while the
subscription is active). To work fast and self-sufficiently — and to have a reproducible
research record — we MIRROR the futures data we need into our OWN parquet store under
data/norgate_futures/ while subscribed. After that, research reads from the parquet (no NDU
dependency for re-runs during the subscription).

  data/norgate_futures/
    continuous/{MKT}.parquet   # both &MKT (unadjusted) + &MKT_CCB (back-adjusted), long
                               # format with a price_type column; for trend (back-adj) +
                               # price levels / carry proxy (unadjusted)
    contracts/{MKT}.parquet    # every individual contract for the market (full term
                               # structure) — for precise carry / roll analysis
    _markets.parquet           # per-market metadata (name)

LICENSE NOTE: this mirror is for our own use WHILE SUBSCRIBED (speed + reproducibility).
Norgate licenses access for the subscription period and prohibits redistribution; retaining/
using the raw mirror after the subscription lapses is governed by their license. The DERIVED
research artifacts (signals, backtests, verdicts) are ours regardless.

All extract_* functions are RESUME-SAFE (skip a market whose parquet already exists unless
force=True) and never raise on a single-symbol failure (logged + skipped).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import pandas as pd

log = logging.getLogger(__name__)

NORGATE_DIR = os.path.join("data", "norgate_futures")
CONTINUOUS_DIR = os.path.join(NORGATE_DIR, "continuous")
CONTRACTS_DIR = os.path.join(NORGATE_DIR, "contracts")
MARKETS_META = os.path.join(NORGATE_DIR, "_markets.parquet")

_COL_MAP = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Volume": "volume", "Open Interest": "open_interest", "Delivery Month": "delivery_month",
    "Turnover": "turnover", "Unadjusted Close": "unadjusted_close",
}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=_COL_MAP).copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out


def _series(symbol: str) -> Optional[pd.DataFrame]:
    """One norgatedata price series → normalized DataFrame (None on failure/empty)."""
    import norgatedata as nd
    try:
        df = nd.price_timeseries(symbol, format="pandas-dataframe")
    except Exception as exc:
        log.debug("norgate: price_timeseries failed for %s: %s", symbol, exc)
        return None
    if df is None or len(df) == 0:
        return None
    return _norm(df)


def list_markets() -> List[str]:
    import norgatedata as nd
    return list(nd.futures_market_symbols())


# ── continuous (trend backbone + carry proxy) ───────────────────────────────────────
def extract_continuous(markets: Optional[Sequence[str]] = None, *,
                       force: bool = False, log_every: int = 20) -> Dict[str, int]:
    """Mirror both continuous series (&MKT back-adjusted=CCB, &MKT unadjusted) per market
    → data/norgate_futures/continuous/{MKT}.parquet (long, price_type ∈ {backadjusted,
    unadjusted}). Resume-safe."""
    os.makedirs(CONTINUOUS_DIR, exist_ok=True)
    markets = list(markets) if markets is not None else list_markets()
    summary = {"saved": 0, "skipped": 0, "failed": 0, "rows": 0}
    for i, m in enumerate(markets):
        path = os.path.join(CONTINUOUS_DIR, f"{m}.parquet")
        if os.path.exists(path) and not force:
            summary["skipped"] += 1
            continue
        frames = []
        for sym, ptype in ((f"&{m}_CCB", "backadjusted"), (f"&{m}", "unadjusted")):
            s = _series(sym)
            if s is not None:
                s = s.reset_index()
                s["price_type"] = ptype
                frames.append(s)
        if not frames:
            summary["failed"] += 1
            log.warning("norgate: no continuous data for %s", m)
            continue
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(path, index=False)
        summary["saved"] += 1
        summary["rows"] += len(df)
        if log_every and (i + 1) % log_every == 0:
            log.info("norgate continuous: %d/%d markets (%d saved, %d rows)",
                     i + 1, len(markets), summary["saved"], summary["rows"])
    return summary


# ── individual contracts (full term structure → precise carry) ───────────────────────
def extract_contracts(markets: Optional[Sequence[str]] = None, *,
                      force: bool = False, log_every: int = 5) -> Dict[str, int]:
    """Mirror EVERY individual contract per market → data/norgate_futures/contracts/{MKT}
    .parquet (long, with a `contract` column). Resume-safe at the market grain. Large:
    ~27k contracts across 105 markets."""
    import norgatedata as nd
    os.makedirs(CONTRACTS_DIR, exist_ok=True)
    markets = list(markets) if markets is not None else list_markets()
    summary = {"markets_saved": 0, "markets_skipped": 0, "contracts": 0, "rows": 0,
               "failed_contracts": 0}
    for i, m in enumerate(markets):
        path = os.path.join(CONTRACTS_DIR, f"{m}.parquet")
        if os.path.exists(path) and not force:
            summary["markets_skipped"] += 1
            continue
        try:
            contracts = nd.futures_market_session_contracts(m)
        except Exception as exc:
            log.warning("norgate: cannot list contracts for %s: %s", m, exc)
            continue
        frames = []
        for c in contracts:
            s = _series(c)
            if s is None:
                summary["failed_contracts"] += 1
                continue
            s = s.reset_index()
            s["contract"] = c
            frames.append(s)
        if not frames:
            log.warning("norgate: no contract data for %s", m)
            continue
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(path, index=False)
        summary["markets_saved"] += 1
        summary["contracts"] += len(frames)
        summary["rows"] += len(df)
        if log_every and (i + 1) % log_every == 0:
            log.info("norgate contracts: %d/%d markets (%d contracts, %d rows so far)",
                     i + 1, len(markets), summary["contracts"], summary["rows"])
    return summary


def extract_market_metadata(markets: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per-market metadata (name) → _markets.parquet."""
    import norgatedata as nd
    os.makedirs(NORGATE_DIR, exist_ok=True)
    markets = list(markets) if markets is not None else list_markets()
    rows = []
    for m in markets:
        try:
            name = nd.futures_market_name(m)
        except Exception:
            name = None
        rows.append({"market": m, "name": name})
    df = pd.DataFrame(rows)
    df.to_parquet(MARKETS_META, index=False)
    return df


# ── readers (research reads from parquet, NOT NDU) ───────────────────────────────────
def load_continuous(market: str, *, price_type: str = "backadjusted") -> pd.DataFrame:
    """Read a mirrored continuous series (default back-adjusted) as a date-indexed frame."""
    path = os.path.join(CONTINUOUS_DIR, f"{market}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no mirrored continuous data for {market} ({path})")
    df = pd.read_parquet(path)
    df = df[df["price_type"] == price_type].drop(columns=["price_type"])
    return df.set_index("date").sort_index()


def load_contracts(market: str) -> pd.DataFrame:
    """Read all mirrored individual contracts for a market (long, with `contract`)."""
    path = os.path.join(CONTRACTS_DIR, f"{market}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no mirrored contract data for {market} ({path})")
    return pd.read_parquet(path)


def mirror_status() -> Dict[str, object]:
    """Coverage of the local mirror (counts + total size)."""
    def _count(d):
        return len([f for f in os.listdir(d) if f.endswith(".parquet")]) if os.path.isdir(d) else 0

    def _size(d):
        if not os.path.isdir(d):
            return 0
        return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                   if f.endswith(".parquet"))
    cont, contr = _count(CONTINUOUS_DIR), _count(CONTRACTS_DIR)
    mb = (_size(CONTINUOUS_DIR) + _size(CONTRACTS_DIR)) / 1e6
    return {"continuous_markets": cont, "contract_markets": contr,
            "total_size_mb": round(mb, 1)}
