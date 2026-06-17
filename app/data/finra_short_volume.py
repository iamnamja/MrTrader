"""
finra_short_volume.py — Alpha-v9 P3-5: FINRA daily off-exchange short-volume.

FINRA publishes a FREE daily consolidated (CNMS) short-sale volume file for every NMS
security: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market. The CDN serves
~2019-01-02 -> present (~1850 trading days, ~8-12k names/day) — i.e. ~10x the power of the
bi-monthly NYSE/Nasdaq short-INTEREST that the Alpha-v8 G2 overlay was KILLED on (for power,
~190 obs). Same economic idea (short-selling pressure), a completely different power regime.

This provider downloads + caches each daily file and distils it to:
  * the AGGREGATE market short-volume ratio  R_t = sum(ShortVolume)/sum(TotalVolume)
    over all NMS names that day (the market-wide short-selling-pressure series), and
  * the per-symbol short-volume ratio for a small target set (e.g. SPY/QQQ/IWM).

Storage is light (one row per trading day): we aggregate on download and never persist the
~10k-name daily cross-section. Incremental: only missing business days are fetched. Cross-
sectional (per-name) short-volume is deliberately NOT the free-data test — it needs a
survivorship-free universe (Norgate, P4-1); the aggregate timing series has no survivorship
bias (it is a market-wide sum) and is the clean free-data signal.

PIT: R_t is computed from day-t trading and is knowable after t's close — a signal for t+1.
"""
from __future__ import annotations

import io
import logging
import os
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Sequence

import pandas as pd

log = logging.getLogger(__name__)

FINRA_BASE = "https://cdn.finra.org/equity/regsho/daily"
FINRA_START = date(2019, 1, 2)              # earliest date the CDN serves
DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "DIA", "EEM", "EFA")
CACHE_PATH = os.path.join("data", "finra_short_volume.parquet")
_REQUEST_TIMEOUT = 30


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _file_url(d: date) -> str:
    return f"{FINRA_BASE}/CNMSshvol{d.strftime('%Y%m%d')}.txt"


def fetch_daily(d: date, *, session=None) -> Optional[pd.DataFrame]:
    """Download + parse ONE day's CNMS short-volume file. Returns a tidy DataFrame
    [symbol, short_volume, total_volume] for that date, or None if unavailable
    (weekend/holiday/not-yet-published -> 404 or an S3 AccessDenied XML stub)."""
    import requests
    url = _file_url(d)
    try:
        resp = (session or requests).get(url, timeout=_REQUEST_TIMEOUT)
    except Exception as exc:
        log.debug("finra: GET failed for %s: %s", d, exc)
        return None
    if resp.status_code != 200:
        return None
    text = resp.text or ""
    if not text.startswith("Date|"):     # AccessDenied XML / empty -> no file that day
        return None
    try:
        df = pd.read_csv(io.StringIO(text), sep="|")
    except Exception as exc:
        log.warning("finra: parse failed for %s: %s", d, exc)
        return None
    df.columns = [str(c).strip() for c in df.columns]
    need = {"Symbol", "ShortVolume", "TotalVolume"}
    if not need.issubset(df.columns):
        return None
    out = pd.DataFrame({
        "symbol": df["Symbol"].astype(str).str.strip(),
        "short_volume": pd.to_numeric(df["ShortVolume"], errors="coerce"),
        "total_volume": pd.to_numeric(df["TotalVolume"], errors="coerce"),
    })
    # Drop the footer count line + any malformed rows (non-numeric volumes).
    out = out.dropna(subset=["short_volume", "total_volume"])
    out = out[(out["total_volume"] > 0) & (out["symbol"].str.len() > 0)]
    return out if not out.empty else None


def _distil(d: date, day_df: pd.DataFrame, symbols: Sequence[str]) -> Dict[str, float]:
    """One cache row: the aggregate market short-vol ratio + each target symbol's ratio."""
    agg_short = float(day_df["short_volume"].sum())
    agg_total = float(day_df["total_volume"].sum())
    row: Dict[str, float] = {
        "date": pd.Timestamp(d),
        "agg_short_vol": agg_short,
        "agg_total_vol": agg_total,
        "agg_short_ratio": (agg_short / agg_total) if agg_total > 0 else float("nan"),
    }
    by_sym = day_df.set_index("symbol")
    for s in symbols:
        if s in by_sym.index:
            r = by_sym.loc[s]
            # a symbol can appear once; guard against accidental dups
            sv = float(r["short_volume"].sum() if hasattr(r["short_volume"], "sum") else r["short_volume"])
            tv = float(r["total_volume"].sum() if hasattr(r["total_volume"], "sum") else r["total_volume"])
            row[f"short_ratio_{s}"] = (sv / tv) if tv > 0 else float("nan")
        else:
            row[f"short_ratio_{s}"] = float("nan")
    return row


def _load_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        log.warning("finra: cache load failed (%s): %s", path, exc)
        return None


def build_panel(*, start: date = FINRA_START, end: Optional[date] = None,
                symbols: Sequence[str] = DEFAULT_SYMBOLS,
                cache_path: str = CACHE_PATH,
                max_days: Optional[int] = None,
                log_every: int = 100) -> pd.DataFrame:
    """Incrementally download the daily short-volume files in [start, end] and return the
    distilled daily panel (date + agg ratio + per-symbol ratios). Only business days NOT
    already cached are fetched; weekends/holidays/missing days are skipped. `max_days` bounds
    a single run's downloads (resume-friendly)."""
    end = end or _today()
    cached = _load_cache(cache_path)
    have = set(cached["date"].dt.date) if cached is not None else set()

    business_days = pd.bdate_range(start, end).date
    todo = [d for d in business_days if d not in have]
    if max_days is not None:
        todo = todo[:max_days]
    if not todo:
        log.info("finra: cache up to date (%d days cached)", len(have))
        return cached if cached is not None else pd.DataFrame()

    import requests
    session = requests.Session()
    rows: List[Dict[str, float]] = []
    fetched = 0
    for i, d in enumerate(todo):
        day_df = fetch_daily(d, session=session)
        if day_df is not None:
            rows.append(_distil(d, day_df, symbols))
            fetched += 1
        if log_every and (i + 1) % log_every == 0:
            log.info("finra: %d/%d processed (%d fetched)", i + 1, len(todo), fetched)

    if not rows:
        log.info("finra: no new files fetched (%d attempted)", len(todo))
        return cached if cached is not None else pd.DataFrame()

    new = pd.DataFrame(rows)
    combined = (pd.concat([cached, new], ignore_index=True) if cached is not None else new)
    combined = (combined.dropna(subset=["date"]).drop_duplicates("date", keep="last")
                .sort_values("date").reset_index(drop=True))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    log.info("finra: wrote %d total rows (%d new) -> %s", len(combined), len(new), cache_path)
    return combined


def load_aggregate_ratio(cache_path: str = CACHE_PATH) -> pd.Series:
    """The cached daily AGGREGATE market short-volume ratio, indexed by date (naive)."""
    df = _load_cache(cache_path)
    if df is None or df.empty:
        return pd.Series(dtype=float, name="agg_short_ratio")
    s = df.set_index("date")["agg_short_ratio"].dropna()
    s.index = pd.to_datetime(s.index)
    return s.rename("agg_short_ratio")


def cache_status(cache_path: str = CACHE_PATH) -> Dict[str, object]:
    df = _load_cache(cache_path)
    if df is None or df.empty:
        return {"n_days": 0, "first": None, "last": None}
    return {"n_days": int(len(df)), "first": str(df["date"].min().date()),
            "last": str(df["date"].max().date())}
