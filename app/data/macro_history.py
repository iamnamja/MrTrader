"""
Persistent daily OHLCV cache for macro/regime tickers.

Stores ^VIX, ^VIX3M, HYG, IEF, RSP, SPY closes in a single Parquet file at
``data/macro/macro_history.parquet`` so live inference and training can read
PIT-safe regime data without hitting yfinance every call.

Schema
------
date   : str (YYYY-MM-DD, sorted ascending)
vix    : float (^VIX close)
vix3m  : float (^VIX3M close)
hyg    : float (HYG close)
ief    : float (IEF close)
rsp    : float (RSP close)
spy    : float (SPY close)
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

MACRO_DIR = Path("data/macro")
MACRO_PATH = MACRO_DIR / "macro_history.parquet"

# yfinance ticker → column name in our parquet
TICKER_COLUMNS = {
    "^VIX": "vix",
    "^VIX3M": "vix3m",
    "HYG": "hyg",
    "IEF": "ief",
    "RSP": "rsp",
    "SPY": "spy",
}

INITIAL_START = "2018-01-01"

_COLUMNS = ["date"] + list(TICKER_COLUMNS.values())

# FRED fallback for the volatility series. yfinance returns a frame for ^VIX3M but with the
# Close almost entirely NaN — measured 1 non-NaN of 8 rows on 2026-09-01, against 8/8 for ^VIX.
# Because the extractor legitimately skips NaN values, the ticker never triggers the
# "No close data returned" warning; the column simply thins out. Coverage of rows carrying BOTH
# vix and vix3m decayed 100% (through Apr 2026) -> 73% (Jun) -> 39% (Jul) -> 5% (Aug), with the
# last complete row on 2026-08-05.
#
# That silently disabled the crash governor, which needs both series on the SAME settled date to
# compute the VIX/VIX3M term-structure ratio. It is fail-safe (returns 1.0 = no de-risk), so the
# only symptom was one WARNING per weekly rebalance — the trend sleeve simply stopped being able
# to cut exposure in backwardation, which is the one regime it exists to protect against.
#
# FRED carries both series complete and free. yfinance stays PRIMARY (intraday-fresh, and the
# other four tickers are fine); FRED only fills values yfinance left missing. FRED publishes with
# roughly a one-business-day lag, which is harmless here: the governor deliberately reads only
# SETTLED closes strictly before today.
FRED_SERIES = {
    "vix": "VIXCLS",     # CBOE Volatility Index
    "vix3m": "VXVCLS",   # CBOE S&P 500 3-Month Volatility Index (VIX3M / legacy VXV)
}
_FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_fred_series(series_id: str, start: str, end: str) -> dict:
    """{date -> float} for a FRED series over [start, end]. {} on any failure.

    Deliberately NOT app.macro.fred_client: that client hardcodes `limit: 24` for the regime
    detector's recent-window contract (with a documented desc/asc bug fix), so it cannot serve a
    date-ranged backfill. Never raises — a fallback that can break the primary path is worse than
    no fallback.
    """
    try:
        from app.config import settings
        api_key = getattr(settings, "fred_api_key", None)
        if not api_key:
            logger.debug("FRED fallback skipped for %s: no api key", series_id)
            return {}
        import httpx
        resp = httpx.get(_FRED_OBS_URL, timeout=20, params={
            "series_id": series_id, "api_key": api_key, "file_type": "json",
            "observation_start": start, "observation_end": end,
        })
        if resp.status_code != 200:
            logger.warning("FRED %s HTTP %s", series_id, resp.status_code)
            return {}
        out = {}
        for o in resp.json().get("observations", []):
            v = o.get("value")
            if v in (None, "", "."):      # FRED marks missing observations with "."
                continue
            try:
                out[str(o["date"])[:10]] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    except Exception as exc:  # noqa: BLE001 — fallback must never break the primary fetch
        logger.warning("FRED fallback failed for %s: %s", series_id, exc)
        return {}


def _fill_from_fred(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Fill missing vix/vix3m values in `df` from FRED. Only ever writes where the value is
    absent, so a working yfinance value is never overwritten."""
    if df.empty:
        return df
    for col, series_id in FRED_SERIES.items():
        if col not in df.columns:
            continue
        missing = df[col].isna()
        if not missing.any():
            continue
        obs = _fetch_fred_series(series_id, start, end)
        if not obs:
            continue
        filled = df.loc[missing, "date"].astype(str).str[:10].map(obs)
        n = int(filled.notna().sum())
        if n:
            df.loc[missing, col] = filled
            logger.info("macro_history: filled %d missing %s value(s) from FRED (%s)",
                        n, col, series_id)
    return df


def load_macro_history() -> pd.DataFrame:
    """Load the macro history parquet from disk (no network call).

    Returns an empty DataFrame with the expected schema if the file is missing.
    """
    if not MACRO_PATH.exists():
        return pd.DataFrame(columns=_COLUMNS)
    try:
        df = pd.read_parquet(MACRO_PATH)
        # Ensure date column is string YYYY-MM-DD and sorted ascending
        if "date" in df.columns:
            df["date"] = df["date"].astype(str).str[:10]
            df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning("Failed to read %s: %s", MACRO_PATH, exc)
        return pd.DataFrame(columns=_COLUMNS)


def _fetch_closes(start: str, end: str) -> pd.DataFrame:
    """Fetch close prices for all macro tickers from yfinance.

    Returns a DataFrame indexed by date string with one column per ticker
    (using our short column names: vix, vix3m, hyg, ief, rsp, spy).
    """
    from dotenv import load_dotenv
    load_dotenv()

    import yfinance as yf

    tickers = list(TICKER_COLUMNS.keys())
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as exc:
        logger.error("yfinance download failed for macro tickers: %s", exc)
        return pd.DataFrame(columns=_COLUMNS)

    rows: dict[str, dict] = {}
    for ticker, col in TICKER_COLUMNS.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                sub = raw[ticker]
            else:
                sub = raw
            close = sub["Close"] if "Close" in sub.columns else sub.get("close")
            if close is None or close.empty:
                logger.warning("No close data returned for %s", ticker)
                continue
            for ts, val in close.items():
                if pd.isna(val):
                    continue
                d = pd.Timestamp(ts).strftime("%Y-%m-%d")
                rows.setdefault(d, {"date": d})[col] = float(val)
        except Exception as exc:
            logger.warning("Failed to extract %s from yfinance batch: %s", ticker, exc)

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.DataFrame(list(rows.values()))
    # Reorder/ensure all columns present (missing tickers → NaN)
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[_COLUMNS].sort_values("date").reset_index(drop=True)
    # Backstop the volatility series from FRED where yfinance returned NaN (see FRED_SERIES).
    df = _fill_from_fred(df, start, end)
    return df


def update_macro_history() -> pd.DataFrame:
    """Load existing macro history, append missing tail from yfinance, save.

    Returns the full updated DataFrame.
    """
    MACRO_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_macro_history()

    if existing.empty:
        start = INITIAL_START
        logger.info("No existing macro_history.parquet — initialising from %s", start)
    else:
        last_date = existing["date"].max()
        # Use last *complete* row as the resume point — partial rows (e.g. today's
        # pre-open fill that only has VIX but not HYG/SPY/etc.) must be re-fetched
        # so we don't leave trailing NaNs that break regime_score_pit.
        src_cols = [c for c in TICKER_COLUMNS.values() if c in existing.columns]
        complete_rows = existing.dropna(subset=src_cols, how="any")
        last_complete = complete_rows["date"].max() if not complete_rows.empty else None
        if last_complete and last_complete < last_date:
            logger.info(
                "macro_history has incomplete tail rows (%s..%s) — re-fetching from %s",
                last_complete, last_date, last_complete,
            )
            start_dt = datetime.strptime(last_complete, "%Y-%m-%d").date()
        else:
            start_dt = datetime.strptime(last_date, "%Y-%m-%d").date() + timedelta(days=1)
        start = start_dt.isoformat()
        logger.info("Existing macro_history latest=%s — fetching from %s", last_date, start)

    # yfinance end is exclusive — fetch up to tomorrow to include today
    end = (date.today() + timedelta(days=1)).isoformat()

    if start >= end:
        logger.info("Macro history already up to date (start=%s >= end=%s)", start, end)
        return existing

    new_df = _fetch_closes(start, end)
    if new_df.empty:
        logger.info("No new macro rows fetched")
        return existing

    if existing.empty:
        combined = new_df
    else:
        # Concat and deduplicate — keep last (new_df wins) so partial rows
        # from a prior partial fill get overwritten by the fresh fetch.
        if new_df.empty:
            logger.info("yfinance returned no rows past %s", existing["date"].max())
            return existing
        combined = pd.concat([existing, new_df], ignore_index=True)

    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    combined = combined.reset_index(drop=True)

    # C3: atomic write — prevents corrupt macro_history.parquet if backfill is
    # killed mid-write (live regime gate depends on this file at startup).
    _tmp = MACRO_PATH.with_suffix(MACRO_PATH.suffix + ".tmp")
    combined.to_parquet(_tmp, index=False)
    os.replace(_tmp, MACRO_PATH)
    logger.info(
        "macro_history.parquet updated: %d rows total (added %d new), %s → %s",
        len(combined), len(new_df), combined["date"].min(), combined["date"].max(),
    )
    return combined


def get_macro_as_of(
    as_of: Union[str, date, datetime, pd.Timestamp],
    field: str,
    df: Optional[pd.DataFrame] = None,
) -> Optional[float]:
    """Point-in-time safe lookup: most recent value for ``field`` on or before ``as_of``.

    Args:
        as_of: date string YYYY-MM-DD, date/datetime, or Timestamp.
        field: column name (one of vix, vix3m, hyg, ief, rsp, spy).
        df:    optional pre-loaded DataFrame; loads from disk if None.

    Returns:
        The most recent non-null value with date <= as_of, or None if none.
    """
    if df is None:
        df = load_macro_history()
    if df is None or df.empty or field not in df.columns:
        return None

    if isinstance(as_of, (date, datetime, pd.Timestamp)):
        as_of_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    else:
        as_of_str = str(as_of)[:10]

    sub = df[(df["date"] <= as_of_str) & df[field].notna()]
    if sub.empty:
        return None
    return float(sub[field].iloc[-1])
