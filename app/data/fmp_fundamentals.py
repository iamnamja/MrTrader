"""
Phase 93 — Financial Modeling Prep (FMP) quarterly fundamentals store.

Canonical persistent parquet of point-in-time (PIT) safe quarterly
fundamentals fetched from FMP /stable/ endpoints. Replaces the EDGAR
annual parquet (data/fundamentals/fundamentals_history.parquet) with a
strictly richer dataset:

  - Quarterly cadence (vs annual) — 4× more snapshots per symbol
  - More fields (margins, FCF, OCF, capex, EPS diluted, BVPS, shares out)
  - 25-year history depth on FMP's standard plan

Design — point-in-time correctness
----------------------------------
The `as_of_date` we store is the FMP `filingDate` — the date the 10-Q/10-K
was first publicly available (typically ~45 days after the fiscal period
ends). Using `period_end` would leak information that wasn't yet known to
market participants. EDGAR had this right; we follow the same rule.

PE and PB are NOT stored in the parquet. They depend on price, which
depends on the lookup date — not the filing date. Storing PE at filing
time would yield wrong values for any training window past the filing
date. Instead, `get_fundamentals_as_of()` accepts a `latest_close`
argument and computes PE/PB on the fly:

    PE = price / (eps_diluted * 4)        # annualised quarterly EPS
    PB = price / book_value_per_share

We deliberately use `eps_diluted * 4` (annualised quarterly run-rate)
rather than the annual ratios endpoint because:
  1. The annual endpoint's PE uses the year-end price, not the as_of price.
  2. Quarterly run-rate is more responsive to recent operating changes.
  3. Quarterly endpoint is included on our plan; quarterly ratios are not.

Schema
------
Parquet at ``data/fundamentals/fmp_fundamentals_history.parquet``:

  symbol                  str    ticker
  as_of_date              str    YYYY-MM-DD — filingDate (PIT-safe)
  period_end              str    YYYY-MM-DD — fiscal quarter end (audit only)
  period                  str    Q1/Q2/Q3/Q4/FY
  fiscal_year             int    fiscal year of the period
  revenue                 float  quarterly revenue
  net_income              float  quarterly net income
  profit_margin           float  net_income / revenue
  revenue_growth_yoy      float  vs same quarter prior year
  gross_margin            float  from income statement
  operating_margin        float  operatingIncome / revenue
  fcf_margin              float  freeCashFlow / revenue
  debt_to_equity          float  totalDebt / totalStockholdersEquity
  book_value_per_share    float  from balance sheet
  eps_diluted             float  diluted EPS (quarterly)
  shares_outstanding      float  weightedAverageShsOutDil
  operating_cash_flow     float
  capex                   float
  data_source             str    'fmp_quarterly' | 'fmp_annual'

Transition
----------
Both this parquet and the EDGAR parquet coexist during transition.
training.py loads both; FMP values OVERRIDE EDGAR where present.
This permits gradual A/B validation before retiring EDGAR.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
FUNDAMENTALS_DIR = Path("data/fundamentals")
FMP_PATH = FUNDAMENTALS_DIR / "fmp_fundamentals_history.parquet"

_BASE = "https://financialmodelingprep.com/stable"
_REQ_DELAY_SEC = 0.15           # ~6.7 req/s per worker; ~10 req/s plan headroom
_REQ_TIMEOUT = 15
_REQ_RETRIES = 2
_INCREMENTAL_STALE_DAYS = 45    # quarterly filings appear ~45d after period end

_SCHEMA_COLUMNS: List[str] = [
    "symbol", "as_of_date", "period_end", "period", "fiscal_year",
    "revenue", "net_income", "profit_margin", "revenue_growth_yoy",
    "gross_margin", "operating_margin", "fcf_margin", "debt_to_equity",
    "book_value_per_share", "eps_diluted", "shares_outstanding",
    "operating_cash_flow", "capex", "data_source",
]


def _api_key() -> str:
    from app.config import settings
    return settings.fmp_api_key or ""


# ── Fetch helpers ────────────────────────────────────────────────────────────

def _fetch(endpoint: str, params: Dict) -> Optional[List[Dict]]:
    """GET <BASE>/<endpoint> with apikey, retries, backoff. Returns list or None."""
    url = f"{_BASE}/{endpoint}"
    p = dict(params)
    p["apikey"] = _api_key()
    last_exc: Optional[Exception] = None
    for attempt in range(_REQ_RETRIES + 1):
        try:
            resp = requests.get(url, params=p, timeout=_REQ_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                logger.debug("FMP %s returned non-list: %r", endpoint, data)
                return []
            if resp.status_code == 402:
                logger.warning("FMP %s 402 (premium): %s", endpoint, params)
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.5 * (attempt + 1))
                continue
            logger.debug("FMP %s HTTP %s for %s", endpoint, resp.status_code, params)
            return None
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    if last_exc:
        logger.debug("FMP %s exhausted retries for %s: %s", endpoint, params, last_exc)
    return None


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None:
        return None
    if d == 0:
        return None
    return n / d


def _quarter_label(period: Optional[str], period_end: Optional[str]) -> str:
    """Best-effort fiscal quarter label."""
    if period and isinstance(period, str):
        s = period.upper().strip()
        if s in ("Q1", "Q2", "Q3", "Q4", "FY"):
            return s
    if period_end:
        try:
            m = int(period_end[5:7])
            return {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}.get(m, "FY")
        except Exception:
            pass
    return "FY"


# ── Per-symbol fetch + assemble ──────────────────────────────────────────────

def _fetch_symbol_quarterly(symbol: str, lookback_quarters: int = 100) -> List[Dict]:
    """
    Fetch quarterly income / balance / cash-flow for *symbol* and join into
    one row per (period_end, filingDate). Returns list of normalised dicts
    matching the parquet schema (without revenue_growth_yoy — that requires
    a YoY join across rows and is filled in by `_compute_yoy_growth`).
    """
    inc = _fetch("income-statement",
                 {"symbol": symbol, "period": "quarter", "limit": lookback_quarters})
    bal = _fetch("balance-sheet-statement",
                 {"symbol": symbol, "period": "quarter", "limit": lookback_quarters})
    cf = _fetch("cash-flow-statement",
                {"symbol": symbol, "period": "quarter", "limit": lookback_quarters})

    # If income statement is missing/empty, abort — without revenue we have nothing.
    if not inc:
        return []

    def _key(row: Dict) -> Optional[str]:
        # FMP uses 'date' for period end; some endpoints use 'period' + 'calendarYear'.
        pe = row.get("date") or row.get("period_end")
        return pe[:10] if isinstance(pe, str) else None

    bal_by_pe = {_key(r): r for r in (bal or []) if _key(r)}
    cf_by_pe = {_key(r): r for r in (cf or []) if _key(r)}

    rows: List[Dict] = []
    for ir in inc:
        period_end = _key(ir)
        if period_end is None:
            continue
        # filingDate is the PIT-safe known date. Fall back to acceptedDate or
        # period_end + 45d if filingDate missing.
        filing_date = ir.get("filingDate") or ir.get("acceptedDate")
        if isinstance(filing_date, str):
            filing_date = filing_date[:10]
        if not filing_date:
            try:
                pe_d = datetime.strptime(period_end, "%Y-%m-%d").date()
                filing_date = (pe_d.replace(day=1) if False else pe_d).isoformat()
                # default 45 day shift
                from datetime import timedelta as _td
                filing_date = (pe_d + _td(days=45)).isoformat()
            except Exception:
                filing_date = period_end

        br = bal_by_pe.get(period_end, {})
        cr = cf_by_pe.get(period_end, {})

        revenue = _safe_float(ir.get("revenue"))
        net_income = _safe_float(ir.get("netIncome"))
        gross_margin = _safe_float(ir.get("grossProfitMargin"))
        if gross_margin is None:
            gross_margin = _safe_div(_safe_float(ir.get("grossProfit")), revenue)
        operating_income = _safe_float(ir.get("operatingIncome"))
        operating_margin = _safe_div(operating_income, revenue)

        fcf = _safe_float(cr.get("freeCashFlow"))
        ocf = _safe_float(cr.get("operatingCashFlow"))
        capex = _safe_float(cr.get("capitalExpenditure"))
        fcf_margin = _safe_div(fcf, revenue)

        equity = _safe_float(br.get("totalStockholdersEquity"))
        debt = _safe_float(br.get("totalDebt"))
        debt_to_equity = _safe_div(debt, equity)
        if debt_to_equity is not None:
            # clip to a sane range — negative equity blows this up
            debt_to_equity = max(-10.0, min(10.0, debt_to_equity))

        bvps = _safe_float(br.get("bookValuePerShare"))
        eps_diluted = _safe_float(ir.get("epsDiluted")) or _safe_float(ir.get("eps"))
        shares = _safe_float(ir.get("weightedAverageShsOutDil")) \
            or _safe_float(ir.get("weightedAverageShsOut"))

        profit_margin = _safe_float(ir.get("netProfitMargin"))
        if profit_margin is None:
            profit_margin = _safe_div(net_income, revenue)
        if profit_margin is not None:
            profit_margin = max(-1.0, min(1.0, profit_margin))

        rows.append({
            "symbol": symbol,
            "as_of_date": filing_date,
            "period_end": period_end,
            "period": _quarter_label(ir.get("period"), period_end),
            "fiscal_year": int(ir.get("calendarYear") or ir.get("fiscalYear") or
                               (period_end[:4] if period_end else 0) or 0),
            "revenue": revenue,
            "net_income": net_income,
            "profit_margin": profit_margin,
            "revenue_growth_yoy": None,    # filled by _compute_yoy_growth
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "fcf_margin": fcf_margin,
            "debt_to_equity": debt_to_equity,
            "book_value_per_share": bvps,
            "eps_diluted": eps_diluted,
            "shares_outstanding": shares,
            "operating_cash_flow": ocf,
            "capex": capex,
            "data_source": "fmp_quarterly",
        })

    return rows


def _compute_yoy_growth(rows: List[Dict]) -> List[Dict]:
    """
    Fill `revenue_growth_yoy` by matching each row to the same fiscal quarter
    one year prior. Operates in-place on a sorted-by-period_end copy and
    returns the list.
    """
    if not rows:
        return rows
    # Index: (fiscal_year, period) → row
    by_key: Dict[Tuple[int, str], Dict] = {}
    for r in rows:
        by_key[(r["fiscal_year"], r["period"])] = r

    for r in rows:
        prior = by_key.get((r["fiscal_year"] - 1, r["period"]))
        if prior is None:
            continue
        rev_now = r.get("revenue")
        rev_prev = prior.get("revenue")
        if rev_now is None or rev_prev is None or rev_prev == 0:
            continue
        growth = (rev_now - rev_prev) / abs(rev_prev)
        r["revenue_growth_yoy"] = max(-1.0, min(5.0, growth))
    return rows


# ── Public API ───────────────────────────────────────────────────────────────

# Module-level cache for the parquet — parquet is small (~MB) and loaded
# repeatedly during inference. Invalidated by mtime change.
_LOAD_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}


def load_fmp_fundamentals(force_reload: bool = False) -> pd.DataFrame:
    """Load the FMP parquet from disk. Empty schema-shaped DF if missing.

    Cached by file mtime — concurrent inference calls share one DataFrame.
    """
    if not FMP_PATH.exists():
        return pd.DataFrame(columns=_SCHEMA_COLUMNS)
    try:
        mtime = FMP_PATH.stat().st_mtime
        cached = _LOAD_CACHE.get(str(FMP_PATH))
        if not force_reload and cached and cached[0] == mtime:
            return cached[1]
        df = pd.read_parquet(FMP_PATH)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", FMP_PATH, exc)
        return pd.DataFrame(columns=_SCHEMA_COLUMNS)
    # Normalise date columns to YYYY-MM-DD strings + sort
    for col in ("as_of_date", "period_end"):
        if col in df.columns:
            df[col] = df[col].astype(str).str[:10]
    if {"symbol", "as_of_date"}.issubset(df.columns):
        df = df.sort_values(["symbol", "as_of_date"]).reset_index(drop=True)
    _LOAD_CACHE[str(FMP_PATH)] = (mtime, df)
    return df


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    # On Windows, replace handles existing target atomically.
    tmp.replace(path)


def backfill_fmp_fundamentals(
    symbols: List[str],
    workers: int = 4,
    lookback_quarters: int = 100,
    dry_run: bool = False,
    request_delay: float = _REQ_DELAY_SEC,
) -> pd.DataFrame:
    """
    Full backfill: fetch all quarterly history for *symbols* and write parquet.

    Rate-limited at `request_delay` sec between requests *per worker*.
    Each symbol incurs 3 requests (income / balance / cash-flow).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not symbols:
        logger.warning("backfill: empty symbols list — nothing to do")
        return pd.DataFrame(columns=_SCHEMA_COLUMNS)

    est_seconds = (len(symbols) * 3 * request_delay) / max(workers, 1)
    logger.info(
        "FMP backfill: %d symbols, %d workers, lookback=%dq, est ~%.1f min",
        len(symbols), workers, lookback_quarters, est_seconds / 60.0,
    )

    all_rows: List[Dict] = []
    errors = 0
    completed = 0
    t0 = time.time()

    def _worker(sym: str) -> List[Dict]:
        # serialise this worker's three requests with the rate-limit delay
        rows = _fetch_symbol_quarterly(sym, lookback_quarters=lookback_quarters)
        time.sleep(request_delay)
        return _compute_yoy_growth(rows)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            completed += 1
            try:
                rows = fut.result()
                if rows:
                    all_rows.extend(rows)
                else:
                    logger.debug("%s: no FMP data", sym)
            except Exception as exc:
                errors += 1
                logger.warning("%s: backfill failed — %s", sym, exc)
            if completed % 50 == 0 or completed == len(symbols):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-3)
                eta = (len(symbols) - completed) / max(rate, 1e-3)
                logger.info(
                    "  progress %d / %d  | %d rows  | %d errors  | ETA %.1f min",
                    completed, len(symbols), len(all_rows), errors, eta / 60.0,
                )

    if dry_run:
        logger.info("[DRY-RUN] %d rows assembled. Sample:", len(all_rows))
        for r in all_rows[:3]:
            logger.info("  %s", r)
        return pd.DataFrame(all_rows, columns=_SCHEMA_COLUMNS)

    if not all_rows:
        logger.warning("FMP backfill: 0 rows assembled — not writing parquet")
        return pd.DataFrame(columns=_SCHEMA_COLUMNS)

    df = pd.DataFrame(all_rows, columns=_SCHEMA_COLUMNS)
    df = df.drop_duplicates(subset=["symbol", "as_of_date", "period_end"], keep="last")
    df = df.sort_values(["symbol", "as_of_date"]).reset_index(drop=True)
    _atomic_write_parquet(df, FMP_PATH)
    logger.info(
        "FMP backfill complete: %s (%d rows, %d symbols, errors=%d)",
        FMP_PATH, len(df), df["symbol"].nunique(), errors,
    )
    return df


def update_fmp_fundamentals_incremental(
    symbols: List[str],
    workers: int = 4,
    stale_days: int = _INCREMENTAL_STALE_DAYS,
) -> pd.DataFrame:
    """
    Incremental update: fetch only symbols whose latest stored row is older
    than `stale_days` (default 45 — typical filing lag). Append + dedupe + save.
    """
    existing = load_fmp_fundamentals()
    today = date.today()

    if existing.empty:
        stale_symbols = list(symbols)
        logger.info("Incremental: no existing parquet — fetching all %d symbols", len(stale_symbols))
    else:
        latest_per_sym: Dict[str, str] = (
            existing.groupby("symbol")["as_of_date"].max().to_dict()
        )
        stale_symbols = []
        for s in symbols:
            latest = latest_per_sym.get(s)
            if latest is None:
                stale_symbols.append(s)
                continue
            try:
                latest_d = datetime.strptime(latest, "%Y-%m-%d").date()
            except Exception:
                stale_symbols.append(s)
                continue
            if (today - latest_d).days > stale_days:
                stale_symbols.append(s)
        logger.info("Incremental: %d / %d symbols stale (>%dd)",
                    len(stale_symbols), len(symbols), stale_days)

    if not stale_symbols:
        return existing

    new_df = backfill_fmp_fundamentals(stale_symbols, workers=workers, dry_run=False)
    if new_df.empty:
        return existing

    # Restrict existing to NOT the stale symbols (we'll replace those entirely
    # with freshly fetched rows — simpler than per-row dedup, and captures any
    # late-arriving restatements).
    keep_existing = existing[~existing["symbol"].isin(stale_symbols)] if not existing.empty else existing
    combined = pd.concat([keep_existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["symbol", "as_of_date", "period_end"], keep="last"
    )
    combined = combined.sort_values(["symbol", "as_of_date"]).reset_index(drop=True)
    _atomic_write_parquet(combined, FMP_PATH)
    logger.info(
        "Incremental update complete: %d rows total (%d symbols)",
        len(combined), combined["symbol"].nunique(),
    )
    return combined


def get_fundamentals_as_of(
    symbol: str,
    as_of_date: Union[str, date, datetime, pd.Timestamp],
    df: Optional[pd.DataFrame] = None,
    latest_close: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """
    PIT lookup: most recent FMP fundamentals row with `as_of_date <= as_of_date`.

    Returns a dict of feature values, with PE/PB computed using
    `latest_close` if provided. Returns None if no row available.

    The returned dict is suitable for splatting onto the FeatureEngineer
    output:

        snap = get_fundamentals_as_of(sym, w_end_date, fmp_df, latest_close=close)
        if snap is not None:
            features.update(snap)
    """
    if df is None:
        df = load_fmp_fundamentals()
    if df is None or df.empty:
        return None

    # Coerce as_of to YYYY-MM-DD string
    if isinstance(as_of_date, (date, datetime, pd.Timestamp)):
        as_of_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    else:
        as_of_str = str(as_of_date)[:10]

    sub = df[(df["symbol"] == symbol) & (df["as_of_date"] <= as_of_str)]
    if sub.empty:
        return None

    row = sub.iloc[-1]    # df is sorted ascending by (symbol, as_of_date)

    def _f(name: str, default: float = 0.0) -> float:
        v = row.get(name)
        if v is None or pd.isna(v):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    eps_diluted = _f("eps_diluted")
    bvps = _f("book_value_per_share")

    pe_ratio = 0.0
    pb_ratio = 0.0
    if latest_close is not None and latest_close > 0:
        if eps_diluted > 0:
            pe_ratio = float(min(latest_close / (eps_diluted * 4.0), 200.0))
        if bvps > 0:
            pb_ratio = float(min(latest_close / bvps, 50.0))

    return {
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "profit_margin": _f("profit_margin"),
        "revenue_growth": _f("revenue_growth_yoy"),
        "debt_to_equity": _f("debt_to_equity"),
        "gross_margin": _f("gross_margin"),
        "operating_margin": _f("operating_margin"),
        "fcf_margin": _f("fcf_margin"),
    }


def build_fmp_lookup_index(df: Optional[pd.DataFrame] = None
                           ) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Pre-build a per-symbol sorted (as_of_date, fields_dict) list — the same
    shape as `_fund_hist_by_symbol` used elsewhere in training.py — so the
    worker can do PIT lookups without scanning a DataFrame.
    """
    if df is None:
        df = load_fmp_fundamentals()
    out: Dict[str, List[Tuple[str, Dict]]] = {}
    if df is None or df.empty:
        return out
    for sym, grp in df.groupby("symbol"):
        grp_sorted = grp.sort_values("as_of_date")
        out[sym] = [
            (str(r["as_of_date"]), {
                "profit_margin": float(r["profit_margin"]) if pd.notna(r["profit_margin"]) else 0.0,
                "revenue_growth": float(r["revenue_growth_yoy"]) if pd.notna(r["revenue_growth_yoy"]) else 0.0,
                "debt_to_equity": float(r["debt_to_equity"]) if pd.notna(r["debt_to_equity"]) else 0.0,
                "gross_margin": float(r["gross_margin"]) if pd.notna(r["gross_margin"]) else 0.0,
                "operating_margin": float(r["operating_margin"]) if pd.notna(r["operating_margin"]) else 0.0,
                "fcf_margin": float(r["fcf_margin"]) if pd.notna(r["fcf_margin"]) else 0.0,
                "eps_diluted": float(r["eps_diluted"]) if pd.notna(r["eps_diluted"]) else 0.0,
                "book_value_per_share": float(r["book_value_per_share"]) if pd.notna(r["book_value_per_share"]) else 0.0,
            })
            for _, r in grp_sorted.iterrows()
        ]
    return out


def lookup_pit_from_index(
    fmp_history: List[Tuple[str, Dict]],
    as_of_date: Union[str, date, datetime, pd.Timestamp],
    latest_close: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """
    PIT lookup against a pre-built `(date_str, fields)` list (subprocess-friendly,
    no DataFrame). Computes PE/PB from `latest_close`.
    """
    if not fmp_history:
        return None
    if isinstance(as_of_date, (date, datetime, pd.Timestamp)):
        as_of_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    else:
        as_of_str = str(as_of_date)[:10]

    pit: Optional[Dict] = None
    for snap_date, snap in fmp_history:
        if snap_date <= as_of_str:
            pit = snap
        else:
            break
    if pit is None:
        return None

    eps_diluted = float(pit.get("eps_diluted", 0.0) or 0.0)
    bvps = float(pit.get("book_value_per_share", 0.0) or 0.0)
    pe_ratio = 0.0
    pb_ratio = 0.0
    if latest_close is not None and latest_close > 0:
        if eps_diluted > 0:
            pe_ratio = float(min(latest_close / (eps_diluted * 4.0), 200.0))
        if bvps > 0:
            pb_ratio = float(min(latest_close / bvps, 50.0))

    return {
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "profit_margin": float(pit.get("profit_margin", 0.0) or 0.0),
        "revenue_growth": float(pit.get("revenue_growth", 0.0) or 0.0),
        "debt_to_equity": float(pit.get("debt_to_equity", 0.0) or 0.0),
        "gross_margin": float(pit.get("gross_margin", 0.0) or 0.0),
        "operating_margin": float(pit.get("operating_margin", 0.0) or 0.0),
        "fcf_margin": float(pit.get("fcf_margin", 0.0) or 0.0),
    }
