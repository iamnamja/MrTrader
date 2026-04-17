"""
Fundamental data fetcher — yfinance + Alpha Vantage.

All methods return safe defaults on failure so the ML pipeline
never breaks due to a missing data source.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Sector ETF map ────────────────────────────────────────────────────────────
SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Financial Services": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# ── In-process cache (TTL = 24 h for fundamentals, 1 h for ETF momentum) ─────
_fund_cache: Dict[str, tuple] = {}   # symbol → (data_dict, fetched_at)
_etf_cache: Dict[str, tuple] = {}    # etf_ticker → (momentum, fetched_at)
_FUND_TTL = 86_400   # 24 h
_ETF_TTL = 3_600     # 1 h


# ── Fundamentals ──────────────────────────────────────────────────────────────

def get_fundamentals(symbol: str) -> Dict[str, float]:
    """
    Return fundamental features for *symbol* via yfinance.

    Features:
      pe_ratio, pb_ratio, profit_margin, revenue_growth,
      debt_to_equity, earnings_proximity_days
    All values are floats; missing data → 0.0.

    Cache hierarchy: in-memory → disk (TTL 24 h) → yfinance API.
    """
    now = time.time()

    # 1. In-memory cache
    cached = _fund_cache.get(symbol)
    if cached and now - cached[1] < _FUND_TTL:
        return cached[0]

    # 2. Disk cache
    try:
        from app.data.cache import get_cache
        disk_data = get_cache().get_json(f"fundamentals/{symbol}", ttl=_FUND_TTL)
        if disk_data is not None:
            clean = {k: v for k, v in disk_data.items() if not k.startswith("_")}
            _fund_cache[symbol] = (clean, now)
            return clean
    except Exception:
        pass

    result = {
        "pe_ratio": 0.0,
        "pb_ratio": 0.0,
        "profit_margin": 0.0,
        "revenue_growth": 0.0,
        "debt_to_equity": 0.0,
        "earnings_proximity_days": 90.0,  # neutral default
    }

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        pe = info.get("trailingPE") or info.get("forwardPE")
        result["pe_ratio"] = float(min(pe, 200)) if pe and pe > 0 else 0.0

        pb = info.get("priceToBook")
        result["pb_ratio"] = float(min(pb, 50)) if pb and pb > 0 else 0.0

        pm = info.get("profitMargins")
        result["profit_margin"] = float(pm) if pm is not None else 0.0

        rg = info.get("revenueGrowth")
        result["revenue_growth"] = float(rg) if rg is not None else 0.0

        de = info.get("debtToEquity")
        result["debt_to_equity"] = float(min(de / 100, 10)) if de and de > 0 else 0.0

        # Days until next earnings
        cal = ticker.calendar
        if cal is not None:
            try:
                if hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    ed = cal["Earnings Date"].iloc[0]
                else:
                    ed = cal.get("Earnings Date", [None])[0] if isinstance(cal, dict) else None
                if ed is not None:
                    import pandas as _pd
                    ed_dt = _pd.to_datetime(ed) if not hasattr(ed, "date") else ed
                    days = (ed_dt.date() - datetime.today().date()).days
                    result["earnings_proximity_days"] = float(max(0, min(days, 180)))
            except Exception:
                pass

    except Exception as exc:
        logger.debug("Fundamentals fetch failed for %s: %s", symbol, exc)

    _fund_cache[symbol] = (result, now)

    # 3. Write to disk cache for next run
    try:
        from app.data.cache import get_cache
        get_cache().put_json(f"fundamentals/{symbol}", result)
    except Exception:
        pass

    return result


def prefetch_fundamentals(symbols: list) -> Dict[str, Dict[str, float]]:
    """
    Pre-fetch fundamentals for all symbols once and warm both caches.

    Use this at the start of a training run so each rolling window can
    read from in-memory cache instead of calling the API 20x per symbol.

    Returns a dict of symbol → fundamentals for direct lookup.
    """
    result = {}
    for symbol in symbols:
        try:
            result[symbol] = get_fundamentals(symbol)
        except Exception as exc:
            logger.debug("prefetch_fundamentals failed for %s: %s", symbol, exc)
            result[symbol] = {
                "pe_ratio": 0.0, "pb_ratio": 0.0, "profit_margin": 0.0,
                "revenue_growth": 0.0, "debt_to_equity": 0.0,
                "earnings_proximity_days": 90.0,
            }
        # Warm earnings history and short interest caches in same pass
        try:
            get_earnings_history(symbol)
        except Exception:
            pass
        try:
            get_short_interest(symbol)
        except Exception:
            pass
    logger.info("prefetch_fundamentals: loaded %d symbols", len(result))
    return result


# ── Sector ETF momentum ───────────────────────────────────────────────────────

def get_sector_momentum(sector: str) -> float:
    """
    Return the 20-day price return of the sector ETF.
    Positive → sector is in an uptrend (tailwind).
    Returns 0.0 on failure.
    """
    etf = SECTOR_ETF_MAP.get(sector)
    if not etf:
        return 0.0

    now = time.time()
    cached = _etf_cache.get(etf)
    if cached and now - cached[1] < _ETF_TTL:
        return cached[0]

    try:
        import pandas as pd
        df = yf.download(etf, period="30d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns or len(df) < 21:
            return 0.0
        close = df["close"].tolist()
        momentum = (close[-1] - close[-21]) / close[-21] if close[-21] else 0.0
        momentum = float(momentum)
    except Exception as exc:
        logger.debug("ETF momentum fetch failed for %s: %s", etf, exc)
        momentum = 0.0

    _etf_cache[etf] = (momentum, now)
    return momentum


# ── SEC EDGAR insider activity ────────────────────────────────────────────────

_insider_cache: Dict[str, tuple] = {}
_INSIDER_TTL = 86_400   # 24 h

# EDGAR full-text search API (no key required)
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_EDGAR_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_EDGAR_SEARCH = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22{symbol}%22&dateRange=custom&startdt={start}&enddt={end}&forms=4"
)
_HEADERS = {"User-Agent": "MrTrader research@example.com"}


def get_insider_score(symbol: str) -> float:
    """
    Net insider buy score for the last 60 days using SEC Form 4 filings.

    Returns a float in [-1, 1]:
      +1 = all insiders buying
      -1 = all insiders selling
       0 = neutral / no data
    """
    now = time.time()
    cached = _insider_cache.get(symbol)
    if cached and now - cached[1] < _INSIDER_TTL:
        return cached[0]

    score = 0.0
    try:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        url = _EDGAR_SEARCH.format(symbol=symbol, start=start, end=end)
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        if resp.status_code == 200:
            hits = resp.json().get("hits", {}).get("hits", [])
            buys = sum(
                1 for h in hits
                if "purchase" in (h.get("_source", {}).get("period_of_report", "") or "").lower()
                or "P" in (h.get("_source", {}).get("transaction_code", "") or "")
            )
            sells = sum(
                1 for h in hits
                if "sale" in (h.get("_source", {}).get("period_of_report", "") or "").lower()
                or "S" in (h.get("_source", {}).get("transaction_code", "") or "")
            )
            total = buys + sells
            if total > 0:
                score = (buys - sells) / total
    except Exception as exc:
        logger.debug("Insider score fetch failed for %s: %s", symbol, exc)

    _insider_cache[symbol] = (score, now)
    return score


# ── Alpha Vantage earnings surprise ──────────────────────────────────────────

_av_cache: Dict[str, tuple] = {}
_AV_TTL = 86_400

_AV_BASE = "https://www.alphavantage.co/query"


def get_earnings_surprise(symbol: str, api_key: Optional[str] = None) -> float:
    """
    Return the most recent quarterly earnings surprise as a fraction.
    e.g. +0.05 = beat by 5%, -0.03 = missed by 3%.
    Returns 0.0 if key absent or call fails.
    """
    if not api_key:
        try:
            from app.config import settings
            api_key = settings.alpha_vantage_api_key or settings.alpha_advantage_api_key
        except Exception:
            pass
    if not api_key:
        return 0.0

    now = time.time()
    cached = _av_cache.get(symbol)
    if cached and now - cached[1] < _AV_TTL:
        return cached[0]

    surprise = 0.0
    try:
        params = {"function": "EARNINGS", "symbol": symbol, "apikey": api_key}
        resp = requests.get(_AV_BASE, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            quarterly = data.get("quarterlyEarnings", [])
            if quarterly:
                latest = quarterly[0]
                est = latest.get("estimatedEPS")
                actual = latest.get("reportedEPS")
                if est and actual:
                    est_f, actual_f = float(est), float(actual)
                    if est_f != 0:
                        surprise = (actual_f - est_f) / abs(est_f)
                        surprise = float(max(-1.0, min(1.0, surprise)))
    except Exception as exc:
        logger.debug("Alpha Vantage earnings fetch failed for %s: %s", symbol, exc)

    _av_cache[symbol] = (surprise, now)
    return surprise


# ── Earnings history (yfinance — no API key needed) ───────────────────────────

_earnh_cache: Dict[str, tuple] = {}
_EARNH_TTL = 86_400  # 24 h


def get_earnings_history(symbol: str) -> Dict[str, float]:
    """
    Return earnings history features from yfinance (free, no API key).

    Returns dict with:
      earnings_surprise_1q  — most recent quarterly EPS surprise (actual-est)/|est|
      earnings_surprise_2q_avg — mean surprise over last 2 quarters
      days_since_earnings   — calendar days since last earnings report
    """
    now = time.time()
    cached = _earnh_cache.get(symbol)
    if cached and now - cached[1] < _EARNH_TTL:
        return cached[0]

    result = {
        "earnings_surprise_1q": 0.0,
        "earnings_surprise_2q_avg": 0.0,
        "days_since_earnings": 90.0,
    }

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.earnings_history
        if hist is None or hist.empty:
            # Fallback: try quarterly_earnings
            hist = ticker.quarterly_earnings
            if hist is None or hist.empty:
                _earnh_cache[symbol] = (result, now)
                return result

        # earnings_history has columns: EPS Estimate, Reported EPS, Surprise(%)
        # quarterly_earnings has columns: Revenue, Earnings (no surprise)
        surprises = []
        report_dates = []

        if "Reported EPS" in hist.columns and "EPS Estimate" in hist.columns:
            for idx_val, row in hist.iterrows():
                est = row.get("EPS Estimate")
                actual = row.get("Reported EPS")
                if est is not None and actual is not None:
                    try:
                        est_f, actual_f = float(est), float(actual)
                        if abs(est_f) > 0.001:
                            surprises.append(
                                float(max(-1.0, min(1.0, (actual_f - est_f) / abs(est_f))))
                            )
                    except (ValueError, TypeError):
                        pass
                # Track report date for days_since
                try:
                    report_dates.append(pd.to_datetime(idx_val))
                except Exception:
                    pass

        if surprises:
            result["earnings_surprise_1q"] = surprises[0]
            result["earnings_surprise_2q_avg"] = float(np.mean(surprises[:2]))

        if report_dates:
            latest = max(report_dates)
            result["days_since_earnings"] = float(
                max(0, min(365, (datetime.now() - latest).days))
            )

    except Exception as exc:
        logger.debug("Earnings history fetch failed for %s: %s", symbol, exc)

    _earnh_cache[symbol] = (result, now)
    return result


# ── Short interest (yfinance — free) ─────────────────────────────────────────

_si_cache: Dict[str, tuple] = {}
_SI_TTL = 86_400  # 24 h


def get_short_interest(symbol: str) -> float:
    """
    Return short interest as a fraction of float (e.g. 0.05 = 5% short).
    Uses yfinance info.shortPercentOfFloat.  Returns 0.0 on failure.
    """
    now = time.time()
    cached = _si_cache.get(symbol)
    if cached and now - cached[1] < _SI_TTL:
        return cached[0]

    value = 0.0
    try:
        info = yf.Ticker(symbol).info or {}
        spf = info.get("shortPercentOfFloat") or info.get("shortRatio")
        if spf is not None:
            value = float(min(spf, 1.0))  # cap at 100%
    except Exception as exc:
        logger.debug("Short interest fetch failed for %s: %s", symbol, exc)

    _si_cache[symbol] = (value, now)
    return value
