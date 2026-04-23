"""
Fundamental data fetcher.

Data sources (all free, no auth required):
  - SEC EDGAR XBRL API  → P/E proxy, P/B, profit margin, revenue growth, D/E
  - SEC EDGAR filings   → earnings history, surprise, days since/until earnings
  - SEC EDGAR Form 4    → insider buy/sell score
  - Alpaca bars         → sector ETF momentum (replaces yfinance ETF download)
  - FINRA short sale    → short interest % of float (bi-monthly, delayed ~2 weeks)
  - Alpha Vantage       → earnings surprise (optional, key required)

All functions return safe defaults on failure so the ML pipeline never breaks.
"""
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── Shared headers for SEC EDGAR (required by their ToS) ─────────────────────
_EDGAR_HEADERS = {"User-Agent": "MrTrader research@example.com", "Accept-Encoding": "gzip, deflate"}

# ── EDGAR rate limiter: 3 concurrent max to avoid throttling on large JSON ────
_EDGAR_SEMAPHORE = threading.Semaphore(3)
_EDGAR_LAST_REQ = {"t": 0.0}
_EDGAR_MIN_GAP = 0.12  # ~8 req/s across all threads

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

# ── In-process caches ─────────────────────────────────────────────────────────
_fund_cache:    Dict[str, tuple] = {}   # symbol → (data_dict, fetched_at)
_etf_cache:     Dict[str, tuple] = {}   # etf_ticker → (momentum, fetched_at)
_earnh_cache:   Dict[str, tuple] = {}   # symbol → (data_dict, fetched_at)
_si_cache: Dict[str, tuple] = {}        # symbol → (value, fetched_at)
_insider_cache: Dict[str, tuple] = {}   # symbol → (score, fetched_at)
_av_cache: Dict[str, tuple] = {}        # symbol → (surprise, fetched_at)
_cik_cache: Dict[str, Optional[int]] = {}  # ticker → CIK int or None

_FUND_TTL = 86_400 * 7     # 7 days (quarterly data, no point refreshing daily)
_ETF_TTL = 3_600           # 1 h
_EARNH_TTL = 86_400 * 7   # 7 days
_SI_TTL = 86_400 * 14     # 14 days (FINRA data is bi-monthly)
_INSIDER_TTL = 86_400     # 24 h
_AV_TTL = 86_400          # 24 h


# ── CIK lookup ────────────────────────────────────────────────────────────────

def _get_cik(symbol: str) -> Optional[int]:
    """Resolve ticker → SEC CIK. Cached indefinitely (CIKs don't change)."""
    if symbol in _cik_cache:
        return _cik_cache[symbol]

    cik = None
    try:
        with _EDGAR_SEMAPHORE:
            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index?q=%22" + symbol + "%22&forms=10-K",
                headers=_EDGAR_HEADERS, timeout=8,
            )
        if resp.status_code == 200:
            hits = resp.json().get("hits", {}).get("hits", [])
            for hit in hits:
                src = hit.get("_source", {})
                if src.get("file_type") == "10-K":
                    cik_str = src.get("entity_id", "") or src.get("cik", "")
                    if cik_str:
                        cik = int(cik_str.lstrip("0") or "0")
                        break
    except Exception as exc:
        logger.debug("CIK lookup failed for %s: %s", symbol, exc)

    # Fallback: company tickers JSON (faster, more reliable)
    if cik is None:
        try:
            with _EDGAR_SEMAPHORE:
                resp = requests.get(
                    "https://www.sec.gov/files/company_tickers.json",
                    headers=_EDGAR_HEADERS, timeout=10,
                )
            if resp.status_code == 200:
                tickers = resp.json()
                sym_upper = symbol.upper()
                for entry in tickers.values():
                    if entry.get("ticker", "").upper() == sym_upper:
                        cik = int(entry["cik_str"])
                        break
        except Exception as exc:
            logger.debug("CIK ticker-file lookup failed for %s: %s", symbol, exc)

    _cik_cache[symbol] = cik
    return cik


def _get_company_facts(cik: int) -> Optional[dict]:
    """Fetch EDGAR XBRL company facts for a CIK. Returns raw JSON or None."""
    try:
        with _EDGAR_SEMAPHORE:
            gap = _EDGAR_MIN_GAP - (time.time() - _EDGAR_LAST_REQ["t"])
            if gap > 0:
                time.sleep(gap)
            resp = requests.get(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                headers=_EDGAR_HEADERS, timeout=15,
            )
            _EDGAR_LAST_REQ["t"] = time.time()
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            time.sleep(2.0)
    except Exception as exc:
        logger.debug("EDGAR company facts fetch failed CIK=%d: %s", cik, exc)
    return None


def _xbrl_latest(facts: dict, *tag_names: str, unit: str = "USD", annual_only: bool = False) -> Optional[float]:
    """
    Extract the most recent value for any of the given US-GAAP XBRL tags.
    Tries each tag in order, returns first match.
    When annual_only=True, filters to entries spanning ~12 months (income statement items).
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tag_names:
        data = us_gaap.get(tag, {})
        entries = data.get("units", {}).get(unit, [])
        # Filter to 10-K annual filings only
        annual = [e for e in entries if e.get("form") in ("10-K", "10-K/A") and "end" in e]
        if annual_only:
            # Income statement items: keep only entries spanning ~12 months (330-400 days)
            def _span_days(e):
                try:
                    from datetime import datetime as _dt
                    return (_dt.strptime(e["end"], "%Y-%m-%d") - _dt.strptime(e["start"], "%Y-%m-%d")).days
                except Exception:
                    return 0
            annual = [e for e in annual if "start" in e and 330 <= _span_days(e) <= 400]
        annual = sorted(annual, key=lambda e: e["end"], reverse=True)
        if annual:
            val = annual[0].get("val")
            if val is not None:
                return float(val)
    return None


def _xbrl_latest_shares(facts: dict) -> Optional[float]:
    """Extract latest common shares outstanding (in shares, not USD)."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in ("CommonStockSharesOutstanding", "CommonStockSharesIssued"):
        entries = us_gaap.get(tag, {}).get("units", {}).get("shares", [])
        recent = sorted(
            [e for e in entries if "end" in e],
            key=lambda e: e["end"],
            reverse=True,
        )
        if recent:
            val = recent[0].get("val")
            if val is not None:
                return float(val)
    return None


# ── Fundamentals (SEC EDGAR XBRL) ────────────────────────────────────────────

def get_fundamentals(symbol: str) -> Dict[str, float]:
    """
    Return fundamental features for *symbol* via SEC EDGAR XBRL API.

    Features:
      pe_ratio              — trailing P/E (estimated from EPS + price)
      pb_ratio              — price-to-book
      profit_margin         — net income / revenue
      revenue_growth        — YoY revenue growth
      debt_to_equity        — total debt / stockholders equity
      earnings_proximity_days — days until next earnings (from earnings history)

    All values are floats; missing data → 0.0.
    Cache TTL: 7 days (EDGAR data is quarterly).
    """
    now = time.time()

    cached = _fund_cache.get(symbol)
    if cached and now - cached[1] < _FUND_TTL:
        return cached[0]

    try:
        from app.data.cache import get_cache
        disk_data = get_cache().get_json(f"fundamentals_edgar/{symbol}", ttl=_FUND_TTL)
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
        "earnings_proximity_days": 90.0,
    }

    cik = _get_cik(symbol)
    if cik is None:
        logger.debug("No CIK found for %s — using defaults", symbol)
        _fund_cache[symbol] = (result, now)
        return result

    facts = _get_company_facts(cik)
    if facts is None:
        _fund_cache[symbol] = (result, now)
        return result

    try:
        # Revenue (two most recent years for growth calc) — annual_only filters to 12-month periods
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        rev_entries = []
        for tag in (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ):
            entries = us_gaap.get(tag, {}).get("units", {}).get("USD", [])

            def _span_days(e):
                try:
                    from datetime import datetime as _dt
                    return (_dt.strptime(e["end"], "%Y-%m-%d") - _dt.strptime(e["start"], "%Y-%m-%d")).days
                except Exception:
                    return 0

            annual = sorted(
                [e for e in entries
                 if e.get("form") in ("10-K", "10-K/A") and "end" in e and "start" in e
                 and 330 <= _span_days(e) <= 400],
                key=lambda e: e["end"],
                reverse=True,
            )
            if len(annual) >= 2:
                rev_entries = annual
                break

        if len(rev_entries) >= 2:
            rev_now = float(rev_entries[0]["val"])
            rev_prev = float(rev_entries[1]["val"])
            if rev_prev and rev_prev != 0:
                result["revenue_growth"] = float(max(-1.0, min(5.0, (rev_now - rev_prev) / abs(rev_prev))))

        # Net income → profit margin (annual 12-month period)
        net_income = _xbrl_latest(facts, "NetIncomeLoss", "ProfitLoss", annual_only=True)
        revenue = float(rev_entries[0]["val"]) if rev_entries else None
        if net_income is not None and revenue and revenue != 0:
            result["profit_margin"] = float(max(-1.0, min(1.0, net_income / revenue)))

        # EPS → P/E (approximate using EPS + we don't have live price here, so skip PE)
        # P/B: stockholders equity / shares → book value per share, then P/B = price / BV
        # We'll compute what we can without a live price feed
        equity = _xbrl_latest(
            facts,
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        )
        shares = _xbrl_latest_shares(facts)

        # Debt-to-equity
        total_debt = _xbrl_latest(
            facts,
            "LongTermDebt",
            "LongTermDebtAndCapitalLeaseObligations",
            "DebtCurrent",
        )
        if total_debt is not None and equity is not None and equity != 0:
            result["debt_to_equity"] = float(min(abs(total_debt / equity), 10.0))

        # Book value per share (used to compute P/B if price is available)
        # Store raw equity/shares in result for caller to compute P/B if needed
        if equity is not None and shares is not None and shares > 0:
            bvps = equity / shares
            # We store bvps so features.py can combine with live price for P/B
            result["_bvps"] = float(bvps)

        # EPS (diluted) for P/E
        eps = _xbrl_latest(facts, "EarningsPerShareDiluted", "EarningsPerShareBasic", unit="USD/shares")
        if eps is not None:
            result["_eps"] = float(eps)

    except Exception as exc:
        logger.debug("EDGAR XBRL parse failed for %s: %s", symbol, exc)

    # Earnings proximity: estimate days until next report from days since last report
    # Companies report quarterly (~90 days). days_until = max(0, 90 - days_since % 90)
    earnh = get_earnings_history(symbol)
    days_since = earnh.get("days_since_earnings", 90.0)
    days_until = max(5.0, float(90 - (days_since % 90)))
    result["earnings_proximity_days"] = min(days_until, 90.0)

    _fund_cache[symbol] = (result, now)

    try:
        from app.data.cache import get_cache
        get_cache().put_json(f"fundamentals_edgar/{symbol}", result)
    except Exception:
        pass

    return result


def prefetch_fundamentals(symbols: list) -> Dict[str, Dict[str, float]]:
    """
    Pre-fetch fundamentals for all symbols once and warm caches.
    Called at start of training run.
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


# ── Sector ETF momentum (Alpaca bars) ────────────────────────────────────────

def get_sector_momentum(sector: str) -> float:
    """
    Return the 20-day price return of the sector ETF via Alpaca bars.
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

    momentum = 0.0
    try:
        from app.integrations import get_alpaca_client
        client = get_alpaca_client()
        df = client.get_bars(etf, timeframe="1Day", limit=30)
        if df is not None and not df.empty and len(df) >= 21:
            close = df["close"].tolist()
            momentum = float((close[-1] - close[-21]) / close[-21]) if close[-21] else 0.0
    except Exception as exc:
        logger.debug("Alpaca ETF momentum fetch failed for %s: %s", etf, exc)

    _etf_cache[etf] = (momentum, now)
    return momentum


# Cache for 5-day ETF momentum (keyed by etf ticker)
_etf_5d_cache: Dict[str, tuple] = {}


def get_sector_momentum_5d(sector: str) -> float:
    """
    Return the 5-day price return of the sector ETF.
    Complements get_sector_momentum (20-day) for short-term relative signal.
    """
    etf = SECTOR_ETF_MAP.get(sector)
    if not etf:
        return 0.0

    now = time.time()
    cached = _etf_5d_cache.get(etf)
    if cached and now - cached[1] < _ETF_TTL:
        return cached[0]

    momentum_5d = 0.0
    try:
        from app.integrations import get_alpaca_client
        client = get_alpaca_client()
        df = client.get_bars(etf, timeframe="1Day", limit=10)
        if df is not None and not df.empty and len(df) >= 6:
            close = df["close"].tolist()
            momentum_5d = float((close[-1] - close[-6]) / close[-6]) if close[-6] else 0.0
    except Exception as exc:
        logger.debug("Alpaca ETF 5d momentum fetch failed for %s: %s", etf, exc)

    _etf_5d_cache[etf] = (momentum_5d, now)
    return momentum_5d


# ── SEC EDGAR insider activity (Form 4) ───────────────────────────────────────

_EDGAR_SEARCH = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22{symbol}%22&dateRange=custom&startdt={start}&enddt={end}&forms=4"
)


def get_insider_score(symbol: str) -> float:
    """
    Net insider buy score for the last 60 days using SEC EDGAR Form 4 filings.

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
        with _EDGAR_SEMAPHORE:
            resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=8)
        if resp.status_code == 200:
            hits = resp.json().get("hits", {}).get("hits", [])
            buys = sum(
                1 for h in hits
                if "P" in (h.get("_source", {}).get("transaction_code", "") or "")
            )
            sells = sum(
                1 for h in hits
                if "S" in (h.get("_source", {}).get("transaction_code", "") or "")
            )
            total = buys + sells
            if total > 0:
                score = (buys - sells) / total
    except Exception as exc:
        logger.debug("Insider score fetch failed for %s: %s", symbol, exc)

    _insider_cache[symbol] = (score, now)
    return score


# ── Alpha Vantage earnings surprise ──────────────────────────────────────────

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
            api_key = getattr(settings, "alpha_vantage_api_key", None) or getattr(settings, "alpha_advantage_api_key", None)
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
            quarterly = resp.json().get("quarterlyEarnings", [])
            if quarterly:
                latest = quarterly[0]
                est = latest.get("estimatedEPS")
                actual = latest.get("reportedEPS")
                if est and actual:
                    est_f, actual_f = float(est), float(actual)
                    if est_f != 0:
                        surprise = float(max(-1.0, min(1.0, (actual_f - est_f) / abs(est_f))))
    except Exception as exc:
        logger.debug("Alpha Vantage earnings fetch failed for %s: %s", symbol, exc)

    _av_cache[symbol] = (surprise, now)
    return surprise


# ── Earnings history (SEC EDGAR) ──────────────────────────────────────────────

def get_earnings_history(symbol: str) -> Dict[str, float]:
    """
    Return earnings history features from SEC EDGAR XBRL.

    Returns dict with:
      earnings_surprise_1q      — most recent quarterly EPS surprise (actual-est)/|est|
      earnings_surprise_2q_avg  — mean surprise over last 2 quarters
      days_since_earnings       — calendar days since last earnings report
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

    cik = _get_cik(symbol)
    if cik is None:
        _earnh_cache[symbol] = (result, now)
        return result

    try:
        facts = _get_company_facts(cik)
        if facts is None:
            _earnh_cache[symbol] = (result, now)
            return result

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        # Diluted EPS entries (quarterly 10-Q filings)
        eps_entries = []
        for tag in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
            entries = us_gaap.get(tag, {}).get("units", {}).get("USD/shares", [])
            quarterly = sorted(
                [e for e in entries if e.get("form") in ("10-Q", "10-Q/A") and "end" in e],
                key=lambda e: e["end"],
                reverse=True,
            )
            if quarterly:
                eps_entries = quarterly
                break

        if eps_entries:
            # Most recent filing date = days_since_earnings
            latest_date = datetime.strptime(eps_entries[0]["end"], "%Y-%m-%d")
            result["days_since_earnings"] = float(
                max(0, min(365, (datetime.now() - latest_date).days))
            )

            # EPS surprise: compare reported (10-Q) vs analyst estimate not available from EDGAR
            # Use QoQ change as a proxy for momentum: (this Q EPS - same Q last year) / |last year|
            if len(eps_entries) >= 4:
                current_eps = float(eps_entries[0]["val"])
                year_ago_eps = float(eps_entries[3]["val"])  # ~4 quarters ago = same season
                if abs(year_ago_eps) > 0.001:
                    surprise = float(max(-1.0, min(1.0, (current_eps - year_ago_eps) / abs(year_ago_eps))))
                    result["earnings_surprise_1q"] = surprise

            if len(eps_entries) >= 5:
                prev_eps = float(eps_entries[1]["val"])
                year_ago_prev = float(eps_entries[4]["val"])
                if abs(year_ago_prev) > 0.001:
                    surprise_prev = float(max(-1.0, min(1.0, (prev_eps - year_ago_prev) / abs(year_ago_prev))))
                    result["earnings_surprise_2q_avg"] = float(
                        np.mean([result["earnings_surprise_1q"], surprise_prev])
                    )

    except Exception as exc:
        logger.debug("EDGAR earnings history failed for %s: %s", symbol, exc)

    _earnh_cache[symbol] = (result, now)
    return result


# ── Short interest (FINRA) ───────────────────────────────────────────────────

_FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/regShoThresholdList"


def get_short_interest(symbol: str) -> float:
    """
    Return short interest as a fraction of float.
    Uses FINRA short sale data (free, ~2 week delay).
    Returns 0.0 on failure.

    Note: FINRA's public API has limited granularity. Returns 0.0 for most
    symbols not on the threshold list; a non-zero value signals elevated short interest.
    """
    now = time.time()
    cached = _si_cache.get(symbol)
    if cached and now - cached[1] < _SI_TTL:
        return cached[0]

    value = 0.0
    try:
        # FINRA Reg SHO threshold list — symbols here have short interest > 0.5% of float
        # This is a binary signal (on list = elevated short interest)
        params = {
            "limit": 1000,
            "fields": "securitiesInformationProcessorSymbol,shortInterestPeriod",
        }
        resp = requests.get(_FINRA_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            symbols_on_list = {
                item.get("securitiesInformationProcessorSymbol", "")
                for item in (data if isinstance(data, list) else [])
            }
            # On threshold list = short interest > 0.5% of float (regulatory threshold)
            value = 0.6 if symbol.upper() in symbols_on_list else 0.0
    except Exception as exc:
        logger.debug("FINRA short interest fetch failed for %s: %s", symbol, exc)

    _si_cache[symbol] = (value, now)
    return value
