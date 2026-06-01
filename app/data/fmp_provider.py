"""
Financial Modeling Prep (FMP) data provider.

Supplies point-in-time fundamental features for the swing ML model:

  - Historical earnings surprises (EPS actual vs estimate, per quarter)
  - Analyst upgrade/downgrade history
  - Institutional holdings changes (13F)

Point-in-time correctness
-------------------------
For a training window ending on date D, we return only the data that
was publicly known on or before D.  This eliminates the look-ahead bias
that caused the swing model AUC to be stuck at ~0.51 when using current
fundamentals for all historical windows.

All endpoints use FMP's /stable/ API (v4 — post-Aug 2025 plan).
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com/stable"
_CACHE_TTL = 86_400  # 24 h in-process cache

# In-process cache: symbol → (data, fetched_at)
_earnings_cache: Dict[str, tuple] = {}
_grades_cache: Dict[str, tuple] = {}
_institutional_cache: Dict[str, tuple] = {}
_insider_cache: Dict[str, tuple] = {}

# Insider-trading fetch paging (page cap). Purchases are sparse vs sales/awards;
# high-volume names need ≤3 pages of 1000 to reach 2016. 6 pages is a safe ceiling
# covering the 6yr CPCV window + trailing cluster lookback without per-day spam.
_INSIDER_MAX_PAGES = 6
_INSIDER_PAGE_LIMIT = 1000


def _api_key() -> str:
    from app.config import settings
    return settings.fmp_api_key or ""


# ── Earnings history ──────────────────────────────────────────────────────────

def get_earnings_history_fmp(symbol: str) -> List[Dict]:
    """
    Return all available quarterly earnings records for *symbol*.

    Each record:
      date         — report date (YYYY-MM-DD string)
      epsActual    — reported EPS (float or None)
      epsEstimated — consensus estimate (float or None)
      surprise_pct — (actual - est) / |est|, clipped to [-1, 1]
    """
    now = time.time()
    cached = _earnings_cache.get(symbol)
    if cached and now - cached[1] < _CACHE_TTL:
        return cached[0]

    records = []
    try:
        resp = requests.get(
            f"{_BASE}/earnings",
            params={"symbol": symbol, "limit": 20, "apikey": _api_key()},
            timeout=10,
        )
        if resp.status_code == 200:
            for row in resp.json():
                actual = row.get("epsActual")
                est = row.get("epsEstimated")
                surprise = None
                if actual is not None and est is not None:
                    try:
                        a, e = float(actual), float(est)
                        if abs(e) > 0.001:
                            surprise = float(max(-1.0, min(1.0, (a - e) / abs(e))))
                    except (TypeError, ValueError):
                        pass
                records.append({
                    "date": row.get("date", ""),
                    "epsActual": actual,
                    "epsEstimated": est,
                    "surprise_pct": surprise,
                })
    except Exception as exc:
        logger.debug("FMP earnings fetch failed for %s: %s", symbol, exc)

    _earnings_cache[symbol] = (records, now)
    return records


def get_earnings_features_at(symbol: str, as_of: date) -> Dict[str, float] | None:
    """
    Return earnings features using only data known on or before *as_of*.

    Features:
      fmp_surprise_1q       — most recent quarterly EPS surprise before as_of
      fmp_surprise_2q_avg   — mean of last 2 surprises before as_of
      fmp_days_since_earnings — calendar days since last report before as_of

    Returns None when no earnings records exist for the symbol (explicit sentinel
    so callers can distinguish "no data" from "zero surprise").
    """
    try:
        records = get_earnings_history_fmp(symbol)
        # Filter to reports on or before as_of
        past = [
            r for r in records
            if r["date"] and r["surprise_pct"] is not None
            and datetime.strptime(r["date"], "%Y-%m-%d").date() <= as_of
        ]
        if not past:
            return None

        last_date = datetime.strptime(past[0]["date"], "%Y-%m-%d").date()
        return {
            "fmp_surprise_1q": float(past[0]["surprise_pct"]),
            "fmp_surprise_2q_avg": float(
                sum(r["surprise_pct"] for r in past[:2]) / min(len(past), 2)
            ),
            "fmp_days_since_earnings": float(max(0, min(365, (as_of - last_date).days))),
        }
    except Exception as exc:
        logger.debug("FMP earnings features failed for %s: %s", symbol, exc)
        return None


# ── Analyst grades ────────────────────────────────────────────────────────────

def get_analyst_grades_fmp(symbol: str) -> List[Dict]:
    """
    Return all available analyst upgrade/downgrade records for *symbol*.

    Each record:
      date           — grade date (YYYY-MM-DD)
      gradingCompany — analyst firm
      previousGrade  — prior rating
      newGrade       — new rating
      action         — 'upgrade', 'downgrade', 'maintain', 'init'
    """
    now = time.time()
    cached = _grades_cache.get(symbol)
    if cached and now - cached[1] < _CACHE_TTL:
        return cached[0]

    records = []
    try:
        resp = requests.get(
            f"{_BASE}/grades",
            params={"symbol": symbol, "limit": 50, "apikey": _api_key()},
            timeout=10,
        )
        if resp.status_code == 200:
            for row in resp.json():
                records.append({
                    "date": row.get("date", ""),
                    "gradingCompany": row.get("gradingCompany", ""),
                    "previousGrade": row.get("previousGrade", ""),
                    "newGrade": row.get("newGrade", ""),
                    "action": _classify_action(
                        row.get("previousGrade", ""),
                        row.get("newGrade", ""),
                        row.get("action", ""),
                    ),
                })
    except Exception as exc:
        logger.debug("FMP grades fetch failed for %s: %s", symbol, exc)

    _grades_cache[symbol] = (records, now)
    return records


def get_analyst_features_at(symbol: str, as_of: date, lookback_days: int = 30) -> Dict[str, float]:
    """
    Return analyst momentum features using only data known on or before *as_of*.

    Features:
      fmp_analyst_upgrades_30d   — upgrades in last *lookback_days* days
      fmp_analyst_downgrades_30d — downgrades in last *lookback_days* days
      fmp_analyst_momentum_30d   — net (upgrades - downgrades), clipped [-5, 5]
    """
    result = {
        "fmp_analyst_upgrades_30d": 0.0,
        "fmp_analyst_downgrades_30d": 0.0,
        "fmp_analyst_momentum_30d": 0.0,
    }

    try:
        records = get_analyst_grades_fmp(symbol)
        cutoff = as_of - timedelta(days=lookback_days)
        window = [
            r for r in records
            if r["date"]
            and cutoff <= datetime.strptime(r["date"], "%Y-%m-%d").date() <= as_of
        ]
        upgrades = sum(1 for r in window if r["action"] == "upgrade")
        downgrades = sum(1 for r in window if r["action"] == "downgrade")
        result["fmp_analyst_upgrades_30d"] = float(upgrades)
        result["fmp_analyst_downgrades_30d"] = float(downgrades)
        result["fmp_analyst_momentum_30d"] = float(
            max(-5, min(5, upgrades - downgrades))
        )
    except Exception as exc:
        logger.debug("FMP analyst features failed for %s: %s", symbol, exc)

    return result


# ── Institutional holdings ────────────────────────────────────────────────────

def get_institutional_features_at(symbol: str, as_of: date) -> Dict[str, float]:
    """
    Return institutional ownership change features as of *as_of*.

    Features:
      fmp_inst_ownership_pct    — % of float held by institutions (latest before as_of)
      fmp_inst_change_pct       — change in ownership vs prior quarter
    """
    result = {
        "fmp_inst_ownership_pct": 0.0,
        "fmp_inst_change_pct": 0.0,
    }

    now = time.time()
    cached = _institutional_cache.get(symbol)
    if not cached or now - cached[1] >= _CACHE_TTL:
        try:
            resp = requests.get(
                f"{_BASE}/institutional-ownership/symbol",
                params={"symbol": symbol, "limit": 8, "apikey": _api_key()},
                timeout=10,
            )
            if resp.status_code == 200:
                _institutional_cache[symbol] = (resp.json(), now)
            else:
                _institutional_cache[symbol] = ([], now)
        except Exception as exc:
            logger.debug("FMP institutional fetch failed for %s: %s", symbol, exc)
            _institutional_cache[symbol] = ([], now)

    records = _institutional_cache[symbol][0]

    try:
        # Filter to filings on or before as_of
        past = [
            r for r in records
            if r.get("dateReported")
            and datetime.strptime(r["dateReported"][:10], "%Y-%m-%d").date() <= as_of
        ]
        if past:
            latest = past[0]
            pct = latest.get("ownershipPercent") or latest.get("percentOwned") or 0.0
            result["fmp_inst_ownership_pct"] = float(min(pct, 100.0))
            if len(past) >= 2:
                prev_pct = past[1].get("ownershipPercent") or past[1].get("percentOwned") or 0.0
                result["fmp_inst_change_pct"] = float(
                    max(-20, min(20, float(pct) - float(prev_pct)))
                )
    except Exception as exc:
        logger.debug("FMP inst ownership features failed for %s: %s", symbol, exc)

    return result


# ── Insider trading (Form 4 open-market purchases) ────────────────────────────

# Academic basis: Cohen, Malloy & Pomorski (2012, J. Finance) — "opportunistic"
# insider purchases predict positive abnormal returns over 1-6 months; Lakonishok
# & Lee (2001) — aggregate/cluster insider BUYING (not selling) is informative.
# We use OPEN-MARKET PURCHASES only (transactionType starts "P"); option exercises
# (M), awards (A), gifts (G), in-kind (F), and sales (S) are excluded.

def get_insider_trades_fmp(symbol: str) -> List[Dict]:
    """
    Return all available insider OPEN-MARKET PURCHASE records for *symbol*.

    Source: FMP /stable/insider-trading/search (Form 4 filings). We paginate back
    far enough to cover the backtest window, then keep only open-market purchases.

    Each record:
      filing_date     — date the Form 4 became public (PIT date; YYYY-MM-DD)
      transaction_date— date of the trade (1-2 days before filing; not yet public)
      reporting_name  — insider name (used to count DISTINCT buyers in a cluster)
      type_of_owner   — role (director / officer / 10% owner)
      shares          — securities transacted (float)
      price           — transaction price (float)
      notional        — shares * price (USD)

    PIT correctness: callers must filter on filing_date <= as_of. The transaction
    date is NOT public until the Form 4 is filed (insiders have 2 business days),
    so acting on transaction_date would be look-ahead.
    """
    now = time.time()
    cached = _insider_cache.get(symbol)
    if cached and now - cached[1] < _CACHE_TTL:
        return cached[0]

    records: List[Dict] = []
    try:
        for page in range(_INSIDER_MAX_PAGES):
            resp = requests.get(
                f"{_BASE}/insider-trading/search",
                params={
                    "symbol": symbol,
                    "limit": _INSIDER_PAGE_LIMIT,
                    "page": page,
                    "apikey": _api_key(),
                },
                timeout=15,
            )
            if resp.status_code != 200:
                break
            rows = resp.json()
            if not rows:
                break
            for row in rows:
                ttype = str(row.get("transactionType") or "")
                # Open-market purchases only (e.g. "P-Purchase"). Exclude S/M/A/G/F.
                if not ttype.startswith("P"):
                    continue
                filing = row.get("filingDate") or ""
                if not filing:
                    continue
                try:
                    shares = float(row.get("securitiesTransacted") or 0.0)
                    price = float(row.get("price") or 0.0)
                except (TypeError, ValueError):
                    shares, price = 0.0, 0.0
                records.append({
                    "filing_date": filing[:10],
                    "transaction_date": (row.get("transactionDate") or "")[:10],
                    "reporting_name": (row.get("reportingName") or "").strip().upper(),
                    "type_of_owner": (row.get("typeOfOwner") or "").strip().lower(),
                    "shares": shares,
                    "price": price,
                    "notional": shares * price,
                })
            if len(rows) < _INSIDER_PAGE_LIMIT:
                break  # last page reached
    except Exception as exc:
        logger.debug("FMP insider fetch failed for %s: %s", symbol, exc)

    _insider_cache[symbol] = (records, now)
    return records


# Cluster thresholds (a-priori, academic basis cited above — NOT tuned to results).
# Window: cluster = multiple insiders buying within a trailing window. CMP/Lakonishok
# use ~quarterly aggregation; we use 30 calendar days (conservative, recent cluster).
INSIDER_CLUSTER_WINDOW_DAYS = 30
# A cluster requires >=2 DISTINCT insiders buying in the window (the documented
# strong "consensus" signal — multiple insiders is far more informative than one).
INSIDER_MIN_DISTINCT_BUYERS = 2
# Single large buy alternative: one insider buying >= this notional also fires
# (large dollar conviction is the other documented strong signal). $1M is a
# defensible "material" threshold for an individual open-market purchase.
INSIDER_LARGE_BUY_NOTIONAL = 1_000_000.0
# Only act within this many days of the cluster's latest filing (signal freshness;
# enter near the event, not weeks later). Matches PEAD's max-days-after pattern.
INSIDER_MAX_DAYS_AFTER_FILING = 5


def get_insider_features_at(symbol: str, as_of: date) -> Dict[str, float] | None:
    """
    Return insider-buying-cluster features using only Form 4 filings that were
    PUBLIC on or before *as_of* (filing_date <= as_of). PIT-safe.

    Looks at open-market purchases filed within the trailing
    INSIDER_CLUSTER_WINDOW_DAYS window ending at as_of.

    Features:
      insider_distinct_buyers   — # distinct insiders who bought in the window
      insider_buy_count         — # purchase filings in the window
      insider_total_notional    — total $ purchased in the window
      insider_max_notional      — largest single purchase $ in the window
      insider_days_since_filing — days since the most recent purchase filing
      insider_is_cluster        — 1.0 if cluster criteria met, else 0.0

    Returns None when the symbol has NO purchase filings at all (sentinel so the
    scorer can skip cheaply). A symbol with purchases but no recent cluster
    returns a dict with insider_is_cluster=0.0.
    """
    try:
        records = get_insider_trades_fmp(symbol)
        if not records:
            return None

        cutoff = as_of - timedelta(days=INSIDER_CLUSTER_WINDOW_DAYS)
        window = []
        for r in records:
            try:
                fd = datetime.strptime(r["filing_date"], "%Y-%m-%d").date()
            except (ValueError, KeyError):
                continue
            # PIT: only filings public on or before as_of, within the trailing window.
            if cutoff <= fd <= as_of:
                window.append((fd, r))

        if not window:
            return {
                "insider_distinct_buyers": 0.0,
                "insider_buy_count": 0.0,
                "insider_total_notional": 0.0,
                "insider_max_notional": 0.0,
                "insider_days_since_filing": float(INSIDER_CLUSTER_WINDOW_DAYS + 1),
                "insider_is_cluster": 0.0,
            }

        distinct_buyers = {r["reporting_name"] for _, r in window if r["reporting_name"]}
        notionals = [r["notional"] for _, r in window]
        total_notional = float(sum(notionals))
        max_notional = float(max(notionals)) if notionals else 0.0
        latest_fd = max(fd for fd, _ in window)
        days_since = float((as_of - latest_fd).days)

        is_cluster = (
            len(distinct_buyers) >= INSIDER_MIN_DISTINCT_BUYERS
            or max_notional >= INSIDER_LARGE_BUY_NOTIONAL
        )

        return {
            "insider_distinct_buyers": float(len(distinct_buyers)),
            "insider_buy_count": float(len(window)),
            "insider_total_notional": total_notional,
            "insider_max_notional": max_notional,
            "insider_days_since_filing": days_since,
            "insider_is_cluster": 1.0 if is_cluster else 0.0,
        }
    except Exception as exc:
        logger.debug("FMP insider features failed for %s: %s", symbol, exc)
        return None


# ── Convenience: all FMP features at once ────────────────────────────────────

def get_fmp_features_at(symbol: str, as_of: date) -> Dict[str, float]:
    """
    Return all FMP features for *symbol* as of *as_of* date.
    Safe — always returns a complete dict of defaults on any error.
    """
    features: Dict[str, float] = {}
    _earnings = get_earnings_features_at(symbol, as_of)
    if _earnings:
        features.update(_earnings)
    else:
        # Stable sentinel columns so feature matrix has consistent shape
        features.update({"fmp_surprise_1q": 0.0, "fmp_surprise_2q_avg": 0.0,
                         "fmp_days_since_earnings": 90.0})
    features.update(get_analyst_features_at(symbol, as_of))
    features.update(get_institutional_features_at(symbol, as_of))
    return features


def prefetch_fmp(symbols: List[str]) -> None:
    """
    Pre-warm all FMP caches for *symbols* in one pass.
    Call this once before a training run to avoid per-window API calls.
    """
    logger.info("Pre-fetching FMP data for %d symbols...", len(symbols))
    for sym in symbols:
        try:
            get_earnings_history_fmp(sym)
        except Exception:
            pass
        try:
            get_analyst_grades_fmp(sym)
        except Exception:
            pass
        try:
            get_insider_trades_fmp(sym)
        except Exception:
            pass
    logger.info("FMP prefetch complete")


# ── Helpers ───────────────────────────────────────────────────────────────────

_BULLISH_GRADES = {"buy", "outperform", "overweight", "strong buy", "accumulate", "positive"}
_BEARISH_GRADES = {"sell", "underperform", "underweight", "reduce", "negative", "strong sell"}


def _classify_action(prev: str, new: str, action_hint: str) -> str:
    """Map grade change to upgrade / downgrade / maintain / init."""
    hint = action_hint.lower()
    if "upgrade" in hint:
        return "upgrade"
    if "downgrade" in hint:
        return "downgrade"
    if "init" in hint or "initiate" in hint:
        return "init"
    p, n = prev.lower(), new.lower()
    if not p:
        return "init"
    p_bull = any(g in p for g in _BULLISH_GRADES)
    n_bull = any(g in n for g in _BULLISH_GRADES)
    p_bear = any(g in p for g in _BEARISH_GRADES)
    n_bear = any(g in n for g in _BEARISH_GRADES)
    if n_bull and not p_bull:
        return "upgrade"
    if n_bear and not p_bear:
        return "downgrade"
    return "maintain"
