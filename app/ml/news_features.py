"""
News sentiment features via Polygon.io news API.

Polygon returns per-article, per-ticker sentiment (positive/neutral/negative)
with AI-generated reasoning.  We aggregate into rolling windows suitable for
the swing model.

Features computed:
  news_sentiment_3d    — avg sentiment score (positive=1, neutral=0, neg=-1) over 3 days
  news_sentiment_7d    — avg sentiment score over 7 days
  news_article_count_7d — article count (signal volume / attention)
  news_sentiment_momentum — 3d score minus 7d score (recent shift)

Point-in-time safe: we pass published_utc lte=as_of_date so training windows
only see articles that existed on that date.
"""

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://api.polygon.io"
_CACHE_TTL = 3600  # 1 h — news changes throughout the day

_news_cache: dict = {}   # symbol → (articles, fetched_at)

_SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _api_key() -> str:
    from app.config import settings
    return settings.polygon_api_key or ""


# ── News fetch ────────────────────────────────────────────────────────────────

def fetch_news(symbol: str, days_back: int = 14) -> List[Dict]:
    """
    Fetch recent news articles for *symbol* from Polygon.

    Returns list of dicts with:
      published_date  — date object
      sentiment       — 'positive', 'neutral', or 'negative'
      score           — float: 1.0 / 0.0 / -1.0
    """
    now = time.time()
    cached = _news_cache.get(symbol)
    if cached and now - cached[1] < _CACHE_TTL:
        return cached[0]

    articles = []
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "ticker": symbol,
            "published_utc.gte": cutoff,
            "limit": 50,
            "order": "desc",
            "sort": "published_utc",
            "apiKey": _api_key(),
        }
        resp = requests.get(f"{_BASE}/v2/reference/news", params=params, timeout=10)
        if resp.status_code != 200:
            logger.debug("Polygon news %s for %s", resp.status_code, symbol)
            _news_cache[symbol] = (articles, now)
            return articles

        for item in resp.json().get("results", []):
            pub_str = item.get("published_utc", "")[:10]
            try:
                pub_date = datetime.strptime(pub_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            # Find sentiment insight for this specific ticker
            insights = item.get("insights", [])
            ticker_sentiment = None
            for ins in insights:
                if ins.get("ticker", "").upper() == symbol.upper():
                    ticker_sentiment = ins.get("sentiment", "neutral")
                    break
            if ticker_sentiment is None and insights:
                ticker_sentiment = insights[0].get("sentiment", "neutral")

            articles.append({
                "published_date": pub_date,
                "sentiment": ticker_sentiment or "neutral",
                "score": _SENTIMENT_SCORE.get(ticker_sentiment or "neutral", 0.0),
            })
    except Exception as exc:
        logger.debug("Polygon news fetch failed for %s: %s", symbol, exc)

    _news_cache[symbol] = (articles, now)
    return articles


_live_cache: dict = {}   # symbol → (articles, fetched_at); short TTL for intraday use
_LIVE_CACHE_TTL = 290    # ~5 min, just under NewsMonitor poll interval


def fetch_news_live(symbol: str, hours_back: int = 2) -> List[Dict]:
    """
    Fetch recent articles with full UTC datetime objects (not date-only).
    Used by NewsMonitor for live negative-news detection.

    Returns list of dicts with:
      published_utc  — tz-aware datetime
      sentiment      — 'positive', 'neutral', or 'negative'
      score          — float
      title          — str
    """
    now = time.time()
    cached = _live_cache.get(symbol)
    if cached and now - cached[1] < _LIVE_CACHE_TTL:
        return cached[0]

    articles = []
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "ticker": symbol,
            "published_utc.gte": cutoff,
            "limit": 20,
            "order": "desc",
            "sort": "published_utc",
            "apiKey": _api_key(),
        }
        resp = requests.get(f"{_BASE}/v2/reference/news", params=params, timeout=10)
        if resp.status_code != 200:
            logger.debug("Polygon live news %s for %s", resp.status_code, symbol)
            _live_cache[symbol] = (articles, now)
            return articles

        for item in resp.json().get("results", []):
            pub_str = item.get("published_utc", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                continue

            insights = item.get("insights", [])
            ticker_sentiment = None
            for ins in insights:
                if ins.get("ticker", "").upper() == symbol.upper():
                    ticker_sentiment = ins.get("sentiment", "neutral")
                    break
            if ticker_sentiment is None and insights:
                ticker_sentiment = insights[0].get("sentiment", "neutral")

            articles.append({
                "published_utc": pub_dt,
                "sentiment": ticker_sentiment or "neutral",
                "score": _SENTIMENT_SCORE.get(ticker_sentiment or "neutral", 0.0),
                "title": item.get("title", ""),
            })
    except Exception as exc:
        logger.debug("Polygon live news fetch failed for %s: %s", symbol, exc)

    _live_cache[symbol] = (articles, now)
    return articles


def fetch_news_historical(symbol: str, as_of: date, days_back: int = 14) -> List[Dict]:
    """
    Fetch news articles published on or before *as_of* (point-in-time safe).
    Used during training to avoid look-ahead bias.
    """
    articles = []
    try:
        cutoff = (as_of - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "ticker": symbol,
            "published_utc.gte": cutoff,
            "published_utc.lte": as_of.strftime("%Y-%m-%d"),
            "limit": 50,
            "order": "desc",
            "sort": "published_utc",
            "apiKey": _api_key(),
        }
        resp = requests.get(f"{_BASE}/v2/reference/news", params=params, timeout=10)
        if resp.status_code != 200:
            return articles

        for item in resp.json().get("results", []):
            pub_str = item.get("published_utc", "")[:10]
            try:
                pub_date = datetime.strptime(pub_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            insights = item.get("insights", [])
            ticker_sentiment = None
            for ins in insights:
                if ins.get("ticker", "").upper() == symbol.upper():
                    ticker_sentiment = ins.get("sentiment", "neutral")
                    break
            if ticker_sentiment is None and insights:
                ticker_sentiment = insights[0].get("sentiment", "neutral")

            articles.append({
                "published_date": pub_date,
                "sentiment": ticker_sentiment or "neutral",
                "score": _SENTIMENT_SCORE.get(ticker_sentiment or "neutral", 0.0),
            })
    except Exception as exc:
        logger.debug("Polygon news historical failed for %s: %s", symbol, exc)

    return articles


# ── Feature computation ───────────────────────────────────────────────────────

def compute_news_features(articles: List[Dict], as_of: date) -> Dict[str, float]:
    """
    Aggregate article list into rolling-window features.

    Args:
        articles: list from fetch_news() or fetch_news_historical()
        as_of:    reference date for computing windows

    Returns dict with 4 features.
    """
    result = {
        "news_sentiment_3d": 0.0,
        "news_sentiment_7d": 0.0,
        "news_article_count_7d": 0.0,
        "news_sentiment_momentum": 0.0,
    }

    if not articles:
        return result

    cutoff_3d = as_of - timedelta(days=3)
    cutoff_7d = as_of - timedelta(days=7)

    scores_3d = [a["score"] for a in articles if cutoff_3d <= a["published_date"] <= as_of]
    scores_7d = [a["score"] for a in articles if cutoff_7d <= a["published_date"] <= as_of]

    if scores_3d:
        result["news_sentiment_3d"] = float(sum(scores_3d) / len(scores_3d))
    if scores_7d:
        result["news_sentiment_7d"] = float(sum(scores_7d) / len(scores_7d))
        result["news_article_count_7d"] = float(len(scores_7d))

    result["news_sentiment_momentum"] = float(
        result["news_sentiment_3d"] - result["news_sentiment_7d"]
    )

    return result


def get_news_features(
    symbol: str, as_of_date: Optional[date] = None
) -> Dict[str, float]:
    """
    Top-level function: fetch + compute news features for *symbol*.

    - Live inference (as_of_date=None): uses cached recent articles
    - Training (as_of_date set): fetches point-in-time articles for that date

    Returns safe defaults on any failure.
    """
    _default = {
        "news_sentiment_3d": 0.0,
        "news_sentiment_7d": 0.0,
        "news_article_count_7d": 0.0,
        "news_sentiment_momentum": 0.0,
    }
    try:
        ref = as_of_date or date.today()
        if as_of_date is not None:
            articles = fetch_news_historical(symbol, as_of=as_of_date, days_back=14)
        else:
            articles = fetch_news(symbol, days_back=14)
        return compute_news_features(articles, ref)
    except Exception as exc:
        logger.debug("get_news_features failed for %s: %s", symbol, exc)
        return _default
