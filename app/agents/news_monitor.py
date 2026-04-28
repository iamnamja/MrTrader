"""
Phase 53: Live news monitor — dual source (Polygon + Alpaca News).

Runs as a background async task during market hours. Polls both Polygon and
Alpaca News APIs every 5 minutes for watched symbols. Caches the most recent
negative article per symbol; PM/RM query `has_negative_news()` to gate entries
and flag exits.

Alpaca News has no pre-labeled sentiment, so we apply a lightweight keyword
classifier on the headline + summary.  Polygon provides its own AI sentiment
label which we trust directly.

Fails open: `has_negative_news()` returns False on any error so it never
blocks trading due to a news API outage.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Negative-keyword classifier for Alpaca News (no sentiment labels) ─────────

_NEG_KEYWORDS = {
    "downgrade", "cut", "miss", "misses", "missed", "below", "disappointed",
    "disappoints", "disappointing", "loss", "losses", "loses", "lower",
    "drop", "drops", "dropped", "fell", "fall", "tumble", "tumbles", "tumbled",
    "plunge", "plunges", "plunged", "decline", "declines", "declined",
    "warning", "warns", "warned", "recall", "layoff", "layoffs", "bankrupt",
    "bankruptcy", "lawsuit", "probe", "investigation", "fraud", "default",
    "guidance", "cut guidance", "misses estimates", "beats estimates" # "beats" is NOT negative
}
_POS_OVERRIDES = {"beats", "beat", "exceeds", "exceeded", "surpasses", "raised guidance"}


def _classify_negative(headline: str, summary: str = "") -> bool:
    """Return True if the text looks negative based on keyword matching."""
    text = (headline + " " + (summary or "")).lower()
    words = set(text.split())
    # A "beats estimates" headline is positive despite containing some neg words
    if words & _POS_OVERRIDES:
        return False
    return bool(words & _NEG_KEYWORDS)


# ── Cache entry ────────────────────────────────────────────────────────────────

class _CacheEntry:
    __slots__ = ("article", "fetched_at")

    def __init__(self, article: Optional[Dict], fetched_at: datetime):
        self.article = article        # None → no negative news found
        self.fetched_at = fetched_at


# ── NewsMonitor ────────────────────────────────────────────────────────────────

_POLL_INTERVAL = 300          # seconds between full sweeps
_NEG_WINDOW_MINUTES = 30      # how far back to look for negative articles
_MARKET_OPEN_UTC = 13         # 09:00 ET = 13:00 UTC (DST approx)
_MARKET_CLOSE_UTC = 21        # 17:00 ET = 21:00 UTC


class NewsMonitor:
    def __init__(self):
        self._cache: Dict[str, _CacheEntry] = {}
        self._watched: Set[str] = set()
        self._lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def watch(self, symbol: str) -> None:
        self._watched.add(symbol.upper())

    def unwatch(self, symbol: str) -> None:
        self._watched.discard(symbol.upper())

    def has_negative_news(self, symbol: str, window_minutes: int = _NEG_WINDOW_MINUTES) -> bool:
        """Return True iff a negative article was published within *window_minutes*."""
        try:
            entry = self._cache.get(symbol.upper())
            if entry is None or entry.article is None:
                return False
            pub = entry.article.get("published_utc")
            if pub is None:
                return False
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(minutes=window_minutes)
            return pub >= cutoff
        except Exception:
            return False  # fail open

    def get_latest_article(self, symbol: str) -> Optional[Dict]:
        entry = self._cache.get(symbol.upper())
        return entry.article if entry else None

    # ── Background loop ────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("NewsMonitor started")
        while True:
            try:
                await self._sweep()
            except Exception as exc:
                logger.debug("NewsMonitor sweep error: %s", exc)
            await asyncio.sleep(_POLL_INTERVAL)

    async def _sweep(self) -> None:
        now_utc = datetime.now(timezone.utc)
        # Only poll during market hours (Mon–Fri, roughly 09:00–17:00 ET)
        if now_utc.weekday() >= 5:
            return
        if not (_MARKET_OPEN_UTC <= now_utc.hour < _MARKET_CLOSE_UTC):
            return

        symbols = list(self._watched)
        if not symbols:
            return

        logger.debug("NewsMonitor polling %d symbols", len(symbols))
        for sym in symbols:
            try:
                article = await asyncio.to_thread(self._poll_one, sym)
                async with self._lock:
                    self._cache[sym] = _CacheEntry(article, now_utc)
            except Exception as exc:
                logger.debug("NewsMonitor poll failed for %s: %s", sym, exc)

    # ── Per-symbol poll (runs in thread) ──────────────────────────────────────

    @staticmethod
    def _poll_one(symbol: str) -> Optional[Dict]:
        """
        Fetch recent articles from both Polygon and Alpaca News.
        Returns the most recent negative article dict, or None.
        """
        candidates: List[Dict] = []
        candidates.extend(NewsMonitor._poll_polygon(symbol))
        candidates.extend(NewsMonitor._poll_alpaca(symbol))

        if not candidates:
            return None

        # Sort descending by publish time, return most recent negative
        candidates.sort(key=lambda a: a["published_utc"], reverse=True)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=_NEG_WINDOW_MINUTES)
        for article in candidates:
            if article["published_utc"] >= cutoff:
                return article
        return None

    @staticmethod
    def _poll_polygon(symbol: str) -> List[Dict]:
        """Fetch from Polygon (pre-labeled sentiment)."""
        from app.ml.news_features import fetch_news_live
        try:
            raw = fetch_news_live(symbol, hours_back=2)
            results = []
            for a in raw:
                if a.get("sentiment") == "negative":
                    results.append({
                        "published_utc": a["published_utc"],
                        "title": a.get("title", ""),
                        "source": "polygon",
                        "sentiment": "negative",
                    })
            return results
        except Exception as exc:
            logger.debug("Polygon news poll failed for %s: %s", symbol, exc)
            return []

    @staticmethod
    def _poll_alpaca(symbol: str) -> List[Dict]:
        """Fetch from Alpaca News (keyword-classified sentiment)."""
        try:
            from alpaca.data.historical import NewsClient
            from alpaca.data.requests import NewsRequest
            from app.config import settings

            client = NewsClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )
            start = datetime.now(timezone.utc) - timedelta(hours=2)
            req = NewsRequest(symbols=symbol.upper(), limit=10, start=start)
            news_set = client.get_news(req)
            articles = news_set.data.get("news", [])

            results = []
            for a in articles:
                pub = a.created_at
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                headline = a.headline or ""
                summary = a.summary or ""
                if _classify_negative(headline, summary):
                    results.append({
                        "published_utc": pub,
                        "title": headline,
                        "source": "alpaca",
                        "sentiment": "negative",
                    })
            return results
        except Exception as exc:
            logger.debug("Alpaca news poll failed for %s: %s", symbol, exc)
            return []


# Module singleton
news_monitor = NewsMonitor()

# Expose _watched for daily reset in portfolio_manager
_watched = news_monitor._watched
