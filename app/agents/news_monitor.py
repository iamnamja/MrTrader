"""
Phase 53: Live News Monitor — intraday news signal.

Polls Polygon news every 5 minutes during market hours for:
  - All currently held intraday symbols (exit flag)
  - Candidate symbols about to be entered (entry block)

Public API (thread-safe, async-safe):
  news_monitor.has_negative_news(symbol, window_minutes=30) -> bool
  news_monitor.get_latest_article(symbol) -> Optional[dict]

Start/stop is handled by the orchestrator (background asyncio task).
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 300   # 5 minutes
NEGATIVE_WINDOW_MINUTES = 30  # entry block / exit flag window
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16

# symbol → (most_recent_negative_utc, latest_article_dict)
_cache: Dict[str, Tuple[Optional[datetime], Optional[dict]]] = {}
_watched: set = set()   # symbols currently being polled


class NewsMonitor:
    """
    Background async poller for intraday news signals.

    Usage:
        # In orchestrator / main.py:
        asyncio.create_task(news_monitor.run())

        # In PM entry gate:
        if news_monitor.has_negative_news(symbol):
            skip entry
    """

    def __init__(self):
        self.status = "idle"
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def watch(self, symbol: str) -> None:
        """Add a symbol to the polling watchlist."""
        _watched.add(symbol.upper())

    def unwatch(self, symbol: str) -> None:
        """Remove a symbol from the watchlist (e.g. after position closes)."""
        _watched.discard(symbol.upper())

    def has_negative_news(self, symbol: str, window_minutes: int = NEGATIVE_WINDOW_MINUTES) -> bool:
        """
        Return True if a negative-sentiment article was published for *symbol*
        within the last *window_minutes* minutes.

        Fails open (returns False) if no data is available — never blocks entries
        due to a missing Polygon key or network error.
        """
        entry = _cache.get(symbol.upper())
        if entry is None:
            return False
        neg_ts, _ = entry
        if neg_ts is None:
            return False
        age = datetime.now(tz=timezone.utc) - neg_ts
        return age <= timedelta(minutes=window_minutes)

    def get_latest_article(self, symbol: str) -> Optional[dict]:
        """Return the most recent negative article dict, or None."""
        entry = _cache.get(symbol.upper())
        if entry is None:
            return None
        _, article = entry
        return article

    # ── Background loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main polling loop — runs as a background asyncio task."""
        logger.info("NewsMonitor started — polling every %ds", POLL_INTERVAL_SECONDS)
        self.status = "running"

        while self.status == "running":
            try:
                now = datetime.now(tz=timezone.utc)
                # Only poll during market hours (ET = UTC-4 or UTC-5)
                # Use a simple UTC window: 13:30–21:00 UTC covers 9:30–17:00 ET
                market_open = now.hour >= 13 and not (now.hour == 13 and now.minute < 30)
                market_close = now.hour < 21
                is_weekday = now.weekday() < 5

                if is_weekday and market_open and market_close and _watched:
                    await self._poll_all()
                else:
                    logger.debug(
                        "NewsMonitor idle — market_open=%s weekday=%s watching=%d",
                        market_open and market_close, is_weekday, len(_watched),
                    )

            except asyncio.CancelledError:
                logger.info("NewsMonitor cancelled — shutting down")
                self.status = "stopped"
                break
            except Exception as exc:
                logger.error("NewsMonitor poll error: %s", exc, exc_info=True)

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _poll_all(self) -> None:
        """Poll all watched symbols concurrently (via threadpool — Polygon is sync)."""
        symbols = list(_watched)
        if not symbols:
            return

        logger.debug("NewsMonitor polling %d symbol(s): %s", len(symbols), symbols)

        results = await asyncio.gather(
            *[asyncio.to_thread(self._poll_one, sym) for sym in symbols],
            return_exceptions=True,
        )

        neg_count = 0
        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.debug("NewsMonitor poll failed for %s: %s", sym, result)
                continue
            neg_ts, article = result
            async with self._lock:
                _cache[sym] = (neg_ts, article)
            if neg_ts is not None:
                neg_count += 1
                logger.info(
                    "NEWS NEGATIVE %s: '%s' published at %s",
                    sym, (article or {}).get("title", "?")[:60], neg_ts.strftime("%H:%M UTC"),
                )

        if neg_count:
            logger.warning(
                "NewsMonitor: %d/%d watched symbols have negative news in last %dmin",
                neg_count, len(symbols), NEGATIVE_WINDOW_MINUTES,
            )
        else:
            logger.debug("NewsMonitor: no negative news across %d symbols", len(symbols))

    @staticmethod
    def _poll_one(symbol: str) -> Tuple[Optional[datetime], Optional[dict]]:
        """
        Fetch live news for *symbol*.
        Returns (most_recent_negative_utc, article_dict) or (None, None).
        """
        from app.ml.news_features import fetch_news_live
        articles = fetch_news_live(symbol, hours_back=2)
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=NEGATIVE_WINDOW_MINUTES)

        for article in articles:  # already sorted desc by published_utc
            if article["sentiment"] == "negative" and article["published_utc"] >= cutoff:
                return article["published_utc"], article

        return None, None


# Module-level singleton — imported by PM and main.py
news_monitor = NewsMonitor()
