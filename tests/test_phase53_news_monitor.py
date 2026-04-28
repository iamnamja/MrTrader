"""Phase 53: NewsMonitor unit tests — no network, no DB."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.agents.news_monitor import NewsMonitor, _cache, _watched, NEGATIVE_WINDOW_MINUTES


@pytest.fixture(autouse=True)
def clear_state():
    _cache.clear()
    _watched.clear()
    yield
    _cache.clear()
    _watched.clear()


def _make_article(minutes_ago: int, sentiment: str) -> dict:
    return {
        "published_utc": datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_ago),
        "sentiment": sentiment,
        "score": -1.0 if sentiment == "negative" else 0.0,
        "title": f"Test article ({sentiment})",
    }


class TestHasNegativeNews:
    def test_no_cache_returns_false(self):
        monitor = NewsMonitor()
        assert monitor.has_negative_news("AAPL") is False

    def test_cached_negative_within_window_returns_true(self):
        monitor = NewsMonitor()
        neg_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
        article = {"title": "Bad news", "sentiment": "negative"}
        _cache["AAPL"] = (neg_ts, article)
        assert monitor.has_negative_news("AAPL") is True

    def test_cached_negative_outside_window_returns_false(self):
        monitor = NewsMonitor()
        neg_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=NEGATIVE_WINDOW_MINUTES + 5)
        _cache["AAPL"] = (neg_ts, {"title": "Old bad news"})
        assert monitor.has_negative_news("AAPL") is False

    def test_cached_no_negative_returns_false(self):
        monitor = NewsMonitor()
        _cache["AAPL"] = (None, None)
        assert monitor.has_negative_news("AAPL") is False

    def test_case_insensitive(self):
        monitor = NewsMonitor()
        neg_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
        _cache["MSFT"] = (neg_ts, {"title": "Bad"})
        assert monitor.has_negative_news("msft") is True

    def test_custom_window(self):
        monitor = NewsMonitor()
        neg_ts = datetime.now(tz=timezone.utc) - timedelta(minutes=20)
        _cache["GOOG"] = (neg_ts, {"title": "Bad"})
        assert monitor.has_negative_news("GOOG", window_minutes=15) is False
        assert monitor.has_negative_news("GOOG", window_minutes=25) is True


class TestWatchUnwatch:
    def test_watch_adds_symbol(self):
        monitor = NewsMonitor()
        monitor.watch("TSLA")
        assert "TSLA" in _watched

    def test_unwatch_removes_symbol(self):
        monitor = NewsMonitor()
        monitor.watch("TSLA")
        monitor.unwatch("TSLA")
        assert "TSLA" not in _watched

    def test_unwatch_missing_symbol_ok(self):
        monitor = NewsMonitor()
        monitor.unwatch("NONEXISTENT")  # should not raise


class TestPollOne:
    def test_poll_one_returns_none_on_positive_news(self):
        articles = [_make_article(5, "positive"), _make_article(15, "neutral")]
        with patch("app.ml.news_features.fetch_news_live", return_value=articles):
            result = NewsMonitor._poll_one("AAPL")
        assert result == (None, None)

    def test_poll_one_returns_negative_within_window(self):
        articles = [_make_article(10, "negative")]
        with patch("app.ml.news_features.fetch_news_live", return_value=articles):
            neg_ts, article = NewsMonitor._poll_one("AAPL")
        assert neg_ts is not None
        assert article["sentiment"] == "negative"

    def test_poll_one_ignores_old_negative(self):
        articles = [_make_article(NEGATIVE_WINDOW_MINUTES + 10, "negative")]
        with patch("app.ml.news_features.fetch_news_live", return_value=articles):
            result = NewsMonitor._poll_one("AAPL")
        assert result == (None, None)

    def test_poll_one_picks_most_recent_negative(self):
        articles = [
            _make_article(5, "negative"),   # most recent — should be returned
            _make_article(20, "negative"),
        ]
        with patch("app.ml.news_features.fetch_news_live", return_value=articles):
            neg_ts, article = NewsMonitor._poll_one("AAPL")
        # articles are sorted desc, so first negative hit is the most recent
        age = datetime.now(tz=timezone.utc) - neg_ts
        assert age < timedelta(minutes=10)
