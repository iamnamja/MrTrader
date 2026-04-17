"""Tests for Phase 38: Polygon news sentiment features."""
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pytest


def _polygon_news_response(ticker="AAPL", n=5):
    """Fake Polygon /v2/reference/news response."""
    from datetime import date, timedelta
    today = date.today()
    results = []
    sentiments = ["positive", "neutral", "negative", "positive", "neutral"]
    for i in range(n):
        pub = (today - timedelta(days=i)).strftime("%Y-%m-%dT12:00:00Z")
        results.append({
            "title": f"Article {i}",
            "published_utc": pub,
            "insights": [{"ticker": ticker, "sentiment": sentiments[i % len(sentiments)]}],
        })
    return {"results": results, "status": "OK"}


class TestFetchNews:

    def test_returns_list_with_scores(self):
        from app.ml.news_features import fetch_news
        import app.ml.news_features as nm
        nm._news_cache.clear()
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _polygon_news_response("AAPL_N1")
            result = fetch_news("AAPL_N1", days_back=14)
        assert isinstance(result, list)
        assert len(result) == 5
        assert all("score" in a for a in result)
        assert all("published_date" in a for a in result)

    def test_positive_article_score_is_1(self):
        from app.ml.news_features import fetch_news
        import app.ml.news_features as nm
        nm._news_cache.clear()
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "results": [{
                    "published_utc": "2026-04-10T12:00:00Z",
                    "insights": [{"ticker": "AAPL_N2", "sentiment": "positive"}],
                }]
            }
            result = fetch_news("AAPL_N2", days_back=14)
        assert result[0]["score"] == 1.0

    def test_negative_article_score_is_minus1(self):
        from app.ml.news_features import fetch_news
        import app.ml.news_features as nm
        nm._news_cache.clear()
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "results": [{
                    "published_utc": "2026-04-10T12:00:00Z",
                    "insights": [{"ticker": "AAPL_N3", "sentiment": "negative"}],
                }]
            }
            result = fetch_news("AAPL_N3", days_back=14)
        assert result[0]["score"] == -1.0

    def test_returns_empty_on_api_error(self):
        from app.ml.news_features import fetch_news
        import app.ml.news_features as nm
        nm._news_cache.clear()
        with patch("requests.get", side_effect=Exception("timeout")):
            result = fetch_news("FAIL_NEWS")
        assert result == []

    def test_uses_cache(self):
        from app.ml.news_features import fetch_news
        import app.ml.news_features as nm
        nm._news_cache.clear()
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _polygon_news_response("CACHE_NEWS")
            fetch_news("CACHE_NEWS")
            fetch_news("CACHE_NEWS")
        assert mock_get.call_count == 1

    def test_returns_empty_on_non_200(self):
        from app.ml.news_features import fetch_news
        import app.ml.news_features as nm
        nm._news_cache.clear()
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 403
            result = fetch_news("FORBIDDEN_NEWS")
        assert result == []


class TestComputeNewsFeatures:

    def _articles(self, today=None):
        from datetime import date, timedelta
        today = today or date.today()
        return [
            {"published_date": today - timedelta(days=0), "score": 1.0, "sentiment": "positive"},
            {"published_date": today - timedelta(days=1), "score": 1.0, "sentiment": "positive"},
            {"published_date": today - timedelta(days=4), "score": -1.0, "sentiment": "negative"},
            {"published_date": today - timedelta(days=6), "score": 0.0, "sentiment": "neutral"},
            {"published_date": today - timedelta(days=10), "score": -1.0, "sentiment": "negative"},
        ]

    def test_returns_4_keys(self):
        from app.ml.news_features import compute_news_features
        result = compute_news_features(self._articles(), date.today())
        assert set(result.keys()) == {
            "news_sentiment_3d", "news_sentiment_7d",
            "news_article_count_7d", "news_sentiment_momentum",
        }

    def test_3d_sentiment_only_recent(self):
        from app.ml.news_features import compute_news_features
        today = date.today()
        result = compute_news_features(self._articles(today), today)
        # Only 2 articles in last 3 days (both positive = score 1.0)
        assert result["news_sentiment_3d"] == 1.0

    def test_7d_includes_more_articles(self):
        from app.ml.news_features import compute_news_features
        today = date.today()
        result = compute_news_features(self._articles(today), today)
        # 4 articles in 7 days: +1, +1, -1, 0 → avg = 0.25
        assert abs(result["news_sentiment_7d"] - 0.25) < 0.01

    def test_article_count_7d(self):
        from app.ml.news_features import compute_news_features
        today = date.today()
        result = compute_news_features(self._articles(today), today)
        assert result["news_article_count_7d"] == 4.0

    def test_momentum_is_3d_minus_7d(self):
        from app.ml.news_features import compute_news_features
        today = date.today()
        result = compute_news_features(self._articles(today), today)
        expected = result["news_sentiment_3d"] - result["news_sentiment_7d"]
        assert abs(result["news_sentiment_momentum"] - expected) < 0.001

    def test_empty_articles_returns_zeros(self):
        from app.ml.news_features import compute_news_features
        result = compute_news_features([], date.today())
        assert result["news_sentiment_3d"] == 0.0
        assert result["news_article_count_7d"] == 0.0

    def test_point_in_time_as_of_excludes_future(self):
        from app.ml.news_features import compute_news_features
        from datetime import date, timedelta
        past_date = date(2023, 6, 1)
        articles = [
            {"published_date": date(2023, 5, 30), "score": 1.0, "sentiment": "positive"},
            {"published_date": date(2023, 6, 5), "score": -1.0, "sentiment": "negative"},  # future!
        ]
        result = compute_news_features(articles, past_date)
        # Only the May 30 article should count in 7d window
        assert result["news_sentiment_7d"] == 1.0


class TestFetchNewsHistorical:

    def test_sends_lte_param(self):
        from app.ml.news_features import fetch_news_historical
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"results": []}
            fetch_news_historical("AAPL", as_of=date(2023, 6, 1))
        call_params = mock_get.call_args[1]["params"]
        assert "published_utc.lte" in call_params
        assert call_params["published_utc.lte"] == "2023-06-01"

    def test_returns_empty_on_error(self):
        from app.ml.news_features import fetch_news_historical
        with patch("requests.get", side_effect=Exception("fail")):
            result = fetch_news_historical("AAPL", date(2023, 6, 1))
        assert result == []


class TestFeatureEngineerNewsIntegration:

    def _make_bars(self, n=100):
        import pandas as pd
        import numpy as np
        idx = pd.bdate_range("2023-01-02", periods=n)
        p = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({
            "open": p * 0.999, "high": p * 1.005,
            "low": p * 0.995, "close": p,
            "volume": 1_000_000 * p / p,
        }, index=idx)

    def test_news_features_in_output(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        df = self._make_bars()
        news_mock = {
            "news_sentiment_3d": 0.5, "news_sentiment_7d": 0.3,
            "news_article_count_7d": 5.0, "news_sentiment_momentum": 0.2,
        }
        with patch("app.ml.news_features.get_news_features", return_value=news_mock):
            result = fe.engineer_features(
                "AAPL", df, fetch_fundamentals=True, as_of_date=date(2023, 4, 30)
            )
        assert result is not None
        assert result["news_sentiment_3d"] == 0.5
        assert result["news_article_count_7d"] == 5.0

    def test_news_zeros_when_fetch_disabled(self):
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        df = self._make_bars()
        result = fe.engineer_features("AAPL", df, fetch_fundamentals=False, as_of_date=date(2023, 4, 30))
        assert result is not None
        assert result["news_sentiment_3d"] == 0.0
        assert result["news_article_count_7d"] == 0.0
