"""
Tests for the News Intelligence Service (NIS).

Covers:
- Tier 1: macro_classify() — LLM call, cache, fallback, empty events
- Tier 2: stock_score() — LLM call, cache, fallback, empty articles, clamp
- NewsIntelligenceService — wiring, fail-open, batch helper
- Finnhub source adapter — event classification, calendar parsing
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ── Tier 1 — macro_classify ───────────────────────────────────────────────────

class TestMacroClassify:
    def _make_events(self):
        return [
            {
                "event_type": "FOMC",
                "event_name": "Federal Funds Rate",
                "event_time": datetime(2026, 4, 29, 18, 0, tzinfo=timezone.utc),
                "importance": "high",
                "estimate": 4.25,
                "prior": 4.25,
                "actual": None,
                "country": "US",
                "currency": "%",
                "source": "finnhub",
            }
        ]

    def test_empty_events_returns_low_risk(self):
        from app.news.llm_scorer import macro_classify
        result = macro_classify([])
        assert result["risk_level"] == "LOW"
        assert result["block_new_entries"] is False
        assert result["sizing_factor"] == 1.0

    def test_llm_result_cached_on_second_call(self):
        from app.news import llm_scorer
        llm_scorer._cache.clear()

        mock_result = {
            "risk_level": "MEDIUM",
            "direction": "NEUTRAL",
            "sizing_factor": 0.75,
            "block_new_entries": False,
            "consensus_summary": "Hold expected",
            "rationale": "High consensus, priced in",
        }

        with patch.object(llm_scorer, "_call_llm", return_value=mock_result) as mock_llm:
            events = self._make_events()
            r1 = llm_scorer.macro_classify(events)
            r2 = llm_scorer.macro_classify(events)
            assert mock_llm.call_count == 1
            assert r1 == r2 == mock_result

    def test_llm_failure_returns_none(self):
        from app.news import llm_scorer
        llm_scorer._cache.clear()

        with patch.object(llm_scorer, "_call_llm", return_value=None):
            result = llm_scorer.macro_classify(self._make_events())
            assert result is None

    def test_high_risk_uncertain_sets_block(self):
        from app.news import llm_scorer
        llm_scorer._cache.clear()

        mock_result = {
            "risk_level": "HIGH",
            "direction": "BEARISH",
            "sizing_factor": 0.5,
            "block_new_entries": True,
            "consensus_summary": "Split consensus",
            "rationale": "Genuine uncertainty",
        }
        with patch.object(llm_scorer, "_call_llm", return_value=mock_result):
            result = llm_scorer.macro_classify(self._make_events())
            assert result["block_new_entries"] is True
            assert result["sizing_factor"] == 0.5


# ── Tier 2 — stock_score ──────────────────────────────────────────────────────

class TestStockScore:
    def _make_articles(self, n=3):
        return [
            {
                "headline": f"Headline {i}: AAPL beats expectations",
                "summary": "Strong results reported.",
                "source": "Reuters",
                "url": "https://example.com",
                "published_at": datetime(2026, 4, 29, 10 + i, 0, tzinfo=timezone.utc),
                "sentiment": "positive",
            }
            for i in range(n)
        ]

    def test_empty_articles_returns_neutral(self):
        from app.news.llm_scorer import stock_score
        result = stock_score("AAPL", [])
        assert result["action_policy"] == "ignore"
        assert result["materiality_score"] == 0.0
        assert result["confidence"] == 0.2

    def test_llm_result_values_clamped(self):
        from app.news import llm_scorer
        llm_scorer._cache.clear()

        mock_result = {
            "direction_score": 9.9,          # should clamp to 1.0
            "materiality_score": -0.5,        # should clamp to 0.0
            "downside_risk_score": 0.3,
            "upside_catalyst_score": 0.8,
            "confidence": 0.9,
            "already_priced_in_score": 0.2,
            "action_policy": "size_up_light",
            "sizing_multiplier": 99.0,        # should clamp to 1.25
            "rationale": "Strong beat",
        }
        with patch.object(llm_scorer, "_call_llm", return_value=mock_result):
            result = llm_scorer.stock_score("AAPL", self._make_articles())
            assert result["direction_score"] == 1.0
            assert result["materiality_score"] == 0.0
            assert result["sizing_multiplier"] == 1.25

    def test_llm_cached_on_repeated_headlines(self):
        from app.news import llm_scorer
        llm_scorer._cache.clear()

        mock_result = {
            "direction_score": 0.5,
            "materiality_score": 0.7,
            "downside_risk_score": 0.1,
            "upside_catalyst_score": 0.6,
            "confidence": 0.8,
            "already_priced_in_score": 0.2,
            "action_policy": "watch",
            "sizing_multiplier": 1.0,
            "rationale": "Positive news",
        }
        articles = self._make_articles()
        with patch.object(llm_scorer, "_call_llm", return_value=mock_result) as mock_llm:
            llm_scorer.stock_score("AAPL", articles)
            llm_scorer.stock_score("AAPL", articles)
            assert mock_llm.call_count == 1

    def test_llm_failure_returns_none(self):
        from app.news import llm_scorer
        llm_scorer._cache.clear()

        with patch.object(llm_scorer, "_call_llm", return_value=None):
            result = llm_scorer.stock_score("AAPL", self._make_articles())
            assert result is None


# ── NewsIntelligenceService ───────────────────────────────────────────────────

class TestNewsIntelligenceService:
    def setup_method(self):
        from app.news import intelligence_service
        intelligence_service._macro_cache.clear()
        intelligence_service._stock_cache.clear()

    def test_get_macro_context_neutral_on_nis_failure(self):
        from app.news.intelligence_service import NewsIntelligenceService
        svc = NewsIntelligenceService()
        with patch("app.news.intelligence_service.nis", svc):
            with patch.object(svc, "_build_macro_context", side_effect=Exception("fail")):
                # The public method catches and returns neutral
                # Test _build_macro_context fail path directly
                pass

        # Test that _build_macro_context returns neutral when llm_scorer returns None
        with patch("app.news.llm_scorer.macro_classify", return_value=None), \
             patch("app.news.sources.finnhub_source.fetch_economic_calendar", return_value=[]):
            ctx = svc._build_macro_context()
            assert ctx.overall_risk == "LOW"
            assert ctx.block_new_entries is False
            assert ctx.global_sizing_factor == 1.0

    def test_get_stock_signal_neutral_on_failure(self):
        from app.news.intelligence_service import NewsIntelligenceService
        svc = NewsIntelligenceService()

        with patch("app.news.sources.finnhub_source.fetch_company_news", side_effect=Exception("api down")):
            sig = svc._build_stock_signal("TSLA", "Auto", 24, None)
            assert sig.symbol == "TSLA"
            assert sig.action_policy == "ignore"
            assert sig.sizing_multiplier == 1.0

    def test_stock_signal_cache_ttl(self):
        from app.news.intelligence_service import NewsIntelligenceService, _stock_cache
        from app.news.signal import NewsSignal

        svc = NewsIntelligenceService()
        stale_sig = NewsSignal.neutral("MSFT")
        # Set evaluated_at to old timestamp (2 hours ago)
        old_dt = datetime.fromtimestamp(time.time() - 7200, tz=timezone.utc)
        object.__setattr__(stale_sig, "evaluated_at", old_dt)
        _stock_cache["MSFT"] = stale_sig

        fresh_result = {
            "direction_score": 0.3,
            "materiality_score": 0.5,
            "downside_risk_score": 0.1,
            "upside_catalyst_score": 0.4,
            "confidence": 0.7,
            "already_priced_in_score": 0.1,
            "action_policy": "watch",
            "sizing_multiplier": 1.0,
            "rationale": "Fresh signal",
        }
        with patch("app.news.sources.finnhub_source.fetch_company_news", return_value=[]), \
             patch("app.news.llm_scorer.stock_score", return_value=None):
            sig = svc.get_stock_signal("MSFT")
            assert sig.rationale == "No recent news"  # neutral (articles=[])

    def test_batch_returns_all_symbols(self):
        from app.news.intelligence_service import NewsIntelligenceService
        svc = NewsIntelligenceService()

        with patch.object(svc, "_build_stock_signal") as mock_build:
            from app.news.signal import NewsSignal
            mock_build.side_effect = lambda sym, *a, **kw: NewsSignal.neutral(sym)
            result = svc.get_stock_signals_batch(["AAPL", "TSLA", "MSFT"])
            assert set(result.keys()) == {"AAPL", "TSLA", "MSFT"}
            assert all(v.action_policy == "ignore" for v in result.values())


# ── Finnhub source: event classification ─────────────────────────────────────

class TestFinnhubSource:
    def test_classify_fomc(self):
        from app.news.sources.finnhub_source import _classify_event
        assert _classify_event("Federal Funds Rate Decision") == "FOMC"
        assert _classify_event("FOMC Meeting Minutes") == "FOMC"

    def test_classify_nfp(self):
        from app.news.sources.finnhub_source import _classify_event
        assert _classify_event("Nonfarm Payrolls") == "NFP"
        assert _classify_event("Non-Farm Employment") == "NFP"

    def test_classify_cpi(self):
        from app.news.sources.finnhub_source import _classify_event
        assert _classify_event("Consumer Price Index MoM") == "CPI"

    def test_classify_unknown_returns_none(self):
        from app.news.sources.finnhub_source import _classify_event
        assert _classify_event("Michigan Consumer Sentiment") is None

    def test_fetch_economic_calendar_empty_on_no_key(self):
        from app.news.sources.finnhub_source import fetch_economic_calendar
        with patch("app.news.sources.finnhub_source._key", return_value=None):
            result = fetch_economic_calendar()
            assert result == []

    def test_fetch_company_news_empty_on_no_key(self):
        from app.news.sources.finnhub_source import fetch_company_news
        with patch("app.news.sources.finnhub_source._key", return_value=None):
            result = fetch_company_news("AAPL")
            assert result == []

    def test_fetch_economic_calendar_filters_past_events(self):
        from app.news.sources.finnhub_source import fetch_economic_calendar
        old_time = "2020-01-01 14:00:00"
        mock_data = {
            "economicCalendar": [
                {
                    "time": old_time,
                    "event": "Nonfarm Payrolls",
                    "impact": "high",
                    "country": "US",
                    "estimate": 200,
                    "prev": 180,
                    "actual": None,
                    "unit": "K",
                }
            ]
        }
        with patch("app.news.sources.finnhub_source._get", return_value=mock_data):
            result = fetch_economic_calendar()
            assert result == []

    def test_fetch_economic_calendar_parses_valid_event(self):
        from app.news.sources.finnhub_source import fetch_economic_calendar
        from datetime import datetime, timezone, timedelta

        future_dt = datetime.now(timezone.utc) + timedelta(hours=1)
        future_str = future_dt.strftime("%Y-%m-%d %H:%M:%S")

        mock_data = {
            "economicCalendar": [
                {
                    "time": future_str,
                    "event": "Consumer Price Index MoM",
                    "impact": "high",
                    "country": "US",
                    "estimate": 0.3,
                    "prev": 0.2,
                    "actual": None,
                    "unit": "%",
                }
            ]
        }
        with patch("app.news.sources.finnhub_source._get", return_value=mock_data):
            result = fetch_economic_calendar()
            assert len(result) == 1
            assert result[0]["event_type"] == "CPI"
            assert result[0]["importance"] == "high"
            assert result[0]["estimate"] == 0.3


# ── NewsSignal dataclass ──────────────────────────────────────────────────────

class TestNewsSignal:
    def test_neutral_factory(self):
        from app.news.signal import NewsSignal
        sig = NewsSignal.neutral("GOOG")
        assert sig.symbol == "GOOG"
        assert sig.action_policy == "ignore"
        assert sig.sizing_multiplier == 1.0

    def test_news_score_overlay_positive_catalyst(self):
        from app.news.signal import NewsSignal
        sig = NewsSignal.neutral("GOOG")
        object.__setattr__(sig, "upside_catalyst_score", 0.8)
        object.__setattr__(sig, "downside_risk_score", 0.0)
        object.__setattr__(sig, "confidence", 1.0)
        object.__setattr__(sig, "already_priced_in_score", 0.0)
        overlay = sig.news_score_overlay()
        assert overlay > 0  # positive catalyst → positive overlay

    def test_news_score_overlay_downside_reduces(self):
        from app.news.signal import NewsSignal
        sig = NewsSignal.neutral("TSLA")
        object.__setattr__(sig, "upside_catalyst_score", 0.0)
        object.__setattr__(sig, "downside_risk_score", 0.8)
        object.__setattr__(sig, "confidence", 1.0)
        object.__setattr__(sig, "already_priced_in_score", 0.0)
        overlay = sig.news_score_overlay()
        assert overlay < 0  # downside risk → negative overlay

    def test_news_score_overlay_priced_in_dampens(self):
        from app.news.signal import NewsSignal
        sig = NewsSignal.neutral("AAPL")
        object.__setattr__(sig, "upside_catalyst_score", 0.8)
        object.__setattr__(sig, "downside_risk_score", 0.0)
        object.__setattr__(sig, "confidence", 1.0)
        object.__setattr__(sig, "already_priced_in_score", 1.0)  # fully priced in
        overlay_priced = sig.news_score_overlay()

        object.__setattr__(sig, "already_priced_in_score", 0.0)
        overlay_fresh = sig.news_score_overlay()
        assert overlay_priced < overlay_fresh  # priced-in dampens the overlay
