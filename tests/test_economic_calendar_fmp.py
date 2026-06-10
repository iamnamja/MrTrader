"""Tests for the FMP economic-calendar source + the provider-agnostic dispatcher.

The economic calendar moved from Finnhub (premium-only → 403 on the free tier) to FMP
(free-tier capable). These cover: FMP response normalization to the shared schema, the
fail-open None contract, token non-leakage, and the dispatcher's FMP-primary /
Finnhub-fallback logic.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import app.news.sources.fmp_source as fmp
import app.news.sources.economic_calendar as ecal

_KEY = "FMP_SECRET_do_not_log"


def _near_future_event(**over):
    """An FMP event ~2h out (inside the default lookahead window)."""
    t = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    base = {
        "event": "Fed Interest Rate Decision",
        "impact": "High",
        "date": t,
        "country": "US",
        "currency": "USD",
        "previous": "5.25",
        "estimate": "5.50",
        "actual": None,
    }
    base.update(over)
    return base


def _resp(status, json_data):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data
    return r


# ───────────────────────── fmp_source ──────────────────────────────────────────

def test_fmp_no_key_returns_none_without_call():
    with patch.object(fmp, "_key", return_value=None), \
         patch.object(fmp.requests, "get") as mock_get:
        assert fmp.fetch_economic_calendar() is None
    mock_get.assert_not_called()


def test_fmp_normalizes_to_shared_schema():
    with patch.object(fmp, "_key", return_value=_KEY), \
         patch.object(fmp.requests, "get", return_value=_resp(200, [_near_future_event()])):
        out = fmp.fetch_economic_calendar(days_ahead=1, min_impact="medium")
    assert out is not None and len(out) == 1
    ev = out[0]
    assert ev["event_type"] == "FOMC"
    assert ev["importance"] == "high"
    assert ev["prior"] == "5.25"          # FMP "previous" -> normalized "prior"
    assert ev["estimate"] == "5.50"
    assert ev["country"] == "US"
    assert ev["source"] == "fmp"
    assert isinstance(ev["event_time"], datetime) and ev["event_time"].tzinfo is not None
    assert ev["id"]  # stable hash present


def test_fmp_filters_low_impact_and_foreign():
    events = [
        _near_future_event(event="Some Minor Print", impact="Low"),
        _near_future_event(event="German CPI", country="DE", impact="High"),
    ]
    with patch.object(fmp, "_key", return_value=_KEY), \
         patch.object(fmp.requests, "get", return_value=_resp(200, events)):
        out = fmp.fetch_economic_calendar(days_ahead=1, min_impact="medium")
    assert out == []  # low-impact dropped; foreign-country dropped


def test_fmp_non_200_returns_none_no_token_leak(caplog):
    with patch.object(fmp, "_key", return_value=_KEY), \
         patch.object(fmp.requests, "get", return_value=_resp(403, [])):
        with caplog.at_level("WARNING"):
            assert fmp.fetch_economic_calendar() is None
    assert all(_KEY not in r.getMessage() for r in caplog.records)


def test_fmp_exception_returns_none_logs_type(caplog):
    import requests
    with patch.object(fmp, "_key", return_value=_KEY), \
         patch.object(fmp.requests, "get",
                      side_effect=requests.ConnectionError(f"x ?apikey={_KEY}")):
        with caplog.at_level("WARNING"):
            assert fmp.fetch_economic_calendar() is None
    assert all(_KEY not in r.getMessage() for r in caplog.records)
    assert any("ConnectionError" in r.getMessage() for r in caplog.records)


# ───────────────────────── dispatcher ──────────────────────────────────────────

def test_dispatcher_prefers_fmp_when_available():
    sentinel = [{"id": "fmp1", "event_type": "CPI"}]
    with patch.object(ecal_fmp(), "fetch_economic_calendar", return_value=sentinel) as f, \
         patch.object(ecal_finnhub(), "fetch_economic_calendar") as fh:
        assert ecal.fetch_economic_calendar() == sentinel
        f.assert_called_once()
        fh.assert_not_called()  # FMP succeeded → no fallback


def test_dispatcher_empty_fmp_is_used_no_fallback():
    """FMP returning [] means 'available, no events today' — must NOT fall back."""
    with patch.object(ecal_fmp(), "fetch_economic_calendar", return_value=[]), \
         patch.object(ecal_finnhub(), "fetch_economic_calendar") as fh:
        assert ecal.fetch_economic_calendar() == []
        fh.assert_not_called()


def test_dispatcher_falls_back_to_finnhub_when_fmp_unavailable():
    fb = [{"id": "fh1", "event_type": "NFP"}]
    with patch.object(ecal_fmp(), "fetch_economic_calendar", return_value=None), \
         patch.object(ecal_finnhub(), "fetch_economic_calendar", return_value=fb) as fh:
        assert ecal.fetch_economic_calendar() == fb
        fh.assert_called_once()


# Helpers to patch the modules the dispatcher imports lazily.
def ecal_fmp():
    import app.news.sources.fmp_source as m
    return m


def ecal_finnhub():
    import app.news.sources.finnhub_source as m
    return m
