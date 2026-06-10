"""Tests for the Finnhub adapter's error handling (app.news.sources.finnhub_source._get).

Two production problems this guards:
  1. A premium-only endpoint returns 403 to a free-tier token on EVERY poll (~once/minute),
     spamming the live log forever. A 403 is permanent — retrying never helps — so the
     adapter must disable the endpoint after the first failure and log it ONCE.
  2. The previous handler logged the raw requests exception, whose string embeds the full
     request URL **including `?token=...`** — leaking the API key into the log. Error logs
     must never contain the token.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

import app.news.sources.finnhub_source as fh

_TOKEN = "SECRET_TOKEN_do_not_log"


@pytest.fixture(autouse=True)
def _reset_state():
    """The disabled-endpoint set is module-global; reset it around each test."""
    fh._DISABLED_ENDPOINTS.clear()
    yield
    fh._DISABLED_ENDPOINTS.clear()


def _resp(status, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data if json_data is not None else {}
    if status >= 400:
        err = requests.HTTPError(f"{status} error for url ...?token={_TOKEN}")
        err.response = MagicMock(status_code=status)
        r.raise_for_status.side_effect = err
    else:
        r.raise_for_status.return_value = None
    return r


def test_403_disables_endpoint_and_logs_once(caplog):
    """A 403 disables the endpoint, returns None, and the SECOND call short-circuits
    without another HTTP request."""
    with patch.object(fh, "_key", return_value=_TOKEN), \
         patch.object(fh.requests, "get", return_value=_resp(403)) as mock_get:
        with caplog.at_level("WARNING"):
            assert fh._get("calendar/economic", {}) is None
            assert fh._get("calendar/economic", {}) is None  # short-circuits
    assert mock_get.call_count == 1, "forbidden endpoint was retried instead of disabled"
    assert "calendar/economic" in fh._DISABLED_ENDPOINTS
    # logged exactly once
    assert sum("returned 403" in r.message for r in caplog.records) == 1


def test_403_does_not_leak_token(caplog):
    with patch.object(fh, "_key", return_value=_TOKEN), \
         patch.object(fh.requests, "get", return_value=_resp(403)):
        with caplog.at_level("WARNING"):
            fh._get("calendar/economic", {})
    assert all(_TOKEN not in r.getMessage() for r in caplog.records), "token leaked into log"


def test_transient_5xx_does_not_disable(caplog):
    """A 503 is transient — must NOT disable the endpoint (it may recover), and must not
    leak the token even though the exception string contains it."""
    with patch.object(fh, "_key", return_value=_TOKEN), \
         patch.object(fh.requests, "get", return_value=_resp(503)) as mock_get:
        with caplog.at_level("WARNING"):
            assert fh._get("calendar/economic", {}) is None
            assert fh._get("calendar/economic", {}) is None  # retried, not disabled
    assert mock_get.call_count == 2
    assert "calendar/economic" not in fh._DISABLED_ENDPOINTS
    assert all(_TOKEN not in r.getMessage() for r in caplog.records)


def test_network_error_logs_type_not_message(caplog):
    """A connection error's str embeds the token-bearing URL — log the type only."""
    with patch.object(fh, "_key", return_value=_TOKEN), \
         patch.object(fh.requests, "get",
                      side_effect=requests.ConnectionError(f"failed ...?token={_TOKEN}")):
        with caplog.at_level("WARNING"):
            assert fh._get("company-news", {}) is None
    assert all(_TOKEN not in r.getMessage() for r in caplog.records)
    assert any("ConnectionError" in r.getMessage() for r in caplog.records)


def test_success_returns_json():
    with patch.object(fh, "_key", return_value=_TOKEN), \
         patch.object(fh.requests, "get", return_value=_resp(200, {"ok": 1})):
        assert fh._get("calendar/earnings", {"from": "x", "to": "y"}) == {"ok": 1}


def test_no_key_skips_without_call():
    with patch.object(fh, "_key", return_value=None), \
         patch.object(fh.requests, "get") as mock_get:
        assert fh._get("calendar/economic", {}) is None
    mock_get.assert_not_called()
