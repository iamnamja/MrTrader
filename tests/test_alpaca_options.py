"""Alpaca options market-data client (app/data/alpaca_options.py) — FUSE A.

Pagination, credential headers, expiration filters, and field passthrough against a
mocked requests session (no network). The live field shapes were confirmed by the
2026-06-11 R1 spike (indicative feed, bp/ap/bs/as quote keys).
"""
from __future__ import annotations

from datetime import date

import pytest

from app.config import settings
from app.data.alpaca_options import (
    fetch_latest_underlying_prices, fetch_option_snapshots,
)


class _FakeResp:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class _FakeSession:
    """Queue of responses; records every (url, headers, params) call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}),
                           "params": dict(params or {})})
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _fake_creds(monkeypatch):
    monkeypatch.setattr(settings, "alpaca_api_key", "test-key-id")
    monkeypatch.setattr(settings, "alpaca_secret_key", "test-secret")


def _snap(bid, ask):
    return {"latestQuote": {"bp": bid, "ap": ask, "bs": 10, "as": 12,
                            "t": "2026-06-11T19:59:59Z"},
            "dailyBar": {"c": 0.5 * (bid + ask), "v": 100}}


class TestFetchOptionSnapshots:
    def test_pagination_merges_pages_and_passes_token(self):
        sess = _FakeSession([
            _FakeResp({"snapshots": {"SPY260717C00600000": _snap(1.0, 1.2)},
                       "next_page_token": "tok-1"}),
            _FakeResp({"snapshots": {"SPY260717P00600000": _snap(2.0, 2.4)},
                       "next_page_token": None}),
        ])
        out = fetch_option_snapshots("SPY", session=sess)
        assert set(out) == {"SPY260717C00600000", "SPY260717P00600000"}
        assert len(sess.calls) == 2
        assert "page_token" not in sess.calls[0]["params"]
        assert sess.calls[1]["params"]["page_token"] == "tok-1"

    def test_headers_and_feed_and_expiration_params(self):
        sess = _FakeSession([_FakeResp({"snapshots": {}})])
        fetch_option_snapshots("spy", feed="indicative",
                               exp_lo=date(2026, 6, 11), exp_hi=date(2026, 8, 20),
                               session=sess)
        call = sess.calls[0]
        assert call["url"].endswith("/v1beta1/options/snapshots/SPY")
        assert call["headers"]["APCA-API-KEY-ID"] == "test-key-id"
        assert call["headers"]["APCA-API-SECRET-KEY"] == "test-secret"
        assert call["params"]["feed"] == "indicative"
        assert call["params"]["expiration_date_gte"] == "2026-06-11"
        assert call["params"]["expiration_date_lte"] == "2026-08-20"

    def test_max_pages_cap_truncates_instead_of_looping(self):
        pages = [_FakeResp({"snapshots": {f"SPY26071{i}C00600000": _snap(1, 1.1)},
                            "next_page_token": f"tok-{i}"}) for i in range(5)]
        sess = _FakeSession(pages)
        out = fetch_option_snapshots("SPY", max_pages=3, session=sess)
        assert len(sess.calls) == 3
        assert len(out) == 3

    def test_http_error_raises(self):
        sess = _FakeSession([_FakeResp({}, status=403)])
        with pytest.raises(RuntimeError):
            fetch_option_snapshots("SPY", session=sess)


class TestFetchLatestUnderlyingPrices:
    def test_batched_symbols_and_price_extraction(self):
        sess = _FakeSession([_FakeResp({"trades": {
            "SPY": {"p": 601.25, "t": "2026-06-11T19:59:59Z"},
            "QQQ": {"p": 0},          # zero price -> dropped
            "IWM": {"p": None},       # missing -> dropped
        }})])
        out = fetch_latest_underlying_prices(["SPY", "QQQ", "IWM"], session=sess)
        assert out == {"SPY": 601.25}
        assert sess.calls[0]["params"]["symbols"] == "SPY,QQQ,IWM"
        assert sess.calls[0]["params"]["feed"] == "iex"
