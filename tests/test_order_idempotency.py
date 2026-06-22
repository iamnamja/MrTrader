"""Alpha-v10 H6 — per-order idempotency: deterministic client_order_id + idempotent placement.

A retry of an order we ALREADY placed (same client_order_id) is rejected by Alpaca as a duplicate.
H6 makes place_market_order treat that as SUCCESS — fetch + return the existing order — so a
crash-retry never double-fills, never logs a spurious order_error, and never orphans the position.
Also pins the centralized id scheme (byte-for-byte the historical trend/cash strings).
"""
from __future__ import annotations

import pytest

from app.integrations import alpaca as al
from app.integrations.alpaca import AlpacaClient
from app.live_trading.order_ids import idempotency_key


class _FakeOrder:
    def __init__(self, oid="oid-1", symbol="SPY", qty="10", side="BUY", status="accepted"):
        # qty is a STRING like real alpaca-py Order.qty -> exercises the int() coercion
        self.id = oid
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.status = status
        self.created_at = None


class _FakeTradingClient:
    def __init__(self, *, dup=False, generic_error=False, existing=None, lookup_none=False):
        self._dup = dup
        self._generic_error = generic_error
        self._existing = existing or _FakeOrder()
        self._lookup_none = lookup_none
        self.submit_calls = 0
        self.lookup_calls = 0

    def submit_order(self, req):
        self.submit_calls += 1
        if self._generic_error:
            raise al.APIError('{"code": 40310000, "message": "insufficient buying power"}')
        if self._dup:
            raise al.APIError('{"code": 42210000, "message": "client_order_id must be unique"}')
        return _FakeOrder()

    def get_order_by_client_id(self, cid):
        self.lookup_calls += 1
        return None if self._lookup_none else self._existing


def _client(fake_tc):
    c = AlpacaClient.__new__(AlpacaClient)   # bypass __init__ (no creds/network)
    c.trading_client = fake_tc
    return c


# ── the id scheme ─────────────────────────────────────────────────────────────
def test_idempotency_key_matches_historical_scheme():
    assert idempotency_key("trend", "SPY", day="20260622") == "trend-20260622-SPY"
    assert idempotency_key("cash", "SGOV", side="buy", day="20260622") == "cash-20260622-SGOV-buy"


def test_idempotency_key_today_path_format():
    # the prod path uses day=None (today) — assert the exact shape the old inline f-strings produced
    from datetime import date
    stamp = date.today().strftime("%Y%m%d")
    assert idempotency_key("trend", "QQQ") == f"trend-{stamp}-QQQ"
    assert idempotency_key("cash", "BIL", side="sell") == f"cash-{stamp}-BIL-sell"


def test_duplicate_detector():
    f = al._is_duplicate_client_order_id
    assert f(Exception("client_order_id must be unique"))
    assert f(Exception("duplicate client_order_id"))
    assert f(Exception("client_order_id already exists"))
    assert not f(Exception("insufficient buying power"))
    assert not f(Exception("client_order_id"))          # bare mention is not a dup signal


# ── idempotent placement ──────────────────────────────────────────────────────
def test_success_is_not_idempotent_reuse(monkeypatch):
    tc = _FakeTradingClient(dup=False)
    cb = []
    monkeypatch.setattr(al, "_notify_circuit_breaker", lambda: cb.append(1))
    r = _client(tc).place_market_order("SPY", 10, "buy", client_order_id="trend-20260622-SPY")
    assert r["idempotent_reuse"] is False and tc.submit_calls == 1 and cb == []


def test_duplicate_rejection_returns_existing_order_idempotently(monkeypatch):
    # the crux: a retry hits a dup -> we fetch the existing order, return it, DON'T raise, DON'T trip CB
    tc = _FakeTradingClient(dup=True, existing=_FakeOrder(oid="existing-123"))
    cb = []
    monkeypatch.setattr(al, "_notify_circuit_breaker", lambda: cb.append(1))
    r = _client(tc).place_market_order("SPY", 10, "buy", client_order_id="trend-20260622-SPY")
    assert r["idempotent_reuse"] is True and r["order_id"] == "existing-123"
    assert tc.lookup_calls == 1 and cb == []            # looked up existing; circuit breaker NOT tripped


def test_non_duplicate_apierror_still_raises_and_trips_circuit_breaker(monkeypatch):
    tc = _FakeTradingClient(generic_error=True)
    cb = []
    monkeypatch.setattr(al, "_notify_circuit_breaker", lambda: cb.append(1))
    with pytest.raises(al.APIError):
        _client(tc).place_market_order("SPY", 10, "buy", client_order_id="trend-20260622-SPY")
    assert tc.lookup_calls == 0 and cb == [1]           # not a dup -> no lookup, CB tripped


def test_dup_but_lookup_fails_raises_and_trips_circuit_breaker(monkeypatch):
    # dup detected but the existing-order lookup comes back None -> must NOT fabricate success;
    # fall through to raise the original APIError + trip the circuit breaker (fail-closed safety path)
    tc = _FakeTradingClient(dup=True, lookup_none=True)
    cb = []
    monkeypatch.setattr(al, "_notify_circuit_breaker", lambda: cb.append(1))
    with pytest.raises(al.APIError):
        _client(tc).place_market_order("SPY", 10, "buy", client_order_id="trend-20260622-SPY")
    assert tc.lookup_calls == 1 and cb == [1]


def test_duplicate_without_client_order_id_does_not_idempotent_reuse(monkeypatch):
    # a dup signal with no client_order_id provided is not our idempotency case -> raise as normal
    tc = _FakeTradingClient(dup=True)
    monkeypatch.setattr(al, "_notify_circuit_breaker", lambda: None)
    with pytest.raises(al.APIError):
        _client(tc).place_market_order("SPY", 10, "buy")
    assert tc.lookup_calls == 0
