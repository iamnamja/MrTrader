"""Phase H — H3: pre-trade per-order fat-finger backstop (fail-closed).

A coarse absolute cap at the Alpaca order chokepoint (`_assert_order_within_caps`, enforced inside
place_market_order/place_limit_order) that REJECTS an absurdly-large single order BEFORE it reaches
the broker. Distinct from the NAV-relative portfolio caps in whole_book_gate.
"""
from __future__ import annotations

import pytest

from app.integrations.alpaca import _assert_order_within_caps, OrderSanityError, AlpacaClient


def _caps(monkeypatch, *, enabled=True, notional=500_000.0, shares=100_000):
    monkeypatch.setattr("app.ml.retrain_config.H3_PRETRADE_CAP_ENABLED", enabled)
    monkeypatch.setattr("app.ml.retrain_config.H3_MAX_ORDER_NOTIONAL_USD", notional)
    monkeypatch.setattr("app.ml.retrain_config.H3_MAX_ORDER_SHARES", shares)


# ── the guard helper ─────────────────────────────────────────────────────────────
def test_within_caps_passes(monkeypatch):
    _caps(monkeypatch)
    _assert_order_within_caps("SGOV", 489, "buy", 100.6)   # ~$49k, well within → no raise


def test_notional_over_cap_raises(monkeypatch):
    _caps(monkeypatch, notional=10_000.0)
    with pytest.raises(OrderSanityError):
        _assert_order_within_caps("TSLA", 100, "buy", 150.0)   # $15k > $10k


def test_shares_over_cap_raises(monkeypatch):
    _caps(monkeypatch, shares=1_000)
    with pytest.raises(OrderSanityError):
        _assert_order_within_caps("F", 5_000, "buy", 12.0)     # 5000 > 1000 shares


def test_nonpositive_or_malformed_qty_raises(monkeypatch):
    _caps(monkeypatch)
    with pytest.raises(OrderSanityError):
        _assert_order_within_caps("SPY", 0, "buy", 500.0)
    with pytest.raises(OrderSanityError):
        _assert_order_within_caps("SPY", -10, "sell", 500.0)
    with pytest.raises(OrderSanityError):
        _assert_order_within_caps("SPY", "oops", "buy", 500.0)


def test_no_price_falls_back_to_shares_cap_only(monkeypatch):
    _caps(monkeypatch, notional=1.0, shares=1_000)   # tiny notional cap, but no price -> not checked
    _assert_order_within_caps("SPY", 100, "buy", None)   # within shares -> OK despite tiny notional cap
    with pytest.raises(OrderSanityError):
        _assert_order_within_caps("SPY", 5_000, "buy", None)   # shares cap still bites


def test_zero_price_treated_as_uncheckable(monkeypatch):
    _caps(monkeypatch, notional=1.0)
    _assert_order_within_caps("SPY", 100, "buy", 0.0)    # price 0 -> notional not checked, shares OK


def test_disabled_flag_allows_anything(monkeypatch):
    _caps(monkeypatch, enabled=False, notional=1.0, shares=1)
    _assert_order_within_caps("TSLA", 1_000_000, "buy", 9999.0)   # disabled -> no raise


# ── enforced at the order chokepoint, BEFORE submit ───────────────────────────────
class _BoomClient:
    """submit_order must NEVER be reached when the guard rejects."""
    def submit_order(self, *a, **k):
        raise AssertionError("submit_order reached — H3 guard did NOT block the order")


def _client():
    c = AlpacaClient.__new__(AlpacaClient)
    c.trading_client = _BoomClient()
    return c


def test_place_market_order_blocks_before_submit(monkeypatch):
    _caps(monkeypatch, notional=10_000.0)
    c = _client()
    with pytest.raises(OrderSanityError):
        c.place_market_order("TSLA", 100, "buy", est_price=150.0)   # $15k > $10k → blocked pre-submit


def test_place_limit_order_blocks_before_submit(monkeypatch):
    _caps(monkeypatch, notional=10_000.0)
    c = _client()
    with pytest.raises(OrderSanityError):
        c.place_limit_order("TSLA", 100, "buy", limit_price=150.0)  # uses limit_price → blocked


def test_place_market_order_without_est_price_uses_shares_cap(monkeypatch):
    # no est_price → notional not checked; a within-shares order is NOT blocked by the guard
    # (it proceeds to submit_order, which here raises AssertionError → proves the guard let it pass)
    _caps(monkeypatch, notional=1.0, shares=1_000)
    c = _client()
    with pytest.raises(AssertionError):       # reached submit_order = guard allowed it
        c.place_market_order("SPY", 100, "buy")
    # but an over-shares order is still blocked pre-submit
    with pytest.raises(OrderSanityError):
        c.place_market_order("SPY", 5_000, "buy")
