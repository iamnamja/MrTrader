"""R0.3 tests — canonical instrument master + read-only Alpaca broker adapter.

Guards: the canonical mapping (incl. cash-equivalent flagging + fail-closed misses), the read-only
normalization of broker truth into canonical objects, and the structural read-only safety (no order
methods on the R0.3 adapter).
"""
from __future__ import annotations

import pytest

from app.live_trading import instrument_master as im
from app.live_trading import broker_adapter as ba


# ---- instrument master ----
def test_live_etfs_and_cash_seeded():
    spy = im.get("SPY")
    assert spy is not None and spy.asset_class == im.ETF and spy.multiplier == 1.0
    assert spy.broker_symbol(im.ALPACA) == "SPY"
    sgov = im.get("SGOV")
    assert sgov is not None and sgov.is_cash_equivalent and sgov.asset_class == im.CASH_ETF
    assert im.is_cash_equivalent("SGOV") and not im.is_cash_equivalent("SPY")


def test_futures_are_placeholders_pending_verify_on_connect():
    es = im.get("FUT.ES")
    assert es is not None and es.asset_class == im.FUTURE and es.root == "ES"
    assert es.multiplier == 50.0 and es.broker_symbol(im.IBKR) == "ES"
    assert es.verified is False                    # must be verify-on-connect in R1


def test_lookup_and_fail_closed_miss():
    assert im.lookup(im.ALPACA, "SPY") == "SPY"
    assert im.lookup(im.IBKR, "ES") == "FUT.ES"
    assert im.lookup(im.ALPACA, "NOPE") is None     # fail-closed: unknown -> None
    assert im.get("NOPE") is None
    assert im.is_cash_equivalent("NOPE") is False


# ---- read-only Alpaca adapter ----
class _FakeAlpaca:
    def get_clock(self):
        return {"is_open": True}

    def get_account(self):
        return {"cash": 50000.0, "buying_power": 100000.0, "portfolio_value": 100000.0,
                "equity": 100000.0}

    def get_positions(self):
        return [
            {"symbol": "SPY", "qty": 100, "current_price": 500.0, "market_value": 50000.0},
            {"symbol": "SGOV", "qty": 400, "current_price": 100.0, "market_value": 40000.0},
            {"symbol": "ZZZZ", "qty": 10, "current_price": 5.0, "market_value": 50.0},  # unmapped
        ]


def test_adapter_health_and_account_mapping():
    ad = ba.AlpacaReadOnlyAdapter(client=_FakeAlpaca())
    h = ad.health()
    assert h.connected and h.clock_ok and h.venue == im.ALPACA
    acct = ad.get_account()
    assert acct.venue == im.ALPACA and acct.nav == 100000.0 and acct.cash == 50000.0
    assert acct.margin_used is None                # cash account -> no margin fields


def test_adapter_positions_normalized_to_canonical():
    ad = ba.AlpacaReadOnlyAdapter(client=_FakeAlpaca())
    pos = {p.broker_symbol: p for p in ad.get_positions()}
    spy = pos["SPY"]
    assert spy.instrument_id == "SPY" and spy.mapped and spy.asset_class == im.ETF
    assert spy.multiplier == 1.0 and spy.notional == 100 * 500.0
    assert not spy.is_cash_equivalent
    sgov = pos["SGOV"]
    assert sgov.is_cash_equivalent and sgov.asset_class == im.CASH_ETF
    zzzz = pos["ZZZZ"]                              # unmapped -> mapped=False, falls back to symbol
    assert zzzz.mapped is False and zzzz.instrument_id == "ZZZZ" and zzzz.asset_class == im.EQUITY


def test_adapter_is_structurally_read_only():
    ad = ba.AlpacaReadOnlyAdapter(client=_FakeAlpaca())
    for forbidden in ("place_order", "submit_order", "cancel_order", "flatten_all", "liquidate_all"):
        assert not hasattr(ad, forbidden)          # R0.3 adapter cannot trade
    assert isinstance(ad, ba.BrokerAdapter)         # satisfies the read-side Protocol


def test_readonly_client_proxy_blocks_non_read_methods():
    # defense-in-depth: the WRAPPED client cannot be used to trade either
    proxy = ba._ReadOnlyClientProxy(_FakeAlpaca())
    assert proxy.get_account()["equity"] == 100000.0    # whitelisted read works
    for forbidden in ("submit_order", "place_order", "cancel_order", "close_all_positions"):
        with pytest.raises(AttributeError):
            getattr(proxy, forbidden)
    with pytest.raises(AttributeError):
        proxy.anything = 1                              # immutable
