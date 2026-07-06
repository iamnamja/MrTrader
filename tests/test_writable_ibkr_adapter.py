"""R1.0c — WritableIBKRAdapter: shadow-first, contract construction (multiplier from instrument_master),
fail-closed guards. Uses a fake connection + real instrument_master; the live thread/gateway path is
validated separately against the paper gateway during the build.
"""
import asyncio
import inspect

import pytest

from app.live_trading.writable_ibkr_adapter import WritableIBKRAdapter
from app.live_trading.writable_broker_adapter import OrderIntent, WritableBrokerAdapter


class _FakeConn:
    """Stand-in for IBKRConnectionManager: runs the thunk inline (awaiting awaitables like the real one)."""
    def __init__(self, ib=None):
        self._ib = ib
        self.calls = 0

    def is_connected(self):
        return self._ib is not None

    def call(self, thunk, *, timeout=None):
        self.calls += 1
        r = thunk(self._ib)
        if inspect.isawaitable(r):
            # Run the awaitable on a private, throwaway loop WITHOUT touching the thread's
            # current-loop registration (unlike asyncio.run, which sets the current loop to None
            # on exit). That teardown is what made these tests fail in the full suite: another
            # async module on the same xdist worker leaves a loop registered, and asyncio.run's
            # cleanup then poisons it. new_event_loop + run_until_complete + close is inert.
            loop = asyncio.new_event_loop()
            try:
                r = loop.run_until_complete(r)
            finally:
                loop.close()
        return r


def _intent(**kw):
    d = dict(venue="IBKR", instrument_id="SPY", sec_type="ETF", side="BUY", quantity=1, client_ref="r1c")
    d.update(kw)
    return OrderIntent(**d)


def test_shadow_place_places_nothing():
    conn = _FakeConn()                              # no ib
    res = WritableIBKRAdapter(conn, mode="shadow").place(_intent())
    assert res.accepted_status == "shadow" and res.broker_order_id is None
    assert res.raw["symbol"] == "SPY" and res.raw["multiplier"] == 1.0
    assert conn.calls == 0                          # PLACED NOTHING — no dispatch to the connection


def test_futures_contract_multiplier_from_instrument_master():
    # THE #1-killer guard: ES multiplier comes from the verified spec (50), never a caller value.
    adapter = WritableIBKRAdapter(_FakeConn(), mode="shadow")
    res = adapter.place(_intent(instrument_id="FUT.ES", sec_type="FUT"))
    assert res.raw["symbol"] == "ES" and res.raw["multiplier"] == 50.0 and res.raw["exchange"] == "CME"
    # Assert on what's ACTUALLY bound onto the IBKR contract (a string), not just the master echo —
    # this is what would catch a str(int(...)) stringification regression.
    assert res.raw["contract_multiplier"] == "50"
    contract = adapter._build_contract(adapter._instrument("FUT.ES"))
    assert contract.multiplier == "50" and contract.symbol == "ES" and contract.exchange == "CME"


def test_futures_zero_or_missing_multiplier_fails_closed(monkeypatch):
    # The killer-guard must FAIL CLOSED (raise), never emit an empty multiplier that lets IBKR
    # substitute a default. Simulate a corrupt master entry with no multiplier.
    import dataclasses
    adapter = WritableIBKRAdapter(_FakeConn(), mode="shadow")
    bad = dataclasses.replace(adapter._instrument("FUT.ES"), multiplier=0)
    with pytest.raises(ValueError):
        adapter._build_contract(bad)


def test_unmapped_instrument_fails_closed():
    with pytest.raises(ValueError):
        WritableIBKRAdapter(_FakeConn(), mode="shadow").place(_intent(instrument_id="NOPE.XYZ"))


def test_non_integer_quantity_rejected():
    with pytest.raises(ValueError):
        WritableIBKRAdapter(_FakeConn(), mode="shadow").place(_intent(quantity=1.5))


def test_non_market_order_rejected():
    with pytest.raises(NotImplementedError):
        WritableIBKRAdapter(_FakeConn(), mode="shadow").place(_intent(order_type="LIMIT"))


def test_venue_mismatch_rejected():
    with pytest.raises(ValueError):
        WritableIBKRAdapter(_FakeConn(), mode="shadow").place(_intent(venue="ALPACA"))


def test_cancel_shadow_does_nothing():
    conn = _FakeConn()
    assert WritableIBKRAdapter(conn, mode="shadow").cancel("o1").accepted_status == "shadow"
    assert conn.calls == 0


def test_conforms_to_writable_protocol():
    assert isinstance(WritableIBKRAdapter(_FakeConn()), WritableBrokerAdapter)


def test_preview_ok_false_when_gateway_returns_empty_state():
    # Read-Only API returns an EMPTY OrderState (no margins) rather than raising → ok must be False.
    class _IB:
        async def whatIfOrderAsync(self, contract, order):
            class _EmptyState:
                pass
            return _EmptyState()
    mi = WritableIBKRAdapter(_FakeConn(_IB()), mode="shadow").preview(_intent())
    assert mi.ok is False and mi.init_margin is None
