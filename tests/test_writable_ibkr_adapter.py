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


# ── R1.0c-2: front-month qualification (reqContractDetails, read-only-safe) ──────────

class _Contract:
    def __init__(self, tradingClass, expiry, multiplier="50", symbol="ES", exchange="CME"):
        self.tradingClass = tradingClass
        self.lastTradeDateOrContractMonth = expiry
        self.multiplier = multiplier
        self.symbol = symbol
        self.exchange = exchange


class _CD:
    def __init__(self, contract):
        self.contract = contract


class _QualifyIB:
    def __init__(self, cds):
        self._cds = cds

    async def reqContractDetailsAsync(self, base):
        return self._cds


def test_qualify_future_picks_nearest_unexpired_front_month():
    cds = [
        _CD(_Contract("ES", "20200320")),   # expired → filtered out
        _CD(_Contract("ES", "20310320")),   # further out
        _CD(_Contract("ES", "20301200")),   # nearest non-expired → the front month
        _CD(_Contract("MES", "20301200")),  # wrong tradingClass → filtered out
    ]
    c = WritableIBKRAdapter(_FakeConn(_QualifyIB(cds)), mode="shadow").qualify_future("FUT.ES")
    assert c.lastTradeDateOrContractMonth == "20301200" and c.tradingClass == "ES"


def test_qualify_future_no_matching_tradingclass_fails_closed():
    cds = [_CD(_Contract("MES", "20301200"))]   # only the micro — none match root ES
    with pytest.raises(ValueError):
        WritableIBKRAdapter(_FakeConn(_QualifyIB(cds)), mode="shadow").qualify_future("FUT.ES")


def test_qualify_future_multiplier_mismatch_on_resolved_contract_fails_closed():
    # THE #1-killer guard on the RESOLVED contract: broker says 5, master says 50 → must raise.
    cds = [_CD(_Contract("ES", "20301200", multiplier="5"))]
    with pytest.raises(ValueError):
        WritableIBKRAdapter(_FakeConn(_QualifyIB(cds)), mode="shadow").qualify_future("FUT.ES")


def test_qualify_future_rejects_non_future():
    with pytest.raises(ValueError):
        WritableIBKRAdapter(_FakeConn(_QualifyIB([])), mode="shadow").qualify_future("SPY")


# ── R1.0c-2: async fill capture (reqExecutions → FillEvent; disconnect-gap recovery) ──

class _Exec:
    def __init__(self, execId, side, shares, price, orderRef, time):
        self.execId, self.side, self.shares = execId, side, shares
        self.price, self.orderRef, self.time = price, orderRef, time


class _Comm:
    def __init__(self, commission):
        self.commission = commission


class _Fill:
    def __init__(self, execution, commissionReport, symbol, secType="STK"):
        self.execution = execution
        self.commissionReport = commissionReport
        self.contract = type("C", (), {"symbol": symbol, "secType": secType})()


class _FillsIB:
    def __init__(self, fills, raise_exc=False):
        self._fills, self._raise = fills, raise_exc

    async def reqExecutionsAsync(self, flt):
        if self._raise:
            raise RuntimeError("gateway down")
        return self._fills


def _fills_fixture():
    from datetime import datetime
    t = datetime(2026, 7, 6, 14, 30, 0)
    return [
        _Fill(_Exec("e1", "BOT", 3, 501.2, "r1c", t), _Comm(1.05), "SPY", secType="STK"),
        _Fill(_Exec("e2", "SLD", 1, 5300.0, "es1", t), _Comm(2.10), "ES", secType="FUT"),
    ]


def test_get_fills_maps_executions_to_fillevents():
    evs = WritableIBKRAdapter(_FakeConn(_FillsIB(_fills_fixture())), mode="shadow").get_fills()
    assert len(evs) == 2
    buy = next(e for e in evs if e.exec_id == "e1")
    assert (buy.side == "BUY" and buy.client_ref == "r1c" and buy.instrument_id == "SPY"
            and buy.filled_qty == 3.0 and buy.avg_price == 501.2 and buy.commission == 1.05
            and buy.venue == "IBKR" and buy.ts.startswith("2026-07-06T14:30:00"))
    sell = next(e for e in evs if e.exec_id == "e2")
    assert sell.side == "SELL" and sell.instrument_id == "FUT.ES"   # ES symbol → canonical instrument


def test_get_fills_filters_by_client_ref():
    evs = WritableIBKRAdapter(_FakeConn(_FillsIB(_fills_fixture())), mode="shadow").get_fills(client_ref="es1")
    assert [e.exec_id for e in evs] == ["e2"]


def test_get_fills_degrades_to_empty_on_error():
    evs = WritableIBKRAdapter(_FakeConn(_FillsIB([], raise_exc=True)), mode="shadow").get_fills()
    assert evs == []


def test_get_fills_fail_closed_drops_unusable_fills():
    from datetime import datetime
    t = datetime(2026, 7, 6, 14, 30, 0)
    bad = [
        _Fill(_Exec("", "BOT", 1, 100.0, "r", t), _Comm(1), "SPY", secType="STK"),          # no execId
        _Fill(_Exec("e", "XXX", 1, 100.0, "r", t), _Comm(1), "SPY", secType="STK"),         # unknown side
        _Fill(_Exec("e", "BOT", 0, 100.0, "r", t), _Comm(1), "SPY", secType="STK"),         # zero qty
        _Fill(_Exec("e", "BOT", 1, 100.0, "r", t), _Comm(1), "MES", secType="FUT"),         # unmapped FUT
    ]
    evs = WritableIBKRAdapter(_FakeConn(_FillsIB(bad)), mode="shadow").get_fills()
    assert evs == []       # every corrupting fill dropped, never emitted with a null/guessed field


def test_get_fills_nulls_unset_commission_sentinel():
    from datetime import datetime
    t = datetime(2026, 7, 6, 14, 30, 0)
    f = [_Fill(_Exec("e1", "BOT", 1, 100.0, "r", t), _Comm(1.7976931348623157e308), "SPY", secType="STK")]
    ev = WritableIBKRAdapter(_FakeConn(_FillsIB(f)), mode="shadow").get_fills()[0]
    assert ev.commission is None      # IBKR UNSET_DOUBLE => not-yet-reported, not a giant real charge


def test_norm_expiry_does_not_pad_month_to_first():
    # Locks the front-month fix: a 6-digit YYYYMM must NOT be padded to the 1st (which would judge the
    # true front month expired mid-month and roll early).
    c = _Contract("ES", "202609")
    assert WritableIBKRAdapter._norm_expiry(c) == "202609"
