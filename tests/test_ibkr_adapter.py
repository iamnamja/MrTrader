"""Alpha-v10 P2.2 — IBKR read-only adapter + verify-on-connect (mocked; no live gateway).

A fake `ib` (duck-typed ib_insync.IB) is injected so these run offline in CI. Pins: the canonical
normalization of account/positions, the broker-is-reality multiplier + notional handling, the
verify-on-connect mismatch detection (incl. the SI micro-vs-full tradingClass disambiguation that
was a real bug), the corrected contract master (ZC/ZS=5000, FX/VIX request symbols), and the
structural read-only guarantee.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from app.live_trading import instrument_master as im
from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter, ContractMismatch


def _av(tag, value, currency="USD"):
    return NS(tag=tag, value=value, currency=currency, account="DUTEST")


def _contract(symbol, mult, exchange, tradingClass=None, secType="FUT", currency="USD"):
    return NS(symbol=symbol, multiplier=str(mult), exchange=exchange,
              tradingClass=tradingClass or symbol, secType=secType, currency=currency,
              lastTradeDateOrContractMonth="20261218", conId=1)


def _pf(contract, position, marketPrice, marketValue):
    return NS(contract=contract, position=position, marketPrice=marketPrice, marketValue=marketValue,
              averageCost=0.0, unrealizedPNL=0.0, realizedPNL=0.0)


class FakeIB:
    def __init__(self, *, account_values=None, portfolio=None, cd_by_symbol=None, connected=True,
                 raise_on_connect=False):
        self._av = account_values or []
        self._pf = portfolio or []
        self._cd = cd_by_symbol or {}
        self._connected = connected
        self._raise = raise_on_connect

    def isConnected(self):
        return self._connected

    def connect(self, *a, **k):
        if self._raise:
            raise ConnectionRefusedError("no gateway")
        self._connected = True

    def disconnect(self):
        self._connected = False

    def reqMarketDataType(self, n):
        pass

    def managedAccounts(self):
        return ["DUTEST"]

    def accountValues(self):
        return self._av

    def portfolio(self):
        return self._pf

    def reqContractDetails(self, contract):
        return [NS(contract=c) for c in self._cd.get(contract.symbol, [])]


# ── structural read-only guarantee ─────────────────────────────────────────────
def test_adapter_has_no_order_methods():
    for forbidden in ("place_order", "submit_order", "placeOrder", "cancel_order", "flatten_all"):
        assert not hasattr(IBKRReadOnlyAdapter, forbidden)


# ── account ─────────────────────────────────────────────────────────────────────
def test_get_account_maps_usd_values():
    ib = FakeIB(account_values=[
        _av("NetLiquidation", "100000.00"), _av("TotalCashValue", "100000.00"),
        _av("BuyingPower", "400000.00"), _av("MaintMarginReq", "0.00"),
        _av("AvailableFunds", "100000.00"),
        _av("NetLiquidation", "999", currency="EUR"),   # non-USD must be ignored
        _av("AccountType", "INDIVIDUAL"),               # non-numeric must not crash
    ])
    a = IBKRReadOnlyAdapter(ib=ib).get_account()
    assert a.venue == im.IBKR
    assert a.nav == 100000.0 and a.cash == 100000.0 and a.buying_power == 400000.0
    assert a.maintenance_margin == 0.0 and a.margin_available == 100000.0


# ── positions: broker multiplier + notional = |qty|*px*mult (NOT marketValue) ────
def test_get_positions_uses_broker_multiplier_and_full_notional():
    es = _contract("ES", 50, "CME", tradingClass="ES")
    # marketValue is deliberately tiny (the futures daily-P&L trap) — notional must NOT use it
    ib = FakeIB(portfolio=[_pf(es, position=2, marketPrice=5000.0, marketValue=12.5)])
    pos = IBKRReadOnlyAdapter(ib=ib).get_positions()
    assert len(pos) == 1
    p = pos[0]
    assert p.instrument_id == "FUT.ES" and p.mapped is True and p.asset_class == im.FUTURE
    assert p.multiplier == 50.0 and p.quantity == 2
    assert p.notional == 2 * 5000.0 * 50.0           # full notional, not the 12.5 marketValue
    assert p.market_value == 12.5


def test_get_positions_unknown_symbol_is_unmapped():
    unk = _contract("XYZ", 1, "CME", tradingClass="XYZ", secType="FUT")
    ib = FakeIB(portfolio=[_pf(unk, position=1, marketPrice=100.0, marketValue=100.0)])
    p = IBKRReadOnlyAdapter(ib=ib).get_positions()[0]
    assert p.mapped is False and p.instrument_id == "XYZ"


# ── verify-on-connect ────────────────────────────────────────────────────────────
def test_verify_works_with_no_current_event_loop():
    # Regression: a worker/thread with no current event loop must NOT raise at the `from ib_insync
    # import Future` in verify_contracts (Python 3.12 RuntimeError; CI shard-ordering surfaced it).
    import asyncio
    asyncio.set_event_loop(None)                       # simulate the loop-less worker
    cd = {}
    for iid, inst in im.futures_instruments().items():
        cd[inst.broker_symbol(im.IBKR)] = [
            _contract(inst.broker_symbol(im.IBKR), inst.multiplier, inst.exchange,
                      tradingClass=inst.root)]
    mm = IBKRReadOnlyAdapter(ib=FakeIB(cd_by_symbol=cd)).verify_contracts()
    assert mm == []                                    # ran cleanly (the guard set a loop)


def test_verify_clean_when_multipliers_match():
    # one entry per futures root, each matching the master multiplier/exchange, tradingClass==root
    cd = {}
    for iid, inst in im.futures_instruments().items():
        cd[inst.broker_symbol(im.IBKR)] = [
            _contract(inst.broker_symbol(im.IBKR), inst.multiplier, inst.exchange,
                      tradingClass=inst.root)]
    mm = IBKRReadOnlyAdapter(ib=FakeIB(cd_by_symbol=cd)).verify_contracts()
    assert mm == [], f"unexpected mismatches: {mm}"


def test_verify_flags_wrong_multiplier_as_critical():
    cd = {}
    for iid, inst in im.futures_instruments().items():
        mult = 999.0 if inst.root == "ES" else inst.multiplier   # corrupt ES only
        cd[inst.broker_symbol(im.IBKR)] = [
            _contract(inst.broker_symbol(im.IBKR), mult, inst.exchange, tradingClass=inst.root)]
    mm = IBKRReadOnlyAdapter(ib=FakeIB(cd_by_symbol=cd)).verify_contracts()
    es = [m for m in mm if m.instrument_id == "FUT.ES"]
    assert len(es) == 1 and es[0].field == "multiplier" and es[0].critical is True
    assert es[0].expected == 50.0 and es[0].actual == 999.0


def test_verify_disambiguates_micro_vs_full_by_trading_class():
    # SI symbol returns BOTH the micro SIL (1000) and the full SI (5000); must pick tradingClass==SI.
    cd = {}
    for iid, inst in im.futures_instruments().items():
        cd[inst.broker_symbol(im.IBKR)] = [
            _contract(inst.broker_symbol(im.IBKR), inst.multiplier, inst.exchange,
                      tradingClass=inst.root)]
    cd["SI"] = [
        _contract("SI", 1000, "COMEX", tradingClass="SIL"),   # micro first (the cds[0] trap)
        _contract("SI", 5000, "COMEX", tradingClass="SI"),    # full — the one we must match
    ]
    mm = IBKRReadOnlyAdapter(ib=FakeIB(cd_by_symbol=cd)).verify_contracts()
    assert not any(m.instrument_id == "FUT.SI" for m in mm), f"SI false-positive: {mm}"


def test_verify_flags_unresolvable_contract():
    cd = {}
    for iid, inst in im.futures_instruments().items():
        cd[inst.broker_symbol(im.IBKR)] = [
            _contract(inst.broker_symbol(im.IBKR), inst.multiplier, inst.exchange,
                      tradingClass=inst.root)]
    cd["EUR"] = []   # FX won't resolve
    mm = IBKRReadOnlyAdapter(ib=FakeIB(cd_by_symbol=cd)).verify_contracts()
    eur = [m for m in mm if m.instrument_id == "FUT.6E"]
    assert len(eur) == 1 and eur[0].field == "resolve" and eur[0].critical is True


def test_verify_checks_all_matched_expiries_not_just_first():
    # two ES contracts, same tradingClass, but a later expiry has a wrong multiplier -> must be caught
    cd = {}
    for iid, inst in im.futures_instruments().items():
        cd[inst.broker_symbol(im.IBKR)] = [
            _contract(inst.broker_symbol(im.IBKR), inst.multiplier, inst.exchange,
                      tradingClass=inst.root)]
    cd["ES"] = [
        _contract("ES", 50, "CME", tradingClass="ES"),    # front (correct) — the matched[0] trap
        _contract("ES", 5, "CME", tradingClass="ES"),     # a re-spec'd month with wrong multiplier
    ]
    mm = IBKRReadOnlyAdapter(ib=FakeIB(cd_by_symbol=cd)).verify_contracts()
    es = [m for m in mm if m.instrument_id == "FUT.ES" and m.field == "multiplier"]
    assert len(es) == 1 and es[0].critical is True and es[0].actual == 5.0


def test_verify_exchange_mismatch_is_warn_not_critical():
    cd = {}
    for iid, inst in im.futures_instruments().items():
        exch = "GLOBEX" if inst.root == "ES" else inst.exchange   # alias differs for ES only
        cd[inst.broker_symbol(im.IBKR)] = [
            _contract(inst.broker_symbol(im.IBKR), inst.multiplier, exch, tradingClass=inst.root)]
    mm = IBKRReadOnlyAdapter(ib=FakeIB(cd_by_symbol=cd)).verify_contracts()
    es = [m for m in mm if m.instrument_id == "FUT.ES"]
    assert len(es) == 1 and es[0].field == "exchange" and es[0].critical is False


# ── reads fail-closed when not connected (silent-wrong-state guard) ───────────────
def test_reads_fail_closed_when_not_connected():
    ad = IBKRReadOnlyAdapter(ib=FakeIB(connected=False))
    import pytest
    for call in (ad.get_account, ad.get_positions, ad.verify_contracts):
        with pytest.raises(ConnectionError):
            call()


# ── connection fail-closed ───────────────────────────────────────────────────────
def test_connect_fail_closed_returns_unhealthy_not_raises():
    h = IBKRReadOnlyAdapter(ib=FakeIB(connected=False, raise_on_connect=True)).connect()
    assert h.connected is False and h.venue == im.IBKR


def test_health_never_raises():
    class Boom:
        def isConnected(self):
            raise RuntimeError("boom")
    h = IBKRReadOnlyAdapter(ib=Boom()).health()
    assert h.connected is False


# ── the broker-confirmed contract-master corrections ─────────────────────────────
def test_contract_master_corrections_from_live_verify():
    # ZC/ZS corrected 50 -> 5000 (broker truth)
    assert im.get("FUT.ZC").multiplier == 5000.0
    assert im.get("FUT.ZS").multiplier == 5000.0
    # FX/VIX request symbols differ from the canonical root
    assert im.get("FUT.6E").broker_symbol(im.IBKR) == "EUR" and im.get("FUT.6E").root == "6E"
    assert im.get("FUT.6J").broker_symbol(im.IBKR) == "JPY"
    assert im.get("FUT.VX").broker_symbol(im.IBKR) == "VIX"
    assert im.lookup(im.IBKR, "EUR") == "FUT.6E"        # reverse index uses the IBKR symbol
    # sec_type derives from asset class
    assert im.get("FUT.ES").sec_type == "FUT" and im.get("SPY").sec_type == "STK"
    # SI stayed correct (5000) — was a verify false-positive, not a real error
    assert im.get("FUT.SI").multiplier == 5000.0
