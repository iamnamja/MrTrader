"""Alpha-v10 audit Wave 5f — live-path MAJOR fixes (batch 2).

Pins:
  #1 IBKR get_account fails CLOSED (raises) on anything but exactly one managed account, instead of
     last-row-wins-clobbering a mixed multi-account NAV.
  #4 Trader._execute_exit never computes P&L off a non-positive price — a 3:45pm force-close that
     passes price=0 (no quote) falls back to a real reference, not a phantom -entry*qty loss.
  #5 premarket overnight-gap AUTO_EXIT only fires on a RELIABLE (tight-spread) quote; a wide/missing
     quote downgrades the destructive exit to a PM re-eval.
  #6 PortfolioManager._get_deployed_by_type classifies positions by the DB Trade.trade_type map, so
     intraday deployed capital is actually counted toward its per-sleeve budget (was always 'swing').
"""
from __future__ import annotations

import asyncio
import logging
import types


# ── #1 IBKR multi-account NAV fails closed ───────────────────────────────────────
def _make_ibkr(accounts, values):
    from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter
    a = IBKRReadOnlyAdapter.__new__(IBKRReadOnlyAdapter)
    a._ib = types.SimpleNamespace(
        isConnected=lambda: True,
        managedAccounts=lambda: accounts,
        accountValues=lambda: values,
    )
    return a


def _av(tag, value, account, currency="USD"):
    return types.SimpleNamespace(tag=tag, value=value, account=account, currency=currency)


def test_ibkr_single_account_maps_nav():
    a = _make_ibkr(["U1"], [_av("NetLiquidation", "100000", "U1"),
                            _av("TotalCashValue", "40000", "U1")])
    st = a.get_account()
    assert st.nav == 100000.0 and st.cash == 40000.0


def test_ibkr_multi_account_fails_closed():
    a = _make_ibkr(["U1", "U2"], [_av("NetLiquidation", "100000", "U1"),
                                  _av("NetLiquidation", "250000", "U2")])
    try:
        a.get_account()
        assert False, "expected a fail-closed raise on multiple managed accounts"
    except ValueError:
        pass


# ── #4 force-close never books P&L off price<=0 ──────────────────────────────────
def test_execute_exit_falls_back_when_price_zero(monkeypatch):
    from app.agents.trader import Trader
    t = Trader.__new__(Trader)
    t.logger = logging.getLogger("t")
    t.active_positions = {"SPY": {"trade_id": 7, "entry_price": 100.0, "shares": 10,
                                  "direction": "BUY", "bars_held": 1, "_partial_pnl": 0.0}}

    async def _fill(*a, **k):
        return None     # fill price unreadable -> caller's price (0) would otherwise be used

    monkeypatch.setattr(t, "_read_order_fill_price", _fill)

    class _Trade:
        status = "ACTIVE"
        quantity = 10
        direction = "BUY"
        pnl = None
        exit_price = None

    trade = _Trade()

    class _DB:
        def query(self, *a, **k):
            return self

        def filter_by(self, **k):
            return self

        def first(self):
            return trade

        def add(self, *a, **k):
            pass

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("app.agents.trader.get_session", lambda: _DB())
    monkeypatch.setattr("app.database.models.recompute_partial_pnl",
                        lambda *a, **k: 0.0, raising=False)

    alpaca = types.SimpleNamespace(
        get_position=lambda s, **k: {"qty": 10},
        get_latest_price=lambda s: 95.0,                 # the real fallback reference
        place_market_order=lambda *a, **k: {"order_id": "x"})

    # price passed in is 0 (force-close with no quote) and the fill readback is None
    asyncio.run(t._execute_exit("SPY", 0, "FORCE_CLOSE_EOD", alpaca))
    # if 0 had been used: pnl = (0-100)*10 = -1000 (phantom). Fallback 95 -> (95-100)*10 = -50.
    assert trade.exit_price == 95.0
    assert trade.pnl == -50.0


# ── #5 gap auto-exit confirmed by the executable bid (robust to wide-spread crashes) ─
def test_executable_sell_price_reads_bid():
    from app.agents.premarket import PremarketIntelligence
    pa = PremarketIntelligence.__new__(PremarketIntelligence)
    pa.logger = logging.getLogger("t")
    # a wide spread (real crash) still returns a usable bid — no tight-spread suppression
    wide = types.SimpleNamespace(get_quote=lambda s: {"bid": 80.0, "ask": 100.0, "mid": 90.0})
    assert pa._executable_sell_price(wide, "XYZ") == 80.0
    # no quote / no bid -> None
    assert pa._executable_sell_price(types.SimpleNamespace(get_quote=lambda s: None), "X") is None
    assert pa._executable_sell_price(
        types.SimpleNamespace(get_quote=lambda s: {"bid": 0}), "X") is None


def test_gap_real_crash_wide_spread_still_auto_exits(monkeypatch):
    # The key regression: a REAL crash widens the spread; the auto-exit must STILL fire (a tight-spread
    # gate would have wrongly suppressed it). bid is far below prior close -> confirmed -> AUTO_EXIT.
    from app.agents import premarket as pm
    pa = pm.PremarketIntelligence.__new__(pm.PremarketIntelligence)
    pa.logger = logging.getLogger("t")
    pa._prior_session_close = lambda bars: 100.0
    pa._current_price = lambda client, symbol: 88.0          # mid -12%
    client = types.SimpleNamespace(
        get_bars=lambda *a, **k: [1, 2, 3],
        get_quote=lambda s: {"bid": 85.0, "ask": 95.0, "mid": 90.0})  # WIDE spread, bid -15%
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: client)
    sent = []
    gaps = pa._check_overnight_gaps(["XYZ"], redis_send_fn=lambda q, m: sent.append((q, m)))
    assert gaps["XYZ"]["action"] == "AUTO_EXIT"
    assert any(q == "trader_exit_requests" for q, _ in sent)


def test_gap_unconfirmed_by_bid_downgrades_to_reeval(monkeypatch):
    # mid implies a big gap but the BID is near prior close (one-sided/illiquid bad mid) -> REEVAL,
    # not a blind sell.
    from app.agents import premarket as pm
    pa = pm.PremarketIntelligence.__new__(pm.PremarketIntelligence)
    pa.logger = logging.getLogger("t")
    pa._prior_session_close = lambda bars: 100.0
    pa._current_price = lambda client, symbol: 90.0          # mid -10%
    client = types.SimpleNamespace(
        get_bars=lambda *a, **k: [1, 2, 3],
        get_quote=lambda s: {"bid": 99.0, "ask": 81.0, "mid": 90.0})  # bid only -1% -> not confirmed
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: client)
    sent = []
    gaps = pa._check_overnight_gaps(["XYZ"], redis_send_fn=lambda q, m: sent.append((q, m)))
    assert gaps["XYZ"]["action"] == "REEVAL"
    assert any(q == "pm_reeval_requests" for q, _ in sent)
    assert not any(q == "trader_exit_requests" for q, _ in sent)


def test_gap_one_sided_book_uses_raw_bid(monkeypatch):
    # one-sided book during a crash: get_quote returns None (needs both sides) but the RAW bid exists
    # far below prior close -> must STILL AUTO_EXIT via get_bid (not stay stuck at REEVAL).
    from app.agents import premarket as pm
    pa = pm.PremarketIntelligence.__new__(pm.PremarketIntelligence)
    pa.logger = logging.getLogger("t")
    pa._prior_session_close = lambda bars: 100.0
    pa._current_price = lambda client, symbol: 85.0
    client = types.SimpleNamespace(
        get_bars=lambda *a, **k: [1, 2, 3],
        get_quote=lambda s: None,           # two-sided quote nulled (no ask)
        get_bid=lambda s: 84.0)             # but the executable bid is real, -16%
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: client)
    sent = []
    gaps = pa._check_overnight_gaps(["XYZ"], redis_send_fn=lambda q, m: sent.append((q, m)))
    assert gaps["XYZ"]["action"] == "AUTO_EXIT"
    assert any(q == "trader_exit_requests" for q, _ in sent)


def test_gap_no_bid_downgrades_to_reeval(monkeypatch):
    from app.agents import premarket as pm
    pa = pm.PremarketIntelligence.__new__(pm.PremarketIntelligence)
    pa.logger = logging.getLogger("t")
    pa._prior_session_close = lambda bars: 100.0
    pa._current_price = lambda client, symbol: 90.0
    client = types.SimpleNamespace(
        get_bars=lambda *a, **k: [1, 2, 3], get_quote=lambda s: None)   # no NBBO bid
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: client)
    sent = []
    gaps = pa._check_overnight_gaps(["XYZ"], redis_send_fn=lambda q, m: sent.append((q, m)))
    assert gaps["XYZ"]["action"] == "REEVAL"
    assert not any(q == "trader_exit_requests" for q, _ in sent)


# ── #6 deployed-by-type counts intraday via the DB trade_type map ────────────────
def test_deployed_by_type_counts_intraday(monkeypatch):
    from app.agents.portfolio_manager import PortfolioManager
    pm_obj = PortfolioManager.__new__(PortfolioManager)
    pm_obj.logger = logging.getLogger("t")
    # _alpaca is a property returning get_alpaca_client() -> patch that
    monkeypatch.setattr("app.integrations.get_alpaca_client",
                        lambda: types.SimpleNamespace(get_positions=lambda: [
                            {"symbol": "AAPL", "market_value": "10000"},   # intraday in DB
                            {"symbol": "MSFT", "market_value": "20000"},   # swing in DB
                            {"symbol": "SGOV", "market_value": "50000"},   # cash ETF -> excluded
                        ]))

    class _Rows:
        def filter(self, *a, **k):
            return self

        def all(self):
            return [("AAPL", "intraday"), ("MSFT", "swing")]

    db = types.SimpleNamespace(query=lambda *a, **k: _Rows(), close=lambda: None)
    monkeypatch.setattr("app.database.session.get_session", lambda: db)
    monkeypatch.setattr("app.live_trading.cash_sleeve.CASH_ETFS", {"SGOV"}, raising=False)

    dep = pm_obj._get_deployed_by_type()
    assert dep["intraday"] == 10000.0     # was always 0 before (Alpaca dict has no trade_type)
    assert dep["swing"] == 20000.0
    assert dep["total"] == 30000.0        # cash ETF excluded


def test_deployed_by_type_raises_on_broker_error(monkeypatch):
    # an indeterminate positions read must RAISE (caller fails closed) — not return all-zeros, which
    # would make the per-sleeve + gross caps read as 'nothing deployed' and over-deploy.
    from app.agents.portfolio_manager import PortfolioManager
    pm_obj = PortfolioManager.__new__(PortfolioManager)
    pm_obj.logger = logging.getLogger("t")

    def _boom():
        raise RuntimeError("503 upstream")

    monkeypatch.setattr("app.integrations.get_alpaca_client",
                        lambda: types.SimpleNamespace(get_positions=_boom))

    class _Rows:
        def filter(self, *a, **k):
            return self

        def all(self):
            return []

    monkeypatch.setattr("app.database.session.get_session",
                        lambda: types.SimpleNamespace(
                            query=lambda *a, **k: _Rows(), close=lambda: None))
    try:
        pm_obj._get_deployed_by_type()
        assert False, "expected a raise on an indeterminate positions read"
    except RuntimeError:
        pass


def test_calculate_quantity_fails_closed_on_deployed_error(monkeypatch):
    from app.agents.portfolio_manager import PortfolioManager
    pm_obj = PortfolioManager.__new__(PortfolioManager)
    pm_obj.logger = logging.getLogger("t")
    monkeypatch.setattr(pm_obj, "_get_deployed_by_type",
                        lambda: (_ for _ in ()).throw(RuntimeError("read failed")))
    # fail CLOSED -> the 1-share floor, never a full-budget size on a transient read
    qty = pm_obj._calculate_quantity(price=100.0, account_value=100_000.0, trade_type="intraday")
    assert qty == 1
