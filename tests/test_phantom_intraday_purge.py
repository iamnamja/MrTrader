"""Hardening: Trader._purge_phantom_intraday_positions drops in-memory intraday positions with no
backing ACTIVE intraday DB trade.

Root cause of the recurring spurious "INTRADAY_FORCE_CLOSED AAPL": an intraday entry left an
in-memory position whose order never filled / DB row was never committed, so it got force-closed
every cycle. This purge is DB-authoritative and independent of the Alpaca-positions fetch.
"""
from __future__ import annotations

import logging
import sys
import types


def _trader():
    from app.agents.trader import Trader
    t = Trader.__new__(Trader)
    t.logger = logging.getLogger("t")
    t.active_positions = {}
    return t


def _stub_db(monkeypatch, active_intraday_symbols):
    """Patch app.database.session.get_session so Trade query returns the given ACTIVE intraday symbols."""
    class _Q:
        def filter(self, *a, **k):
            return self

        def all(self):
            return [(s,) for s in active_intraday_symbols]

    class _DB:
        def query(self, *a, **k):
            return _Q()

        def close(self):
            pass
    monkeypatch.setattr("app.database.session.get_session", lambda: _DB())
    # ensure `from app.database.models import Trade` resolves without a real model dependency
    if "app.database.models" not in sys.modules:
        m = types.ModuleType("app.database.models")
        m.Trade = type("Trade", (), {"symbol": None, "status": None, "trade_type": None})
        monkeypatch.setitem(sys.modules, "app.database.models", m)


def test_purges_unbacked_phantom(monkeypatch):
    t = _trader()
    t.active_positions = {
        "AAPL": {"trade_type": "intraday", "stop_price": 1.0},   # phantom: no DB trade
        "MSFT": {"trade_type": "intraday", "stop_price": 1.0},   # real: backed below
    }
    _stub_db(monkeypatch, active_intraday_symbols=["MSFT"])
    purged = t._purge_phantom_intraday_positions()
    assert purged == ["AAPL"]
    assert "AAPL" not in t.active_positions
    assert "MSFT" in t.active_positions


def test_keeps_backed_intraday(monkeypatch):
    t = _trader()
    t.active_positions = {"MSFT": {"trade_type": "intraday", "stop_price": 1.0}}
    _stub_db(monkeypatch, active_intraday_symbols=["MSFT"])
    assert t._purge_phantom_intraday_positions() == []
    assert "MSFT" in t.active_positions


def test_never_touches_swing_or_cash(monkeypatch):
    t = _trader()
    t.active_positions = {
        "SPY": {"trade_type": "swing", "stop_price": 1.0},     # not intraday — untouched
        "SGOV": {"trade_type": "cash", "stop_price": 1.0},     # not intraday — untouched
        "AAPL": {"trade_type": "intraday", "stop_price": 1.0},  # phantom intraday
    }
    _stub_db(monkeypatch, active_intraday_symbols=[])           # no backing intraday trades
    purged = t._purge_phantom_intraday_positions()
    assert purged == ["AAPL"]
    assert set(t.active_positions) == {"SPY", "SGOV"}


def test_no_intraday_is_noop(monkeypatch):
    t = _trader()
    t.active_positions = {"SPY": {"trade_type": "swing"}}
    # should early-return without touching the DB
    monkeypatch.setattr("app.database.session.get_session",
                        lambda: (_ for _ in ()).throw(AssertionError("DB must not be queried")))
    assert t._purge_phantom_intraday_positions() == []


def test_fails_safe_on_db_error(monkeypatch):
    # a DB error must NOT raise and must NOT drop anything (conservative)
    t = _trader()
    t.active_positions = {"AAPL": {"trade_type": "intraday"}}
    monkeypatch.setattr("app.database.session.get_session",
                        lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    assert t._purge_phantom_intraday_positions() == []
    assert "AAPL" in t.active_positions   # not dropped on error
