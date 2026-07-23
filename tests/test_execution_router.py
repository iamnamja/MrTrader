"""Tests for the R1.2 execution venue router (app/live_trading/execution_router.py).

Pins the fail-safe contract: default 'alpaca', unknown/error → 'alpaca' (never route a live order to
an unintended venue), and get_execution_adapter raises on an unrecognized venue."""
from __future__ import annotations

import pytest

from app.live_trading import execution_router as er
from app.live_trading.writable_broker_adapter import WritableAlpacaAdapter


def _cfg(monkeypatch, value):
    monkeypatch.setattr("app.database.agent_config.get_agent_config",
                        lambda db, k: value if k == "pm.trend_venue" else None)


def test_default_is_alpaca_when_unset(monkeypatch):
    _cfg(monkeypatch, None)
    assert er.resolve_venue(object(), "trend") == "alpaca"


def test_explicit_ibkr(monkeypatch):
    _cfg(monkeypatch, "ibkr")
    assert er.resolve_venue(object(), "trend") == "ibkr"


def test_case_and_whitespace_normalized(monkeypatch):
    _cfg(monkeypatch, "  IBKR ")
    assert er.resolve_venue(object(), "trend") == "ibkr"


def test_unknown_venue_falls_back_to_alpaca(monkeypatch):
    _cfg(monkeypatch, "etrade")           # a typo must NOT route live orders anywhere unexpected
    assert er.resolve_venue(object(), "trend") == "alpaca"


def test_config_error_falls_back_to_alpaca(monkeypatch):
    monkeypatch.setattr("app.database.agent_config.get_agent_config",
                        lambda db, k: (_ for _ in ()).throw(RuntimeError("db down")))
    assert er.resolve_venue(object(), "trend") == "alpaca"


def test_get_adapter_alpaca_returns_alpaca_adapter():
    sentinel = object()
    adapter = er.get_execution_adapter("alpaca", alpaca_client=sentinel)
    assert isinstance(adapter, WritableAlpacaAdapter)
    assert adapter._client is sentinel        # wraps the passed live client (no fresh construction)


def test_get_adapter_unknown_venue_raises():
    with pytest.raises(ValueError):
        er.get_execution_adapter("etrade")


def test_get_adapter_ibkr_builds_live_adapter(monkeypatch):
    # 'ibkr' → WritableIBKRAdapter(mode='live') on a from_config connection (both mocked; no gateway).
    import app.live_trading.writable_ibkr_adapter as wia
    import app.live_trading.ibkr_connection as ic
    built = {}
    monkeypatch.setattr(ic.IBKRConnectionManager, "from_config", classmethod(lambda cls, db: "CONN"))
    monkeypatch.setattr(wia, "WritableIBKRAdapter",
                        lambda conn, mode="shadow": built.update(conn=conn, mode=mode) or "IBKR_ADAPTER")
    out = er.get_execution_adapter("ibkr", db=object())
    assert out == "IBKR_ADAPTER" and built == {"conn": "CONN", "mode": "live"}
