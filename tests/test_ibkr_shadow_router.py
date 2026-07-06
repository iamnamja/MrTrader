"""R1.1 — ibkr_shadow_router: reconstruct live ETF/cash orders as IBKR orders in SHADOW, compare to
Alpaca, place NOTHING. Gated OFF by default; structurally cannot reach the gateway; never raises."""
import pytest

from app.live_trading import ibkr_shadow_router as isr


def _orders():
    return [
        {"symbol": "SPY", "side": "BUY", "qty": 10, "client_ref": "trend-SPY", "price": 500.0},
        {"symbol": "SGOV", "side": "SELL", "qty": 3, "client_ref": "cash-SGOV", "price": 100.0},
    ]


def test_disabled_by_default_is_noop():
    # No flag / no db -> OFF -> returns [] and does nothing.
    assert isr.route_shadow(_orders(), sleeve="trend", db=None) == []


def test_enabled_reconstructs_and_matches_alpaca():
    rows = isr.route_shadow(_orders(), sleeve="trend", db=None, enabled=True)
    assert len(rows) == 2
    spy = next(r for r in rows if r["symbol"] == "SPY")
    assert (spy["alpaca_qty"] == 10 and spy["ibkr_qty"] == 10 and spy["match"] is True
            and spy["status"] == "shadow")
    assert all(r["match"] for r in rows)


def test_unmapped_symbol_fails_closed_per_order_not_raising():
    rows = isr.route_shadow(
        [{"symbol": "NOPE", "side": "BUY", "qty": 1}], sleeve="trend", db=None, enabled=True)
    assert len(rows) == 1 and rows[0]["match"] is False
    assert rows[0]["ibkr_qty"] is None and rows[0]["status"].startswith("error")


def test_bad_quantity_recorded_not_raised():
    rows = isr.route_shadow(
        [{"symbol": "SPY", "side": "BUY", "qty": None}], sleeve="cash", db=None, enabled=True)
    assert rows[0]["status"].startswith("error") and rows[0]["match"] is False


def test_one_bad_item_does_not_blind_the_batch():
    # A malformed item (missing 'symbol') must be recorded, not drop the good order (MINOR-2).
    rows = isr.route_shadow(
        [{"side": "BUY", "qty": 1},                                   # missing symbol -> error row
         {"symbol": "SPY", "side": "BUY", "qty": 7}], sleeve="trend", db=None, enabled=True)
    assert len(rows) == 2
    good = next(r for r in rows if r["symbol"] == "SPY")
    assert good["match"] is True and good["ibkr_qty"] == 7


def test_client_ref_callable_is_used_per_item():
    seen = []
    rows = isr.route_shadow(
        [{"symbol": "SPY", "side": "BUY", "qty": 1}], sleeve="trend",
        client_ref=lambda it: seen.append(it["symbol"]) or f"ref-{it['symbol']}",
        db=None, enabled=True)
    assert seen == ["SPY"] and rows[0]["status"] == "shadow"


def test_shadow_router_never_dispatches_to_gateway():
    # The stub connection RAISES on dispatch; a happy-path shadow route must never trigger it.
    conn = isr._ShadowOnlyConn()
    with pytest.raises(RuntimeError):
        conn.call(lambda ib: None)          # proves dispatch is fail-loud
    # ...yet routing many orders never calls it (placed nothing / no dispatch):
    rows = isr.route_shadow(_orders() * 5, sleeve="trend", db=None, enabled=True)
    assert len(rows) == 10 and all(r["status"] == "shadow" for r in rows)


def test_shadow_routing_flag_registered_and_settable(db_session):
    # The R1.1 flag must be a first-class, schema-registered config: default OFF, settable to ON,
    # and is_enabled must read it end-to-end (no schema entry => set_agent_config would reject it).
    from app.database.agent_config import set_agent_config, get_agent_config
    assert get_agent_config(db_session, "ibkr.shadow_routing") == "false"   # schema default OFF
    assert isr.is_enabled(db_session) is False
    set_agent_config(db_session, "ibkr.shadow_routing", "true")
    assert get_agent_config(db_session, "ibkr.shadow_routing") == "true"
    assert isr.is_enabled(db_session) is True


def test_is_enabled_reads_flag_truthy_values(monkeypatch):
    import app.database.agent_config as ac
    vals = {"ibkr.shadow_routing": "on"}
    monkeypatch.setattr(ac, "get_agent_config", lambda db, k: vals.get(k))
    assert isr.is_enabled(db=object()) is True
    vals["ibkr.shadow_routing"] = "0"
    assert isr.is_enabled(db=object()) is False
    vals.pop("ibkr.shadow_routing")
    assert isr.is_enabled(db=object()) is False        # default OFF
