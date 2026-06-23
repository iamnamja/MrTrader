"""Alpha-v10 P2.3 — the IBKR futures rebalance EXECUTOR (shadow-first, default-off).

Pins: DORMANT by default (no connect); SHADOW computes would-be orders but PLACES NOTHING (no order
path exists) while writing audit rows; verify-on-connect CRITICAL mismatches drop the instrument;
kill-switch / not-connected / NAV<=0 fail-closed; run_id/orderRef are deterministic; trading_mode=
'live' still places nothing in P2.3.
"""
from __future__ import annotations

import json

import app.live_trading.futures_sleeve as fs
from app.live_trading import instrument_master as im
from app.live_trading.broker_adapter import AccountState, BrokerHealth, CanonicalPosition
from app.live_trading.ibkr_adapter import ContractMismatch
from app.live_trading.order_ids import futures_order_ref, futures_run_id


def _pos(iid, qty, price):
    inst = im.get(iid)
    mult = inst.multiplier if inst else 1.0
    return CanonicalPosition(
        instrument_id=iid, venue=im.IBKR, broker_symbol=inst.broker_symbol(im.IBKR),
        asset_class=im.FUTURE, quantity=qty, price=price, multiplier=mult, currency="USD",
        market_value=qty * price * mult, notional=abs(qty) * price * mult, mapped=True)


class _FakeIBKR:
    venue = im.IBKR

    def __init__(self, nav=1_000_000.0, positions=None, mismatches=None, connected=True):
        self._nav = nav
        self._pos = positions or []
        self._mm = mismatches or []
        self._conn = connected
        self.connect_calls = 0
        self.disconnected = False

    def connect(self):
        self.connect_calls += 1
        return BrokerHealth(im.IBKR, self._conn, self._conn, "ok" if self._conn else "down")

    def verify_contracts(self):
        return self._mm

    def get_account(self):
        return AccountState(im.IBKR, self._nav, 0.0, 0.0)

    def get_positions(self):
        return list(self._pos)

    def disconnect(self):
        self.disconnected = True


def _cfg(monkeypatch, audits, *, enabled=True, futures_enabled=True, mode="shadow",
         weights=None, kill=False):
    conf = {
        "ibkr.enabled": "true" if enabled else "false",
        "ibkr.futures_enabled": "true" if futures_enabled else "false",
        "ibkr.trading_mode": mode,
        "ibkr.futures_target_weights_json": json.dumps(weights or {}),
        "pm.reconciliation_mode": "shadow",
        "pm.whole_book_gate_mode": "shadow",
    }
    monkeypatch.setattr("app.database.agent_config.get_agent_config",
                        lambda db, key: conf.get(key))
    from app.live_trading.kill_switch import kill_switch
    monkeypatch.setattr(kill_switch, "_active", kill, raising=False)
    monkeypatch.setattr("app.database.decision_audit.write_decision",
                        lambda **kw: audits.append(kw))


# ── dormancy / fail-closed ───────────────────────────────────────────────────────
def test_dormant_when_flag_off_does_not_connect(monkeypatch):
    audits = []
    _cfg(monkeypatch, audits, futures_enabled=False, weights={"FUT.ES": 0.5})
    fake = _FakeIBKR(positions=[_pos("FUT.ES", 1, 5000.0)])
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert summary["status"] == "dormant"
    assert fake.connect_calls == 0          # never connected
    assert audits == []


def test_kill_switch_blocks(monkeypatch):
    audits = []
    _cfg(monkeypatch, audits, weights={"FUT.ES": 0.5}, kill=True)
    fake = _FakeIBKR(positions=[_pos("FUT.ES", 1, 5000.0)])
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert summary["status"] == "blocked" and summary["block_reason"] == "kill_switch"
    assert fake.connect_calls == 0


def test_not_connected_fails_closed(monkeypatch):
    audits = []
    _cfg(monkeypatch, audits, weights={"FUT.ES": 0.5})
    fake = _FakeIBKR(positions=[_pos("FUT.ES", 1, 5000.0)], connected=False)
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert summary["status"] == "failed" and summary["block_reason"] == "ibkr_not_connected"


def test_nav_zero_fails_closed(monkeypatch):
    audits = []
    _cfg(monkeypatch, audits, weights={"FUT.ES": 0.5})
    fake = _FakeIBKR(nav=0.0, positions=[_pos("FUT.ES", 1, 5000.0)])
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert summary["status"] == "failed" and summary["block_reason"] == "nav_unavailable"


# ── shadow: computes would-be orders, PLACES NOTHING ──────────────────────────────
def test_shadow_computes_orders_but_places_nothing(monkeypatch):
    audits = []
    _cfg(monkeypatch, audits, weights={"FUT.ES": 0.5})    # 0.5*1M/(5000*50)=2 lots
    fake = _FakeIBKR(positions=[_pos("FUT.ES", 1, 5000.0)])   # held 1 -> target 2 -> buy 1
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert summary["status"] == "ok" and summary["mode"] == "shadow"
    assert summary["target_lots"] == {"FUT.ES": 2}
    assert len(summary["would_orders"]) == 1
    o = summary["would_orders"][0]
    assert o["instrument_id"] == "FUT.ES" and o["side"] == "buy" and o["qty"] == 1
    assert o["order_ref"] == futures_order_ref(summary["run_id"], "FUT.ES", "buy")
    assert summary["placed"] == 0                       # nothing placed
    # a shadow audit row was written for the would-be order
    assert any(a["strategy"] == fs.STRATEGY_ID and a["block_reason"] == "shadow" for a in audits)
    assert fake.disconnected is False                  # injected adapter not auto-disconnected


def test_verify_critical_mismatch_drops_instrument(monkeypatch):
    audits = []
    _cfg(monkeypatch, audits, weights={"FUT.ES": 0.5, "FUT.GC": 0.5})
    fake = _FakeIBKR(positions=[_pos("FUT.ES", 0, 5000.0), _pos("FUT.GC", 0, 2000.0)],
                     mismatches=[ContractMismatch("FUT.ES", "multiplier", 50.0, 5.0, True)])
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert "FUT.ES" in summary["verify_blocked"]
    assert "FUT.ES" not in summary["target_lots"]      # blocked instrument never sized


def test_empty_weights_no_orders(monkeypatch):
    audits = []
    _cfg(monkeypatch, audits, weights={})
    fake = _FakeIBKR(positions=[_pos("FUT.ES", 0, 5000.0)])
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert summary["status"] == "ok" and summary["would_orders"] == [] and audits == []


def test_live_mode_in_p23_still_places_nothing(monkeypatch):
    audits = []
    _cfg(monkeypatch, audits, mode="live", weights={"FUT.ES": 0.5})
    fake = _FakeIBKR(positions=[_pos("FUT.ES", 1, 5000.0)])
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert summary["mode"] == "live"
    assert summary["placed"] == 0                       # P2.3 has no order path even in 'live'
    assert len(summary["would_orders"]) == 1


# ── run-id idempotency ────────────────────────────────────────────────────────────
def test_run_id_deterministic_and_distinct():
    a = futures_run_id("futures_book", "2026-06-22", "2026-06-22", "h1", "p2.3")
    b = futures_run_id("futures_book", "2026-06-22", "2026-06-22", "h1", "p2.3")
    c = futures_run_id("futures_book", "2026-06-23", "2026-06-23", "h1", "p2.3")
    assert a == b and a != c and len(a) == 16


def test_order_ref_shape():
    assert futures_order_ref("abc123", "FUT.ES", "buy") == "abc123-FUT.ES-buy"


def test_executor_has_no_placement_method():
    for forbidden in ("place_order", "place_market_order", "submit_order"):
        assert not hasattr(fs, forbidden)


def test_executor_never_touches_alpaca_client(monkeypatch):
    audits = []
    _cfg(monkeypatch, audits, weights={"FUT.ES": 0.5})

    def _boom():
        raise AssertionError("futures executor must never construct an Alpaca client")
    monkeypatch.setattr("app.integrations.get_alpaca_client", _boom)
    fake = _FakeIBKR(positions=[_pos("FUT.ES", 1, 5000.0)])
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert summary["status"] == "ok"        # ran to completion without ever calling Alpaca


def test_reconcile_is_ibkr_scoped_not_alpaca_book(monkeypatch):
    # The futures reconcile must pass expected={} so it does NOT pull the whole Alpaca DB book in as
    # phantom "missing" breaks. Prove db_expected_positions (the Alpaca-book source) is never called.
    audits = []
    _cfg(monkeypatch, audits, weights={"FUT.ES": 0.5})
    called = {"db_expected": 0}

    def _track(db):
        called["db_expected"] += 1
        return {("ALPACA", "SPY"): 100.0}     # a non-empty Alpaca book that must NOT leak in
    monkeypatch.setattr("app.live_trading.reconciliation.db_expected_positions", _track)
    fake = _FakeIBKR(positions=[_pos("FUT.ES", 1, 5000.0)])
    summary = fs.run_futures_rebalance(db=object(), adapter=fake)
    assert called["db_expected"] == 0         # Alpaca book never consulted by the futures executor
    assert summary.get("recon_mode") == "shadow"
