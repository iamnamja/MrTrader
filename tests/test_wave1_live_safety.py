"""Alpha-v10 audit Wave 1 — live order-path fail-OPEN + idempotency fixes.

Pins the "don't blow up" cluster: get_position distinguishes not-found from an indeterminate error
(entry guard fail-CLOSED); get_positions tolerates fractional/malformed rows; place_limit_order is
idempotent on a duplicate client_order_id; the trend sleeve fails CLOSED when current positions are
undeterminable; the kill switch flattens once (no re-flatten double-sell) and verifies its persist;
the close-position route covers SHORTS; exit order ids are deterministic; the notifier escalates a
dropped CATASTROPHIC alert instead of silently orphaning it.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from app.integrations import alpaca as al


def _client(fake_trading):
    c = al.AlpacaClient.__new__(al.AlpacaClient)   # bypass __init__ (no creds needed)
    c.trading_client = fake_trading
    c.data_client = None
    return c


def _raise(exc):
    def _fn(*a, **k):
        raise exc
    return _fn


# ── get_position: not-found vs indeterminate (BLOCKER #1) ────────────────────────
def test_get_position_not_found_returns_none_even_when_raising():
    c = _client(types.SimpleNamespace(get_open_position=_raise(Exception("position does not exist"))))
    assert c.get_position("SPY") is None
    assert c.get_position("SPY", raise_on_error=True) is None   # confirmed-flat is still None


def test_get_position_indeterminate_raises_only_when_requested():
    c = _client(types.SimpleNamespace(get_open_position=_raise(RuntimeError("503 upstream"))))
    assert c.get_position("SPY") is None                        # legacy default: swallow
    with pytest.raises(RuntimeError):
        c.get_position("SPY", raise_on_error=True)              # entry guard fail-closed


def test_is_position_not_found_classifier():
    assert al._is_position_not_found(Exception("position does not exist"))
    assert not al._is_position_not_found(RuntimeError("connection reset"))


# ── get_positions: fractional + malformed-row tolerance ──────────────────────────
def test_get_positions_handles_fractional_and_skips_malformed():
    good = types.SimpleNamespace(symbol="SPY", qty="1.5", avg_entry_price="100",
                                 market_value="150", unrealized_pl="0", unrealized_plpc="0",
                                 current_price="100")
    bad = types.SimpleNamespace(symbol="BAD", qty="notanum", avg_entry_price="1",
                                market_value="1", unrealized_pl="0", unrealized_plpc="0",
                                current_price="1")
    c = _client(types.SimpleNamespace(get_all_positions=lambda: [good, bad]))
    res = c.get_positions()
    assert len(res) == 1 and res[0]["symbol"] == "SPY" and res[0]["qty"] == 1   # int(float("1.5"))


# ── place_limit_order idempotency ────────────────────────────────────────────────
def test_place_limit_order_idempotent_reuse_on_duplicate():
    existing = types.SimpleNamespace(id="oid1", symbol="SPY", qty="10", side="buy",
                                     limit_price="100", status="new", created_at=None)
    fake = types.SimpleNamespace(
        submit_order=_raise(al.APIError('{"code":42210000,"message":"client_order_id must be unique"}')),
        get_order_by_client_id=lambda coid: existing)
    out = _client(fake).place_limit_order("SPY", 10, "buy", 100.0, client_order_id="abc")
    assert out["idempotent_reuse"] is True and out["order_id"] == "oid1"


# ── trend sleeve: current positions fail-CLOSED ──────────────────────────────────
class _Q:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        return _Q(self._rows)


def test_current_trend_positions_fail_closed_on_broker_error():
    import app.live_trading.trend_sleeve as ts
    db = _FakeDB([types.SimpleNamespace(symbol="SPY")])
    fake_alpaca = types.SimpleNamespace(get_positions=_raise(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        ts._current_trend_positions(db, fake_alpaca)


def test_current_trend_positions_empty_is_genuinely_flat():
    import app.live_trading.trend_sleeve as ts
    assert ts._current_trend_positions(_FakeDB([]), object()) == {}   # query OK, no rows


# ── kill switch: no re-flatten, retry on failed flatten ──────────────────────────
def _ks(monkeypatch, flatten_results):
    from app.live_trading.kill_switch import KillSwitch
    ks = KillSwitch()
    seq = list(flatten_results)
    calls = {"n": 0}

    def fake_flatten(*, execute, alpaca):
        calls["n"] += 1
        return seq[min(calls["n"] - 1, len(seq) - 1)]
    monkeypatch.setattr("app.live_trading.emergency_flatten.flatten_alpaca", fake_flatten)
    monkeypatch.setattr(ks, "_alpaca", lambda: object())
    monkeypatch.setattr(ks, "_persist_state", lambda: True)
    monkeypatch.setattr(ks, "_audit", lambda *a, **k: None)
    return ks, calls


def test_kill_switch_reactivation_does_not_reflatten(monkeypatch):
    ks, calls = _ks(monkeypatch, [{"ok": True, "positions": [{"symbol": "SPY"}], "errors": []}])
    r1 = ks.activate("first")
    assert r1["status"] == "activated" and calls["n"] == 1
    r2 = ks.activate("again")
    assert r2["status"] == "already_active" and calls["n"] == 1   # NOT re-flattened


def test_kill_switch_failed_flatten_can_retry(monkeypatch):
    ks, calls = _ks(monkeypatch, [
        {"ok": False, "positions": [], "errors": ["x"]},
        {"ok": True, "positions": [], "errors": []},
    ])
    assert ks.activate()["flatten_ok"] is False
    assert ks.activate()["flatten_ok"] is True and calls["n"] == 2   # retried (not yet flattened)


# ── close-position route: short-aware ────────────────────────────────────────────
def test_close_position_route_covers_shorts(monkeypatch):
    from app.api import routes
    calls = {}
    fake = types.SimpleNamespace(
        get_position=lambda s: {"qty": -100},
        place_market_order=lambda sym, qty, side: calls.update(sym=sym, qty=qty, side=side) or {"order_id": "x"})
    monkeypatch.setattr(routes, "_alpaca", lambda: fake)
    res = asyncio.run(routes.close_position("spy"))
    assert calls["side"] == "buy" and calls["qty"] == 100 and res["side"] == "buy"


# ── exit order id ────────────────────────────────────────────────────────────────
def test_exit_order_id_deterministic_and_distinct():
    from app.live_trading.order_ids import exit_order_id
    a = exit_order_id("tid-1", "SPY", "full")
    assert a == exit_order_id("tid-1", "SPY", "full")     # stable retry (clock-independent)
    assert a != exit_order_id("tid-1", "SPY", "partial")  # phase distinct
    assert a != exit_order_id("tid-2", "SPY", "full")     # trade distinct


# ── notifier: critical never silently dropped ────────────────────────────────────
def test_notifier_escalates_dropped_critical(monkeypatch, tmp_path):
    from app.notifications import notifier
    monkeypatch.setattr(notifier, "DB_PATH", tmp_path / "n.db")
    rid = notifier.enqueue("kill_switch", {"reason": "x"})
    assert rid
    with notifier._conn() as c:
        c.execute("UPDATE notification_queue SET attempts=? WHERE id=?",
                  (notifier._CRITICAL_MAX_ATTEMPTS, rid))
    assert notifier.escalate_dropped_critical() == 1
    assert notifier.escalate_dropped_critical() == 0           # idempotent (marked ESCALATED)
    # the dropped critical row is no longer surfaced for normal sending
    assert all(r[0] != rid for r in notifier.pending())
