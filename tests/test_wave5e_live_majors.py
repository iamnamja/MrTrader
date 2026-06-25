"""Alpha-v10 audit Wave 5e — live-path MAJOR fixes (fail-CLOSED on indeterminate state).

Pins:
  1. CapitalManager persists/restores the operator PAUSE flag across a restart (else a paused
     ramp silently resumes deploying capital after a daemon bounce).
  2. cash_sleeve._current_cash_positions RAISES on an indeterminate read (DB/broker error) so the
     rebalance fail-closes instead of treating "can't read" as "no positions" (re-buy / under-sell).
  3. emergency_flatten only reports ok=True after VERIFYING the book is flat — a 2xx ack is not a
     confirmed fill; lingering positions => ok=False.
"""
from __future__ import annotations

import types


# ── 1. capital-ramp pause survives a restart ─────────────────────────────────────
def test_capital_pause_persists_across_restart(monkeypatch):
    from app.live_trading import capital_manager as cm

    store: dict = {}
    monkeypatch.setattr("app.database.config_store.set_config",
                        lambda db, k, v, *a, **kw: store.__setitem__(k, v))
    monkeypatch.setattr("app.database.config_store.get_config",
                        lambda db, k: store.get(k))
    monkeypatch.setattr("app.database.session.get_session",
                        lambda: types.SimpleNamespace(close=lambda: None))
    # force the non-test persistence path
    monkeypatch.setattr(cm.CapitalManager, "_running_under_pytest", lambda self: False)

    mgr = cm.CapitalManager()
    mgr.start()
    mgr.pause()
    assert mgr._paused is True
    assert store.get(cm._CFG_PAUSED) == "1"

    # simulate a daemon restart: fresh instance restores from the same store
    fresh = cm.CapitalManager()
    assert fresh._paused is False          # default before load
    fresh.load_state()
    assert fresh._paused is True           # pause restored -> ramp stays held
    assert fresh.can_advance(0.0, 0.0) is False

    # and resume clears it persistently
    fresh.resume()
    assert store.get(cm._CFG_PAUSED) == "0"
    again = cm.CapitalManager()
    again.load_state()
    assert again._paused is False


# ── 2. cash positions fail CLOSED on an indeterminate read ───────────────────────
def test_current_cash_positions_raises_on_broker_error():
    from app.live_trading.cash_sleeve import _current_cash_positions, _PositionsUnavailable

    class _Row:
        symbol = "SGOV"

    class _Q:
        def filter(self, *a, **k):
            return self

        def all(self):
            return [_Row()]

    db = types.SimpleNamespace(query=lambda *a, **k: _Q())

    def _boom():
        raise RuntimeError("503 upstream")

    alpaca = types.SimpleNamespace(get_positions=_boom)
    try:
        _current_cash_positions(db, alpaca)
        assert False, "expected _PositionsUnavailable"
    except _PositionsUnavailable:
        pass


def test_current_cash_positions_empty_when_truly_flat():
    from app.live_trading.cash_sleeve import _current_cash_positions

    class _Q:
        def filter(self, *a, **k):
            return self

        def all(self):
            return []          # no cash trades -> genuinely flat

    db = types.SimpleNamespace(query=lambda *a, **k: _Q())
    alpaca = types.SimpleNamespace(get_positions=lambda: [])
    assert _current_cash_positions(db, alpaca) == {}


# ── 3. emergency_flatten verifies the book is actually flat ───────────────────────
def test_emergency_flatten_ok_false_when_positions_remain():
    from app.live_trading.emergency_flatten import flatten_alpaca

    class _Resp:
        def __init__(self, sym):
            self.symbol = sym
            self.status = 200

    calls = {"n": 0}

    def _get_positions():
        # first call (pre-flatten snapshot) shows one position; the post-flatten verify ALSO
        # shows it still open -> the 2xx ack lied -> must report ok=False
        calls["n"] += 1
        return [{"symbol": "SPY", "qty": 10, "market_value": 1000.0}]

    tc = types.SimpleNamespace(
        close_all_positions=lambda cancel_orders=True: [_Resp("SPY")])
    alpaca = types.SimpleNamespace(get_positions=_get_positions, trading_client=tc)

    report = flatten_alpaca(execute=True, alpaca=alpaca)
    assert report["ok"] is False
    assert any("STILL OPEN" in e for e in report["errors"])


def test_emergency_flatten_ok_true_when_verified_flat():
    from app.live_trading.emergency_flatten import flatten_alpaca

    class _Resp:
        def __init__(self, sym):
            self.symbol = sym
            self.status = 200

    seq = [[{"symbol": "SPY", "qty": 10, "market_value": 1000.0}], []]  # pre, then post=flat

    def _get_positions():
        return seq.pop(0) if seq else []

    tc = types.SimpleNamespace(
        close_all_positions=lambda cancel_orders=True: [_Resp("SPY")])
    alpaca = types.SimpleNamespace(get_positions=_get_positions, trading_client=tc)

    report = flatten_alpaca(execute=True, alpaca=alpaca)
    assert report["ok"] is True
