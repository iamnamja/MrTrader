"""Alpha-v10 audit Wave 5d (hotfix) — exit path fail-CLOSED on an indeterminate broker read.

Re-audit #3 BLOCKER: _execute_exit treated an indeterminate get_position read as 'flat' and marked
the trade closed / dropped it WITHOUT placing an exit order — abandoning a still-open live position
(stop/target no longer enforced). Same root in _execute_partial_exit. Both now fail CLOSED: on an
indeterminate read they abort and keep the position monitored (retry next cycle).
"""
from __future__ import annotations

import asyncio
import logging
import types


def test_execute_exit_aborts_on_indeterminate_read():
    from app.agents.trader import Trader
    t = Trader.__new__(Trader)
    t.logger = logging.getLogger("t")
    t.active_positions = {"SPY": {"trade_id": 1, "entry_price": 100.0, "shares": 10,
                                  "direction": "BUY"}}
    placed = []

    def _gp(sym, raise_on_error=False):
        raise RuntimeError("503 upstream")     # indeterminate read

    alpaca = types.SimpleNamespace(
        get_position=_gp,
        place_market_order=lambda *a, **k: placed.append(a) or {"order_id": "x"})
    asyncio.run(t._execute_exit("SPY", 100.0, "STOP", alpaca))
    # the position must NOT be marked closed / dropped, and no (wrong) exit order placed
    assert "SPY" in t.active_positions
    assert placed == []


def test_partial_exit_resets_guard_on_indeterminate_read(monkeypatch):
    from app.agents.trader import Trader
    t = Trader.__new__(Trader)
    t.logger = logging.getLogger("t")
    pos = {"_partial_exited": False, "direction": "BUY", "entry_price": 100.0, "stop_price": 98.0}
    t.active_positions = {"SPY": pos}
    monkeypatch.setattr("app.database.session.get_session",
                        lambda: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr("app.database.agent_config.get_agent_config", lambda db, k: 0.5)

    def _gp(sym, raise_on_error=False):
        raise RuntimeError("503 upstream")

    alpaca = types.SimpleNamespace(get_position=_gp)
    asyncio.run(t._execute_partial_exit("SPY", 110.0, 2.0, alpaca))
    # one-shot guard reset so the partial retries next cycle (not permanently skipped)
    assert pos["_partial_exited"] is False
