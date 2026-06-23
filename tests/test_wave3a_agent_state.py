"""Alpha-v10 audit Wave 3a — live agent-path state-corruption fixes.

Pins: exit P&L uses the ACTUAL fill price (not a fabricated fallback); a partial exit decrements the
in-memory share count (no double-counted P&L in the no-position fallback); and the 30-min pending-
approval rescore only re-scores SWING proposals (never withdraws a valid intraday entry with the
swing model).
"""
from __future__ import annotations

import asyncio
import types

from app.agents.trader import Trader


class _FakeDB:
    def query(self, *a):
        return self

    def filter_by(self, **k):
        return self

    def first(self):
        return None

    def add(self, *a):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


# ── _read_order_fill_price ───────────────────────────────────────────────────────
def test_read_order_fill_price_returns_fill():
    t = Trader.__new__(Trader)
    t.logger = __import__("logging").getLogger("t")
    alpaca = types.SimpleNamespace(
        get_order_status=lambda oid: {"filled_avg_price": 123.45, "filled_qty": 10})
    assert asyncio.run(t._read_order_fill_price(alpaca, "oid", attempts=1, delay=0)) == 123.45


def test_read_order_fill_price_none_when_unfilled():
    t = Trader.__new__(Trader)
    t.logger = __import__("logging").getLogger("t")
    alpaca = types.SimpleNamespace(get_order_status=lambda oid: {"filled_avg_price": None})
    assert asyncio.run(t._read_order_fill_price(alpaca, "oid", attempts=2, delay=0)) is None


# ── partial-exit decrements in-memory shares ─────────────────────────────────────
def test_partial_exit_decrements_pos_shares(monkeypatch):
    t = Trader.__new__(Trader)
    t.logger = __import__("logging").getLogger("t")
    pos = {"direction": "BUY", "entry_price": 100.0, "stop_price": 98.0,
           "trade_id": None, "shares": 100, "_partial_pnl": 0.0, "_partial_exited": False}
    t.active_positions = {"AAPL": pos}

    monkeypatch.setattr("app.agents.trader.get_session", lambda: _FakeDB())
    monkeypatch.setattr("app.database.session.get_session", lambda: _FakeDB())
    monkeypatch.setattr("app.database.agent_config.get_agent_config", lambda db, k: 0.50)
    monkeypatch.setattr("app.agents.compliance.compliance_tracker",
                        types.SimpleNamespace(record_sale_proceeds=lambda *a, **k: None))

    async def _noop(*a, **k):
        return None
    monkeypatch.setattr(t, "log_decision", _noop)

    alpaca = types.SimpleNamespace(
        get_position=lambda s: {"qty": 100},
        place_market_order=lambda *a, **k: {"order_id": "x"})

    asyncio.run(t._execute_partial_exit("AAPL", 110.0, 2.0, alpaca))
    assert pos["shares"] == 50          # 100 - 50% partial; was never decremented before


# ── 30-min rescore only re-scores SWING pendings ─────────────────────────────────
def test_rescore_skips_intraday_pendings(monkeypatch):
    from app.agents.portfolio_manager import PortfolioManager
    pm = PortfolioManager.__new__(PortfolioManager)
    pm.logger = __import__("logging").getLogger("t")
    import time as _t
    # one SWING pending (should be rescored -> low score -> withdrawn) + one INTRADAY pending
    # (must NOT be rescored/withdrawn by the swing model).
    pm._pending_approvals = {
        "SWG": (_t.monotonic(), "swing"),
        "INTRA": (_t.monotonic(), "intraday"),
    }
    rescored = []

    class _Model:
        feature_names = ["f"]
        is_trained = True

        def predict(self, x):
            return None, [0.0]      # below any MIN_SCORE -> withdraw if rescored
    pm.model = _Model()
    monkeypatch.setattr(
        "app.integrations.get_alpaca_client",
        lambda: types.SimpleNamespace(get_bars=lambda *a, **k: (rescored.append(a[0]) or _bars())))
    pm.feature_engineer = types.SimpleNamespace(
        engineer_features=lambda *a, **k: {"f": 1.0})
    pm._normalize_for_inference = lambda x, syms, model: x
    sent = []
    pm.send_message = lambda q, m: sent.append(m)

    async def _noop(*a, **k):
        return None
    pm.log_decision = _noop
    monkeypatch.setattr("app.database.agent_config.get_agent_config", lambda db, k: None)
    monkeypatch.setattr("app.database.session.get_session", lambda: _FakeDB())

    asyncio.run(pm._rescore_pending_approvals())

    withdrawn = {m["symbol"] for m in sent if m.get("action") == "WITHDRAW"}
    assert "SWG" in withdrawn          # swing rescored -> withdrawn on low score
    assert "INTRA" not in withdrawn    # intraday NOT withdrawn by the swing model
    assert "INTRA" not in rescored     # and not even rescored


def _bars():
    import pandas as pd
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]})
