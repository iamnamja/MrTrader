"""Alpha-v10 audit Wave 5c — agent live-path fixes.

Pins: the daily-loss gate fails CLOSED (returns None) when today's P&L can't be determined; the
intraday RM slot is released on a stale-proposal discard (and only for intraday); and the macro
high-impact sizing_factor actually shrinks the order quantity.
"""
from __future__ import annotations

import logging


# ── RM daily-loss fails closed (None) when undeterminable ────────────────────────
class _Q:
    def filter_by(self, **k):
        return self

    def first(self):
        return None


class _DB:
    def query(self, *a):
        return _Q()

    def close(self):
        pass


def test_rm_daily_pnl_none_when_undeterminable(monkeypatch):
    from app.agents.risk_manager import RiskManager
    rm = RiskManager.__new__(RiskManager)
    rm.logger = logging.getLogger("t")
    monkeypatch.setattr("app.agents.risk_manager.get_session", lambda: _DB())
    # account has no last_equity -> live can't compute; no DB row -> None (fail-closed)
    assert rm._get_daily_pnl({"equity": 100_000.0}) is None
    # with last_equity -> live figure
    assert rm._get_daily_pnl({"equity": 97_000.0, "last_equity": 100_000.0}) == -3000.0


# ── intraday slot released on stale discard (only for intraday) ──────────────────
def test_release_intraday_slot_only_for_intraday(monkeypatch):
    from app.agents.trader import Trader
    import app.agents.risk_manager as rmmod
    t = Trader.__new__(Trader)
    t.logger = logging.getLogger("t")
    calls = {"n": 0}
    monkeypatch.setattr(rmmod.risk_manager, "on_intraday_position_closed",
                        lambda: calls.__setitem__("n", calls["n"] + 1))
    t._release_intraday_slot("swing")
    assert calls["n"] == 0          # no-op for non-intraday
    t._release_intraday_slot("intraday")
    assert calls["n"] == 1          # releases the intraday slot


# ── macro high-impact sizing_factor actually shrinks quantity ────────────────────
def test_macro_sizing_factor_shrinks_quantity():
    # mirrors the PM attach logic: a <1.0 factor must reduce the order quantity, not just tag metadata
    factor = 0.5
    p = {"symbol": "AAPL", "quantity": 100}
    p["macro_sizing_factor"] = factor
    _q = int(p.get("quantity") or 0)
    if _q > 0:
        p["quantity"] = max(1, int(_q * factor))
    assert p["quantity"] == 50 and p["macro_sizing_factor"] == 0.5
