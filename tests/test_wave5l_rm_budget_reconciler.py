"""Alpha-v10 audit Wave 5l — two fail-OPEN gates closed.

  RM strategy-budget: when the per-symbol trade_type map can't be read, the per-sleeve budget cap
    previously defaulted every position to 'swing' (intraday deployed read ~0 -> cap passed). Now it
    fails CLOSED (rejects), mirroring the PM-side hardening (Wave 5f).
  Reconciler _is_broker_view_trusted: expected exposure is the broker GROSS (|long|+|short| market
    value), not equity-cash — which is ~0 for a SHORT book and would 'trust' an empty/partial snapshot
    and ghost-close a real short.
"""
from __future__ import annotations

from unittest.mock import MagicMock


# ── reconciler: a short book is not trusted on an empty/partial snapshot ──────────
def _acct(**kw):
    a = MagicMock()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_short_book_distrusts_empty_snapshot():
    from app.startup_reconciler import _is_broker_view_trusted
    alpaca = MagicMock()
    # a real short: equity ~= cash (short proceeds inflate cash), but short_market_value is large.
    # equity-cash ~= 0 would have WRONGLY trusted an empty snapshot; gross uses |short_mv|.
    alpaca.trading_client.get_account.return_value = _acct(
        equity="100000.00", cash="100000.00", long_market_value="0.0",
        short_market_value="-40000.00")
    # empty snapshot while a 40k short is open -> must DISTRUST (don't ghost-close the short)
    assert _is_broker_view_trusted(alpaca, {}) is False


def test_short_book_trusted_when_snapshot_complete():
    from app.startup_reconciler import _is_broker_view_trusted
    alpaca = MagicMock()
    alpaca.trading_client.get_account.return_value = _acct(
        equity="100000.00", cash="100000.00", long_market_value="0.0",
        short_market_value="-40000.00")
    # snapshot accounts for the short exposure -> trusted
    assert _is_broker_view_trusted(alpaca, {"XYZ": {"market_value": -40000.0}}) is True


def test_genuine_flat_book_still_trusted():
    from app.startup_reconciler import _is_broker_view_trusted
    alpaca = MagicMock()
    alpaca.trading_client.get_account.return_value = _acct(
        equity="100000.00", cash="100000.00", long_market_value="0.0", short_market_value="0.0")
    assert _is_broker_view_trusted(alpaca, {}) is True   # no exposure -> empty snapshot is genuine


# ── RM strategy-budget fails CLOSED when the trade_type map is unreadable ─────────
def _rm():
    import logging
    from app.agents.risk_manager import RiskManager
    rm = RiskManager.__new__(RiskManager)
    rm.logger = logging.getLogger("t")
    return rm


def test_rm_budget_fails_closed_on_db_error(monkeypatch):
    rm = _rm()

    def _boom():
        raise RuntimeError("db down")
    monkeypatch.setattr("app.agents.risk_manager.get_session", _boom)

    reasoning = {"checks": []}
    positions = [{"symbol": "AAPL", "market_value": "10000"}]
    # trade_type map unreadable -> must REJECT (not default-to-swing and pass)
    ok = rm._strategy_budget_ok("intraday", positions, account_value=100_000.0,
                                trade_cost=1_000.0, reasoning=reasoning)
    assert ok is False
    assert reasoning["failed_rule"] == "strategy_budget_unavailable"


def test_rm_budget_passes_within_cap(monkeypatch):
    rm = _rm()

    class _T:
        def __init__(self, sym, tt):
            self.symbol, self.trade_type, self.status = sym, tt, "ACTIVE"

    class _Q:
        def filter(self, *a, **k):
            return self

        def all(self):
            return [_T("MSFT", "swing")]            # only a swing position is open

    db = type("DB", (), {"query": lambda self, *a: _Q(), "close": lambda self: None})()
    monkeypatch.setattr("app.agents.risk_manager.get_session", lambda: db)
    reasoning = {"checks": []}
    # an INTRADAY proposal: no intraday deployed yet -> within the intraday budget -> pass
    ok = rm._strategy_budget_ok("intraday", [{"symbol": "MSFT", "market_value": "20000"}],
                                account_value=100_000.0, trade_cost=1_000.0, reasoning=reasoning)
    assert ok is True
    assert reasoning["checks"][-1]["ok"] is True


def test_rm_budget_rejects_over_cap(monkeypatch):
    rm = _rm()

    class _T:
        def __init__(self, sym, tt):
            self.symbol, self.trade_type, self.status = sym, tt, "ACTIVE"

    class _Q:
        def filter(self, *a, **k):
            return self

        def all(self):
            return [_T("AAPL", "intraday")]

    db = type("DB", (), {"query": lambda self, *a: _Q(), "close": lambda self: None})()
    monkeypatch.setattr("app.agents.risk_manager.get_session", lambda: db)
    reasoning = {"checks": []}
    # AAPL intraday already deploys 35% of a 100k account; another 1k intraday trade -> over the
    # intraday budget -> reject
    ok = rm._strategy_budget_ok("intraday", [{"symbol": "AAPL", "market_value": "35000"}],
                                account_value=100_000.0, trade_cost=1_000.0, reasoning=reasoning)
    assert ok is False
    assert reasoning["failed_rule"] == "strategy_budget_cap"
