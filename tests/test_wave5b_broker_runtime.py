"""Alpha-v10 audit Wave 5b — broker classifiers, idempotent-reuse verify, go-live capital, ghost guard.

Pins: the error classifiers never raise on a non-JSON error body; idempotent-reuse fails CLOSED on a
symbol/side mismatch; the go-live metrics use the real configured capital (not a hardcoded $20k);
and the reconciler distrusts a partial position snapshot (won't ghost-close real positions).
"""
from __future__ import annotations

import types

import pytest

from app.integrations import alpaca as al


class _RaisingErr(Exception):
    """An error whose .message/.code property raises (non-JSON body) — must not crash classifiers."""
    @property
    def message(self):
        raise ValueError("non-JSON body")

    @property
    def code(self):
        raise ValueError("non-JSON body")


def test_classifiers_never_raise_on_non_json_error():
    e = _RaisingErr("503 <html>gateway</html>")
    assert al._is_duplicate_client_order_id(e) is False     # no crash -> falls through to normal error
    assert al._is_position_not_found(e) is False
    assert al._err_text(e)  # returns something, never raises
    assert al._err_code(e) is None


def _client(fake_trading):
    c = al.AlpacaClient.__new__(al.AlpacaClient)
    c.trading_client = fake_trading
    c.data_client = None
    return c


def test_idempotent_reuse_mismatch_fails_closed():
    # the existing order under this client_order_id is a DIFFERENT symbol -> must NOT be booked as
    # a phantom fill; the call fails (caller treats as order failure = fail-closed).
    existing = types.SimpleNamespace(id="oid", symbol="QQQ", qty="10", side="OrderSide.SELL",
                                     status="new", created_at=None)

    def _submit(req):
        raise al.APIError('{"code":1,"message":"client_order_id must be unique"}')
    fake = types.SimpleNamespace(submit_order=_submit, get_order_by_client_id=lambda c: existing)
    with pytest.raises(Exception):
        _client(fake).place_market_order("SPY", 10, "buy", client_order_id="abc")


def test_idempotent_reuse_match_returns_reuse():
    existing = types.SimpleNamespace(id="oid", symbol="SPY", qty="10", side="OrderSide.BUY",
                                     status="new", created_at=None)

    def _submit(req):
        raise al.APIError('{"code":1,"message":"client_order_id must be unique"}')
    fake = types.SimpleNamespace(submit_order=_submit, get_order_by_client_id=lambda c: existing)
    out = _client(fake).place_market_order("SPY", 10, "buy", client_order_id="abc")
    assert out["idempotent_reuse"] is True and out["symbol"] == "SPY"


def test_go_live_empty_metrics_uses_real_capital():
    from app.approval_workflow import approval_workflow
    from app.config import settings
    m = approval_workflow._empty_metrics()
    assert m["initial_capital"] == float(settings.initial_capital)   # not the hardcoded 20_000


# ── reconciler partial-snapshot guard ────────────────────────────────────────────
def _acct(equity, cash):
    return types.SimpleNamespace(equity=equity, cash=cash)


def _alpaca(equity, cash):
    return types.SimpleNamespace(trading_client=types.SimpleNamespace(get_account=lambda: _acct(equity, cash)))


def test_reconciler_distrusts_partial_snapshot():
    from app.startup_reconciler import _is_broker_view_trusted
    # account implies ~$50k held, but the snapshot only shows $1k -> partial -> distrust
    snap = {"SPY": {"market_value": 1000.0}}
    assert _is_broker_view_trusted(_alpaca(equity=100_000, cash=50_000), snap) is False


def test_reconciler_trusts_complete_snapshot():
    from app.startup_reconciler import _is_broker_view_trusted
    snap = {"SPY": {"market_value": 48_000.0}, "QQQ": {"market_value": 3_000.0}}
    assert _is_broker_view_trusted(_alpaca(equity=100_000, cash=50_000), snap) is True


def test_reconciler_trusts_all_cash_account():
    from app.startup_reconciler import _is_broker_view_trusted
    # genuinely flat (equity≈cash) -> empty snapshot is real -> trust (so real ghosts get detected)
    assert _is_broker_view_trusted(_alpaca(equity=100_000, cash=100_000), {}) is True
