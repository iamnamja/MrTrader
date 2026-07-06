"""R1.0a — WritableAlpacaAdapter wraps place_market_order byte-identically + returns an ACK-only result.

The seam every venue plugs into for the all-IBKR migration. This pins: (1) the intent maps to the
EXISTING order call verbatim (so H3/H6 are preserved), (2) client_ref passes through unchanged (must
NOT be re-derived — that would defeat H6 idempotency), (3) OrderResult is ack-only (no fill fields),
(4) unsupported order/sec types fail loud.
"""
from unittest.mock import MagicMock

import pytest

from app.live_trading.writable_broker_adapter import (
    WritableAlpacaAdapter, OrderIntent, OrderResult,
)


def _intent(**kw):
    d = dict(venue="ALPACA", instrument_id="SPY", sec_type="ETF", side="BUY", quantity=3,
             client_ref="trend:SPY", est_price=740.0)
    d.update(kw)
    return OrderIntent(**d)


def _client():
    c = MagicMock()
    c.place_market_order.return_value = {
        "order_id": "o-123", "symbol": "SPY", "qty": 3, "side": "OrderSide.BUY",
        "status": "accepted", "created_at": None, "idempotent_reuse": False,
    }
    return c


def test_place_maps_intent_to_place_market_order_verbatim():
    c = _client()
    res = WritableAlpacaAdapter(client=c).place(_intent())
    c.place_market_order.assert_called_once_with(
        symbol="SPY", quantity=3, side="buy", client_order_id="trend:SPY", est_price=740.0)
    assert isinstance(res, OrderResult)
    assert res.broker_order_id == "o-123" and res.accepted_status == "accepted"
    assert res.idempotent_reuse is False


def test_place_propagates_idempotent_reuse():
    c = _client()
    c.place_market_order.return_value = {**c.place_market_order.return_value, "idempotent_reuse": True}
    assert WritableAlpacaAdapter(client=c).place(_intent()).idempotent_reuse is True


def test_place_client_ref_passed_verbatim():
    c = _client()
    WritableAlpacaAdapter(client=c).place(_intent(client_ref="cash:SGOV:sell"))
    assert c.place_market_order.call_args.kwargs["client_order_id"] == "cash:SGOV:sell"


def test_place_rejects_non_market():
    with pytest.raises(NotImplementedError):
        WritableAlpacaAdapter(client=_client()).place(_intent(order_type="LIMIT"))


def test_place_rejects_non_equity():
    with pytest.raises(NotImplementedError):
        WritableAlpacaAdapter(client=_client()).place(_intent(sec_type="FUT"))


def test_place_rejects_non_integer_quantity():
    # Fail loud instead of silently truncating (whole shares/lots only) — protects the futures path.
    with pytest.raises(ValueError):
        WritableAlpacaAdapter(client=_client()).place(_intent(quantity=2.9))


def test_ack_only_result_has_exactly_the_ack_fields():
    from dataclasses import fields
    res = WritableAlpacaAdapter(client=_client()).place(_intent())
    assert {f.name for f in fields(res)} == {"broker_order_id", "accepted_status", "idempotent_reuse", "raw"}
    assert "filled_qty" not in res.raw and "avg_price" not in res.raw    # no fill leakage via raw


def test_raw_is_a_copy_not_the_clients_dict():
    c = _client()
    res = WritableAlpacaAdapter(client=c).place(_intent())
    assert res.raw is not c.place_market_order.return_value


def test_get_open_orders_queries_open_only():
    c = _client()
    c.get_orders.return_value = [{"order_id": "o1", "status": "open"}]
    out = WritableAlpacaAdapter(client=c).get_open_orders()
    c.get_orders.assert_called_once_with(limit=200, status="open")
    assert out[0]["order_id"] == "o1"


def test_cancel_maps_success_and_failure():
    c = _client()
    c.cancel_order.return_value = True
    assert WritableAlpacaAdapter(client=c).cancel("o-9").accepted_status == "canceled"
    c.cancel_order.return_value = False
    assert WritableAlpacaAdapter(client=c).cancel("o-9").accepted_status == "cancel_failed"


def test_conforms_to_writable_protocol():
    from app.live_trading.writable_broker_adapter import WritableBrokerAdapter
    assert isinstance(WritableAlpacaAdapter(client=_client()), WritableBrokerAdapter)
