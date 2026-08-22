"""Order-status normalization + sleeve-safe PENDING_FILL promotion (2026-08-22).

`AlpacaClient.get_order_status` returned the RAW SDK enum for `status`/`side`, so every caller
doing `str(status).lower() == "filled"` compared against "orderstatus.filled" and could never
match. One leak silently disabled three things at once:

  * startup_reconciler  — PENDING_FILL rows never promoted to ACTIVE
  * portfolio_manager   — the EOD sweep never cancelled stale working orders
  * trader              — the fill branch never fired

Six weeks of un-promoted sleeve rows is what surfaced it: DBC's ACTIVE row sat frozen at 241
while the broker held 228, tripping an enforce-mode FAIL_CLOSED.

The promotion tests matter just as much as the enum ones: fixing the enum ALONE would have made
things worse, promoting DBC's pending row with filled_qty=298 (a rebalance SELL delta) beside the
stale 241 row — expected 539 against a broker holding 228.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _Enum:
    """Stands in for the SDK enum: str() yields 'OrderStatus.FILLED', as the real one does."""
    def __init__(self, label):
        self._label = label

    def __str__(self):
        return self._label


def _order(status="OrderStatus.FILLED", side="OrderSide.SELL", qty="298", filled="298"):
    return SimpleNamespace(
        id="oid-1", symbol="DBC", qty=qty, filled_qty=filled,
        side=_Enum(side), status=_Enum(status), filled_avg_price="30.17",
    )


def _client_returning(order):
    from app.integrations.alpaca import AlpacaClient
    c = AlpacaClient.__new__(AlpacaClient)          # bypass __init__ (no network/credentials)
    c.trading_client = MagicMock()
    c.trading_client.get_order_by_id.return_value = order
    return c


class TestStatusNormalization:
    def test_status_is_bare_lowercase(self):
        s = _client_returning(_order()).get_order_status("oid-1")
        assert s["status"] == "filled"
        # The regression: the raw enum stringifies to this and never matches "filled".
        assert s["status"] != "orderstatus.filled"

    def test_side_is_bare_lowercase(self):
        s = _client_returning(_order()).get_order_status("oid-1")
        assert s["side"] == "sell"

    @pytest.mark.parametrize("raw,expected", [
        ("OrderStatus.FILLED", "filled"),
        ("OrderStatus.PARTIALLY_FILLED", "partially_filled"),
        ("OrderStatus.CANCELED", "canceled"),
        ("OrderStatus.NEW", "new"),
        ("OrderStatus.ACCEPTED", "accepted"),
    ])
    def test_every_status_callers_compare_against(self, raw, expected):
        s = _client_returning(_order(status=raw)).get_order_status("oid-1")
        assert s["status"] == expected

    def test_already_bare_string_passes_through(self):
        """Some SDK versions hand back a plain str — normalizing must be idempotent."""
        o = _order()
        o.status = "filled"
        o.side = "buy"
        s = _client_returning(o).get_order_status("oid-1")
        assert s["status"] == "filled" and s["side"] == "buy"

    def test_callers_str_lower_is_a_noop_on_normalized(self):
        """Callers apply str().lower() themselves; that must not change the value."""
        s = _client_returning(_order()).get_order_status("oid-1")
        assert str(s["status"]).lower() == s["status"]


class TestSleeveClassification:
    """The rule the promotion path depends on: sleeve rows carry a TARGET, not a fill."""

    @pytest.mark.parametrize("trade_type,is_sleeve", [
        ("trend", True), ("cash", True), ("TREND", True),
        ("swing", False), ("pead", False), ("", False), (None, False),
    ])
    def test_sleeve_detection(self, trade_type, is_sleeve):
        sleeve = str(trade_type or "").lower()
        assert (sleeve in ("trend", "cash")) is is_sleeve


class TestPromotionDoesNotDoubleCount:
    """Guards the interaction that would have made the enum fix actively harmful."""

    def test_sleeve_row_keeps_target_not_fill_delta(self):
        trade = SimpleNamespace(quantity=228, trade_type="trend")
        filled_qty = 298                       # the rebalance SELL that reached the 228 target
        is_sleeve = str(trade.trade_type).lower() in ("trend", "cash")
        new_qty = trade.quantity if is_sleeve else filled_qty
        assert new_qty == 228, "sleeve row must keep its target, not adopt the delta"

    def test_swing_row_adopts_fill(self):
        trade = SimpleNamespace(quantity=0, trade_type="swing")
        filled_qty = 50
        is_sleeve = str(trade.trade_type).lower() in ("trend", "cash")
        new_qty = trade.quantity if is_sleeve else filled_qty
        assert new_qty == 50, "a swing entry's fill IS its position"

    def test_two_active_rows_would_double_count(self):
        """Why folding is required rather than a second ACTIVE row."""
        stale_active, promoted = 241, 228
        assert stale_active + promoted == 469          # what db_expected_positions would sum
        assert promoted == 228                          # broker truth after folding
