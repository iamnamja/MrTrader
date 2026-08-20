"""FIFO realized P&L for the execution blotter (2026-08-20).

The cases that matter here are the ones found against the live account during design, because each
was silent: a wrong number, not an error. Every test is pure-Python — no broker, no DB.
"""

import pytest

from app.analytics.execution_pnl import (
    BASIS_INCOMPLETE,
    BASIS_OPENING,
    BASIS_REALIZED,
    attach_realized_pnl,
    compute_realized_pnl,
    iter_fills,
)


def order(oid, symbol, side, filled_qty, price, *, status="filled", at=None, qty=None):
    return {
        "order_id": oid, "symbol": symbol, "side": side,
        "qty": qty if qty is not None else filled_qty,
        "filled_qty": filled_qty, "filled_avg_price": price, "status": status,
        "filled_at": at or f"2026-01-{int(oid[1:]) + 1:02d}T14:00:00", "submitted_at": None,
    }


class TestFillSelection:
    def test_partially_filled_canceled_order_counts(self):
        """The MP case: buy 152 -> filled 117, CANCELED. It moved the position by 117 shares.

        Selecting on status == 'filled' drops it and corrupts every later basis for that symbol.
        """
        orders = [order("o1", "MP", "buy", 117, 10.0, status="canceled", qty=152)]
        assert len(iter_fills(orders)) == 1

    def test_zero_fill_and_missing_price_excluded(self):
        orders = [
            order("o1", "X", "buy", 0, 10.0, status="canceled"),
            {"order_id": "o2", "symbol": "X", "side": "buy", "qty": 5,
             "filled_qty": 5, "filled_avg_price": None, "status": "filled",
             "filled_at": "2026-01-02T14:00:00", "submitted_at": None},
        ]
        assert iter_fills(orders) == []

    def test_fills_sorted_oldest_first(self):
        a = order("o1", "X", "buy", 1, 10.0, at="2026-03-01T10:00:00")
        b = order("o2", "X", "buy", 1, 11.0, at="2026-01-01T10:00:00")
        assert [f["order_id"] for f in iter_fills([a, b])] == ["o2", "o1"]


class TestFifoBasics:
    def test_buy_realizes_nothing(self):
        r = compute_realized_pnl([order("o1", "X", "buy", 10, 100.0)], {"X": 10})
        assert r["o1"]["realized_pnl"] is None
        assert r["o1"]["pnl_basis"] == BASIS_OPENING

    def test_sell_realizes_against_oldest_lot_first(self):
        """FIFO, not average: selling 10 after buying 10@100 then 10@200 realizes off the 100 lot."""
        orders = [
            order("o1", "X", "buy", 10, 100.0),
            order("o2", "X", "buy", 10, 200.0),
            order("o3", "X", "sell", 10, 150.0),
        ]
        r = compute_realized_pnl(orders, {"X": 10})
        assert r["o3"]["realized_pnl"] == pytest.approx(500.0)   # 10 * (150-100); avg would be 0
        assert r["o3"]["pnl_basis"] == BASIS_REALIZED

    def test_sell_spanning_multiple_lots(self):
        orders = [
            order("o1", "X", "buy", 10, 100.0),
            order("o2", "X", "buy", 10, 200.0),
            order("o3", "X", "sell", 15, 150.0),
        ]
        r = compute_realized_pnl(orders, {"X": 5})
        # 10 @100 -> +500, then 5 @200 -> -250
        assert r["o3"]["realized_pnl"] == pytest.approx(250.0)

    def test_realized_pct_is_return_on_released_basis(self):
        orders = [order("o1", "X", "buy", 10, 100.0), order("o2", "X", "sell", 10, 110.0)]
        r = compute_realized_pnl(orders, {"X": 0})
        assert r["o2"]["realized_pnl"] == pytest.approx(100.0)
        assert r["o2"]["realized_pct"] == pytest.approx(10.0)

    def test_scale_in_then_partial_sell(self):
        """The shape the user actually asked about: only portions are sold."""
        orders = [
            order("o1", "DBC", "buy", 257, 28.98),
            order("o2", "DBC", "buy", 7, 29.32),
            order("o3", "DBC", "sell", 23, 28.62),
        ]
        r = compute_realized_pnl(orders, {"DBC": 241})
        assert r["o3"]["realized_pnl"] == pytest.approx(23 * (28.62 - 28.98))
        assert r["o1"]["realized_pnl"] is None and r["o2"]["realized_pnl"] is None


class TestShorts:
    def test_short_then_cover_realizes_correctly(self):
        """A FIFO replay that ignores shorts strands lots and OVERSTATES realized P&L."""
        orders = [order("o1", "X", "sell", 10, 100.0), order("o2", "X", "buy", 10, 90.0)]
        r = compute_realized_pnl(orders, {"X": 0})
        assert r["o1"]["realized_pnl"] is None            # opening the short
        assert r["o2"]["realized_pnl"] == pytest.approx(100.0)   # covered 10 lower

    def test_flip_through_zero_opens_other_side(self):
        orders = [
            order("o1", "X", "buy", 10, 100.0),
            order("o2", "X", "sell", 15, 110.0),   # closes 10 long (+100), opens 5 short @110
            order("o3", "X", "buy", 5, 105.0),     # covers the short (+25)
        ]
        r = compute_realized_pnl(orders, {"X": 0})
        assert r["o2"]["realized_pnl"] == pytest.approx(100.0)
        assert r["o3"]["realized_pnl"] == pytest.approx(25.0)


class TestReconciliationIdentity:
    def test_realized_equals_cashflow_plus_remaining_basis(self):
        """realized == (sells - buys) + remaining_basis. This identity is what let the live
        numbers be checked against account equity, so it must hold for any fill sequence."""
        orders = [
            order("o1", "X", "buy", 100, 10.0),
            order("o2", "X", "buy", 50, 12.0),
            order("o3", "X", "sell", 120, 11.0),
            order("o4", "X", "buy", 30, 9.0),
        ]
        r = compute_realized_pnl(orders, {"X": 60})
        realized = sum(v["realized_pnl"] or 0 for v in r.values())
        buys = 100 * 10.0 + 50 * 12.0 + 30 * 9.0
        sells = 120 * 11.0
        remaining = 30 * 12.0 + 30 * 9.0        # 30 left of the 12.0 lot, plus the 9.0 lot
        assert realized == pytest.approx(sells - buys + remaining)


class TestIncompleteHistoryGuard:
    def test_replay_disagreeing_with_broker_reports_unknown(self):
        """A sell whose opening buy predates the retrievable window must NOT produce a number."""
        orders = [order("o1", "OLD", "sell", 50, 20.0)]
        r = compute_realized_pnl(orders, {"OLD": 0})   # replay ends -50, broker says 0
        assert r["o1"]["realized_pnl"] is None
        assert r["o1"]["pnl_basis"] == BASIS_INCOMPLETE

    def test_one_bad_symbol_does_not_poison_others(self):
        orders = [
            order("o1", "OLD", "sell", 50, 20.0),
            order("o2", "GOOD", "buy", 10, 100.0),
            order("o3", "GOOD", "sell", 10, 110.0),
        ]
        r = compute_realized_pnl(orders, {"OLD": 0, "GOOD": 0})
        assert r["o1"]["pnl_basis"] == BASIS_INCOMPLETE
        assert r["o3"]["realized_pnl"] == pytest.approx(100.0)

    def test_no_broker_positions_skips_validation(self):
        orders = [order("o1", "X", "buy", 10, 100.0), order("o2", "X", "sell", 10, 110.0)]
        r = compute_realized_pnl(orders, None)
        assert r["o2"]["realized_pnl"] == pytest.approx(100.0)


class TestAttach:
    def test_basis_comes_from_full_history_not_the_page(self):
        """THE reason this is server-side: the page lacks the opening buy."""
        history = [
            order("o1", "X", "buy", 10, 100.0),
            order("o2", "X", "sell", 10, 130.0),
        ]
        page = [dict(history[1])]          # only the sell is on screen
        out = attach_realized_pnl(page, history, {"X": 0})
        assert out[0]["realized_pnl"] == pytest.approx(300.0)

    def test_unknown_order_defaults_to_opening(self):
        out = attach_realized_pnl(
            [order("o9", "X", "buy", 1, 1.0, at="2026-01-02T14:00:00")], [], {})
        assert out[0]["realized_pnl"] is None
        assert out[0]["pnl_basis"] == BASIS_OPENING
