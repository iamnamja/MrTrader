"""Per-sleeve daily P&L reconstruction (2026-09-01).

Three months of live paper trading recorded NO P&L: `trend_daily` had 14 rows with every P&L
column NULL (the only caller passed none, and ran weekly), and `cash_daily` had no P&L columns at
all. This module rebuilds the series from the daily back_validation snapshots plus the fill
blotter.

Every test below pins a mistake that was actually made and caught by reconciling the result
against the account's own NAV move — none are hypothetical:

  * row 0 absorbing ALL pre-window realized P&L        (+$1,051.95 of contamination)
  * day-0's own fills double-counted vs the baseline   (-$52.61)
  * the unrealized SEED starting at 0 rather than the inherited level  (-$199.33)

Each is silent: the numbers stay plausible, which is precisely why the reconciliation criterion
exists and why these are pinned.
"""

import pytest

from app.live_trading.sleeve_pnl import (
    compute_daily_pnl,
    daily_pnl_series,
    sleeve_of_fill,
)


def fill(sym, side, qty, px, date, coid=None):
    return {"order_id": f"{sym}-{date}-{side}-{qty}", "symbol": sym, "side": side,
            "filled_qty": qty, "filled_avg_price": px, "status": "filled",
            "filled_at": f"{date}T14:00:00", "submitted_at": f"{date}T13:59:00",
            "client_order_id": coid}


def snaps(*pairs):
    return [{"date": d, "prices": p} for d, p in pairs]


class TestSleeveAttribution:
    @pytest.mark.parametrize("coid,expected", [
        ("trend-20260831-DBC", "trend"),
        ("cash-20260831-SGOV-buy", "cash"),
        ("TREND-20260831-SPY", "trend"),
    ])
    def test_prefix_attribution(self, coid, expected):
        assert sleeve_of_fill(fill("DBC", "buy", 1, 1.0, "2026-08-31", coid)) == expected

    def test_uuid_coid_falls_back_to_symbol(self):
        """A real 2026-06-17 QQQ sell carried a raw UUID. Dropping it into an unattributed
        bucket would silently break the reconciliation, so it must classify by symbol."""
        got = sleeve_of_fill(fill("QQQ", "sell", 5, 729.44, "2026-06-17",
                                  "1c9ffa3e-e69c-4496-8abc-000000000000"))
        assert got != "unknown"


class TestWarmUp:
    """Fills up to AND INCLUDING the first snapshot build the opening book but book no P&L."""

    def test_pre_window_realized_is_excluded(self):
        fills = [
            fill("SPY", "buy", 10, 100.0, "2026-01-01"),
            fill("SPY", "sell", 10, 150.0, "2026-01-02"),   # +$500 realized, BEFORE the window
            fill("SPY", "buy", 10, 200.0, "2026-01-03"),
        ]
        rows = compute_daily_pnl(fills, snaps(("2026-06-01", {"SPY": 200.0}),
                                              ("2026-06-02", {"SPY": 201.0})))
        assert rows[0]["realized"] == 0.0, "pre-window realized P&L leaked into row 0"
        assert sum(r["realized"] for r in rows) == 0.0

    def test_day_zero_fills_are_baseline_not_pnl(self):
        """Day-0 trades had already executed when day-0's NAV was recorded."""
        fills = [
            fill("SPY", "buy", 10, 100.0, "2026-05-01"),
            fill("SPY", "sell", 5, 150.0, "2026-06-01"),    # ON the window-start date
        ]
        rows = compute_daily_pnl(fills, snaps(("2026-06-01", {"SPY": 150.0}),
                                              ("2026-06-02", {"SPY": 151.0})))
        assert rows[0]["realized"] == 0.0
        assert rows[0]["positions"] == {"SPY": 5.0}        # book still established

    def test_opening_book_carries_correct_cost_basis(self):
        fills = [fill("SPY", "buy", 10, 100.0, "2026-05-01")]
        rows = compute_daily_pnl(fills, snaps(("2026-06-01", {"SPY": 120.0}),))
        assert rows[0]["cost_basis"] == pytest.approx(1000.0)
        assert rows[0]["unrealized"] == pytest.approx(200.0)   # LEVEL, not a daily change


class TestSeeding:
    """The series must not book an inherited unrealized level as day-one P&L."""

    def test_inherited_level_is_not_day_one_pnl(self):
        fills = [fill("SPY", "buy", 10, 100.0, "2026-05-01")]
        rows = daily_pnl_series(compute_daily_pnl(
            fills, snaps(("2026-06-01", {"SPY": 80.0}),      # inherited -$200 unrealized
                         ("2026-06-02", {"SPY": 80.0}))))    # flat day
        assert rows[0]["daily"] == pytest.approx(0.0), "booked the inherited level as day one"
        assert rows[1]["daily"] == pytest.approx(0.0)
        assert rows[-1]["cumulative"] == pytest.approx(0.0)

    def test_only_the_move_after_the_baseline_counts(self):
        fills = [fill("SPY", "buy", 10, 100.0, "2026-05-01")]
        rows = daily_pnl_series(compute_daily_pnl(
            fills, snaps(("2026-06-01", {"SPY": 80.0}), ("2026-06-02", {"SPY": 90.0}))))
        assert rows[-1]["cumulative"] == pytest.approx(100.0)   # 10 shares x +$10

    def test_realized_after_baseline_is_counted(self):
        fills = [
            fill("SPY", "buy", 10, 100.0, "2026-05-01"),
            fill("SPY", "sell", 10, 120.0, "2026-06-02"),
        ]
        rows = daily_pnl_series(compute_daily_pnl(
            fills, snaps(("2026-06-01", {"SPY": 100.0}), ("2026-06-02", {"SPY": 120.0}))))
        # Sold the whole book at 120 vs cost 100 -> +$200 realized, unrealized returns to 0.
        assert rows[-1]["cumulative"] == pytest.approx(200.0)


class TestUnmarkedDays:
    def test_unpriced_holding_yields_none_not_a_guess(self):
        fills = [fill("SGOV", "buy", 100, 100.0, "2026-05-01")]
        rows = compute_daily_pnl(fills, snaps(("2026-06-01", {}),))   # no price for SGOV
        assert rows[0]["unrealized"] is None
        assert rows[0]["unpriced"] == ["SGOV"]

    def test_gap_does_not_advance_cumulative(self):
        """A missing mark must not be absorbed as a gain — the 2026-08-24/25 outage dropped
        snapshots, and interpolating across such a gap would invent P&L."""
        rows = daily_pnl_series([
            {"date": "2026-06-01", "realized": 0.0, "unrealized": 0.0},
            {"date": "2026-06-02", "realized": 0.0, "unrealized": None},
            {"date": "2026-06-03", "realized": 0.0, "unrealized": 50.0},
        ])
        assert rows[1]["daily"] is None
        assert rows[1]["cumulative"] == pytest.approx(0.0)
        # The move is attributed once, on the next MARKED day — not lost, not doubled.
        assert rows[2]["daily"] == pytest.approx(50.0)


class TestReconciliationIdentity:
    def test_sleeve_sum_equals_unfiltered_total(self):
        """Attribution must be exhaustive: trend + cash has to equal the whole book, or the
        NAV reconciliation silently absorbs the difference."""
        fills = [
            fill("SPY", "buy", 10, 100.0, "2026-05-01", "trend-x-SPY"),
            fill("SGOV", "buy", 100, 100.0, "2026-05-01", "cash-x-SGOV"),
        ]
        sn = snaps(("2026-06-01", {"SPY": 100.0, "SGOV": 100.0}),
                   ("2026-06-02", {"SPY": 110.0, "SGOV": 100.5}))
        total = daily_pnl_series(compute_daily_pnl(fills, sn))[-1]["cumulative"]
        parts = sum(daily_pnl_series(compute_daily_pnl(fills, sn, sleeve=s))[-1]["cumulative"]
                    for s in ("trend", "cash"))
        assert parts == pytest.approx(total)
        assert total == pytest.approx(10 * 10 + 100 * 0.5)

    def test_no_snapshots_is_safe(self):
        assert compute_daily_pnl([fill("SPY", "buy", 1, 1.0, "2026-01-01")], []) == []
