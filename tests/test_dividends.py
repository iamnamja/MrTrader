"""Uncredited-dividend accrual (2026-09-02).

Alpaca paper credits NO dividends, so the P&L series is price-only and understates the book. Over
2026-06-17..2026-09-01 that gap was **$463.18** — enough to flip the headline from -$41.19 (paper)
to +$421.99 (economic).

The contract these tests pin: the accrual is recorded ALONGSIDE the P&L and never inside it,
because `daily_pnl`/`cumulative_pnl` must keep reconciling against the broker's own NAV. That
reconciliation is the only thing that caught the three construction errors in
`docs/reference/PNL_TRACKING_SCOPE_2026-09-01.md`.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.live_trading.dividends import accrual_for, fetch_dividends
from app.live_trading.sleeve_pnl import compute_daily_pnl, daily_pnl_series


def fill(sym, side, qty, px, date, coid=None):
    return {"order_id": f"{sym}{date}{side}", "symbol": sym, "side": side, "filled_qty": qty,
            "filled_avg_price": px, "status": "filled", "filled_at": f"{date}T14:00:00",
            "submitted_at": None, "client_order_id": coid}


class TestAccrualFor:
    def test_pays_on_the_ex_date_only(self):
        divs = {"SGOV": {"2026-09-01": 0.307}}
        assert accrual_for({"SGOV": 493}, "2026-09-01", divs) == pytest.approx(493 * 0.307)
        assert accrual_for({"SGOV": 493}, "2026-08-31", divs) == 0.0

    def test_scales_with_the_position_held_on_that_date(self):
        divs = {"SGOV": {"2026-09-01": 0.307}}
        assert accrual_for({"SGOV": 100}, "2026-09-01", divs) == pytest.approx(30.7)

    def test_sums_across_symbols(self):
        divs = {"SPY": {"2026-06-20": 1.904}, "QQQ": {"2026-06-20": 0.813}}
        got = accrual_for({"SPY": 10, "QQQ": 8}, "2026-06-20", divs)
        assert got == pytest.approx(10 * 1.904 + 8 * 0.813)

    def test_no_position_no_accrual(self):
        assert accrual_for({}, "2026-09-01", {"SGOV": {"2026-09-01": 0.3}}) == 0.0

    def test_missing_dividend_data_is_zero_not_an_error(self):
        assert accrual_for({"SGOV": 100}, "2026-09-01", None) == 0.0
        assert accrual_for({"SGOV": 100}, "2026-09-01", {}) == 0.0


class TestFetchDividends:
    @pytest.fixture(autouse=True)
    def _clear(self):
        """The memo is process-wide; without this each test would read the previous one's result."""
        from app.live_trading.dividends import clear_cache
        clear_cache()
        yield
        clear_cache()

    def test_handles_dataframe_shaped_response(self):
        """yfinance returns a single-column DataFrame in some versions; .items() on it yields
        (column, Series) and blew up on a truthiness test, silently returning NO dividends."""
        df = pd.DataFrame({"Dividends": [0.296, 0.307]},
                          index=pd.to_datetime(["2026-07-01", "2026-08-03"]))
        tkr = MagicMock(); tkr.dividends = df
        with patch("yfinance.Ticker", return_value=tkr):
            out = fetch_dividends(["SGOV"], "2026-06-17")
        assert out["SGOV"] == {"2026-07-01": 0.296, "2026-08-03": 0.307}

    def test_handles_series_shaped_response(self):
        ser = pd.Series([0.296], index=pd.to_datetime(["2026-07-01"]), name="Dividends")
        tkr = MagicMock(); tkr.dividends = ser
        with patch("yfinance.Ticker", return_value=tkr):
            out = fetch_dividends(["SGOV"], "2026-06-17")
        assert out["SGOV"] == {"2026-07-01": 0.296}

    def test_filters_to_the_window(self):
        ser = pd.Series([0.299, 0.296], index=pd.to_datetime(["2026-06-01", "2026-07-01"]))
        tkr = MagicMock(); tkr.dividends = ser
        with patch("yfinance.Ticker", return_value=tkr):
            out = fetch_dividends(["SGOV"], "2026-06-17")
        assert list(out["SGOV"]) == ["2026-07-01"], "pre-window payment must not be counted"

    def test_cache_key_includes_today_so_it_expires(self):
        """A process runs for weeks; a cache without a date component would pin dividend history
        at boot and never see a newly-declared ex-div."""
        from datetime import date
        import app.live_trading.dividends as dv
        ser = pd.Series([0.296], index=pd.to_datetime(["2026-07-01"]))
        tkr = MagicMock(); tkr.dividends = ser
        with patch("yfinance.Ticker", return_value=tkr):
            dv.fetch_dividends(["SGOV"], "2026-06-17")
        assert any(date.today().isoformat() in k for k in dv._CACHE)

    def test_lookup_failure_never_raises(self):
        """A dividend lookup must not break P&L reconstruction — that series is the one that
        actually reconciles."""
        with patch("yfinance.Ticker", side_effect=RuntimeError("network")):
            assert fetch_dividends(["SGOV"], "2026-06-17") == {}


class TestSeparationFromPnL:
    """THE contract: dividends never enter the reconciling series."""

    def _series(self, divs):
        fills = [fill("SGOV", "buy", 100, 100.0, "2026-05-01", "cash-x-SGOV")]
        snaps = [{"date": "2026-06-01", "prices": {"SGOV": 100.0}},
                 {"date": "2026-06-02", "prices": {"SGOV": 100.0}}]
        return daily_pnl_series(compute_daily_pnl(fills, snaps, divs=divs))

    def test_dividend_excluded_from_daily_and_cumulative(self):
        rows = self._series({"SGOV": {"2026-06-02": 0.30}})
        last = rows[-1]
        assert last["daily"] == pytest.approx(0.0), "dividend leaked into daily_pnl"
        assert last["cumulative"] == pytest.approx(0.0), "dividend leaked into cumulative_pnl"

    def test_dividend_reported_separately(self):
        rows = self._series({"SGOV": {"2026-06-02": 0.30}})
        last = rows[-1]
        assert last["dividend"] == pytest.approx(30.0)
        assert last["cumulative_dividend"] == pytest.approx(30.0)
        assert last["cumulative_economic"] == pytest.approx(30.0)

    def test_economic_equals_paper_plus_dividends(self):
        rows = self._series({"SGOV": {"2026-06-02": 0.30}})
        last = rows[-1]
        assert last["cumulative_economic"] == pytest.approx(
            last["cumulative"] + last["cumulative_dividend"])

    def test_no_dividends_leaves_economic_equal_to_paper(self):
        last = self._series({})[-1]
        assert last["cumulative_economic"] == pytest.approx(last["cumulative"])
