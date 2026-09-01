"""FRED fallback for the macro volatility series (2026-09-01).

yfinance returns a frame for ^VIX3M whose Close is almost entirely NaN (measured 1 non-NaN of 8
rows, against 8/8 for ^VIX). Because `_fetch_closes` legitimately skips NaN values, the ticker
never tripped its "No close data returned" warning — the column just thinned out:

    rows carrying BOTH vix and vix3m:  100% (through Apr) -> 73% (Jun) -> 39% (Jul) -> 5% (Aug)
    last complete row: 2026-08-05

That silently disabled the crash governor, which needs both series on the SAME settled date to
compute the VIX/VIX3M term-structure ratio. Being fail-safe it returned 1.0 (no de-risk), so the
only symptom was one WARNING per weekly rebalance — five weeks of decay went unnoticed while the
trend sleeve lost its ability to cut exposure in backwardation.

The contract pinned here: FRED fills ONLY what yfinance left missing, and can never break the
primary fetch.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data.macro_history import (
    FRED_SERIES,
    _fetch_fred_series,
    _fill_from_fred,
)


def _resp(observations, status=200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = {"observations": observations}
    return m


class TestFetchFredSeries:
    def test_parses_observations(self):
        obs = [{"date": "2026-08-27", "value": "17.56"},
               {"date": "2026-08-28", "value": "17.48"}]
        with patch("app.config.settings") as s, patch("httpx.get", return_value=_resp(obs)):
            s.fred_api_key = "k"
            out = _fetch_fred_series("VXVCLS", "2026-08-01", "2026-08-31")
        assert out == {"2026-08-27": 17.56, "2026-08-28": 17.48}

    def test_skips_fred_missing_marker(self):
        """FRED encodes a missing observation as '.' — it must not become a float."""
        obs = [{"date": "2026-08-27", "value": "."},
               {"date": "2026-08-28", "value": "17.48"}]
        with patch("app.config.settings") as s, patch("httpx.get", return_value=_resp(obs)):
            s.fred_api_key = "k"
            out = _fetch_fred_series("VXVCLS", "2026-08-01", "2026-08-31")
        assert out == {"2026-08-28": 17.48}

    def test_no_api_key_returns_empty(self):
        with patch("app.config.settings") as s:
            s.fred_api_key = None
            assert _fetch_fred_series("VXVCLS", "a", "b") == {}

    def test_http_error_returns_empty(self):
        with patch("app.config.settings") as s, patch("httpx.get", return_value=_resp([], 500)):
            s.fred_api_key = "k"
            assert _fetch_fred_series("VXVCLS", "a", "b") == {}

    def test_exception_never_propagates(self):
        """A fallback that can break the primary fetch is worse than no fallback."""
        with patch("app.config.settings") as s, patch("httpx.get", side_effect=RuntimeError("net")):
            s.fred_api_key = "k"
            assert _fetch_fred_series("VXVCLS", "a", "b") == {}


class TestFillFromFred:
    def test_fills_only_missing_never_overwrites(self):
        """THE core contract: a working yfinance value must survive untouched."""
        df = pd.DataFrame({
            "date": ["2026-08-27", "2026-08-28"],
            "vix": [14.51, 14.43],
            "vix3m": [float("nan"), 99.99],     # second row already has a real value
        })
        with patch("app.data.macro_history._fetch_fred_series",
                   return_value={"2026-08-27": 17.56, "2026-08-28": 17.48}):
            out = _fill_from_fred(df.copy(), "2026-08-01", "2026-08-31")
        assert out.loc[0, "vix3m"] == 17.56      # gap filled
        assert out.loc[1, "vix3m"] == 99.99      # existing value preserved

    def test_restores_a_usable_pair(self):
        """The point of the fix: the governor needs BOTH on the same settled date."""
        df = pd.DataFrame({
            "date": ["2026-08-28"], "vix": [14.43], "vix3m": [float("nan")],
        })
        assert not (df["vix"].notna() & df["vix3m"].notna()).any()
        with patch("app.data.macro_history._fetch_fred_series",
                   return_value={"2026-08-28": 17.48}):
            out = _fill_from_fred(df.copy(), "a", "b")
        assert (out["vix"].notna() & out["vix3m"].notna()).all()

    def test_dates_align_no_shift(self):
        """Fill must key on date, not row order — a shift would corrupt the ratio."""
        df = pd.DataFrame({
            "date": ["2026-08-26", "2026-08-27", "2026-08-28"],
            "vix": [15.21, 14.51, 14.43],
            "vix3m": [float("nan"), float("nan"), float("nan")],
        })
        with patch("app.data.macro_history._fetch_fred_series",
                   return_value={"2026-08-26": 17.99, "2026-08-27": 17.56, "2026-08-28": 17.48}):
            out = _fill_from_fred(df.copy(), "a", "b")
        assert list(out["vix3m"]) == [17.99, 17.56, 17.48]

    def test_date_absent_from_fred_stays_nan(self):
        """FRED lags ~1 business day; today's row simply stays empty rather than being guessed."""
        df = pd.DataFrame({
            "date": ["2026-08-28", "2026-08-31"],
            "vix": [14.43, 15.28], "vix3m": [float("nan"), float("nan")],
        })
        with patch("app.data.macro_history._fetch_fred_series",
                   return_value={"2026-08-28": 17.48}):
            out = _fill_from_fred(df.copy(), "a", "b")
        assert out.loc[0, "vix3m"] == 17.48
        assert pd.isna(out.loc[1, "vix3m"])

    def test_nothing_missing_makes_no_call(self):
        df = pd.DataFrame({"date": ["2026-08-28"], "vix": [14.43], "vix3m": [17.48]})
        with patch("app.data.macro_history._fetch_fred_series") as f:
            _fill_from_fred(df.copy(), "a", "b")
        f.assert_not_called()

    def test_empty_frame_is_safe(self):
        out = _fill_from_fred(pd.DataFrame(), "a", "b")
        assert out.empty


class TestSeriesMapping:
    @pytest.mark.parametrize("col,series", [("vix", "VIXCLS"), ("vix3m", "VXVCLS")])
    def test_series_ids(self, col, series):
        """VXVCLS is the 3-month index (VIX3M / legacy VXV) — the ratio is wrong if this drifts."""
        assert FRED_SERIES[col] == series

    def test_only_volatility_series_are_backstopped(self):
        """HYG/IEF/RSP/SPY come from yfinance fine; don't silently add sources for them."""
        assert set(FRED_SERIES) == {"vix", "vix3m"}
