"""Phase RB.1 smoke tests: _make_regime_gate_fn and regime label in rebalance trades."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward_tier3 import _make_regime_gate_fn


def _make_symbols_data(spy_prices, vix_prices, start="2022-01-03"):
    """Build minimal symbols_data dict with SPY and ^VIX."""
    idx = pd.date_range(start=start, periods=len(spy_prices), freq="B")
    spy_df = pd.DataFrame({"close": spy_prices, "open": spy_prices,
                           "high": spy_prices, "low": spy_prices,
                           "volume": [1_000_000] * len(spy_prices)}, index=idx)
    vix_df = pd.DataFrame({"close": vix_prices, "open": vix_prices,
                           "high": vix_prices, "low": vix_prices,
                           "volume": [0] * len(vix_prices)}, index=idx)
    return {"SPY": spy_df, "^VIX": vix_df}


class TestMakeRegimeGateFn:
    def _fn(self, spy_prices, vix_prices, start="2022-01-03", **kwargs):
        sd = _make_symbols_data(spy_prices, vix_prices, start=start)
        return _make_regime_gate_fn(sd, **kwargs)

    def test_bull_regime(self):
        # SPY trending up well above MA, VIX low
        spy = [100.0 + i * 0.1 for i in range(250)]
        vix = [15.0] * 250
        fn = self._fn(spy, vix)
        # Ask for a date well into the series (enough MA history)
        day = date(2022, 12, 30)
        assert fn(day) == pytest.approx(1.0)

    def test_bear_vix_high(self):
        # VIX spiked above 30 — should be BEAR regardless of SPY
        spy = [100.0 + i * 0.1 for i in range(250)]
        vix = [35.0] * 250
        fn = self._fn(spy, vix)
        day = date(2022, 12, 30)
        assert fn(day) == pytest.approx(0.3)

    def test_bear_spy_below_ma(self):
        # SPY falling, clearly below its 200d MA
        spy = [200.0 - i * 0.5 for i in range(250)]
        vix = [18.0] * 250
        fn = self._fn(spy, vix)
        day = date(2022, 12, 30)
        result = fn(day)
        assert result == pytest.approx(0.3)

    def test_neutral_regime(self):
        # SPY flat (just above MA), VIX between 20-30
        spy = [100.0] * 250
        spy[-1] = 100.5  # just above flat MA
        vix = [25.0] * 250
        fn = self._fn(spy, vix)
        day = date(2022, 12, 30)
        result = fn(day)
        assert result == pytest.approx(0.7)

    def test_pit_safe_excludes_current_day(self):
        # Gate queried on the last bar's date should NOT use that bar (strict < filter)
        # SPY trends up from 100→120 (clearly above short MA), then last bar crashes to 50.
        # When we query on last_date, the crash bar is excluded → should still see BULL.
        spy = [100.0 + i * 0.2 for i in range(200)] + [50.0]  # crash on last bar
        vix = [15.0] * 201
        sd = _make_symbols_data(spy, vix, start="2022-01-03")
        fn = _make_regime_gate_fn(sd, spy_ma_days=50)  # short MA so SPY is clearly above
        last_date = sd["SPY"].index[-1].date()
        # Pre-crash: SPY ≈ 139.8 well above 50d MA ≈ ~130 → BULL
        result = fn(last_date)
        assert result == pytest.approx(1.0)

    def test_no_data_returns_neutral(self):
        # Empty symbols_data → neutral (1.0 — no gate)
        fn = _make_regime_gate_fn({})
        result = fn(date(2023, 6, 1))
        assert result == pytest.approx(0.7)

    def test_custom_thresholds(self):
        # Custom: vix_bull=25, vix_bear=35
        spy = [100.0 + i * 0.1 for i in range(250)]
        vix = [22.0] * 250  # between default bull(20) and neutral — but under custom bull(25)
        fn = self._fn(spy, vix, vix_bull=25.0, vix_bear=35.0)
        day = date(2022, 12, 30)
        assert fn(day) == pytest.approx(1.0)
