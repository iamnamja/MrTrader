"""Tests for app/strategy/calendar_premia.py (Alpha-v7 F1) — turn-of-month + overnight."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.calendar_premia import (
    TurnOfMonthConfig, OvernightConfig, IntradayConfig, CalendarPremiaResult,
    turn_of_month_window, turn_of_month_backtest, overnight_premium_backtest,
    intraday_premium_backtest,
)


def _bars(n=600, seed=0, start="2018-01-02", mu=0.0003, sd=0.01, gap_mu=0.0002):
    """Synthetic OHLC: close is a random walk; open carries a small overnight gap from
    the prior close so the overnight premium has a known positive sign."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    rets = rng.normal(mu, sd, size=n)
    close = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    gaps = rng.normal(gap_mu, sd / 3, size=n)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + gaps)
    return pd.DataFrame({"open": open_, "close": close})


# ── turn-of-month window ───────────────────────────────────────────────────────────
def test_tom_window_flags_first_and_last_days():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    win = turn_of_month_window(idx, days_before_end=1, days_after_start=3)
    # January 2020 trading days
    jan = idx[(idx.year == 2020) & (idx.month == 1)]
    first3 = set(jan[:3])
    last1 = set(jan[-1:])
    for d in first3 | last1:
        assert win.loc[d], f"{d.date()} should be in-window"
    # A mid-month day must be OUT.
    mid = jan[len(jan) // 2]
    assert not win.loc[mid]


def test_tom_window_count_per_month():
    idx = pd.bdate_range("2021-01-01", "2021-12-31")
    win = turn_of_month_window(idx, days_before_end=1, days_after_start=3)
    # Each full month contributes <= 4 in-window days (3 first + 1 last).
    per_month = win.groupby([win.index.year, win.index.month]).sum()
    assert per_month.max() <= 4
    assert per_month.min() >= 1


# ── turn-of-month backtest ─────────────────────────────────────────────────────────
def test_tom_backtest_only_holds_in_window():
    bars = _bars()
    res = turn_of_month_backtest(bars, TurnOfMonthConfig(days_before_end=1, days_after_start=3))
    assert isinstance(res, CalendarPremiaResult)
    # Position is non-zero only on in-window days.
    win = turn_of_month_window(bars.index, days_before_end=1, days_after_start=3)
    assert (res.position[~win.reindex(res.position.index).fillna(False)] == 0).all()
    assert 0.0 < res.exposure < 0.5    # turn-of-month is a small slice of the month
    assert np.isfinite(res.sharpe)


def test_tom_cost_reduces_return():
    bars = _bars()
    free = turn_of_month_backtest(bars, TurnOfMonthConfig(cost_bps=0.0))
    costed = turn_of_month_backtest(bars, TurnOfMonthConfig(cost_bps=5.0))
    assert costed.returns.sum() < free.returns.sum()


# ── overnight premium ──────────────────────────────────────────────────────────────
def test_overnight_matches_close_to_open():
    bars = _bars(seed=1)
    res = overnight_premium_backtest(bars, OvernightConfig(cost_bps=0.0))
    expected = (bars["open"] / bars["close"].shift(1) - 1.0).dropna()
    # zero-cost overnight return must equal close[t-1]->open[t] exactly
    pd.testing.assert_series_equal(
        res.returns, expected.rename("overnight_SPY").loc[res.returns.index],
        check_names=False)


def test_overnight_positive_gap_is_profitable_gross():
    bars = _bars(seed=2, gap_mu=0.0005)  # clear positive overnight drift
    res = overnight_premium_backtest(bars, OvernightConfig(cost_bps=0.0))
    assert res.returns.mean() > 0
    assert res.exposure == 1.0           # always long overnight


def test_overnight_cost_is_round_trip():
    bars = _bars(seed=3)
    free = overnight_premium_backtest(bars, OvernightConfig(cost_bps=0.0))
    costed = overnight_premium_backtest(bars, OvernightConfig(cost_bps=2.0))
    # round-trip = 2 * cost_bps -> 4bps/day drag
    drag = (free.returns - costed.returns.reindex(free.returns.index)).dropna()
    assert np.allclose(drag, 2.0 * 2.0 / 1e4)


# ── intraday premium (open -> close) ─────────────────────────────────────────────────
def test_intraday_matches_open_to_close():
    bars = _bars(seed=1)
    res = intraday_premium_backtest(bars, IntradayConfig(cost_bps=0.0))
    expected = (bars["close"] / bars["open"] - 1.0)
    pd.testing.assert_series_equal(
        res.returns, expected.rename("intraday_SPY").loc[res.returns.index],
        check_names=False)


def test_intraday_cost_is_round_trip():
    bars = _bars(seed=3)
    free = intraday_premium_backtest(bars, IntradayConfig(cost_bps=0.0))
    costed = intraday_premium_backtest(bars, IntradayConfig(cost_bps=2.0))
    drag = (free.returns - costed.returns.reindex(free.returns.index)).dropna()
    assert np.allclose(drag, 2.0 * 2.0 / 1e4)   # 2 sides/day


def test_overnight_and_intraday_reconcile_to_close_to_close():
    bars = _bars(seed=5)
    on = overnight_premium_backtest(bars, OvernightConfig(cost_bps=0.0)).returns
    idy = intraday_premium_backtest(bars, IntradayConfig(cost_bps=0.0)).returns
    cc = (bars["close"] / bars["close"].shift(1) - 1.0)
    recon = ((1 + on) * (1 + idy) - 1.0).dropna()
    aligned = cc.reindex(recon.index)
    assert np.allclose(recon.to_numpy(), aligned.to_numpy(), atol=1e-12)


# ── validation ─────────────────────────────────────────────────────────────────────
def test_missing_columns_raise():
    bad = pd.DataFrame({"price": [1, 2, 3]}, index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError):
        overnight_premium_backtest(bad, OvernightConfig())
    with pytest.raises(ValueError):
        turn_of_month_backtest(bad, TurnOfMonthConfig())


def test_result_metrics_finite():
    bars = _bars(seed=4)
    for res in (turn_of_month_backtest(bars, TurnOfMonthConfig()),
                overnight_premium_backtest(bars, OvernightConfig())):
        assert np.isfinite(res.sharpe) and np.isfinite(res.cagr) and np.isfinite(res.ann_vol)
        assert res.n_days > 0
