"""Tests for app/strategy/carry.py (Alpha-v7 F3) — rates duration carry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.carry import (
    RatesCarryConfig, CarryResult, term_spread, rates_carry_backtest,
)


def _yields_and_ief(n=600, seed=0, steep=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n)
    y3m = pd.Series(1.0 + rng.normal(0, 0.02, n).cumsum() * 0.0, index=idx).clip(0.1, 5)
    spread = 1.5 if steep else -0.5
    y10 = (y3m + spread).rename("y10")
    # IEF: when curve steep (positive carry), let duration drift up; tie return sign to spread
    drift = 0.0003 if steep else -0.0003
    ief = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(drift, 0.004, n)), index=idx)
    return pd.DataFrame({"IEF": ief, "SPY": ief * 1.1}), y10, y3m


def test_term_spread_aligns_and_subtracts():
    _, y10, y3m = _yields_and_ief()
    s = term_spread(y10, y3m)
    assert np.allclose(s.dropna(), 1.5, atol=1e-9)


def test_carry_long_when_curve_steep():
    prices, y10, y3m = _yields_and_ief(steep=True)
    res = rates_carry_backtest(prices, y10, y3m, RatesCarryConfig(scale_pct=1.5))
    assert isinstance(res, CarryResult)
    assert res.mean_position > 0.5     # steep curve -> long duration
    assert np.isfinite(res.sharpe)


def test_carry_short_when_inverted_long_short():
    prices, y10, y3m = _yields_and_ief(steep=False)
    res = rates_carry_backtest(prices, y10, y3m, RatesCarryConfig(long_short=True))
    assert res.mean_position < 0       # inverted -> short duration


def test_carry_long_flat_never_short():
    prices, y10, y3m = _yields_and_ief(steep=False)
    res = rates_carry_backtest(prices, y10, y3m, RatesCarryConfig(long_short=False))
    assert (res.position >= 0).all()


def test_carry_pit_lag():
    prices, y10, y3m = _yields_and_ief(steep=True)
    res = rates_carry_backtest(prices, y10, y3m, RatesCarryConfig())
    # position is shift(1)-lagged -> the first aligned day has zero position
    assert res.position.iloc[0] == 0.0


def test_carry_cost_reduces_return():
    prices, y10, y3m = _yields_and_ief(seed=3)
    free = rates_carry_backtest(prices, y10, y3m, RatesCarryConfig(cost_bps=0.0))
    costed = rates_carry_backtest(prices, y10, y3m, RatesCarryConfig(cost_bps=10.0))
    assert costed.returns.sum() <= free.returns.sum()


def test_carry_missing_etf_raises():
    _, y10, y3m = _yields_and_ief()
    bad = pd.DataFrame({"SPY": np.linspace(100, 110, 600)},
                       index=pd.bdate_range("2015-01-02", periods=600))
    with pytest.raises(ValueError):
        rates_carry_backtest(bad, y10, y3m, RatesCarryConfig())
