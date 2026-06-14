"""Tests for app/strategy/etf_relative_value.py (Alpha-v7 F2) — dollar-neutral spread MR."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.etf_relative_value import (
    RelativeValueConfig, RelativeValueResult, DEFAULT_PAIRS,
    _band_positions, pair_spread_backtest, relative_value_backtest,
)


def _mean_reverting_pair(n=900, seed=0, kappa=0.05, sigma=0.02):
    """A cointegrated pair: B is a random walk, A = B * exp(stationary OU spread)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    b = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, n)), index=idx)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = (1 - kappa) * s[t - 1] + rng.normal(0, sigma)   # OU spread
    a = b * np.exp(pd.Series(s, index=idx))
    return a.rename("A"), b.rename("B")


# ── band logic ─────────────────────────────────────────────────────────────────────
def test_band_enters_and_exits():
    z = pd.Series([0, 0, 2.0, 1.0, 0.4, 0.0, -2.0, -1.0, -0.3, 0.0],
                  index=pd.bdate_range("2020-01-01", periods=10))
    pos = _band_positions(z, entry=1.5, exit_=0.5)
    assert pos.iloc[2] == -1.0          # z=2.0 > entry -> short spread
    assert pos.iloc[3] == -1.0          # z=1.0 still > exit -> hold
    assert pos.iloc[4] == 0.0           # z=0.4 < exit -> flat
    assert pos.iloc[6] == 1.0           # z=-2.0 < -entry -> long spread
    assert pos.iloc[8] == 0.0           # z=-0.3 inside exit -> flat


def test_band_holds_between_bands():
    # z stays between exit and entry after entering -> position persists
    z = pd.Series([2.0, 1.2, 1.0, 0.8, 0.6], index=pd.bdate_range("2020-01-01", periods=5))
    pos = _band_positions(z, entry=1.5, exit_=0.5)
    assert (pos == -1.0).all()


def test_band_nan_is_flat():
    z = pd.Series([np.nan, np.nan, 2.0], index=pd.bdate_range("2020-01-01", periods=3))
    pos = _band_positions(z, entry=1.5, exit_=0.5)
    assert pos.iloc[0] == 0.0 and pos.iloc[1] == 0.0


# ── pair backtest ──────────────────────────────────────────────────────────────────
def test_pair_backtest_pit_and_profitable_on_mean_reverter():
    a, b = _mean_reverting_pair(seed=1)
    cfg = RelativeValueConfig(lookback=60, cost_bps=0.0)
    ret = pair_spread_backtest(a, b, cfg, label="A_B")
    assert isinstance(ret, pd.Series)
    # a genuinely mean-reverting spread should be profitable gross of cost
    assert ret.sum() > 0
    # PIT: first lookback rows have no position -> zero return there
    assert abs(ret.iloc[:60].sum()) < 1e-9


def test_pair_cost_reduces_return():
    a, b = _mean_reverting_pair(seed=2)
    free = pair_spread_backtest(a, b, RelativeValueConfig(lookback=60, cost_bps=0.0))
    costed = pair_spread_backtest(a, b, RelativeValueConfig(lookback=60, cost_bps=5.0))
    assert costed.sum() < free.sum()


def test_pair_rejects_short_history():
    a, b = _mean_reverting_pair(n=50)
    with pytest.raises(ValueError):
        pair_spread_backtest(a, b, RelativeValueConfig(lookback=120))


# ── combined sleeve ────────────────────────────────────────────────────────────────
def test_relative_value_combines_pairs():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-02", periods=800)
    syms = sorted({s for p in DEFAULT_PAIRS for s in p})
    prices = pd.DataFrame(
        {s: 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, len(idx))) for s in syms},
        index=idx)
    res = relative_value_backtest(prices, RelativeValueConfig(lookback=60))
    assert isinstance(res, RelativeValueResult)
    assert res.n_days > 0
    assert res.per_pair.shape[1] == len(DEFAULT_PAIRS)
    assert np.isfinite(res.sharpe)
    assert 0.0 <= res.avg_gross_exposure <= 1.0


def test_relative_value_missing_symbol_raises():
    idx = pd.bdate_range("2015-01-02", periods=400)
    prices = pd.DataFrame({"QQQ": np.linspace(100, 120, len(idx)),
                           "SPY": np.linspace(100, 110, len(idx))}, index=idx)
    # default pairs reference HYG/IEF/TLT/EEM/EFA which are absent -> error
    with pytest.raises(ValueError):
        relative_value_backtest(prices, RelativeValueConfig(lookback=60))


def test_market_neutral_is_low_beta_to_legs():
    """A dollar-neutral spread return should be ~uncorrelated with the level of the legs."""
    a, b = _mean_reverting_pair(seed=3)
    ret = pair_spread_backtest(a, b, RelativeValueConfig(lookback=60, cost_bps=0.0))
    ra = a.pct_change().reindex(ret.index)
    # corr to a single leg's direction should be far from 1 (it's long A / short B)
    assert abs(float(ret.corr(ra))) < 0.9
