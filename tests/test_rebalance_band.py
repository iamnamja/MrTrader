"""P4-2c — no-trade-band rebalancing tests (the threshold-rebalance primitive)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.tsmom import _held_with_band, tsmom_backtest, TSMOMConfig


def test_held_with_band_holds_until_drift_exceeds_band():
    idx = pd.bdate_range("2020-01-01", periods=4)
    target = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.05]}, index=idx)   # small drifts, then a jump
    held = _held_with_band(target, band=0.02)
    # 0.01 drift never exceeds the 2% band -> hold 0; the 0.05 jump (>2%) trades
    assert list(held["A"]) == [0.0, 0.0, 0.0, 0.05]


def test_held_with_band_trades_on_first_big_move():
    idx = pd.bdate_range("2020-01-01", periods=3)
    target = pd.DataFrame({"A": [0.10, 0.105, -0.10]}, index=idx)
    held = _held_with_band(target, band=0.02)
    # t0 jump 0->0.10 (>2%) trades; t1 drift 0.005 holds; t2 swing 0.20 (>2%) trades
    assert held["A"].tolist() == [0.10, 0.10, -0.10]


def test_held_with_band_treats_nan_target_as_zero():
    idx = pd.bdate_range("2020-01-01", periods=3)
    target = pd.DataFrame({"A": [0.05, np.nan, 0.05]}, index=idx)
    held = _held_with_band(target, band=0.02)
    # NaN -> 0 target: from 0.05 the move to 0 (0.05>band) trades to 0, then back to 0.05
    assert held["A"].tolist() == [0.05, 0.0, 0.05]


def test_band_is_pit_prefix_invariant():
    # held_t must depend only on target<=t -> recomputing on a prefix gives the same held
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=50)
    target = pd.DataFrame({"A": rng.normal(0, 0.05, 50), "B": rng.normal(0, 0.05, 50)}, index=idx)
    full = _held_with_band(target, band=0.02)
    pref = _held_with_band(target.iloc[:30], band=0.02)
    assert np.allclose(full.iloc[:30].to_numpy(), pref.to_numpy())


def _trend_panel(n=400, k=6, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = {f"M{i}": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)) for i in range(k)}
    return pd.DataFrame(cols, index=idx)


def test_tsmom_default_is_calendar_not_band():
    # rebalance_band=None must use the calendar grid (unchanged engine behavior).
    panel = _trend_panel()
    cfg = TSMOMConfig(universe=list(panel.columns), rebalance_days=5, lookbacks=(21, 63),
                      vol_lookback=30)
    res = tsmom_backtest(panel, cfg)
    assert cfg.rebalance_band is None and len(res.returns) > 0


def test_tsmom_band_runs_and_raises_turnover():
    panel = _trend_panel()
    base = dict(universe=list(panel.columns), lookbacks=(21, 63), vol_lookback=30)
    weekly = tsmom_backtest(panel, TSMOMConfig(rebalance_days=5, **base))
    banded = tsmom_backtest(panel, TSMOMConfig(rebalance_band=0.02, **base))
    assert len(banded.returns) > 0 and np.isfinite(banded.returns).all()
    # daily-recompute band trades more than a weekly calendar -> higher annualized turnover
    assert banded.ann_turnover > weekly.ann_turnover
