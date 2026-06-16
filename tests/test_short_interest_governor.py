"""Tests for app/strategy/short_interest_governor.py (Alpha-v8 G2)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.strategy.short_interest_governor import (
    ShortInterestGovernorConfig, aggregate_short_interest, short_interest_index,
    si_multiplier, live_si_multiplier,
)


def _store(n_settle=60, n_tickers=5, seed=0, trend=0.0):
    """Synthetic per-security SI store mimicking short_interest_provider schema."""
    rng = np.random.default_rng(seed)
    settles = pd.bdate_range("2018-01-15", periods=n_settle, freq="2W-MON")
    rows = []
    for i, sd in enumerate(settles):
        kd = sd + pd.tseries.offsets.BDay(10)            # knowable = settlement + 10 bdays
        for t in range(n_tickers):
            si = 1e6 * (1.0 + trend * i) * (1.0 + rng.normal(0, 0.05))
            rows.append({"ticker": f"T{t}", "settlement_date": sd, "knowable_date": kd,
                         "short_interest": max(1.0, si), "avg_daily_volume": 1e5,
                         "days_to_cover": 10.0})
    return pd.DataFrame(rows)


def test_aggregate_sums_per_settlement_indexed_by_knowable():
    store = _store(n_settle=10, n_tickers=4)
    agg = aggregate_short_interest(store)
    assert len(agg) == 10
    # index is knowable_date (after settlement); strictly increasing
    assert agg.index.is_monotonic_increasing
    # each point = sum across the 4 tickers
    first_settle = store["settlement_date"].min()
    expected = store[store["settlement_date"] == first_settle]["short_interest"].sum()
    assert abs(agg.iloc[0] - expected) < 1e-6


def test_sii_is_trailing_z():
    # a rising aggregate SI should push the trailing-z positive in the later periods
    store = _store(n_settle=60, trend=0.03, seed=1)
    agg = aggregate_short_interest(store)
    sii = short_interest_index(agg, window=24, min_obs=12).dropna()
    assert sii.iloc[-5:].mean() > 0          # crowded vs trailing norm late in a rising series
    # warmup: first (min_obs-1) points are NaN (no look-ahead from a full-sample z)
    raw = short_interest_index(agg, window=24, min_obs=12)
    assert raw.iloc[:11].isna().all()


def test_si_multiplier_derisks_when_crowded_and_is_daily():
    store = _store(n_settle=60, trend=0.05, seed=2)
    cfg = ShortInterestGovernorConfig(window=24, min_obs=12, z_threshold=1.0, derisk_to=0.5)
    mult = si_multiplier(store, cfg)
    assert set(np.unique(mult.round(6))).issubset({0.5, 1.0})
    assert (mult == 0.5).any()               # de-risks once SII gets crowded
    # daily series (ffilled) — far more rows than the ~60 bi-monthly settlements
    assert len(mult) > 200
    assert mult.index.is_monotonic_increasing


def test_si_multiplier_pit_no_future_value():
    """The multiplier on a given day must come from an SII knowable on/before that day."""
    store = _store(n_settle=40, trend=0.05, seed=3)
    cfg = ShortInterestGovernorConfig(window=20, min_obs=10, z_threshold=1.0, derisk_to=0.0)
    mult = si_multiplier(store, cfg)
    from app.strategy.short_interest_governor import _bimonthly_multiplier
    bim = _bimonthly_multiplier(store, cfg)
    # the first daily date cannot precede the first knowable bi-monthly stamp's value
    first_kd = bim.index.min()
    assert (mult.index >= first_kd).all() or mult.loc[mult.index < first_kd].empty


def test_live_si_multiplier_and_thin_guard():
    store = _store(n_settle=40, trend=0.05, seed=4)
    cfg = ShortInterestGovernorConfig(window=20, min_obs=10, z_threshold=1.0, derisk_to=0.5)
    m = live_si_multiplier(store, date(2020, 6, 1), cfg)
    assert m in (0.5, 1.0)
    # before any knowable SII -> None (caller fail-safes to 1.0)
    assert live_si_multiplier(store, date(2017, 1, 1), cfg) is None


def test_derisk_to_validation():
    store = _store(n_settle=30)
    with pytest.raises(ValueError):
        si_multiplier(store, ShortInterestGovernorConfig(derisk_to=1.5))
