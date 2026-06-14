"""Offline tests for the sleeve registry (scripts/walkforward/sleeves.py).

Builders are exercised via the `bars=` injection so no network/data-provider call is made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.walkforward.sleeve_lab import Sleeve, list_sleeves
from scripts.walkforward import sleeves


def _bars(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n)), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + rng.normal(0.0002, 0.003, n))
    return pd.DataFrame({"open": open_, "close": close})


def test_f1_sleeves_registered():
    names = list_sleeves()
    assert "turn_of_month" in names
    assert "overnight" in names


def test_build_turn_of_month_from_bars():
    s = sleeves.build_turn_of_month(bars=_bars())
    assert isinstance(s, Sleeve)
    assert s.component_type == "risk_premium"
    assert s.label == "turn_of_month_SPY"
    assert s.spy_prices is not None          # SPY -> residual-α factor supplied
    assert s.n_trials_registered == 1
    assert len(s.returns) > 100


def test_build_overnight_from_bars():
    s = sleeves.build_overnight(bars=_bars(seed=1))
    assert isinstance(s, Sleeve)
    assert s.component_type == "risk_premium"
    assert s.label == "overnight_SPY"
    assert s.spy_prices is not None
    assert len(s.returns) > 100
