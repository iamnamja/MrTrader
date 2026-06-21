"""Unit tests for the read-only data-quality audit helpers (scripts/audit_data_quality.py).

These guard the detector logic itself -- a stale-run counter that miscounts or an
OHLC-magnitude check that flags cosmetic rounding would make the sweep untrustworthy.
"""
from __future__ import annotations

import pandas as pd

from scripts.audit_data_quality import (
    _ohlc_violation_magnitude,
    _stale_max_run,
)


def test_stale_run_counts_longest_identical_block():
    s = pd.Series([1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    assert _stale_max_run(s) == 3          # the run of three 2.0s


def test_stale_run_all_distinct_is_one():
    assert _stale_max_run(pd.Series([1.0, 2.0, 3.0])) == 1


def test_stale_run_empty_and_singleton():
    assert _stale_max_run(pd.Series([], dtype=float)) == 0
    assert _stale_max_run(pd.Series([5.0])) == 1


def test_ohlc_magnitude_zero_for_valid_bars():
    df = pd.DataFrame({
        "open": [10.0, 11.0],
        "high": [12.0, 13.0],
        "low": [9.0, 10.0],
        "close": [11.0, 12.0],
    })
    assert _ohlc_violation_magnitude(df) == 0.0


def test_ohlc_magnitude_flags_close_above_high():
    # close (12.0) sits 10% above high (10.9..) -> a real broken bar
    df = pd.DataFrame({
        "open": [10.0],
        "high": [10.9],
        "low": [9.0],
        "close": [12.0],
    })
    mag = _ohlc_violation_magnitude(df)
    assert mag > 0.05


def test_ohlc_magnitude_subpenny_rounding_is_small():
    # close a hair above high (adj-close vs raw-OHLC rounding) -> tiny magnitude
    df = pd.DataFrame({
        "open": [100.0],
        "high": [105.0],
        "low": [99.0],
        "close": [105.0001],
    })
    mag = _ohlc_violation_magnitude(df)
    assert 0.0 < mag < 0.005
