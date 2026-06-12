"""
Tests for app/data/options_quality.py — the R1K options-quality coverage filter.

A well-covered name is admitted; a thin / NaN-atm_iv / non-trading name is
rejected; the filter is PIT (uses the latest feature row knowable <= as_of).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.data.options_quality import (
    MIN_VALID_CONTRACTS, filter_options_universe,
)


def _row(underlying, d, *, n_valid, atm_iv, total_vol=5000.0):
    d = pd.Timestamp(d)
    return {
        "underlying": underlying, "date": d,
        "knowable_date": d + pd.offsets.BDay(1),
        "atm_iv_30d": atm_iv, "implied_move_front": 0.05,
        "cpiv_matched_delta": 0.01, "skew_25d_put": 0.03,
        "term_slope_30_60": 0.02, "iv_rv_20d_ratio": 1.1,
        "opt_share_volume_ratio": 2.0, "put_call_volume_ratio": 1.0,
        "opt_volume_z": 0.0, "total_opt_volume": total_vol,
        "n_valid_contracts": n_valid, "coverage_flags": 0,
    }


def _features(rows):
    return pd.DataFrame(rows)


def test_admits_well_covered_name():
    feats = _features([_row("GOOD", "2024-03-01", n_valid=30, atm_iv=0.3)])
    out = filter_options_universe(date(2024, 3, 5), ["GOOD"], feats)
    assert out == ["GOOD"]


def test_rejects_thin_chain():
    feats = _features([
        _row("THIN", "2024-03-01", n_valid=MIN_VALID_CONTRACTS - 1, atm_iv=0.3),
    ])
    assert filter_options_universe(date(2024, 3, 5), ["THIN"], feats) == []


def test_rejects_nan_atm_iv():
    feats = _features([_row("NOIV", "2024-03-01", n_valid=30, atm_iv=np.nan)])
    assert filter_options_universe(date(2024, 3, 5), ["NOIV"], feats) == []


def test_rejects_non_trading_chain():
    feats = _features([
        _row("DEAD", "2024-03-01", n_valid=30, atm_iv=0.3, total_vol=10.0),
    ])
    assert filter_options_universe(date(2024, 3, 5), ["DEAD"], feats) == []


def test_absent_symbol_is_dropped():
    feats = _features([_row("GOOD", "2024-03-01", n_valid=30, atm_iv=0.3)])
    out = filter_options_universe(date(2024, 3, 5), ["GOOD", "MISSING"], feats)
    assert out == ["GOOD"]


def test_pit_uses_latest_row_knowable_by_as_of():
    # Two rows: an OLD good row and a NEW thin row. As of a date BEFORE the new
    # row is knowable, the name qualifies on the old row; after, it fails.
    feats = _features([
        _row("PIT", "2024-03-01", n_valid=30, atm_iv=0.3),     # knowable 03-04 (Mon)
        _row("PIT", "2024-03-15", n_valid=2, atm_iv=np.nan),   # knowable 03-18 (Mon)
    ])
    # as_of 03-10: only the first row is knowable -> qualifies
    assert filter_options_universe(date(2024, 3, 10), ["PIT"], feats) == ["PIT"]
    # as_of 03-20: latest knowable row is the thin/NaN one -> rejected
    assert filter_options_universe(date(2024, 3, 20), ["PIT"], feats) == []


def test_no_row_knowable_yet_is_dropped():
    feats = _features([_row("FUT", "2024-03-15", n_valid=30, atm_iv=0.3)])
    # as_of before the row's knowable_date -> not visible -> dropped
    assert filter_options_universe(date(2024, 3, 1), ["FUT"], feats) == []


def test_order_preserved():
    feats = _features([
        _row("A", "2024-03-01", n_valid=30, atm_iv=0.3),
        _row("B", "2024-03-01", n_valid=30, atm_iv=0.3),
        _row("C", "2024-03-01", n_valid=30, atm_iv=0.3),
    ])
    out = filter_options_universe(date(2024, 3, 5), ["C", "A", "B"], feats)
    assert out == ["C", "A", "B"]


def test_empty_inputs():
    feats = _features([_row("A", "2024-03-01", n_valid=30, atm_iv=0.3)])
    assert filter_options_universe(date(2024, 3, 5), [], feats) == []
    assert filter_options_universe(date(2024, 3, 5), ["A"], pd.DataFrame()) == []
