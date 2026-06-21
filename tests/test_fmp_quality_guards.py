"""Data-quality guards on the FMP fundamentals loader (audit 2026-06-21).

Covers the two on-load defects the data-quality sweep found:
  1. NEGATIVE revenue (FMP non-standard line item) -> NaN field + derived ratios.
     Genuine ZERO revenue is left intact.
  2. Multiple period_ends under one filing date -> the PIT pick is made
     deterministic (latest period_end) WITHOUT dropping the bundled quarters
     (they are real history / YoY bases).
"""
from __future__ import annotations

import pandas as pd

from app.data.fmp_fundamentals import (
    _SCHEMA_COLUMNS,
    _apply_quality_guards,
    build_fmp_lookup_index,
    get_fundamentals_as_of,
    lookup_pit_from_index,
)


def _raw_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            # ADI: ONE filing date, TWO period_ends (restated vs original), different
            # revenue -> PIT pick must deterministically resolve to the LATEST
            # period_end, but BOTH rows must be retained (no data loss).
            {"symbol": "ADI", "as_of_date": "2017-11-22", "period_end": "2017-10-28",
             "period": "Q4", "revenue": 1_003_623_000.0, "profit_margin": 0.10},
            {"symbol": "ADI", "as_of_date": "2017-11-22", "period_end": "2017-10-31",
             "period": "Q4", "revenue": 1_541_170_000.0, "profit_margin": 0.20},
            # ADSK: a late filing bundling TWO DISTINCT quarters under one date --
            # both must survive (they are real intermediate history / YoY bases).
            {"symbol": "ADSK", "as_of_date": "2007-06-04", "period_end": "2006-10-31",
             "period": "Q3", "revenue": 4.0e8, "profit_margin": 0.11},
            {"symbol": "ADSK", "as_of_date": "2007-06-04", "period_end": "2007-01-31",
             "period": "Q4", "revenue": 5.0e8, "profit_margin": 0.12},
            # BEP: NEGATIVE revenue (REIT/MLP line-item artifact) -> NaN + margins.
            {"symbol": "BEP", "as_of_date": "2003-12-31", "period_end": "2003-12-31",
             "period": "Q4", "revenue": -86_742_260.0, "profit_margin": -5.0,
             "gross_margin": -3.0, "operating_margin": -4.0, "fcf_margin": -2.0,
             "revenue_growth_yoy": 9.9},
            # BIO: genuine ZERO revenue (pre-revenue biotech) -> MUST be left intact.
            {"symbol": "BIO", "as_of_date": "2016-08-09", "period_end": "2016-06-30",
             "period": "Q2", "revenue": 0.0, "profit_margin": 0.0},
            # AAA: a clean normal row that must survive untouched.
            {"symbol": "AAA", "as_of_date": "2020-05-01", "period_end": "2020-03-31",
             "period": "Q1", "revenue": 500.0, "profit_margin": 0.15},
        ]
    )
    # ensure the full schema is present (build_fmp_lookup_index indexes columns
    # directly) -- fill any unspecified numeric column with 0.0
    for c in _SCHEMA_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
    return df


def test_no_data_loss_bundled_quarters_retained():
    out = _apply_quality_guards(_raw_frame())
    assert len(out) == len(_raw_frame())                  # nothing dropped
    assert len(out[out["symbol"] == "ADSK"]) == 2         # both quarters kept
    assert len(out[out["symbol"] == "ADI"]) == 2          # restatement pair kept


def test_pit_pick_latest_period_end_deterministic():
    guarded = _apply_quality_guards(_raw_frame())
    # DataFrame consumer
    snap = get_fundamentals_as_of("ADI", "2018-01-01", df=guarded)
    assert snap is not None
    assert snap["profit_margin"] == 0.20                  # latest period (10-31), not 0.10
    # index/list consumer must agree
    idx = build_fmp_lookup_index(guarded)
    snap2 = lookup_pit_from_index(idx["ADI"], "2018-01-01")
    assert snap2["profit_margin"] == 0.20


def test_negative_revenue_and_margins_nulled():
    out = _apply_quality_guards(_raw_frame())
    bep = out[out["symbol"] == "BEP"].iloc[0]
    for col in ("revenue", "profit_margin", "gross_margin",
                "operating_margin", "fcf_margin", "revenue_growth_yoy"):
        assert pd.isna(bep[col]), f"{col} should be NaN for negative-revenue row"
    assert int((pd.to_numeric(out["revenue"], errors="coerce") < 0).sum()) == 0


def test_zero_revenue_left_intact():
    out = _apply_quality_guards(_raw_frame())
    bio = out[out["symbol"] == "BIO"].iloc[0]
    assert bio["revenue"] == 0.0                           # NOT NaN'd
    assert bio["profit_margin"] == 0.0


def test_clean_row_untouched():
    out = _apply_quality_guards(_raw_frame())
    aaa = out[out["symbol"] == "AAA"].iloc[0]
    assert aaa["revenue"] == 500.0
    assert aaa["profit_margin"] == 0.15


def test_guard_is_pure_does_not_mutate_input():
    raw = _raw_frame()
    before = raw.copy(deep=True)
    _apply_quality_guards(raw)
    pd.testing.assert_frame_equal(raw, before)            # caller's frame intact


def test_guard_is_idempotent():
    once = _apply_quality_guards(_raw_frame())
    twice = _apply_quality_guards(once)
    pd.testing.assert_frame_equal(
        once.reset_index(drop=True), twice.reset_index(drop=True)
    )


def test_handles_missing_optional_columns():
    df = pd.DataFrame([{"symbol": "X", "as_of_date": "2020-01-01", "revenue": -1.0}])
    out = _apply_quality_guards(df)
    assert pd.isna(out.iloc[0]["revenue"])
