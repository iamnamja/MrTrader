"""
Tests for app/research/event_options_join.py — the PIT event-time options join.

Pre-event snapshot selection (strictly before announce), staleness guard,
reaction_ratio / iv_runup / post_iv_retention math, and graceful NaN/coverage on
missing data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.event_options_join import (
    MAX_PRE_STALE_DAYS, compute_event_option_features,
)


def _feat_rows(dates, *, atm=0.30, im=0.04, cpiv=0.01, skew=0.03, term=0.02,
               volz=0.0):
    """A symbol's daily feature slice over `dates` (constant unless overridden)."""
    n = len(dates)

    def col(v):
        return v if isinstance(v, (list, tuple)) else [v] * n
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "atm_iv_30d": col(atm),
        "implied_move_front": col(im), "cpiv_matched_delta": col(cpiv),
        "skew_25d_put": col(skew), "term_slope_30_60": col(term),
        "opt_volume_z": col(volz),
    })


def test_pre_event_snapshot_is_last_row_strictly_before_announce():
    dates = pd.bdate_range("2024-03-01", periods=15)   # Fri 03-01 ..
    rows = _feat_rows(dates, cpiv=[0.01 * i for i in range(15)])
    announce = dates[10]                                # an announce date
    out = compute_event_option_features(announce, 0.02, rows)
    assert out["options_coverage_flag"] is True
    # pre = the row at dates[9] (strictly before dates[10]); cpiv there = 0.09.
    assert out["cpiv_pre"] == pytest.approx(0.09)
    # The announce-day row (dates[10]) must NOT be used.
    assert out["cpiv_pre"] != pytest.approx(0.10)


def test_pre_event_gated_on_knowable_date_when_present():
    # Two chains before the Wed announce: date=ann-2 (knowable ann-1) and date=ann-1
    # (knowable ON ann). Gating on knowable_date < ann EXCLUDES the ann-1 chain
    # (its data is only public ON the announce day) -> pre = the ann-2 chain.
    ann = pd.Timestamp("2024-03-06")
    rows = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")],
        "knowable_date": [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-06")],
        "atm_iv_30d": [0.30, 0.30], "implied_move_front": [0.04, 0.04],
        "cpiv_matched_delta": [0.01, 0.99],  # ann-2 chain 0.01; ann-1 chain 0.99
        "skew_25d_put": [0.03, 0.03], "term_slope_30_60": [0.02, 0.02],
        "opt_volume_z": [0.0, 0.0],
    })
    out = compute_event_option_features(ann, 0.02, rows)
    assert out["cpiv_pre"] == pytest.approx(0.01)   # the ann-1 (0.99) chain excluded


def test_reaction_ratio_is_abs_gap_over_implied_move():
    dates = pd.bdate_range("2024-03-01", periods=12)
    rows = _feat_rows(dates, im=0.05)
    out = compute_event_option_features(dates[6], -0.03, rows)  # 3% down gap
    assert out["pre_event_implied_move"] == pytest.approx(0.05)
    assert out["reaction_ratio"] == pytest.approx(0.03 / 0.05)  # |−0.03| / 0.05


def test_iv_runup_uses_ten_rows_prior():
    dates = pd.bdate_range("2024-03-01", periods=15)
    # ATM IV rises from 0.20 to 0.34 over the window (one step per day).
    atm = [0.20 + 0.01 * i for i in range(15)]
    rows = _feat_rows(dates, atm=atm)
    announce = dates[12]                       # pre = dates[11], atm = 0.31
    out = compute_event_option_features(announce, 0.02, rows)
    # iv_runup = atm[pre=11] / atm[pre-10=1] - 1 = 0.31 / 0.21 - 1
    assert out["iv_runup_t10_t1"] == pytest.approx(0.31 / 0.21 - 1.0)


def test_post_iv_retention_is_first_chain_after_announce():
    dates = pd.bdate_range("2024-03-01", periods=12)
    atm = [0.30] * 12
    atm[7] = 0.18                              # IV crush on the first post-announce day
    rows = _feat_rows(dates, atm=atm)
    announce = dates[6]                          # pre = dates[5] (0.30); post = dates[7] (0.18)
    out = compute_event_option_features(announce, 0.02, rows)
    assert out["post_iv_retention_t1"] == pytest.approx(0.18 / 0.30)


def test_no_pre_row_yields_no_coverage():
    dates = pd.bdate_range("2024-03-08", periods=5)
    rows = _feat_rows(dates)
    # announce BEFORE all feature rows -> no pre-event snapshot.
    out = compute_event_option_features(pd.Timestamp("2024-03-01"), 0.02, rows)
    assert out["options_coverage_flag"] is False
    assert np.isnan(out["cpiv_pre"]) and np.isnan(out["reaction_ratio"])


def test_stale_pre_row_is_rejected():
    # The only pre-event row is far (> MAX_PRE_STALE_DAYS) before the announce.
    dates = pd.to_datetime(["2024-03-01"])
    rows = _feat_rows(dates)
    announce = pd.Timestamp("2024-03-01") + pd.Timedelta(days=MAX_PRE_STALE_DAYS + 5)
    out = compute_event_option_features(announce, 0.02, rows)
    assert out["options_coverage_flag"] is False


def test_empty_or_missing_inputs_are_nan_not_fabricated():
    assert compute_event_option_features(pd.Timestamp("2024-03-06"), 0.02,
                                         pd.DataFrame())["options_coverage_flag"] is False
    # implied_move missing -> reaction_ratio NaN, but cpiv_pre still present.
    dates = pd.bdate_range("2024-03-01", periods=12)
    rows = _feat_rows(dates, im=np.nan)
    out = compute_event_option_features(dates[6], -0.03, rows)
    assert out["options_coverage_flag"] is True
    assert out["cpiv_pre"] == pytest.approx(0.01)
    assert np.isnan(out["reaction_ratio"])
    assert np.isnan(out["pre_event_implied_move"])
