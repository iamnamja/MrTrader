"""Unit tests for the short interest / short volume provider (Alpha-v3 A2 data).

Focus: the point-in-time contract. Short interest is settlement-dated but only
knowable ~8 bdays later; the accessors MUST never return a value before its
conservative knowable_date. No network — stores are built in-process from raw
Polygon-shaped rows via the public to_df helpers.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.data import short_interest_provider as sip


# ── publication lag / knowable_date ───────────────────────────────────────────

def test_si_knowable_lag_is_10_bdays_conservative():
    # 2024-05-15 (Wed) + 10 weekday-bdays = 2024-05-29. Real FINRA dissemination
    # (~8 trading days, holiday-aware = 2024-05-28) is EARLIER -> we're conservative.
    assert sip.knowable_date(date(2024, 5, 15), sip.SI_PUBLICATION_LAG_BDAYS) == date(2024, 5, 29)


def test_sv_knowable_lag_is_next_bday():
    assert sip.knowable_date(date(2024, 5, 15), sip.SV_PUBLICATION_LAG_BDAYS) == date(2024, 5, 16)
    # Friday -> Monday
    assert sip.knowable_date(date(2024, 5, 17), sip.SV_PUBLICATION_LAG_BDAYS) == date(2024, 5, 20)


def test_knowable_lag_always_after_settlement():
    for d in ("2020-01-31", "2021-06-15", "2023-12-29", "2024-02-29"):
        sd = date.fromisoformat(d)
        assert sip.knowable_date(sd, sip.SI_PUBLICATION_LAG_BDAYS) > sd


# ── parsing / knowable stamping ───────────────────────────────────────────────

def _si_rows():
    return [
        {"settlement_date": "2024-04-15", "short_interest": 1000, "avg_daily_volume": 500, "days_to_cover": 2.0},
        {"settlement_date": "2024-05-15", "short_interest": 1500, "avg_daily_volume": 500, "days_to_cover": 3.0},
    ]


def test_short_interest_to_df_stamps_knowable():
    df = sip.short_interest_to_df("GME", _si_rows())
    assert list(df.columns) == sip._SI_COLS
    assert len(df) == 2
    row = df[df["settlement_date"] == pd.Timestamp("2024-05-15")].iloc[0]
    assert row["knowable_date"] == pd.Timestamp("2024-05-29")
    assert row["short_interest"] == 1500.0


def test_to_df_handles_missing_and_bad_values():
    rows = [{"settlement_date": "2024-05-15", "short_interest": None, "days_to_cover": "x"},
            {"short_interest": 5}]  # no settlement_date -> dropped
    df = sip.short_interest_to_df("AAA", rows)
    assert len(df) == 1
    assert np.isnan(df.iloc[0]["short_interest"])
    assert np.isnan(df.iloc[0]["days_to_cover"])


# ── PIT accessor: THE no-lookahead guard ──────────────────────────────────────

def test_get_si_at_excludes_undisseminated_settlement():
    """A settlement BEFORE as_of but disseminated AFTER it must be excluded."""
    store = sip.short_interest_to_df("GME", _si_rows())
    # 2024-05-15 settled (< as_of) but knowable 2024-05-29 (> as_of) -> must NOT leak.
    got = sip.get_short_interest_at("GME", date(2024, 5, 20), store=store)
    assert got is not None
    assert got["settlement_date"] == date(2024, 4, 15)   # the older, already-disseminated one
    assert got["short_interest"] == 1000.0


def test_get_si_at_returns_latest_knowable_and_change_pct():
    store = sip.short_interest_to_df("GME", _si_rows())
    got = sip.get_short_interest_at("GME", date(2024, 5, 30), store=store)
    assert got["settlement_date"] == date(2024, 5, 15)
    assert got["short_interest"] == 1500.0
    assert got["si_change_pct"] == pytest.approx((1500 - 1000) / 1000)  # +50% build


def test_get_si_at_none_before_any_dissemination():
    store = sip.short_interest_to_df("GME", _si_rows())
    # before even the first settlement is knowable (2024-04-15 -> ~2024-04-29)
    assert sip.get_short_interest_at("GME", date(2024, 4, 1), store=store) is None


def test_get_si_at_empty_store():
    assert sip.get_short_interest_at("GME", date(2024, 5, 30),
                                     store=pd.DataFrame(columns=sip._SI_COLS)) is None


def test_get_si_at_unknown_symbol():
    store = sip.short_interest_to_df("GME", _si_rows())
    assert sip.get_short_interest_at("ZZZZ", date(2024, 5, 30), store=store) is None


def test_survivorship_delisted_symbol_is_served():
    """The accessor never filters by current membership -> a delisted name's
    history is returned as-of, exactly as a backtest needs."""
    store = sip.short_interest_to_df("DELISTED", _si_rows())
    got = sip.get_short_interest_at("DELISTED", date(2024, 5, 30), store=store)
    assert got is not None and got["short_interest"] == 1500.0


# ── short volume features ─────────────────────────────────────────────────────

def _sv_rows(start="2024-05-01", n=25, ratio0=20.0):
    rows = []
    d = pd.Timestamp(start)
    for i in range(n):
        if d.weekday() < 5:
            rows.append({"date": d.date().isoformat(),
                         "short_volume": 100 + i, "total_volume": 1000,
                         "exempt_volume": 1, "short_volume_ratio": ratio0 + i})
        d += pd.Timedelta(days=1)
    return rows


def test_get_sv_features_pit_and_zscore():
    store = sip.short_volume_to_df("GME", _sv_rows())
    feats = sip.get_short_volume_features_at("GME", date(2024, 6, 1), lookback_days=10, store=store)
    assert feats["n_obs"] > 0
    assert feats["sv_ratio_last"] is not None
    # ratio rises monotonically -> last is the max -> positive z-score
    assert feats["sv_ratio_z"] > 0


def test_get_sv_features_respects_knowable_filter():
    rows = [{"date": "2024-05-31", "short_volume": 9, "total_volume": 10,
             "exempt_volume": 0, "short_volume_ratio": 90.0}]
    store = sip.short_volume_to_df("GME", rows)
    # 2024-05-31 (Fri) knowable 2024-06-03 (Mon). as_of 2024-05-31 -> not yet knowable.
    assert sip.get_short_volume_features_at("GME", date(2024, 5, 31), store=store)["n_obs"] == 0
    assert sip.get_short_volume_features_at("GME", date(2024, 6, 3), store=store)["n_obs"] == 1


def test_get_sv_features_empty_store():
    feats = sip.get_short_volume_features_at("GME", date(2024, 6, 1),
                                             store=pd.DataFrame(columns=sip._SV_COLS))
    assert feats["n_obs"] == 0 and feats["sv_ratio_last"] is None
