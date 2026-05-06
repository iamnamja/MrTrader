"""Phase R1 — Unit tests for RegimeFeatureBuilder.

Tests use mocked yfinance and DB calls so they run without network/DB.
The 2025-04-07 scenario is manually verified:
  - VIX was ~23 (moderate stress)
  - SPY was below MA20 (April drawdown)
  - NFP week (first Friday was Apr 4)
"""
from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.ml.regime_features import REGIME_FEATURE_NAMES, RegimeFeatureBuilder


def _make_spy_df(n: int = 220, last_close: float = 520.0, trend: str = "down") -> pd.DataFrame:
    """Synthetic SPY price series for testing."""
    dates = pd.date_range(end="2025-04-07", periods=n, freq="B")
    if trend == "down":
        # Price declining over last 20 days to put SPY below MA20
        closes = np.linspace(last_close + 30, last_close, n)
    else:
        closes = np.linspace(last_close - 10, last_close, n)
    df = pd.DataFrame(
        {"close": closes, "open": closes * 0.99, "high": closes * 1.01, "low": closes * 0.98},
        index=dates,
    )
    df.index = pd.DatetimeIndex(df.index)
    return df


def _make_vix_s(n: int = 260, last_vix: float = 23.0) -> pd.Series:
    dates = pd.date_range(end="2025-04-07", periods=n, freq="B")
    vals = np.linspace(last_vix - 5, last_vix, n)
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


# ── Core feature tests ────────────────────────────────────────────────────────

class TestRegimeFeatureBuilderCore:
    def _builder_with_mocks(self):
        builder = RegimeFeatureBuilder()
        # Patch out DB (NIS)
        builder._add_nis_features = lambda feats, as_of: None  # skip DB
        return builder

    def test_all_feature_names_present(self):
        builder = self._builder_with_mocks()
        spy = _make_spy_df()
        vix = _make_vix_s()
        feats = builder.build(as_of_date=date(2025, 4, 7), _spy_df=spy, _vix_df=vix)
        for name in REGIME_FEATURE_NAMES:
            assert name in feats, f"Missing feature: {name}"

    def test_vix_level_clipped(self):
        builder = self._builder_with_mocks()
        # VIX of 23
        vix = _make_vix_s(last_vix=23.0)
        feats = builder.build(as_of_date=date(2025, 4, 7), _spy_df=_make_spy_df(), _vix_df=vix)
        assert abs(feats["vix_level"] - 23.0) < 0.5

    def test_vix_extreme_clipped_to_80(self):
        builder = self._builder_with_mocks()
        vix = _make_vix_s(last_vix=90.0)
        feats = builder.build(as_of_date=date(2025, 4, 7), _spy_df=_make_spy_df(), _vix_df=vix)
        assert feats["vix_level"] <= 80.0

    def test_spy_below_ma20_gives_negative_dist(self):
        """April 2025 drawdown: SPY declining → last close below MA20."""
        builder = self._builder_with_mocks()
        spy = _make_spy_df(trend="down", last_close=510.0)
        feats = builder.build(as_of_date=date(2025, 4, 7), _spy_df=spy, _vix_df=_make_vix_s())
        assert feats["spy_ma20_dist"] < 0.0, "Expected SPY below MA20 in downtrend scenario"

    def test_spy_above_ma20_gives_positive_dist(self):
        builder = self._builder_with_mocks()
        spy = _make_spy_df(trend="up", last_close=540.0)
        feats = builder.build(as_of_date=date(2025, 4, 7), _spy_df=spy, _vix_df=_make_vix_s())
        assert feats["spy_ma20_dist"] > 0.0

    def test_vix_percentiles_between_0_and_1(self):
        builder = self._builder_with_mocks()
        feats = builder.build(
            as_of_date=date(2025, 4, 7),
            _spy_df=_make_spy_df(),
            _vix_df=_make_vix_s(n=260, last_vix=20.0),
        )
        for key in ("vix_pct_1y", "vix_pct_60d"):
            val = feats[key]
            assert not math.isnan(val), f"{key} should not be NaN"
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_spy_rvol_positive(self):
        builder = self._builder_with_mocks()
        feats = builder.build(
            as_of_date=date(2025, 4, 7), _spy_df=_make_spy_df(), _vix_df=_make_vix_s()
        )
        assert feats["spy_rvol_5d"] > 0.0
        assert feats["spy_rvol_20d"] > 0.0

    def test_no_data_returns_nan_not_exception(self):
        builder = self._builder_with_mocks()
        # Empty DataFrames
        feats = builder.build(
            as_of_date=date(2025, 4, 7),
            _spy_df=pd.DataFrame(),
            _vix_df=pd.Series([], dtype=float),
        )
        assert math.isnan(feats["vix_level"])
        assert math.isnan(feats["spy_ma20_dist"])


# ── Macro calendar tests ───────────────────────────────────────────────────────

class TestRegimeCalendarFeatures:
    def _builder(self):
        b = RegimeFeatureBuilder()
        b._add_nis_features = lambda feats, as_of: None
        return b

    def test_fomc_day_detected(self):
        """2026-05-07 is an FOMC announcement day."""
        builder = self._builder()
        spy = _make_spy_df()
        spy.index = pd.date_range(end="2026-05-07", periods=len(spy), freq="B")
        vix = _make_vix_s()
        vix.index = pd.date_range(end="2026-05-07", periods=len(vix), freq="B")
        feats = builder.build(as_of_date=date(2026, 5, 7), _spy_df=spy, _vix_df=vix)
        assert feats["is_fomc_day"] == 1.0
        assert feats["days_to_fomc"] == 0.0

    def test_days_to_event_capped_at_30(self):
        builder = self._builder()
        # Far from any event
        feats = builder.build(
            as_of_date=date(2023, 7, 15),
            _spy_df=_make_spy_df(),
            _vix_df=_make_vix_s(),
        )
        for key in ("days_to_fomc", "days_to_cpi", "days_to_nfp"):
            assert feats[key] <= 30.0

    def test_non_event_day_is_zero(self):
        builder = self._builder()
        # 2023-07-15 is not an FOMC day
        feats = builder.build(
            as_of_date=date(2023, 7, 15),
            _spy_df=_make_spy_df(),
            _vix_df=_make_vix_s(),
        )
        assert feats["is_fomc_day"] == 0.0


# ── NIS features ──────────────────────────────────────────────────────────────

class TestRegimeNisFeatures:
    def test_nis_risk_maps_correctly(self):
        builder = RegimeFeatureBuilder()
        mock_row = MagicMock()
        mock_row.overall_risk = "HIGH"
        mock_row.global_sizing_factor = 0.7

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_row

        with patch("app.database.session.get_session", return_value=mock_db):
            feats: dict = {k: float("nan") for k in REGIME_FEATURE_NAMES}
            builder._add_nis_features(feats, date(2025, 10, 1))

        assert feats["nis_risk_numeric"] == 1.0
        assert abs(feats["nis_sizing_factor"] - 0.7) < 0.001

    def test_nis_medium_maps_to_05(self):
        builder = RegimeFeatureBuilder()
        mock_row = MagicMock()
        mock_row.overall_risk = "MEDIUM"
        mock_row.global_sizing_factor = 0.85

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_row

        with patch("app.database.session.get_session", return_value=mock_db):
            feats = {k: float("nan") for k in REGIME_FEATURE_NAMES}
            builder._add_nis_features(feats, date(2025, 10, 1))

        assert feats["nis_risk_numeric"] == 0.5

    def test_nis_missing_row_stays_nan(self):
        builder = RegimeFeatureBuilder()
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        with patch("app.database.session.get_session", return_value=mock_db):
            feats = {k: float("nan") for k in REGIME_FEATURE_NAMES}
            builder._add_nis_features(feats, date(2023, 6, 1))

        assert math.isnan(feats["nis_risk_numeric"])
        assert math.isnan(feats["nis_sizing_factor"])
