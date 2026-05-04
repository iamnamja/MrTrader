"""Tests for Phase 64: NIS features in swing model training."""
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_bars(n=100):
    """Minimal OHLCV DataFrame sufficient for engineer_features."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.clip(close, 1, None)
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })
    return df


def _make_nis_row(direction=0.7, materiality=0.6, already_priced_in=0.2,
                  sizing_mult=1.2, downside_risk=0.3):
    row = MagicMock()
    row.direction_score = direction
    row.materiality_score = materiality
    row.already_priced_in_score = already_priced_in
    row.sizing_multiplier = sizing_mult
    row.downside_risk_score = downside_risk
    return row


class TestNisPitLookup:
    """Unit tests for _get_nis_features_pit."""

    def test_returns_defaults_when_no_as_of_date(self):
        from app.ml.features import _get_nis_features_pit
        result = _get_nis_features_pit("AAPL", as_of_date=None)
        assert result["nis_direction_score"] == 0.0
        assert result["nis_materiality_score"] == 0.0
        assert result["nis_sizing_mult"] == 1.0

    def test_returns_defaults_when_no_row_found(self):
        from app.ml.features import _get_nis_features_pit
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        with (
            patch("app.ml.features.get_session", return_value=mock_db, create=True),
            patch("app.database.session.get_session", return_value=mock_db),
        ):
            result = _get_nis_features_pit("AAPL", as_of_date=date(2025, 1, 10))
        assert result["nis_direction_score"] == 0.0
        assert result["nis_sizing_mult"] == 1.0

    def test_returns_row_values_when_found(self):
        from app.ml.features import _get_nis_features_pit
        row = _make_nis_row(direction=0.8, materiality=0.6, sizing_mult=1.3)
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = row
        with patch("app.database.session.get_session", return_value=mock_db):
            result = _get_nis_features_pit("AAPL", as_of_date=date(2025, 1, 10))
        assert abs(result["nis_direction_score"] - 0.8) < 1e-6
        assert abs(result["nis_sizing_mult"] - 1.3) < 1e-6

    def test_returns_defaults_on_db_exception(self):
        from app.ml.features import _get_nis_features_pit
        with patch("app.database.session.get_session", side_effect=Exception("db down")):
            result = _get_nis_features_pit("AAPL", as_of_date=date(2025, 1, 10))
        assert result["nis_direction_score"] == 0.0
        assert result["nis_sizing_mult"] == 1.0


class TestNisFeaturesInEngineer:
    """Integration: engineer_features includes NIS group when DB row present."""

    def test_nis_features_present_in_output(self):
        from app.ml.features import FeatureEngineer
        bars = _make_bars(120)
        row = _make_nis_row(direction=0.75, materiality=0.55)
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = row
        fe = FeatureEngineer()
        with patch("app.database.session.get_session", return_value=mock_db):
            feats = fe.engineer_features(
                "AAPL", bars,
                fetch_fundamentals=False,
                as_of_date=date(2025, 6, 1),
            )
        assert feats is not None
        assert "nis_direction_score" in feats
        assert "nis_materiality_score" in feats
        assert "nis_already_priced_in" in feats
        assert "nis_sizing_mult" in feats
        assert "nis_downside_risk" in feats
        assert abs(feats["nis_direction_score"] - 0.75) < 1e-6

    def test_nis_defaults_when_no_db_row(self):
        from app.ml.features import FeatureEngineer
        bars = _make_bars(120)
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        fe = FeatureEngineer()
        with patch("app.database.session.get_session", return_value=mock_db):
            feats = fe.engineer_features(
                "TSLA", bars,
                fetch_fundamentals=False,
                as_of_date=date(2025, 6, 1),
            )
        assert feats is not None
        assert feats["nis_direction_score"] == 0.0
        assert feats["nis_sizing_mult"] == 1.0

    def test_nis_defaults_when_no_as_of_date(self):
        from app.ml.features import FeatureEngineer
        bars = _make_bars(120)
        fe = FeatureEngineer()
        feats = fe.engineer_features(
            "MSFT", bars,
            fetch_fundamentals=False,
            as_of_date=None,
        )
        assert feats is not None
        assert feats["nis_direction_score"] == 0.0

    def test_nis_point_in_time_filter(self):
        """DB query must filter as_of_date <= window date (no lookahead)."""
        from app.ml.features import _get_nis_features_pit
        from app.database.models import NewsSignalCache

        captured_filters = []

        class CapturingQuery:
            def filter(self, *args):
                captured_filters.extend(args)
                return self
            def order_by(self, *args):
                return self
            def first(self):
                return None

        mock_db = MagicMock()
        mock_db.query.return_value = CapturingQuery()

        with patch("app.database.session.get_session", return_value=mock_db):
            _get_nis_features_pit("AAPL", as_of_date=date(2025, 3, 15))

        # Should have captured two filter conditions: symbol == and as_of_date <=
        assert len(captured_filters) >= 2
