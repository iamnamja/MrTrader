"""Tests for Phase 36: FMP point-in-time fundamental features."""
from datetime import date
from unittest.mock import patch

import pytest


def _earnings_response(symbol="AAPL"):
    """Pre-processed records as returned by get_earnings_history_fmp()."""
    return [
        {"date": "2024-01-25", "epsActual": 2.18, "epsEstimated": 2.10,
         "surprise_pct": (2.18 - 2.10) / 2.10},
        {"date": "2023-10-26", "epsActual": 1.46, "epsEstimated": 1.39,
         "surprise_pct": (1.46 - 1.39) / 1.39},
        {"date": "2023-07-28", "epsActual": 1.26, "epsEstimated": 1.19,
         "surprise_pct": (1.26 - 1.19) / 1.19},
    ]


def _grades_response(symbol="AAPL"):
    return [
        {"symbol": symbol, "date": "2024-03-10", "gradingCompany": "Wedbush",
         "previousGrade": "Neutral", "newGrade": "Outperform", "action": "upgrade"},
        {"symbol": symbol, "date": "2024-02-15", "gradingCompany": "Goldman",
         "previousGrade": "Buy", "newGrade": "Buy", "action": "maintain"},
        {"symbol": symbol, "date": "2024-01-05", "gradingCompany": "Morgan Stanley",
         "previousGrade": "Overweight", "newGrade": "Equal-Weight", "action": "downgrade"},
    ]


def _inst_response():
    return [
        {"dateReported": "2024-03-31", "ownershipPercent": 58.2},
        {"dateReported": "2023-12-31", "ownershipPercent": 57.1},
    ]


class TestFMPEarningsHistory:

    def test_returns_list_of_records(self):
        from app.data.fmp_provider import get_earnings_history_fmp
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _earnings_response()
            records = get_earnings_history_fmp("AAPL_TEST1")
        assert isinstance(records, list)
        assert len(records) == 3

    def test_surprise_pct_calculated(self):
        from app.data.fmp_provider import get_earnings_history_fmp
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _earnings_response()
            records = get_earnings_history_fmp("AAPL_TEST2")
        assert records[0]["surprise_pct"] is not None
        # (2.18 - 2.10) / 2.10 ≈ 0.038
        assert abs(records[0]["surprise_pct"] - (2.18 - 2.10) / 2.10) < 0.01

    def test_returns_empty_on_api_error(self):
        from app.data.fmp_provider import get_earnings_history_fmp
        with patch("requests.get", side_effect=Exception("timeout")):
            records = get_earnings_history_fmp("FAIL_SYM")
        assert records == []

    def test_uses_in_process_cache(self):
        from app.data import fmp_provider
        import app.data.fmp_provider as fmp_mod
        fmp_mod._earnings_cache.clear()
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _earnings_response()
            fmp_mod.get_earnings_history_fmp("CACHE_TEST")
            fmp_mod.get_earnings_history_fmp("CACHE_TEST")
        # Second call should hit cache — API called once
        assert mock_get.call_count == 1


class TestFMPEarningsFeaturesAt:

    def test_point_in_time_filters_future_reports(self):
        from app.data.fmp_provider import get_earnings_features_at
        with patch("app.data.fmp_provider.get_earnings_history_fmp",
                   return_value=_earnings_response()):
            # as_of = 2023-11-01 → only 2023-10-26 and 2023-07-28 are known
            result = get_earnings_features_at("AAPL", date(2023, 11, 1))
        assert result["fmp_surprise_1q"] is not None
        assert result["fmp_days_since_earnings"] < 10  # 2023-10-26 is 6 days before

    def test_no_future_data_leakage(self):
        from app.data.fmp_provider import get_earnings_features_at
        with patch("app.data.fmp_provider.get_earnings_history_fmp",
                   return_value=_earnings_response()):
            # as_of = 2023-01-01 → all reports are in the future, nothing known
            result = get_earnings_features_at("AAPL", date(2023, 1, 1))
        assert result["fmp_surprise_1q"] == 0.0
        assert result["fmp_days_since_earnings"] == 90.0

    def test_two_quarter_average(self):
        from app.data.fmp_provider import get_earnings_features_at
        with patch("app.data.fmp_provider.get_earnings_history_fmp",
                   return_value=_earnings_response()):
            result = get_earnings_features_at("AAPL", date(2024, 6, 1))
        # All 3 reports are past, avg of first 2
        s1 = (2.18 - 2.10) / 2.10
        s2 = (1.46 - 1.39) / 1.39
        expected_avg = (s1 + s2) / 2
        assert abs(result["fmp_surprise_2q_avg"] - expected_avg) < 0.01

    def test_safe_defaults_on_exception(self):
        from app.data.fmp_provider import get_earnings_features_at
        with patch("app.data.fmp_provider.get_earnings_history_fmp",
                   side_effect=Exception("fail")):
            result = get_earnings_features_at("AAPL", date(2024, 1, 1))
        assert result["fmp_surprise_1q"] == 0.0


class TestFMPAnalystFeatures:

    def test_counts_upgrades_and_downgrades(self):
        from app.data.fmp_provider import get_analyst_features_at
        with patch("app.data.fmp_provider.get_analyst_grades_fmp",
                   return_value=_grades_response()):
            result = get_analyst_features_at("AAPL", date(2024, 4, 1), lookback_days=90)
        assert result["fmp_analyst_upgrades_30d"] >= 0
        assert result["fmp_analyst_downgrades_30d"] >= 0

    def test_point_in_time_excludes_future_grades(self):
        from app.data.fmp_provider import get_analyst_features_at
        with patch("app.data.fmp_provider.get_analyst_grades_fmp",
                   return_value=_grades_response()):
            # as_of = 2024-01-01 → all grades (Jan 5, Feb 15, Mar 10) are in the future
            result = get_analyst_features_at("AAPL", date(2024, 1, 1), lookback_days=30)
        assert result["fmp_analyst_upgrades_30d"] == 0.0
        assert result["fmp_analyst_downgrades_30d"] == 0.0

    def test_momentum_is_net_upgrades_minus_downgrades(self):
        from app.data.fmp_provider import get_analyst_features_at
        with patch("app.data.fmp_provider.get_analyst_grades_fmp",
                   return_value=_grades_response()):
            result = get_analyst_features_at("AAPL", date(2024, 4, 1), lookback_days=90)
        expected = result["fmp_analyst_upgrades_30d"] - result["fmp_analyst_downgrades_30d"]
        assert result["fmp_analyst_momentum_30d"] == expected


class TestFMPClassifyAction:

    def test_upgrade_from_hint(self):
        from app.data.fmp_provider import _classify_action
        assert _classify_action("Neutral", "Outperform", "upgrade") == "upgrade"

    def test_downgrade_from_hint(self):
        from app.data.fmp_provider import _classify_action
        assert _classify_action("Buy", "Neutral", "downgrade") == "downgrade"

    def test_inferred_upgrade(self):
        from app.data.fmp_provider import _classify_action
        assert _classify_action("Neutral", "Buy", "") == "upgrade"

    def test_inferred_downgrade(self):
        from app.data.fmp_provider import _classify_action
        assert _classify_action("Buy", "Sell", "") == "downgrade"

    def test_init_when_no_previous(self):
        from app.data.fmp_provider import _classify_action
        assert _classify_action("", "Buy", "") == "init"


class TestFMPGetAllFeaturesAt:

    def test_returns_all_8_features(self):
        from app.data.fmp_provider import get_fmp_features_at
        with patch("app.data.fmp_provider.get_earnings_history_fmp", return_value=_earnings_response()), \
             patch("app.data.fmp_provider.get_analyst_grades_fmp", return_value=_grades_response()), \
             patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _inst_response()
            result = get_fmp_features_at("AAPL", date(2024, 6, 1))
        expected_keys = {
            "fmp_surprise_1q", "fmp_surprise_2q_avg", "fmp_days_since_earnings",
            "fmp_analyst_upgrades_30d", "fmp_analyst_downgrades_30d", "fmp_analyst_momentum_30d",
            "fmp_inst_ownership_pct", "fmp_inst_change_pct",
        }
        assert expected_keys.issubset(result.keys())

    def test_all_values_are_floats(self):
        from app.data.fmp_provider import get_fmp_features_at
        with patch("app.data.fmp_provider.get_earnings_history_fmp", return_value=_earnings_response()), \
             patch("app.data.fmp_provider.get_analyst_grades_fmp", return_value=_grades_response()), \
             patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _inst_response()
            result = get_fmp_features_at("AAPL", date(2024, 6, 1))
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {v}"


class TestFeatureEngineerFMPIntegration:

    def test_fmp_features_present_in_engineer_features(self):
        import pandas as pd
        import numpy as np
        from app.ml.features import FeatureEngineer
        from unittest.mock import patch

        fe = FeatureEngineer()
        idx = pd.bdate_range("2023-01-02", periods=60)
        prices = 100 + np.cumsum(np.random.randn(60) * 0.5)
        df = pd.DataFrame({
            "open": prices * 0.999, "high": prices * 1.005,
            "low": prices * 0.995, "close": prices,
            "volume": np.ones(60) * 1_000_000,
        }, index=idx)

        fmp_mock = {
            "fmp_surprise_1q": 0.05, "fmp_surprise_2q_avg": 0.04,
            "fmp_days_since_earnings": 30.0, "fmp_analyst_upgrades_30d": 2.0,
            "fmp_analyst_downgrades_30d": 0.0, "fmp_analyst_momentum_30d": 2.0,
            "fmp_inst_ownership_pct": 65.0, "fmp_inst_change_pct": 1.2,
        }
        with patch("app.data.fmp_provider.get_fmp_features_at", return_value=fmp_mock):
            result = fe.engineer_features(
                "AAPL", df, fetch_fundamentals=True, as_of_date=date(2023, 3, 31)
            )
        assert result is not None
        assert "fmp_surprise_1q" in result
        assert result["fmp_surprise_1q"] == 0.05
        assert result["fmp_analyst_momentum_30d"] == 2.0

    def test_fmp_features_zero_when_fetch_disabled(self):
        import pandas as pd
        import numpy as np
        from app.ml.features import FeatureEngineer

        fe = FeatureEngineer()
        idx = pd.bdate_range("2023-01-02", periods=60)
        prices = 100 + np.cumsum(np.random.randn(60) * 0.5)
        df = pd.DataFrame({
            "open": prices * 0.999, "high": prices * 1.005,
            "low": prices * 0.995, "close": prices,
            "volume": np.ones(60) * 1_000_000,
        }, index=idx)
        result = fe.engineer_features("AAPL", df, fetch_fundamentals=False)
        assert result is not None
        assert result["fmp_surprise_1q"] == 0.0
        assert result["fmp_analyst_momentum_30d"] == 0.0
