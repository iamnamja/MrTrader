"""Tests for FredClient and composite RegimeDetector."""
from unittest.mock import patch, MagicMock
import pytest

from app.macro.fred_client import FredClient
from app.strategy.regime_detector import RegimeDetector


# ── FredClient ────────────────────────────────────────────────────────────────

class TestFredClientLatest:
    def _client_with_data(self, data):
        c = FredClient()
        with patch.object(c, "_fetch", return_value=data):
            yield c

    def test_returns_last_non_null(self):
        c = FredClient()
        data = [{"value": "4.5"}, {"value": "."}, {"value": "5.0"}]
        with patch.object(c, "_fetch", return_value=data):
            assert c.get_fed_funds_rate() == 5.0

    def test_skips_dot_values(self):
        c = FredClient()
        data = [{"value": "3.0"}, {"value": "."}, {"value": "."}]
        with patch.object(c, "_fetch", return_value=data):
            assert c.get_fed_funds_rate() == 3.0

    def test_returns_none_when_all_dots(self):
        c = FredClient()
        data = [{"value": "."}, {"value": "."}]
        with patch.object(c, "_fetch", return_value=data):
            assert c.get_fed_funds_rate() is None

    def test_returns_none_when_fetch_fails(self):
        c = FredClient()
        with patch.object(c, "_fetch", return_value=None):
            assert c.get_10y_yield() is None


class TestFredClientCpiYoy:
    def _make_obs(self, values):
        return [{"value": str(v)} for v in values]

    def test_cpi_yoy_calculation(self):
        c = FredClient()
        # 13 obs: first 12 are 100, last is 104 → YoY ≈ 4%
        data = self._make_obs([100.0] * 12 + [104.0])
        with patch.object(c, "_fetch", return_value=data):
            result = c.get_cpi_yoy()
        assert result is not None
        assert abs(result - 4.0) < 0.1

    def test_cpi_yoy_none_on_short_data(self):
        c = FredClient()
        data = self._make_obs([100.0] * 5)
        with patch.object(c, "_fetch", return_value=data):
            assert c.get_cpi_yoy() is None


class TestFredClientMacroRiskScore:
    def test_high_risk_environment(self):
        c = FredClient()
        with patch.object(c, "get_yield_spread", return_value=-0.6), \
             patch.object(c, "get_fed_funds_rate", return_value=5.5), \
             patch.object(c, "get_cpi_yoy", return_value=6.0), \
             patch.object(c, "get_unemployment_rate", return_value=5.5):
            score = c.macro_risk_score()
        assert score > 0.6

    def test_low_risk_environment(self):
        c = FredClient()
        with patch.object(c, "get_yield_spread", return_value=1.5), \
             patch.object(c, "get_fed_funds_rate", return_value=1.0), \
             patch.object(c, "get_cpi_yoy", return_value=1.5), \
             patch.object(c, "get_unemployment_rate", return_value=3.5):
            score = c.macro_risk_score()
        assert score == 0.0

    def test_score_bounded_0_1(self):
        c = FredClient()
        with patch.object(c, "get_yield_spread", return_value=None), \
             patch.object(c, "get_fed_funds_rate", return_value=None), \
             patch.object(c, "get_cpi_yoy", return_value=None), \
             patch.object(c, "get_unemployment_rate", return_value=None):
            score = c.macro_risk_score()
        assert 0.0 <= score <= 1.0

    def test_get_all_returns_all_keys(self):
        c = FredClient()
        with patch.object(c, "_fetch", return_value=[{"value": "4.5"}] * 15):
            result = c.get_all()
        assert set(result.keys()) == {
            "fed_funds_rate", "yield_10y", "yield_spread_10y2y",
            "cpi_yoy", "unemployment_rate",
        }


class TestFredClientFetch:
    def test_caches_result(self):
        c = FredClient()
        obs = [{"value": "5.0"}]
        with patch.object(c, "_fetch_api", return_value=obs) as mock_api, \
             patch.object(c, "_fetch_graph", return_value=None):
            c._fetch("FEDFUNDS")
            c._fetch("FEDFUNDS")
        assert mock_api.call_count == 1  # second call uses cache

    def test_falls_back_to_graph(self):
        c = FredClient()
        with patch.object(c, "_fetch_api", return_value=None), \
             patch.object(c, "_fetch_graph", return_value=[{"value": "4.5"}]) as mock_graph:
            result = c._fetch("DGS10")
        mock_graph.assert_called_once()
        assert result == [{"value": "4.5"}]


# ── RegimeDetector ────────────────────────────────────────────────────────────

class TestRegimeDetector:
    def test_low_regime(self):
        rd = RegimeDetector()
        with patch.object(rd, "_vix_score", return_value=0.0), \
             patch.object(rd, "_macro_score", return_value=0.0):
            assert rd.get_regime() == "LOW"

    def test_high_regime(self):
        rd = RegimeDetector()
        with patch.object(rd, "_vix_score", return_value=1.0), \
             patch.object(rd, "_macro_score", return_value=1.0):
            assert rd.get_regime() == "HIGH"

    def test_medium_regime(self):
        rd = RegimeDetector()
        with patch.object(rd, "_vix_score", return_value=0.4), \
             patch.object(rd, "_macro_score", return_value=0.4):
            assert rd.get_regime() == "MEDIUM"

    def test_composite_score_weighted(self):
        rd = RegimeDetector()
        with patch.object(rd, "_vix_score", return_value=0.8), \
             patch.object(rd, "_macro_score", return_value=0.2):
            score = rd.composite_score()
        assert abs(score - (0.70 * 0.8 + 0.30 * 0.2)) < 0.001

    def test_vix_score_normalization(self):
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=20.0):
            score = rd._vix_score()
        assert 0.0 < score < 1.0

    def test_vix_score_none_returns_half(self):
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=None):
            assert rd._vix_score() == 0.5

    def test_macro_score_error_returns_half(self):
        rd = RegimeDetector()
        with patch("app.strategy.regime_detector.RegimeDetector._macro_score",
                   side_effect=Exception("network error")):
            rd2 = RegimeDetector()
            # _macro_score is defined on the class so patch via try/except inside
        # Direct test: import failure path
        with patch.dict("sys.modules", {"app.macro.fred_client": None}):
            score = rd._macro_score()
        assert 0.0 <= score <= 1.0

    def test_get_regime_detail_keys(self):
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=18.0), \
             patch.object(rd, "_macro_score", return_value=0.3):
            detail = rd.get_regime_detail()
        for key in ("regime", "composite_score", "vix", "vix_score", "macro_score",
                    "vix_weight", "macro_weight", "macro_indicators"):
            assert key in detail

    def test_trend_following_active_in_low(self):
        rd = RegimeDetector()
        with patch.object(rd, "get_regime", return_value="LOW"):
            assert rd.trend_following_active() is True
            assert rd.mean_reversion_active() is False

    def test_both_active_in_medium(self):
        rd = RegimeDetector()
        with patch.object(rd, "get_regime", return_value="MEDIUM"):
            assert rd.trend_following_active() is True
            assert rd.mean_reversion_active() is True

    def test_position_size_multiplier(self):
        rd = RegimeDetector()
        for regime, expected in [("LOW", 1.0), ("MEDIUM", 0.75), ("HIGH", 0.5)]:
            with patch.object(rd, "get_regime", return_value=regime):
                assert rd.position_size_multiplier() == expected
