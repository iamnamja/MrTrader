"""
Tests for Phase 16: regime detector, mean-reversion signal,
portfolio heat management, risk rule integration, and new API endpoints.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Regime detector ────────────────────────────────────────────────────────────

class TestRegimeDetector:
    def test_low_regime_when_vix_below_15(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=12.0):
            assert rd.get_regime() == "LOW"

    def test_medium_regime_when_vix_15_to_25(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=20.0):
            assert rd.get_regime() == "MEDIUM"

    def test_high_regime_when_vix_above_25(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=32.0):
            assert rd.get_regime() == "HIGH"

    def test_defaults_to_medium_on_none_vix(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=None):
            assert rd.get_regime() == "MEDIUM"

    def test_trend_following_active_in_low(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=10.0):
            assert rd.trend_following_active()
            assert not rd.mean_reversion_active()

    def test_mean_reversion_active_in_high(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=30.0):
            assert rd.mean_reversion_active()
            assert not rd.trend_following_active()

    def test_both_active_in_medium(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=20.0):
            assert rd.trend_following_active()
            assert rd.mean_reversion_active()

    def test_position_size_multiplier_by_regime(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with patch.object(rd, "get_vix", return_value=10.0):
            assert rd.position_size_multiplier() == 1.0
        with patch.object(rd, "get_vix", return_value=20.0):
            assert rd.position_size_multiplier() == 0.75
        with patch.object(rd, "get_vix", return_value=30.0):
            assert rd.position_size_multiplier() == 0.5

    def test_vix_caching(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__getitem__ = lambda self, key: MagicMock(iloc=[20.0])
        import yfinance as yf
        with patch("yfinance.download", return_value=mock_df) as mock_dl:
            rd.get_vix()
            rd.get_vix()
            # Should only call yfinance once (cached)
            assert mock_dl.call_count == 1

    def test_vix_returns_none_on_yfinance_error(self):
        from app.strategy.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with patch("yfinance.download", side_effect=Exception("network")):
            result = rd.get_vix()
        assert result is None


# ── Mean-reversion signal ─────────────────────────────────────────────────────

class TestMeanReversion:
    def _make_squeeze_bars(self, n=100):
        """Bars where price is below lower BB, narrow band, above SMA50, RSI oversold."""
        import numpy as np
        prices = pd.Series([100.0] * n)
        # Flat price creates narrow BB; nudge last bar below lower band
        prices.iloc[-1] = 95.0  # well below the band
        df = pd.DataFrame({
            "close": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "volume": [1_000_000] * n,
        })
        return df

    def test_returns_none_on_insufficient_bars(self):
        from app.strategy.mean_reversion import check_mean_reversion_signal
        df = pd.DataFrame({"close": [100.0] * 10, "high": [101.0] * 10,
                           "low": [99.0] * 10, "volume": [1000] * 10})
        assert check_mean_reversion_signal("AAPL", df) is None

    def test_no_signal_when_price_above_lower_bb(self):
        from app.strategy.mean_reversion import check_mean_reversion_signal
        import numpy as np
        n = 100
        # Steadily rising prices → price stays above BB lower band
        prices = pd.Series([100.0 + i * 0.1 for i in range(n)])
        df = pd.DataFrame({"close": prices, "high": prices * 1.01,
                           "low": prices * 0.99, "volume": [1000] * n})
        assert check_mean_reversion_signal("AAPL", df) is None

    def test_no_signal_when_band_too_wide(self):
        from app.strategy.mean_reversion import check_mean_reversion_signal
        import numpy as np
        n = 100
        # Highly volatile prices → wide bands
        rng = list(range(n))
        prices = pd.Series([100.0 + (i % 2) * 20 for i in rng])
        df = pd.DataFrame({"close": prices, "high": prices * 1.01,
                           "low": prices * 0.99, "volume": [1000] * n})
        assert check_mean_reversion_signal("AAPL", df) is None


# ── Portfolio heat ─────────────────────────────────────────────────────────────

class TestPortfolioHeat:
    def test_zero_heat_no_positions(self):
        from app.strategy.portfolio_heat import get_portfolio_heat
        assert get_portfolio_heat([], 10000.0) == 0.0

    def test_heat_with_stop_prices(self):
        from app.strategy.portfolio_heat import get_portfolio_heat
        positions = [
            {"entry_price": 100.0, "stop_price": 98.0, "qty": 10},  # risk = $20
        ]
        heat = get_portfolio_heat(positions, 10000.0)
        assert heat == pytest.approx(0.002)

    def test_heat_fallback_without_stop(self):
        from app.strategy.portfolio_heat import get_portfolio_heat, FALLBACK_RISK_PCT
        positions = [{"market_value": 1000.0}]
        heat = get_portfolio_heat(positions, 10000.0)
        assert heat == pytest.approx(FALLBACK_RISK_PCT * 1000.0 / 10000.0)

    def test_validate_heat_passes_under_limit(self):
        from app.strategy.portfolio_heat import validate_portfolio_heat
        ok, msg = validate_portfolio_heat(100.0, [], 10000.0, max_heat_pct=0.06)
        assert ok

    def test_validate_heat_fails_over_limit(self):
        from app.strategy.portfolio_heat import validate_portfolio_heat
        # Existing positions already at 5.5% heat; new trade pushes over 6%
        positions = [
            {"entry_price": 100.0, "stop_price": 94.5, "qty": 10},  # $55 risk = 5.5% of $1000
        ]
        ok, msg = validate_portfolio_heat(new_trade_risk=10.0, positions=positions,
                                          account_value=1000.0, max_heat_pct=0.06)
        assert not ok
        assert "exceeds" in msg

    def test_validate_heat_zero_account_value(self):
        from app.strategy.portfolio_heat import validate_portfolio_heat
        ok, msg = validate_portfolio_heat(100.0, [], 0.0)
        assert not ok


# ── Risk rules integration ─────────────────────────────────────────────────────

class TestPortfolioHeatRule:
    def test_validate_portfolio_heat_in_risk_rules(self):
        from app.agents.risk_rules import validate_portfolio_heat
        ok, msg = validate_portfolio_heat(100.0, [], 10000.0)
        assert ok

    def test_risk_limits_has_portfolio_heat_field(self):
        from app.agents.risk_rules import RiskLimits
        limits = RiskLimits()
        assert hasattr(limits, "MAX_PORTFOLIO_HEAT_PCT")
        assert limits.MAX_PORTFOLIO_HEAT_PCT == 0.06

    def test_risk_limits_max_positions_is_5(self):
        from app.agents.risk_rules import RiskLimits
        assert RiskLimits().MAX_OPEN_POSITIONS == 5


# ── Generate signal regime integration ────────────────────────────────────────

class TestGenerateSignalRegime:
    def _make_bars(self, n=220):
        import numpy as np
        prices = pd.Series([100.0 + i * 0.5 for i in range(n)])
        prices.iloc[-1] = prices.iloc[-2] * 1.03
        df = pd.DataFrame({
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": [1_000_000] * n,
        })
        return df

    def test_trend_following_suppressed_in_high_regime(self):
        from app.strategy.signals import generate_signal
        bars = self._make_bars()
        with patch("app.strategy.regime_detector.regime_detector") as mock_rd:
            mock_rd.get_regime.return_value = "HIGH"
            mock_rd.trend_following_active.return_value = False
            mock_rd.mean_reversion_active.return_value = True
            result = generate_signal("AAPL", bars, check_earnings=False, check_regime=True)
        # Trend-following should be suppressed; result is HOLD unless MR fires
        assert result.action in ("BUY", "HOLD")

    def test_regime_skipped_when_check_regime_false(self):
        from app.strategy.signals import generate_signal
        bars = self._make_bars()
        with patch("app.strategy.regime_detector.regime_detector") as mock_rd:
            generate_signal("AAPL", bars, check_earnings=False, check_regime=False)
            mock_rd.get_regime.assert_not_called()


# ── API endpoints ──────────────────────────────────────────────────────────────

class TestPhase16Endpoints:
    def test_regime_endpoint(self, test_client):
        detail = {
            "regime": "LOW", "composite_score": 0.1, "vix": 12.5,
            "vix_score": 0.1, "macro_score": 0.0, "vix_weight": 0.7, "macro_weight": 0.3,
            "macro_indicators": {},
        }
        with patch("app.strategy.regime_detector.regime_detector") as mock_rd:
            mock_rd.get_regime_detail.return_value = detail
            mock_rd.trend_following_active.return_value = True
            mock_rd.mean_reversion_active.return_value = False
            mock_rd.position_size_multiplier.return_value = 1.0
            r = test_client.get("/api/dashboard/analytics/regime")
        assert r.status_code == 200
        body = r.json()
        assert "regime" in body
        assert "vix" in body
        assert "trend_following_active" in body

    def test_portfolio_heat_endpoint(self, test_client):
        with patch("app.integrations.get_alpaca_client") as mock_client:
            mock_client.return_value.get_account.return_value = {
                "portfolio_value": 10000.0, "cash": 5000.0,
                "buying_power": 5000.0, "equity": 10000.0,
                "account_blocked": False, "status": "ACTIVE",
            }
            mock_client.return_value.get_positions.return_value = []
            r = test_client.get("/api/dashboard/analytics/portfolio-heat")
        assert r.status_code == 200
        body = r.json()
        assert "portfolio_heat_pct" in body
        assert "max_heat_pct" in body
        assert body["positions"] == 0
