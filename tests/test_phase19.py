"""
Unit tests for Phase 19: Risk Intelligence.

Covers:
- Correlation check (60-day returns, threshold 0.75)
- Beta computation (OLS regression vs SPY)
- Beta-adjusted exposure gate
- Factor/sector concentration gate
- New agent_config keys

All tests are pure-Python — no database, Redis, or Alpaca connections.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import pandas as pd


# ─── AgentConfig: Phase 19 keys ──────────────────────────────────────────────

class TestAgentConfigPhase19Keys:
    def test_keys_present(self):
        from app.database.agent_config import _DEFAULTS
        assert "risk.max_correlation" in _DEFAULTS
        assert "risk.max_portfolio_beta" in _DEFAULTS
        assert "risk.high_beta_threshold" in _DEFAULTS
        assert "risk.max_factor_concentration" in _DEFAULTS

    def test_default_values(self):
        from app.database.agent_config import _DEFAULTS
        assert _DEFAULTS["risk.max_correlation"] == 0.75
        assert _DEFAULTS["risk.max_portfolio_beta"] == 1.30
        assert _DEFAULTS["risk.high_beta_threshold"] == 1.20
        assert _DEFAULTS["risk.max_factor_concentration"] == 0.60


# ─── RiskLimits: new fields ───────────────────────────────────────────────────

class TestRiskLimitsPhase19:
    def test_defaults(self):
        from app.agents.risk_rules import RiskLimits
        limits = RiskLimits()
        assert limits.max_correlation == 0.75
        assert limits.max_portfolio_beta == 1.30
        assert limits.high_beta_threshold == 1.20
        assert limits.max_factor_concentration == 0.60


# ─── Correlation Check ────────────────────────────────────────────────────────

class TestCorrelationCheck:
    def _make_rm(self):
        from app.agents.risk_manager import RiskManager
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            rm = RiskManager.__new__(RiskManager)
            rm.logger = MagicMock()
            rm._sector_map = {}
            from app.agents.risk_rules import RiskLimits
            rm.limits = RiskLimits()
            return rm

    def _make_bars(self, returns):
        """Build a fake daily bars DataFrame from a sequence of returns."""
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        close = pd.Series(prices)
        return pd.DataFrame({"close": close, "volume": 1_000_000})

    def test_no_positions_skips(self):
        rm = self._make_rm()
        result = rm._check_correlation("AAPL", [], max_corr=0.75)
        assert result["ok"]

    def test_low_correlation_passes(self):
        rm = self._make_rm()
        rng = np.random.default_rng(42)
        returns_a = rng.normal(0.001, 0.02, 80)
        returns_b = rng.normal(0.001, 0.02, 80)  # independent

        bars_a = self._make_bars(returns_a)
        bars_b = self._make_bars(returns_b)

        mock_client = MagicMock()
        mock_client.get_bars.side_effect = [bars_a, bars_b]

        positions = [{"symbol": "MSFT"}]
        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            result = rm._check_correlation("AAPL", positions, max_corr=0.75)
        assert result["ok"]

    def test_high_correlation_rejected(self):
        rm = self._make_rm()
        base_returns = np.array([0.01, -0.005, 0.008, -0.003] * 20)
        # Both stocks move almost identically
        bars_a = self._make_bars(base_returns)
        bars_b = self._make_bars(base_returns * 1.001)

        mock_client = MagicMock()
        mock_client.get_bars.side_effect = [bars_a, bars_b]

        positions = [{"symbol": "MSFT"}]
        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            result = rm._check_correlation("AAPL", positions, max_corr=0.75)
        assert not result["ok"]
        assert "MSFT" in result["msg"]

    def test_insufficient_bars_skips(self):
        rm = self._make_rm()
        tiny_bars = pd.DataFrame({"close": [100, 101, 102], "volume": [1000] * 3})
        mock_client = MagicMock()
        mock_client.get_bars.return_value = tiny_bars

        positions = [{"symbol": "MSFT"}]
        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            result = rm._check_correlation("AAPL", positions, max_corr=0.75)
        assert result["ok"]
        assert "skipped" in result["msg"]


# ─── Beta Computation ─────────────────────────────────────────────────────────

class TestComputeBeta:
    def _make_rm(self):
        from app.agents.risk_manager import RiskManager
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            rm = RiskManager.__new__(RiskManager)
            rm.logger = MagicMock()
            rm._sector_map = {}
            return rm

    def _make_bars(self, returns):
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return pd.DataFrame({"close": pd.Series(prices), "volume": 1_000_000})

    def test_beta_of_spy_vs_spy_is_one(self):
        rm = self._make_rm()
        spy_returns = np.array([0.01, -0.005, 0.008, -0.003, 0.004] * 10)
        bars = self._make_bars(spy_returns)

        mock_client = MagicMock()
        mock_client.get_bars.return_value = bars

        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            beta = rm._compute_beta("SPY", lookback=60)
        assert abs(beta - 1.0) < 0.05

    def test_high_beta_stock(self):
        rm = self._make_rm()
        spy_returns = np.array([0.01, -0.005, 0.008, -0.003] * 15)
        # Stock moves 2× SPY
        stock_returns = spy_returns * 2.0

        bars_stock = self._make_bars(stock_returns)
        bars_spy = self._make_bars(spy_returns)

        mock_client = MagicMock()
        mock_client.get_bars.side_effect = [bars_stock, bars_spy]

        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            beta = rm._compute_beta("NVDA", lookback=60)
        assert beta > 1.5  # should be close to 2.0

    def test_returns_one_on_insufficient_data(self):
        rm = self._make_rm()
        tiny = pd.DataFrame({"close": [100, 101], "volume": [1000, 1000]})
        mock_client = MagicMock()
        mock_client.get_bars.return_value = tiny

        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            beta = rm._compute_beta("X", lookback=60)
        assert beta == 1.0

    def test_returns_one_on_exception(self):
        rm = self._make_rm()
        mock_client = MagicMock()
        mock_client.get_bars.side_effect = RuntimeError("API down")

        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            beta = rm._compute_beta("X", lookback=60)
        assert beta == 1.0


# ─── Beta Exposure Check ──────────────────────────────────────────────────────

class TestBetaExposureCheck:
    def _make_rm(self):
        from app.agents.risk_manager import RiskManager
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            rm = RiskManager.__new__(RiskManager)
            rm.logger = MagicMock()
            rm._sector_map = {}
            return rm

    def test_low_portfolio_beta_passes(self):
        rm = self._make_rm()
        from app.agents.risk_rules import RiskLimits
        limits = RiskLimits(max_portfolio_beta=1.30, high_beta_threshold=1.20)
        positions = [{"symbol": "JNJ", "market_value": 10000}]

        # JNJ beta=0.7, new stock beta=0.9 → portfolio well under 1.30
        with patch.object(rm, "_compute_beta", side_effect=[0.7, 0.9]):
            result = rm._check_beta_exposure("AAPL", 5000, positions, 100000, limits)
        assert result["ok"]

    def test_high_portfolio_beta_blocks_high_beta_stock(self):
        rm = self._make_rm()
        from app.agents.risk_rules import RiskLimits
        limits = RiskLimits(max_portfolio_beta=1.30, high_beta_threshold=1.20)
        # 90k deployed in NVDA with beta=1.8 → portfolio_beta = 90k*1.8/100k = 1.62 > 1.30
        positions = [{"symbol": "NVDA", "market_value": 90000}]

        with patch.object(rm, "_compute_beta", side_effect=[1.8, 1.5]):
            result = rm._check_beta_exposure("TSLA", 5000, positions, 100000, limits)
        assert not result["ok"]
        assert "beta" in result["msg"].lower()

    def test_high_portfolio_beta_allows_low_beta_stock(self):
        rm = self._make_rm()
        from app.agents.risk_rules import RiskLimits
        limits = RiskLimits(max_portfolio_beta=1.30, high_beta_threshold=1.20)
        # 90k in NVDA beta=1.8 → portfolio_beta=1.62; new stock beta=0.5 < 1.20 → should pass
        positions = [{"symbol": "NVDA", "market_value": 90000}]

        with patch.object(rm, "_compute_beta", side_effect=[1.8, 0.5]):
            result = rm._check_beta_exposure("JNJ", 5000, positions, 100000, limits)
        assert result["ok"]


# ─── Factor Concentration Check ───────────────────────────────────────────────

class TestFactorConcentrationCheck:
    def _make_rm(self, sector_map=None):
        from app.agents.risk_manager import RiskManager
        with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
            rm = RiskManager.__new__(RiskManager)
            rm.logger = MagicMock()
            rm._sector_map = sector_map or {}
            return rm

    def test_under_limit_passes(self):
        rm = self._make_rm({"AAPL": "Technology", "MSFT": "Technology"})
        positions = [{"symbol": "MSFT", "market_value": 20000}]
        result = rm._check_factor_concentration("AAPL", 10000, positions, 100000, max_factor_conc=0.60)
        assert result["ok"]
        # 30% technology — under 60%
        assert "Technology" in result["msg"]

    def test_over_limit_rejected(self):
        rm = self._make_rm({"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology"})
        positions = [
            {"symbol": "MSFT", "market_value": 40000},
            {"symbol": "NVDA", "market_value": 25000},
        ]
        # adding AAPL $10k → Technology = 75k / 100k = 75% > 60%
        result = rm._check_factor_concentration("AAPL", 10000, positions, 100000, max_factor_conc=0.60)
        assert not result["ok"]
        assert "Technology" in result["msg"]

    def test_unknown_sector_uses_unknown(self):
        rm = self._make_rm({})  # no sector map
        positions = []
        result = rm._check_factor_concentration("XYZ", 5000, [], 100000, max_factor_conc=0.60)
        assert result["ok"]  # 5% in UNKNOWN — well under 60%
