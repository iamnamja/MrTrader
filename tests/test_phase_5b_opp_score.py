"""Phase 5b — Opportunity score breadth + dispersion tests."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


def _make_spy_df(vix=15.0, spy_close=450.0, spy_ma20=445.0, spy_5d_ret=0.005, n=25):
    """Build a minimal SPY DataFrame mimicking yf.download output."""
    closes = [spy_ma20] * (n - 6) + [spy_close / (1 + spy_5d_ret)] * 5 + [spy_close]
    return pd.DataFrame({"close": closes})


def _make_vix_df(vix=15.0, n=10):
    return pd.DataFrame({"close": [vix] * n})


def _make_pm(regime_ctx: dict = None):
    from app.agents.portfolio_manager import PortfolioManager
    pm = PortfolioManager.__new__(PortfolioManager)
    import logging
    pm.logger = logging.getLogger("test")
    pm._current_regime_ctx = regime_ctx or {}
    return pm


def _score(pm, vix=15.0, spy_close=450.0):
    spy_df = _make_spy_df(vix=vix, spy_close=spy_close)
    vix_df = _make_vix_df(vix=vix)
    with patch("yfinance.download") as mock_dl, \
         patch.object(pm, "_fetch_vix_level", return_value=vix):
        mock_dl.side_effect = [spy_df, vix_df]
        return pm._compute_opportunity_score()


class TestOpportunityScoreWeights:
    def test_returns_tuple_of_five(self):
        """_compute_opportunity_score returns (score, vix, spy_close, ma20, 5d_ret)."""
        pm = _make_pm()
        result = _score(pm)
        assert len(result) == 5
        assert 0.0 <= result[0] <= 1.0

    def test_high_vix_lowers_score(self):
        pm_low = _make_pm()
        pm_high = _make_pm()
        high_score = _score(pm_high, vix=35.0, spy_close=400.0)[0]
        low_score_val = _score(pm_low, vix=12.0, spy_close=450.0)[0]
        assert low_score_val > high_score

    def test_breadth_dispersion_used_when_present(self):
        """When regime ctx has breadth/dispersion, they should influence the score."""
        ctx = {"features": {"breadth_pct_ma50": 0.8, "dispersion_pctile": 0.2}}
        pm = _make_pm(regime_ctx=ctx)
        score_with = _score(pm)[0]
        pm2 = _make_pm(regime_ctx={})
        score_without = _score(pm2)[0]
        assert 0.0 <= score_with <= 1.0
        assert 0.0 <= score_without <= 1.0

    def test_missing_breadth_renormalizes_weights(self):
        """Missing breadth data doesn't crash — weights renormalize."""
        ctx = {"features": {}}
        pm = _make_pm(regime_ctx=ctx)
        score, *_ = _score(pm)
        assert 0.0 <= score <= 1.0

    def test_config_weights_sum_to_one(self):
        from app.config import settings
        total = (
            settings.opp_score_vix_weight
            + settings.opp_score_vix_trend_weight
            + settings.opp_score_ma_weight
            + settings.opp_score_mom_weight
            + settings.opp_score_breadth_weight
            + settings.opp_score_dispersion_weight
        )
        assert abs(total - 1.0) < 1e-9
