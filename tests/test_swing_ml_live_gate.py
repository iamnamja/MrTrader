"""Regression: the dead cross-sectional swing ML ranker must be DORMANT in the live path.

The XS-ML swing ranker was validated null (DECISIONS 2026-06-03) and frozen for retraining
(SWING_ENABLED=False), but its LIVE proposing paths (the 30-min _scan_new_opportunities rescan
and the selector='ml_model' premarket fall-through) were never gated, so it produced ~30/32
recent live trades. This locks the default-OFF behavior behind pm.swing_ml_live_enabled, and
confirms it's fail-closed. PEAD/quality_short/factor_portfolio/trend/intraday are not touched.
"""
from __future__ import annotations

import asyncio
from unittest import mock

from app.agents.portfolio_manager import PortfolioManager


def _pm_stub():
    pm = object.__new__(PortfolioManager)
    pm.logger = mock.MagicMock()
    pm.model = mock.MagicMock()
    pm.model.is_trained = True
    pm.log_decision = mock.AsyncMock()
    return pm  # _alpaca is a property -> patch get_alpaca_client in the tests that need it


def test_swing_ml_live_disabled_by_default():
    # No DB value -> get_agent_config returns None -> default OFF (fail-closed).
    pm = _pm_stub()
    with mock.patch("app.database.agent_config.get_agent_config", return_value=None):
        assert pm._swing_ml_live_enabled() is False
    # Explicit 'true' enables; anything else stays off.
    with mock.patch("app.database.agent_config.get_agent_config", return_value="true"):
        assert pm._swing_ml_live_enabled() is True
    with mock.patch("app.database.agent_config.get_agent_config", return_value="false"):
        assert pm._swing_ml_live_enabled() is False
    # Errors fail closed.
    with mock.patch("app.database.agent_config.get_agent_config", side_effect=RuntimeError):
        assert pm._swing_ml_live_enabled() is False


def test_scan_new_opportunities_short_circuits_when_disabled():
    pm = _pm_stub()
    fake_alpaca = mock.MagicMock()
    with mock.patch.object(pm, "_swing_ml_live_enabled", return_value=False), \
         mock.patch("app.integrations.get_alpaca_client", return_value=fake_alpaca):
        asyncio.run(pm._scan_new_opportunities())
    # Returned BEFORE touching the account / scanning (the dead ranker never runs).
    fake_alpaca.get_account.assert_not_called()


def test_scan_new_opportunities_proceeds_when_enabled():
    # When explicitly enabled, it passes the gate and reaches account budgeting (we stop it
    # there by raising from get_account, proving the gate was passed).
    pm = _pm_stub()
    fake_alpaca = mock.MagicMock()
    fake_alpaca.get_account.side_effect = RuntimeError("stop after gate")
    with mock.patch.object(pm, "_swing_ml_live_enabled", return_value=True), \
         mock.patch("app.integrations.get_alpaca_client", return_value=fake_alpaca):
        asyncio.run(pm._scan_new_opportunities())  # get_account raises -> caught by its try/except
    fake_alpaca.get_account.assert_called_once()


def test_pead_still_routes_when_swing_ml_disabled():
    # COLLATERAL-DAMAGE GUARD: with the dead ranker OFF, pm.swing_selector='pead' must still
    # route to the PEAD analyzer (PEAD is the live book and must be unaffected).
    pm = _pm_stub()
    pm._analyze_swing_pead = mock.AsyncMock()
    pm._analyze_swing_factor_portfolio = mock.AsyncMock()
    with mock.patch.object(pm, "_swing_ml_live_enabled", return_value=False), \
         mock.patch("app.database.agent_config.get_agent_config", return_value="pead"):
        asyncio.run(pm._analyze_swing_premarket())
    pm._analyze_swing_pead.assert_awaited_once()
    pm._analyze_swing_factor_portfolio.assert_not_called()


def test_ml_model_fallthrough_skipped_when_disabled():
    # selector='ml_model' + flag OFF -> the dead-ranker fall-through is skipped (no scoring),
    # and a SELECTION_SKIPPED decision is recorded.
    pm = _pm_stub()
    pm._analyze_swing_pead = mock.AsyncMock()
    pm._analyze_swing_factor_portfolio = mock.AsyncMock()
    pm._swing_proposals = ["stale"]
    with mock.patch.object(pm, "_swing_ml_live_enabled", return_value=False), \
         mock.patch("app.database.agent_config.get_agent_config", return_value="ml_model"):
        asyncio.run(pm._analyze_swing_premarket())
    pm.model.predict.assert_not_called()
    pm._analyze_swing_pead.assert_not_called()
    assert pm._swing_proposals == []
    skipped = [c for c in pm.log_decision.await_args_list
               if c.args and c.args[0] == "SELECTION_SKIPPED"]
    assert skipped
