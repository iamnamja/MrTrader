"""Phase R4 — Regime model parallel running analytics tests."""
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch


# ── Weekly summary logger ─────────────────────────────────────────────────────

class TestRegimeWeeklySummary:
    def _make_pm(self):
        from unittest.mock import AsyncMock
        from app.agents.portfolio_manager import PortfolioManager
        pm = PortfolioManager.__new__(PortfolioManager)
        pm.logger = MagicMock()
        return pm

    def test_logs_label_counts(self):
        pm = self._make_pm()
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.group_by.return_value.all.return_value = [
            ("RISK_OFF", 2), ("NEUTRAL", 3), ("RISK_ON", 5),
        ]

        with patch("app.database.session.get_session", return_value=mock_session):
            pm._log_regime_weekly_summary()

        pm.logger.info.assert_called()
        log_msg = str(pm.logger.info.call_args_list[-1])
        assert "RISK_OFF" in log_msg or "WEEKLY" in log_msg

    def test_handles_empty_data(self):
        pm = self._make_pm()
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.group_by.return_value.all.return_value = []

        with patch("app.database.session.get_session", return_value=mock_session):
            pm._log_regime_weekly_summary()  # should not raise


# ── Divergence tracker ────────────────────────────────────────────────────────

class TestRegimeDivergenceLog:
    def _make_pm(self):
        from app.agents.portfolio_manager import PortfolioManager
        pm = PortfolioManager.__new__(PortfolioManager)
        pm.logger = MagicMock()
        return pm

    def _make_row(self, regime_score, opp_score):
        r = MagicMock()
        r.regime_score_at_scan = regime_score
        r.opportunity_score = opp_score
        return r

    def test_logs_divergence_count(self):
        pm = self._make_pm()
        rows = [
            self._make_row(0.20, 0.60),  # regime BLOCK, opp ALLOW — diverge
            self._make_row(0.70, 0.70),  # both allow — agree
            self._make_row(0.20, 0.40),  # both block — agree
        ]
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = rows

        with patch("app.database.session.get_session", return_value=mock_session):
            pm._log_regime_divergence_today()

        pm.logger.info.assert_called()
        log_msg = str(pm.logger.info.call_args_list[-1])
        assert "diverge" in log_msg.lower() or "DIVERGENCE" in log_msg

    def test_handles_no_proposals(self):
        pm = self._make_pm()
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = []

        with patch("app.database.session.get_session", return_value=mock_session):
            pm._log_regime_divergence_today()  # should not raise


# ── API endpoint structure ────────────────────────────────────────────────────

class TestRegimeAnalyticsEndpoint:
    def test_regime_analytics_route_exists(self):
        from app.api.routes import router
        paths = [r.path for r in router.routes]
        assert any("regime/analytics" in p for p in paths)

    def test_regime_current_route_exists(self):
        from app.api.routes import router
        paths = [r.path for r in router.routes]
        assert any("regime/current" in p for p in paths)

    def test_regime_history_route_exists(self):
        from app.api.routes import router
        paths = [r.path for r in router.routes]
        assert any("regime/history" in p for p in paths)


# ── Agreement rate logic ──────────────────────────────────────────────────────

class TestAgreementRateLogic:
    def test_perfect_agreement(self):
        rows = [
            (0.20, 0.30),  # both block
            (0.70, 0.80),  # both allow
            (0.50, 0.55),  # both neutral/allow
        ]
        n_agree = sum(1 for r, o in rows if (r < 0.35) == (o < 0.5))
        assert n_agree == 3
        assert n_agree / len(rows) == 1.0

    def test_full_divergence(self):
        rows = [
            (0.20, 0.70),  # regime blocks, opp allows
            (0.80, 0.30),  # regime allows, opp blocks
        ]
        n_agree = sum(1 for r, o in rows if (r < 0.35) == (o < 0.5))
        assert n_agree == 0

    def test_partial_agreement(self):
        rows = [
            (0.20, 0.70),  # diverge
            (0.70, 0.70),  # agree
        ]
        n_agree = sum(1 for r, o in rows if (r < 0.35) == (o < 0.5))
        assert n_agree / len(rows) == 0.5
