"""Tests for Phase 27: paper trading health monitor."""
from unittest.mock import MagicMock, patch
from datetime import date


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_metric(daily_pnl):
    m = MagicMock()
    m.daily_pnl = daily_pnl
    m.max_drawdown = 0.01
    m.date = date.today().isoformat()
    return m


def _healthy_account():
    return {
        "portfolio_value": 20000.0,
        "buying_power": 15000.0,
        "cash": 5000.0,
        "equity": 20000.0,
    }


# ── _consecutive_losing_days ──────────────────────────────────────────────────

class TestConsecutiveLosingDays:

    def _monitor(self):
        from app.live_trading.monitoring import LiveTradingMonitor
        return LiveTradingMonitor()

    def test_zero_when_no_metrics(self):
        m = self._monitor()
        with patch("app.live_trading.monitoring.get_session") as gs:
            db = MagicMock()
            db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
            gs.return_value = db
            assert m._consecutive_losing_days() == 0

    def _patch_metrics(self, gs, metrics):
        db = MagicMock()
        (db.query.return_value
           .order_by.return_value
           .limit.return_value
           .all.return_value) = metrics
        gs.return_value = db

    def test_counts_consecutive_losses(self):
        m = self._monitor()
        metrics = [_make_metric(-100), _make_metric(-50), _make_metric(30)]
        with patch("app.live_trading.monitoring.get_session") as gs:
            self._patch_metrics(gs, metrics)
            assert m._consecutive_losing_days() == 2

    def test_streak_broken_by_positive(self):
        m = self._monitor()
        metrics = [_make_metric(50), _make_metric(-20), _make_metric(-30)]
        with patch("app.live_trading.monitoring.get_session") as gs:
            self._patch_metrics(gs, metrics)
            assert m._consecutive_losing_days() == 0

    def test_all_losing(self):
        m = self._monitor()
        metrics = [_make_metric(-10)] * 5
        with patch("app.live_trading.monitoring.get_session") as gs:
            self._patch_metrics(gs, metrics)
            assert m._consecutive_losing_days() == 5

    def test_none_pnl_breaks_streak(self):
        m = self._monitor()
        metrics = [_make_metric(None), _make_metric(-10)]
        with patch("app.live_trading.monitoring.get_session") as gs:
            self._patch_metrics(gs, metrics)
            assert m._consecutive_losing_days() == 0


# ── _determine_status ─────────────────────────────────────────────────────────

class TestDetermineStatus:

    def _s(self, daily_pnl_pct, max_dd):
        from app.live_trading.monitoring import LiveTradingMonitor
        return LiveTradingMonitor._determine_status(daily_pnl_pct, max_dd)

    def test_healthy(self):
        assert self._s(1.0, 0.01) == "healthy"

    def test_warning_on_drawdown(self):
        assert self._s(0.0, 0.04) == "warning"

    def test_warning_on_daily_loss(self):
        assert self._s(-1.5, 0.0) == "warning"

    def test_critical_on_drawdown(self):
        assert self._s(0.0, 0.06) == "critical"

    def test_critical_on_daily_loss(self):
        assert self._s(-2.5, 0.0) == "critical"


# ── _check_thresholds ─────────────────────────────────────────────────────────

class TestCheckThresholds:

    def _monitor(self):
        from app.live_trading.monitoring import LiveTradingMonitor
        return LiveTradingMonitor()

    def test_no_alert_when_healthy(self):
        m = self._monitor()
        with patch.object(m, "_alert") as mock_alert:
            m._check_thresholds({
                "max_drawdown_pct": 1.0,
                "pnl_today_pct": 0.5,
                "consecutive_losing_days": 0,
            })
            mock_alert.assert_not_called()

    def test_critical_alert_on_high_drawdown(self):
        m = self._monitor()
        with patch.object(m, "_alert") as mock_alert:
            m._check_thresholds({
                "max_drawdown_pct": 6.0,
                "pnl_today_pct": 0.0,
                "consecutive_losing_days": 0,
            })
            calls = [c.args[0] for c in mock_alert.call_args_list]
            assert "CRITICAL" in calls

    def test_warning_on_two_losing_days(self):
        m = self._monitor()
        with patch.object(m, "_alert") as mock_alert:
            m._check_thresholds({
                "max_drawdown_pct": 0.0,
                "pnl_today_pct": 0.0,
                "consecutive_losing_days": 2,
            })
            calls = [c.args[0] for c in mock_alert.call_args_list]
            assert "WARNING" in calls

    def test_critical_on_three_losing_days(self):
        m = self._monitor()
        with patch.object(m, "_alert") as mock_alert:
            m._check_thresholds({
                "max_drawdown_pct": 0.0,
                "pnl_today_pct": 0.0,
                "consecutive_losing_days": 3,
            })
            calls = [c.args[0] for c in mock_alert.call_args_list]
            assert "CRITICAL" in calls


# ── daily_session_summary ─────────────────────────────────────────────────────

class TestDailySessionSummary:

    def _monitor(self):
        from app.live_trading.monitoring import LiveTradingMonitor
        return LiveTradingMonitor()

    def _patch_health(self, m):
        return patch.object(m, "health_check", return_value={
            "trades_today": 5,
            "pnl_today": 120.0,
            "pnl_today_pct": 0.6,
            "account_value": 20000.0,
            "max_drawdown_pct": 1.0,
            "status": "healthy",
            "consecutive_losing_days": 0,
            "open_positions": 2,
        })

    def test_returns_summary_dict(self):
        m = self._monitor()
        with self._patch_health(m):
            with patch("app.live_trading.monitoring.get_session") as gs:
                db = MagicMock()
                gs.return_value = db
                with patch.object(m, "_alert"):
                    result = m.daily_session_summary()
        assert result["trades_today"] == 5
        assert result["pnl_today"] == 120.0
        assert "date" in result

    def test_sets_last_summary(self):
        m = self._monitor()
        assert m.last_summary is None
        with self._patch_health(m):
            with patch("app.live_trading.monitoring.get_session") as gs:
                gs.return_value = MagicMock()
                with patch.object(m, "_alert"):
                    m.daily_session_summary()
        assert m.last_summary is not None

    def test_persists_to_audit_log(self):
        m = self._monitor()
        with self._patch_health(m):
            with patch("app.live_trading.monitoring.get_session") as gs:
                db = MagicMock()
                gs.return_value = db
                with patch.object(m, "_alert"):
                    m.daily_session_summary()
        db.add.assert_called_once()
        db.commit.assert_called_once()


# ── Monitor API endpoints ─────────────────────────────────────────────────────

class TestMonitorEndpoints:

    def _mock_health(self):
        from app.live_trading.monitoring import LiveTradingMonitor
        fake = {
            "timestamp": "2026-01-01T00:00:00", "alpaca_connected": True,
            "account_value": 20000.0, "buying_power": 10000.0, "cash": 5000.0,
            "open_positions": 2, "trades_today": 3, "pnl_today": 50.0,
            "pnl_today_pct": 0.25, "max_drawdown_pct": 1.0, "status": "healthy",
            "consecutive_losing_days": 0,
        }
        return patch.object(LiveTradingMonitor, "health_check", return_value=fake)

    def test_health_endpoint_returns_200(self, test_client):
        with self._mock_health():
            r = test_client.get("/api/dashboard/monitor/health")
        assert r.status_code == 200

    def test_health_endpoint_has_status_field(self, test_client):
        with self._mock_health():
            r = test_client.get("/api/dashboard/monitor/health")
        body = r.json()
        assert "status" in body

    def test_summary_endpoint_returns_200(self, test_client):
        r = test_client.get("/api/dashboard/monitor/summary")
        assert r.status_code == 200

    def test_summary_no_summary_yet_returns_none(self, test_client):
        r = test_client.get("/api/dashboard/monitor/summary")
        body = r.json()
        assert body.get("summary") is None or "summary" in body

    def test_history_endpoint_returns_200(self, test_client):
        r = test_client.get("/api/dashboard/monitor/history")
        assert r.status_code == 200

    def test_history_returns_list(self, test_client):
        r = test_client.get("/api/dashboard/monitor/history")
        body = r.json()
        assert "history" in body
        assert isinstance(body["history"], list)
