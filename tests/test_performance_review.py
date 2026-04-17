"""Tests for performance review analytics and API endpoint."""
from unittest.mock import patch, MagicMock
from datetime import datetime, date


# ── Core analytics ────────────────────────────────────────────────────────────

class TestGetPerformanceReview:

    def _make_trade(self, pnl: float, signal_type: str = "EMA_CROSSOVER"):
        t = MagicMock()
        t.pnl = pnl
        t.signal_type = signal_type
        t.closed_at = datetime(2026, 4, 1)
        return t

    def _mock_db(self, trades):
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = trades
        mock_db.query.return_value = mock_query
        return mock_db

    def test_empty_returns_structure(self):
        from app.analytics.performance_review import get_performance_review
        mock_db = self._mock_db([])
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=0.0):
            result = get_performance_review(days=30)
        assert "total_trades" in result
        assert "drift" in result
        assert "overall_status" in result
        assert result["total_trades"] == 0

    def test_win_rate_calculation(self):
        from app.analytics.performance_review import get_performance_review
        trades = [self._make_trade(50.0), self._make_trade(-20.0), self._make_trade(30.0)]
        mock_db = self._mock_db(trades)
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=0.0):
            result = get_performance_review(days=30)
        assert result["total_trades"] == 3
        assert result["wins"] == 2
        assert abs(result["win_rate_pct"] - 66.7) < 0.1

    def test_avg_pnl_calculation(self):
        from app.analytics.performance_review import get_performance_review
        trades = [self._make_trade(100.0), self._make_trade(20.0)]
        mock_db = self._mock_db(trades)
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=0.0):
            result = get_performance_review(days=30)
        assert result["avg_pnl_per_trade"] == 60.0

    def test_by_signal_breakdown(self):
        from app.analytics.performance_review import get_performance_review
        trades = [
            self._make_trade(50.0, "EMA_CROSSOVER"),
            self._make_trade(-10.0, "EMA_CROSSOVER"),
            self._make_trade(30.0, "RSI_DIP"),
        ]
        mock_db = self._mock_db(trades)
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=0.0):
            result = get_performance_review(days=30)
        assert "EMA_CROSSOVER" in result["by_signal"]
        assert result["by_signal"]["EMA_CROSSOVER"]["trades"] == 2
        assert result["by_signal"]["RSI_DIP"]["trades"] == 1

    def test_drift_ok_when_above_target(self):
        from app.analytics.performance_review import get_performance_review
        # 10 winning trades out of 10 → 100% win rate (above 55% target)
        trades = [self._make_trade(20.0) for _ in range(10)]
        mock_db = self._mock_db(trades)
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=0.0):
            result = get_performance_review(days=30)
        win_drift = next((d for d in result["drift"] if "Win Rate" in d["metric"]), None)
        assert win_drift is not None
        assert win_drift["status"] == "ok"

    def test_drift_alert_when_far_below_target(self):
        from app.analytics.performance_review import get_performance_review
        # 3 wins out of 10 → 30% win rate (far below 55% target)
        trades = [self._make_trade(10.0) for _ in range(3)] + [self._make_trade(-5.0) for _ in range(7)]
        mock_db = self._mock_db(trades)
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=0.0):
            result = get_performance_review(days=30)
        win_drift = next((d for d in result["drift"] if "Win Rate" in d["metric"]), None)
        assert win_drift is not None
        assert win_drift["status"] == "alert"

    def test_overall_status_alert_when_any_alert(self):
        from app.analytics.performance_review import get_performance_review
        trades = [self._make_trade(-5.0) for _ in range(10)]  # 0% win rate → alert
        mock_db = self._mock_db(trades)
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=0.0):
            result = get_performance_review(days=30)
        assert result["overall_status"] == "alert"

    def test_overall_status_ok(self):
        from app.analytics.performance_review import get_performance_review
        # High win rate + good avg P&L
        trades = [self._make_trade(50.0) for _ in range(10)]
        mock_db = self._mock_db(trades)
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=0.0):
            result = get_performance_review(days=30)
        assert result["overall_status"] == "ok"

    def test_spy_return_included(self):
        from app.analytics.performance_review import get_performance_review
        mock_db = self._mock_db([])
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=3.5):
            result = get_performance_review(days=30)
        assert result["spy_return_pct"] == 3.5

    def test_result_keys_present(self):
        from app.analytics.performance_review import get_performance_review
        mock_db = self._mock_db([])
        with patch("app.database.session.get_session", return_value=mock_db), \
             patch("app.analytics.performance_review._spy_return", return_value=0.0):
            result = get_performance_review(days=7)
        for key in ("period_days", "total_trades", "win_rate_pct", "total_pnl",
                    "avg_pnl_per_trade", "sharpe_estimate", "spy_return_pct",
                    "alpha_pct", "by_signal", "backtest_targets", "drift",
                    "alerts", "warnings", "overall_status"):
            assert key in result, f"Missing key: {key}"


# ── API endpoint ──────────────────────────────────────────────────────────────

class TestPerformanceReviewEndpoint:
    def test_endpoint_returns_200(self, test_client):
        r = test_client.get("/api/dashboard/analytics/performance-review")
        assert r.status_code == 200
        body = r.json()
        assert "overall_status" in body

    def test_endpoint_accepts_days_param(self, test_client):
        r = test_client.get("/api/dashboard/analytics/performance-review?days=7")
        assert r.status_code == 200
        assert r.json()["period_days"] == 7
