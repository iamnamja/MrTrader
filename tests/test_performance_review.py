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
    # Patch _spy_return so these are hermetic — the un-patched route makes a real,
    # date-dependent yfinance call, which is what made this endpoint flake in CI
    # (a NaN off a partial bar -> non-JSON-compliant response -> 500).
    def test_endpoint_returns_200(self, test_client):
        with patch("app.analytics.performance_review._spy_return", return_value=1.5):
            r = test_client.get("/api/dashboard/analytics/performance-review")
        assert r.status_code == 200
        body = r.json()
        assert "overall_status" in body

    def test_endpoint_accepts_days_param(self, test_client):
        with patch("app.analytics.performance_review._spy_return", return_value=1.5):
            r = test_client.get("/api/dashboard/analytics/performance-review?days=7")
        assert r.status_code == 200
        assert r.json()["period_days"] == 7

    def test_endpoint_survives_nan_metric(self, test_client):
        """Regression: a non-finite metric (NaN SPY return off a partial/unpublished
        bar) must NOT 500 the endpoint. FastAPI's strict JSON encoder rejects NaN;
        _json_safe scrubs it to null. This deterministically reproduces the
        date-triggered CI failure that previously only appeared on certain days."""
        with patch("app.analytics.performance_review._spy_return",
                   return_value=float("nan")):
            r = test_client.get("/api/dashboard/analytics/performance-review")
        assert r.status_code == 200
        assert r.json()["spy_return_pct"] is None


class TestSpyReturnRobustness:
    def test_finite_when_latest_bar_nan(self):
        """A NaN latest bar (today's partial session) is dropped, not propagated."""
        import numpy as np
        import pandas as pd
        idx = pd.date_range("2026-05-11", periods=5, freq="B")
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, np.nan]}, index=idx)
        with patch("yfinance.download", return_value=df):
            from app.analytics.performance_review import _spy_return
            v = _spy_return(date(2026, 5, 11), date(2026, 5, 18))
        import math
        assert math.isfinite(v)
        assert v == 3.0  # 103/100*100-100, NaN tail dropped

    def test_zero_when_all_nan(self):
        import numpy as np
        import pandas as pd
        idx = pd.date_range("2026-05-11", periods=3, freq="B")
        df = pd.DataFrame({"Close": [np.nan, np.nan, np.nan]}, index=idx)
        with patch("yfinance.download", return_value=df):
            from app.analytics.performance_review import _spy_return
            assert _spy_return(date(2026, 5, 11), date(2026, 5, 14)) == 0.0

    def test_json_safe_scrubs_non_finite(self):
        from app.analytics.performance_review import _json_safe
        out = _json_safe({"a": float("nan"), "b": float("inf"), "c": 1.5,
                          "d": [float("-inf"), 2]})
        assert out["a"] is None
        assert out["b"] is None
        assert out["c"] == 1.5
        assert out["d"] == [None, 2]
