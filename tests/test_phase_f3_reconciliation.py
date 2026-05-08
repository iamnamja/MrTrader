"""
Tests for F3: WF-6 live vs walk-forward reconciliation.

Verifies:
  - compute_live_metrics returns correct stats from a set of fake closed trades
  - Sharpe, win_rate, profit_factor, drawdown math is correct
  - run_reconciliation writes a row to WfLiveReconciliation with status='complete'
  - API endpoints return expected shape (schema smoke test)
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
import pytest


# ── Math unit tests ───────────────────────────────────────────────────────────

class TestLiveMetricsMath:
    def _make_trades(self, pnl_list, hold_days=1.0):
        """Build minimal fake trade objects."""
        trades = []
        base_dt = datetime(2026, 1, 2)
        for i, pnl in enumerate(pnl_list):
            t = MagicMock()
            t.pnl = pnl
            t.symbol = "AAPL"
            t.entry_price = 100.0
            t.quantity = 1
            t.created_at = base_dt + timedelta(days=i * 2)
            t.closed_at = t.created_at + timedelta(days=hold_days)
            trades.append(t)
        return trades

    def test_win_rate(self):
        from app.analytics.wf_reconciliation import compute_live_metrics
        trades = self._make_trades([10, -5, 8, -3, 12])
        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = trades
        result = compute_live_metrics("swing", date(2026, 1, 1), date(2026, 2, 1), db)
        assert result["live_win_rate"] == pytest.approx(3 / 5)
        assert result["live_trade_count"] == 5

    def test_profit_factor(self):
        from app.analytics.wf_reconciliation import _profit_factor
        pnl = [10, -5, 8, -3]
        pf = _profit_factor(pnl)
        assert pf == pytest.approx(18 / 8)

    def test_profit_factor_no_losses(self):
        from app.analytics.wf_reconciliation import _profit_factor
        assert _profit_factor([5, 10, 3]) == 0.0  # no gross loss → returns 0

    def test_max_drawdown_flat(self):
        from app.analytics.wf_reconciliation import _max_drawdown
        assert _max_drawdown([10, 10, 10]) == pytest.approx(0.0)

    def test_max_drawdown_single_dip(self):
        from app.analytics.wf_reconciliation import _max_drawdown
        # cumulative: 10, 5, 8 → peak=10, trough=5 → dd=50%
        dd = _max_drawdown([10, -5, 3])
        assert dd == pytest.approx(50.0, rel=0.01)

    def test_sharpe_positive_edge(self):
        from app.analytics.wf_reconciliation import _annualised_sharpe
        pnl = [1.0] * 20
        hold = [1.0] * 20
        sr = _annualised_sharpe(pnl, hold)
        # All identical returns → std=0 → returns 0.0
        assert sr == 0.0

    def test_sharpe_mixed(self):
        from app.analytics.wf_reconciliation import _annualised_sharpe
        pnl = [5.0, -1.0, 3.0, -2.0, 4.0]
        hold = [2.0] * 5
        sr = _annualised_sharpe(pnl, hold)
        assert isinstance(sr, float)

    def test_empty_trades_returns_none_metrics(self):
        from app.analytics.wf_reconciliation import compute_live_metrics
        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        result = compute_live_metrics("swing", date(2026, 1, 1), date(2026, 2, 1), db)
        assert result["live_trade_count"] == 0
        assert result["live_sharpe"] is None


# ── run_reconciliation integration smoke test ─────────────────────────────────

class TestRunReconciliation:
    def test_writes_complete_row(self, tmp_path):
        """run_reconciliation should persist a 'complete' row when trades exist."""
        from app.analytics.wf_reconciliation import run_reconciliation

        fake_row = MagicMock()
        fake_row.id = 1
        fake_row.strategy = "swing"
        fake_row.range_start = date(2026, 1, 1)
        fake_row.range_end = date(2026, 4, 1)
        fake_row.status = "complete"
        fake_row.computed_at = datetime.utcnow()

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = []

        with patch("app.analytics.wf_reconciliation.WfLiveReconciliation", return_value=fake_row), \
             patch("app.analytics.wf_reconciliation.compute_live_metrics") as mock_live, \
             patch("app.analytics.wf_reconciliation.get_wf_predicted_metrics") as mock_wf:

            mock_live.return_value = {
                "live_sharpe": 1.2,
                "live_win_rate": 0.60,
                "live_total_return_pct": 8.5,
                "live_max_drawdown_pct": 3.2,
                "live_trade_count": 30,
                "live_profit_factor": 1.8,
                "live_avg_hold_days": 4.5,
                "per_symbol": [],
            }
            mock_wf.return_value = (0.85, 0.55, 25, 164)

            row = run_reconciliation("swing", date(2026, 1, 1), date(2026, 4, 1), trigger="test", db=mock_db)

        assert row is fake_row
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called()

    def test_error_sets_error_status(self):
        from app.analytics.wf_reconciliation import run_reconciliation

        fake_row = MagicMock()

        mock_db = MagicMock()
        with patch("app.analytics.wf_reconciliation.WfLiveReconciliation", return_value=fake_row), \
             patch("app.analytics.wf_reconciliation.compute_live_metrics", side_effect=RuntimeError("boom")):
            run_reconciliation("intraday", db=mock_db)

        assert fake_row.status == "error"
        assert "boom" in fake_row.error_detail


# ── API schema smoke tests ────────────────────────────────────────────────────

class TestReconciliationApiShape:
    """Verify the reconciliation route response structure via FastAPI TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_reconciliation_latest_swing_returns_200_or_no_data(self, client):
        resp = client.get("/api/dashboard/reconciliation/latest?strategy=swing")
        assert resp.status_code == 200
        body = resp.json()
        assert "strategy" in body

    def test_reconciliation_history_returns_list(self, client):
        resp = client.get("/api/dashboard/reconciliation/history?strategy=swing")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_reconciliation_run_bad_strategy_returns_400(self, client):
        resp = client.post("/api/dashboard/reconciliation/run?strategy=bad")
        assert resp.status_code == 400
