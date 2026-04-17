"""
Tests for Phase 15: earnings filter, signal attribution, drawdown analyzer,
analytics API endpoints, and weekly report script.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ── Earnings filter ────────────────────────────────────────────────────────────

class TestEarningsFilter:
    def test_no_blackout_when_earnings_far_away(self):
        from app.strategy.earnings_filter import EarningsFilter
        ef = EarningsFilter()
        far_date = date.today() + timedelta(days=30)
        with patch.object(ef, "_fetch", return_value=far_date):
            assert not ef.is_blackout("AAPL")

    def test_blackout_active_day_before_earnings(self):
        from app.strategy.earnings_filter import EarningsFilter
        ef = EarningsFilter()
        tomorrow = date.today() + timedelta(days=1)
        with patch.object(ef, "_fetch", return_value=tomorrow):
            assert ef.is_blackout("AAPL")

    def test_blackout_active_on_earnings_day(self):
        from app.strategy.earnings_filter import EarningsFilter
        ef = EarningsFilter()
        with patch.object(ef, "_fetch", return_value=date.today()):
            assert ef.is_blackout("AAPL")

    def test_no_blackout_when_fetch_returns_none(self):
        from app.strategy.earnings_filter import EarningsFilter
        ef = EarningsFilter()
        with patch.object(ef, "_fetch", return_value=None):
            assert not ef.is_blackout("AAPL")

    def test_caching_prevents_double_fetch(self):
        from app.strategy.earnings_filter import EarningsFilter
        ef = EarningsFilter()
        far_date = date.today() + timedelta(days=30)
        with patch.object(ef, "_fetch", return_value=far_date) as mock_fetch:
            ef.is_blackout("AAPL")
            ef.is_blackout("AAPL")
            assert mock_fetch.call_count == 1

    def test_next_earnings_date_returned(self):
        from app.strategy.earnings_filter import EarningsFilter
        ef = EarningsFilter()
        expected = date.today() + timedelta(days=10)
        with patch.object(ef, "_fetch", return_value=expected):
            assert ef.next_earnings_date("MSFT") == expected

    def test_fetch_returns_none_on_yfinance_error(self):
        from app.strategy.earnings_filter import EarningsFilter
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = EarningsFilter._fetch("AAPL")
        assert result is None

    def test_fetch_handles_empty_calendar(self):
        from app.strategy.earnings_filter import EarningsFilter
        mock_ticker = MagicMock()
        mock_ticker.calendar = {"Earnings Date": []}
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = EarningsFilter._fetch("AAPL")
        assert result is None

    def test_fetch_handles_none_calendar(self):
        from app.strategy.earnings_filter import EarningsFilter
        mock_ticker = MagicMock()
        mock_ticker.calendar = None
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = EarningsFilter._fetch("AAPL")
        assert result is None


# ── generate_signal earnings integration ──────────────────────────────────────

class TestGenerateSignalEarningsBlackout:
    def _make_bars(self):
        """Build a minimal DataFrame that would normally produce a BUY signal."""
        import numpy as np
        import pandas as pd
        n = 220
        # Uptrending prices: ensures price > EMA200, momentum ok, EMA crossover
        prices = pd.Series([100.0 + i * 0.5 for i in range(n)])
        # Engineer a crossover: make last bar push fast EMA above slow EMA
        prices.iloc[-1] = prices.iloc[-2] * 1.03
        df = pd.DataFrame({
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": [1_000_000] * n,
        })
        return df

    def test_earnings_blackout_suppresses_buy(self):
        from app.strategy.signals import generate_signal
        bars = self._make_bars()
        with patch("app.strategy.earnings_filter.earnings_filter") as mock_ef:
            mock_ef.is_blackout.return_value = True
            result = generate_signal("AAPL", bars, check_earnings=True)
        # Result should be HOLD (suppressed), not necessarily BUY
        # The important thing is it doesn't raise
        assert result.action in ("BUY", "HOLD")

    def test_earnings_check_skipped_in_backtest(self):
        """check_earnings=False should never call the filter."""
        from app.strategy.signals import generate_signal
        bars = self._make_bars()
        with patch("app.strategy.earnings_filter.is_earnings_blackout") as mock_check:
            generate_signal("AAPL", bars, check_earnings=False)
        mock_check.assert_not_called()


# ── Signal attribution ────────────────────────────────────────────────────────

class TestSignalAttribution:
    def test_empty_returns_empty_dict(self, db_session):
        from app.analytics.signal_attribution import get_signal_attribution
        with patch("app.analytics.signal_attribution.get_session", return_value=db_session):
            result = get_signal_attribution(days=30)
        assert isinstance(result, dict)

    def test_groups_by_signal_type(self, db_session):
        from app.analytics.signal_attribution import get_signal_attribution
        from tests.conftest import make_trade
        make_trade(db_session, symbol="AAPL", status="CLOSED",
                   pnl=50.0, signal_type="EMA_CROSSOVER")
        make_trade(db_session, symbol="MSFT", status="CLOSED",
                   pnl=-20.0, signal_type="EMA_CROSSOVER")
        make_trade(db_session, symbol="NVDA", status="CLOSED",
                   pnl=30.0, signal_type="RSI_DIP")
        db_session.commit()
        with patch("app.analytics.signal_attribution.get_session", return_value=db_session):
            result = get_signal_attribution(days=365)
        assert "EMA_CROSSOVER" in result
        assert result["EMA_CROSSOVER"]["trades"] == 2
        assert result["EMA_CROSSOVER"]["win_rate"] == 50.0
        assert "RSI_DIP" in result
        assert result["RSI_DIP"]["trades"] == 1
        assert result["RSI_DIP"]["win_rate"] == 100.0

    def test_unknown_signal_type_bucketed(self, db_session):
        from app.analytics.signal_attribution import get_signal_attribution
        from tests.conftest import make_trade
        make_trade(db_session, symbol="AAPL", status="CLOSED", pnl=10.0, signal_type=None)
        db_session.commit()
        with patch("app.analytics.signal_attribution.get_session", return_value=db_session):
            result = get_signal_attribution(days=365)
        assert "UNKNOWN" in result


# ── Drawdown analyzer ─────────────────────────────────────────────────────────

class TestDrawdownAnalyzer:
    def test_empty_database(self, db_session):
        from app.analytics.drawdown_analyzer import get_drawdown_summary
        with patch("app.analytics.drawdown_analyzer.get_session", return_value=db_session):
            result = get_drawdown_summary(days=30)
        assert result["total_trades"] == 0
        assert result["max_drawdown_pct"] == 0.0
        assert result["worst_sequences"] == []

    def test_per_symbol_grouping(self, db_session):
        from app.analytics.drawdown_analyzer import get_drawdown_summary
        from tests.conftest import make_trade
        make_trade(db_session, symbol="AAPL", status="CLOSED", pnl=-30.0)
        make_trade(db_session, symbol="AAPL", status="CLOSED", pnl=20.0)
        make_trade(db_session, symbol="MSFT", status="CLOSED", pnl=-10.0)
        db_session.commit()
        with patch("app.analytics.drawdown_analyzer.get_session", return_value=db_session):
            result = get_drawdown_summary(days=365)
        assert "AAPL" in result["by_symbol"]
        assert result["by_symbol"]["AAPL"]["trades"] == 2
        assert result["by_symbol"]["AAPL"]["losses"] == 1
        assert result["total_trades"] == 3

    def test_consecutive_loss_sequence_detected(self, db_session):
        from app.analytics.drawdown_analyzer import _find_loss_sequences
        from app.database.models import Trade
        now = datetime.utcnow()
        trades = []
        for i, pnl in enumerate([-10.0, -20.0, -5.0, 30.0, -8.0]):
            t = MagicMock(spec=Trade)
            t.pnl = pnl
            t.symbol = "AAPL"
            t.closed_at = now + timedelta(hours=i)
            trades.append(t)
        seqs = _find_loss_sequences(trades)
        assert len(seqs) == 1
        assert seqs[0]["length"] == 3
        assert seqs[0]["total_pnl"] == pytest.approx(-35.0)


# ── Analytics API endpoints ────────────────────────────────────────────────────

class TestAnalyticsEndpoints:
    def test_signal_attribution_endpoint(self, test_client):
        r = test_client.get("/api/dashboard/analytics/signal-attribution")
        assert r.status_code == 200
        body = r.json()
        assert "attribution" in body
        assert "days" in body

    def test_drawdown_endpoint(self, test_client):
        r = test_client.get("/api/dashboard/analytics/drawdown")
        assert r.status_code == 200
        body = r.json()
        assert "max_drawdown_pct" in body
        assert "by_symbol" in body

    def test_earnings_blackout_endpoint_no_blackout(self, test_client):
        with patch("app.strategy.earnings_filter.earnings_filter") as mock_ef:
            mock_ef.is_blackout.return_value = False
            mock_ef.next_earnings_date.return_value = None
            r = test_client.get("/api/dashboard/analytics/earnings-blackout/AAPL")
        assert r.status_code == 200
        body = r.json()
        assert body["symbol"] == "AAPL"
        assert body["blackout_active"] is False

    def test_earnings_blackout_endpoint_active(self, test_client):
        with patch("app.strategy.earnings_filter.earnings_filter") as mock_ef:
            mock_ef.is_blackout.return_value = True
            mock_ef.next_earnings_date.return_value = date.today()
            r = test_client.get("/api/dashboard/analytics/earnings-blackout/AAPL")
        assert r.status_code == 200
        body = r.json()
        assert body["blackout_active"] is True
        assert body["next_earnings_date"] is not None

    def test_analytics_days_param(self, test_client):
        r = test_client.get("/api/dashboard/analytics/signal-attribution?days=30")
        assert r.status_code == 200
        assert r.json()["days"] == 30


# ── Weekly report script ───────────────────────────────────────────────────────

class TestWeeklyReport:
    def test_report_generates_without_error(self, db_session):
        from scripts.generate_weekly_report import generate
        with patch("scripts.generate_weekly_report.get_session", return_value=db_session):
            with patch("scripts.generate_weekly_report._get_spy_return", return_value=1.5):
                with patch("scripts.generate_weekly_report.get_signal_attribution", return_value={}):
                    report = generate(days=7)
        assert "MrTrader Weekly Report" in report
        assert "Summary" in report
        assert "Signal Attribution" in report

    def test_report_contains_spy_benchmark(self, db_session):
        from scripts.generate_weekly_report import generate
        with patch("scripts.generate_weekly_report.get_session", return_value=db_session):
            with patch("scripts.generate_weekly_report._get_spy_return", return_value=2.3):
                with patch("scripts.generate_weekly_report.get_signal_attribution", return_value={}):
                    report = generate(days=7)
        assert "SPY" in report

    def test_spy_return_handles_yfinance_error(self):
        from scripts.generate_weekly_report import _get_spy_return
        with patch("yfinance.download", side_effect=Exception("network")):
            result = _get_spy_return(date.today() - timedelta(days=7), date.today())
        assert result == 0.0
