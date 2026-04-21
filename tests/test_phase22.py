"""
Unit tests for Phase 22: Performance Intelligence.

Covers:
- Signal quality monitoring: rolling window, win-rate tracking, floor flag
- Benchmark comparison: cumulative returns, alpha, underperform flag
- Model health: avg score, drift detection
- Weekly report structure
"""
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch


def _fresh_monitor():
    from app.agents.performance_monitor import PerformanceMonitor
    return PerformanceMonitor()


# ─── Signal Quality ───────────────────────────────────────────────────────────

class TestSignalQuality:
    def test_empty_returns_empty_dict(self):
        m = _fresh_monitor()
        assert m.signal_quality() == {}

    def test_single_trade_recorded(self):
        m = _fresh_monitor()
        m.record_trade_result("EMA_CROSSOVER", 50.0)
        sq = m.signal_quality()
        assert "EMA_CROSSOVER" in sq
        assert sq["EMA_CROSSOVER"]["trades"] == 1
        assert sq["EMA_CROSSOVER"]["win_rate"] == 100.0

    def test_win_rate_calculation(self):
        m = _fresh_monitor()
        m.record_trade_result("RSI_DIP", 10.0)   # win
        m.record_trade_result("RSI_DIP", -5.0)   # loss
        sq = m.signal_quality()
        assert sq["RSI_DIP"]["win_rate"] == 50.0

    def test_floor_flag_when_below_threshold(self):
        m = _fresh_monitor()
        # 2 wins, 8 losses → 20% win rate < 45% floor
        for _ in range(2):
            m.record_trade_result("ML_RANK", 10.0)
        for _ in range(8):
            m.record_trade_result("ML_RANK", -5.0)
        sq = m.signal_quality()
        assert sq["ML_RANK"]["flagged"] is True

    def test_no_flag_when_above_threshold(self):
        m = _fresh_monitor()
        for _ in range(7):
            m.record_trade_result("EMA_CROSSOVER", 10.0)
        for _ in range(3):
            m.record_trade_result("EMA_CROSSOVER", -5.0)
        sq = m.signal_quality()
        assert sq["EMA_CROSSOVER"]["flagged"] is False

    def test_rolling_window_caps_at_30(self):
        from app.agents.performance_monitor import SIGNAL_QUALITY_WINDOW
        m = _fresh_monitor()
        for i in range(40):
            m.record_trade_result("EMA_CROSSOVER", 10.0 if i % 2 == 0 else -5.0)
        sq = m.signal_quality()
        assert sq["EMA_CROSSOVER"]["trades"] == SIGNAL_QUALITY_WINDOW

    def test_multiple_signal_types_tracked_independently(self):
        m = _fresh_monitor()
        m.record_trade_result("EMA_CROSSOVER", 10.0)
        m.record_trade_result("RSI_DIP", -5.0)
        sq = m.signal_quality()
        assert "EMA_CROSSOVER" in sq
        assert "RSI_DIP" in sq
        assert sq["EMA_CROSSOVER"]["win_rate"] == 100.0
        assert sq["RSI_DIP"]["win_rate"] == 0.0


# ─── Benchmark Comparison ─────────────────────────────────────────────────────

class TestBenchmarkComparison:
    def test_empty_returns_zeros(self):
        m = _fresh_monitor()
        bm = m.benchmark_comparison()
        assert bm["strategy_cum_pct"] == 0.0
        assert bm["spy_cum_pct"] == 0.0
        assert bm["underperform_flag"] is False

    def test_positive_alpha(self):
        m = _fresh_monitor()
        m.record_daily_return(0.02, 0.01)  # strategy +2%, spy +1%
        bm = m.benchmark_comparison()
        assert bm["alpha_pct"] > 0
        assert bm["underperform_flag"] is False

    def test_underperform_flag_triggers(self):
        m = _fresh_monitor()
        # Strategy flat, SPY up 20% → alpha = -20% < -15% threshold
        m.record_daily_return(0.00, 0.20)
        bm = m.benchmark_comparison()
        assert bm["underperform_flag"] is True

    def test_underperform_flag_not_triggered_on_small_gap(self):
        m = _fresh_monitor()
        m.record_daily_return(0.05, 0.10)  # alpha = -5% — above -15% threshold
        bm = m.benchmark_comparison()
        assert bm["underperform_flag"] is False

    def test_days_count(self):
        m = _fresh_monitor()
        m.record_daily_return(0.01, 0.01, for_date=date.today())
        m.record_daily_return(0.02, 0.01, for_date=date.today() - timedelta(days=1))
        bm = m.benchmark_comparison()
        assert bm["days"] == 2

    def test_old_entries_pruned_outside_window(self):
        from app.agents.performance_monitor import BENCHMARK_WINDOW_DAYS
        m = _fresh_monitor()
        old_date = date.today() - timedelta(days=BENCHMARK_WINDOW_DAYS + 5)
        m.record_daily_return(0.01, 0.01, for_date=old_date)
        m.record_daily_return(0.01, 0.01, for_date=date.today())
        bm = m.benchmark_comparison()
        assert bm["days"] == 1  # old entry pruned when new one added


# ─── Model Health ─────────────────────────────────────────────────────────────

class TestModelHealth:
    def test_empty_returns_none_score(self):
        m = _fresh_monitor()
        mh = m.model_health()
        assert mh["avg_score"] is None
        assert mh["drift_flag"] is False

    def test_healthy_scores_no_flag(self):
        m = _fresh_monitor()
        m.record_pm_cycle([0.65, 0.70, 0.60], {"EXIT": 0, "HOLD": 3, "EXTEND_TARGET": 0})
        mh = m.model_health()
        assert mh["avg_score"] > 0.45
        assert mh["drift_flag"] is False

    def test_low_scores_trigger_drift_flag(self):
        m = _fresh_monitor()
        m.record_pm_cycle([0.30, 0.35, 0.40], {"EXIT": 3, "HOLD": 0, "EXTEND_TARGET": 0})
        mh = m.model_health()
        assert mh["drift_flag"] is True

    def test_decision_counts_accumulated(self):
        m = _fresh_monitor()
        m.record_pm_cycle([0.6], {"EXIT": 2, "HOLD": 1, "EXTEND_TARGET": 0})
        m.record_pm_cycle([0.6], {"EXIT": 0, "HOLD": 3, "EXTEND_TARGET": 1})
        mh = m.model_health()
        assert mh["decision_counts"]["EXIT"] == 2
        assert mh["decision_counts"]["HOLD"] == 4
        assert mh["decision_counts"]["EXTEND_TARGET"] == 1

    def test_samples_count(self):
        m = _fresh_monitor()
        m.record_pm_cycle([0.6, 0.7], {"EXIT": 0, "HOLD": 2, "EXTEND_TARGET": 0})
        mh = m.model_health()
        assert mh["samples"] == 2


# ─── Weekly Report ────────────────────────────────────────────────────────────

class TestWeeklyReport:
    def test_report_has_required_keys(self):
        m = _fresh_monitor()
        with patch("app.analytics.performance_review.get_performance_review", return_value={}):
            with patch("app.analytics.signal_attribution.get_signal_attribution", return_value={}):
                with patch("app.analytics.drawdown_analyzer.get_drawdown_summary", return_value={}):
                    report = m.generate_weekly_report()
        assert "performance" in report
        assert "signal_attribution" in report
        assert "signal_quality" in report
        assert "benchmark" in report
        assert "model_health" in report
        assert "generated_at" in report

    def test_report_includes_live_signal_quality(self):
        m = _fresh_monitor()
        m.record_trade_result("EMA_CROSSOVER", 100.0)
        with patch("app.analytics.performance_review.get_performance_review", return_value={}):
            with patch("app.analytics.signal_attribution.get_signal_attribution", return_value={}):
                with patch("app.analytics.drawdown_analyzer.get_drawdown_summary", return_value={}):
                    report = m.generate_weekly_report()
        assert "EMA_CROSSOVER" in report["signal_quality"]
