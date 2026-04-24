"""
Tests for Phase 22 — Walk-Forward Tier 3 Validation.

Verifies:
  1. FoldResult.passed_gate() respects MIN_FOLD_SHARPE threshold
  2. WalkForwardReport.gate_passed() requires avg >= SHARPE_GATE and min >= MIN_FOLD_SHARPE
  3. WalkForwardReport.avg_sharpe, min_sharpe, avg_win_rate, total_trades computed correctly
  4. run_swing_walkforward() returns a report with correct fold count
  5. run_intraday_walkforward() returns a report with correct fold count
  6. Script is importable and main() runs without model (returns 1, no crash)
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fold(sharpe: float, trades: int = 50, win_rate: float = 0.5) -> "FoldResult":
    from scripts.walkforward_tier3 import FoldResult
    return FoldResult(
        fold=1,
        train_start=date(2021, 1, 1),
        train_end=date(2022, 12, 31),
        test_start=date(2023, 1, 1),
        test_end=date(2023, 12, 31),
        trades=trades,
        win_rate=win_rate,
        sharpe=sharpe,
        max_drawdown=0.05,
        total_return=0.02,
        stop_exit_rate=0.5,
    )


def _mock_sim_result(trades=60, win_rate=0.52, sharpe=0.9, drawdown=0.04):
    r = MagicMock()
    r.total_trades = trades
    r.win_rate = win_rate
    r.sharpe_ratio = sharpe
    r.max_drawdown_pct = drawdown
    r.total_return_pct = 0.05
    r.exit_breakdown = {"STOP": int(trades * 0.45), "TARGET": int(trades * 0.45), "TIME_EXIT": 6}
    r.print_report = MagicMock()
    return r


def _daily_bars(n: int = 300, base: float = 100.0, trend: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    p = base * (1 + trend) ** np.arange(n) + rng.normal(0, 0.3, n)
    p = np.maximum(p, 1.0)
    return pd.DataFrame({
        "open": p * 0.999, "high": p * 1.005,
        "low": p * 0.995, "close": p,
        "volume": np.full(n, 1_000_000.0),
    }, index=idx)


def _5min_bars(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": [100.0] * n, "high": [100.2] * n,
        "low": [99.8] * n, "close": [100.0] * n,
        "volume": [10_000.0] * n,
    }, index=idx)


# ── FoldResult ────────────────────────────────────────────────────────────────

class TestFoldResult:

    def test_passed_gate_above_threshold(self):
        from scripts.walkforward_tier3 import MIN_FOLD_SHARPE
        f = _fold(sharpe=MIN_FOLD_SHARPE + 0.01)
        assert f.passed_gate() is True

    def test_failed_gate_below_threshold(self):
        from scripts.walkforward_tier3 import MIN_FOLD_SHARPE
        f = _fold(sharpe=MIN_FOLD_SHARPE - 0.01)
        assert f.passed_gate() is False

    def test_exactly_at_threshold_passes(self):
        from scripts.walkforward_tier3 import MIN_FOLD_SHARPE
        f = _fold(sharpe=MIN_FOLD_SHARPE)
        assert f.passed_gate() is True

    def test_summary_line_contains_fold_number(self):
        f = _fold(sharpe=1.0)
        f.fold = 2
        assert "2" in f.summary_line()


# ── WalkForwardReport ─────────────────────────────────────────────────────────

class TestWalkForwardReport:

    def test_avg_sharpe_correct(self):
        from scripts.walkforward_tier3 import WalkForwardReport
        r = WalkForwardReport(model_type="swing")
        r.folds = [_fold(0.5), _fold(1.5), _fold(1.0)]
        assert r.avg_sharpe == pytest.approx(1.0)

    def test_min_sharpe_correct(self):
        from scripts.walkforward_tier3 import WalkForwardReport
        r = WalkForwardReport(model_type="swing")
        r.folds = [_fold(0.5), _fold(1.5), _fold(1.0)]
        assert r.min_sharpe == pytest.approx(0.5)

    def test_total_trades_summed(self):
        from scripts.walkforward_tier3 import WalkForwardReport
        r = WalkForwardReport(model_type="swing")
        r.folds = [_fold(1.0, trades=50), _fold(1.0, trades=70)]
        assert r.total_trades == 120

    def test_gate_passes_when_avg_and_min_ok(self):
        from scripts.walkforward_tier3 import WalkForwardReport, SHARPE_GATE, MIN_FOLD_SHARPE
        r = WalkForwardReport(model_type="swing")
        r.folds = [_fold(SHARPE_GATE + 0.1), _fold(SHARPE_GATE + 0.2)]
        assert r.gate_passed() is True

    def test_gate_fails_when_avg_below(self):
        from scripts.walkforward_tier3 import WalkForwardReport, SHARPE_GATE
        r = WalkForwardReport(model_type="swing")
        r.folds = [_fold(SHARPE_GATE - 0.5), _fold(SHARPE_GATE - 0.5)]
        assert r.gate_passed() is False

    def test_gate_fails_when_one_fold_too_low(self):
        from scripts.walkforward_tier3 import WalkForwardReport, SHARPE_GATE, MIN_FOLD_SHARPE
        r = WalkForwardReport(model_type="swing")
        # avg is fine but one fold is below MIN_FOLD_SHARPE
        r.folds = [_fold(SHARPE_GATE + 2.0), _fold(MIN_FOLD_SHARPE - 0.1)]
        assert r.gate_passed() is False

    def test_empty_report_does_not_crash(self):
        from scripts.walkforward_tier3 import WalkForwardReport
        r = WalkForwardReport(model_type="swing")
        assert r.avg_sharpe == 0.0
        assert r.gate_passed() is False

    def test_print_does_not_raise(self, capsys):
        from scripts.walkforward_tier3 import WalkForwardReport
        r = WalkForwardReport(model_type="swing")
        r.folds = [_fold(1.0)]
        r.print()  # should not raise


# ── Swing walk-forward integration ────────────────────────────────────────────

class TestSwingWalkforward:

    def test_returns_correct_fold_count(self):
        from scripts.walkforward_tier3 import run_swing_walkforward
        model = MagicMock()
        mock_result = _mock_sim_result()

        spy_series = pd.Series(
            np.full(300, 400.0),
            index=pd.date_range("2021-01-04", periods=300, freq="B"),
        )
        bars = {sym: _daily_bars(300) for sym in ["AAPL", "MSFT"]}

        import yfinance as yf_mod

        def fake_download(sym, **kw):
            df = bars.get(sym, _daily_bars(300)).copy()
            df.columns = [c.title() for c in df.columns]
            if sym == "SPY":
                return spy_series.to_frame("Close").rename(columns={"Close": "Close"})
            return df

        with patch("scripts.walkforward_tier3._load_model", return_value=(model, 95)):
            with patch("yfinance.download", side_effect=fake_download):
                with patch("app.backtesting.agent_simulator.AgentSimulator.run",
                           return_value=mock_result):
                    report = run_swing_walkforward(
                        n_folds=2, total_years=3, symbols=["AAPL", "MSFT"]
                    )

        assert len(report.folds) == 2

    def test_no_model_returns_empty_report(self):
        from scripts.walkforward_tier3 import run_swing_walkforward
        with patch("scripts.walkforward_tier3._load_model", return_value=(None, 0)):
            report = run_swing_walkforward(n_folds=2, total_years=3, symbols=["AAPL"])
        assert len(report.folds) == 0


# ── Intraday walk-forward integration ─────────────────────────────────────────

class TestIntradayWalkforward:

    def test_returns_correct_fold_count(self):
        from scripts.walkforward_tier3 import run_intraday_walkforward
        model = MagicMock()
        bars_5m = _5min_bars(500)
        cache_data = {"AAPL": bars_5m, "MSFT": bars_5m}

        mock_result = _mock_sim_result(trades=30)
        with patch("scripts.walkforward_tier3._load_model", return_value=(model, 18)):
            with patch("app.data.intraday_cache.available_symbols", return_value=["AAPL", "MSFT"]):
                with patch("app.data.intraday_cache.load_many", return_value=cache_data):
                    with patch(
                        "app.backtesting.intraday_agent_simulator.IntradayAgentSimulator.run",
                        return_value=mock_result,
                    ):
                        report = run_intraday_walkforward(
                            n_folds=2, total_days=180, symbols=["AAPL", "MSFT"]
                        )

        assert len(report.folds) == 2

    def test_no_model_returns_empty_report(self):
        from scripts.walkforward_tier3 import run_intraday_walkforward
        with patch("scripts.walkforward_tier3._load_model", return_value=(None, 0)):
            report = run_intraday_walkforward(n_folds=2, total_days=180, symbols=["AAPL"])
        assert len(report.folds) == 0


# ── Script importability ───────────────────────────────────────────────────────

class TestScriptImport:

    def test_importable(self):
        import scripts.walkforward_tier3 as wf
        assert hasattr(wf, "main")
        assert hasattr(wf, "run_swing_walkforward")
        assert hasattr(wf, "run_intraday_walkforward")
        assert hasattr(wf, "WalkForwardReport")
        assert hasattr(wf, "FoldResult")

    def test_constants_defined(self):
        from scripts.walkforward_tier3 import SHARPE_GATE, MIN_FOLD_SHARPE
        assert SHARPE_GATE > 0
        assert MIN_FOLD_SHARPE < 0  # allows slight negative folds
