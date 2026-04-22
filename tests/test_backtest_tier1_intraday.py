"""Tests for Tier 1 intraday backtester — label-aligned 2h cross-sectional simulation."""
import numpy as np
import pandas as pd
import pytest
from datetime import date
from unittest.mock import MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_5min_bars(n: int, start: str, base_price: float = 100.0,
                   drift: float = 0.0, spread_pct: float = 0.001) -> pd.DataFrame:
    """n 5-min bars starting at start with optional drift per bar.
    spread_pct controls high/low distance from close (default 0.1% — tight enough
    to not pierce 0.3% stop or 0.5% target naturally).
    """
    idx = pd.date_range(start, periods=n, freq="5min")
    prices = base_price + np.arange(n) * drift
    return pd.DataFrame({
        "open":   prices * (1 - spread_pct * 0.5),
        "high":   prices * (1 + spread_pct),
        "low":    prices * (1 - spread_pct),
        "close":  prices,
        "volume": np.ones(n) * 100_000,
    }, index=idx)


def _make_uptrend_bars(n: int, start: str, base: float = 100.0) -> pd.DataFrame:
    """Bars that rise 0.03% per bar — should hit 0.5% target in ~17 bars."""
    return _make_5min_bars(n, start, base, drift=base * 0.0003)


def _make_downtrend_bars(n: int, start: str, base: float = 100.0) -> pd.DataFrame:
    """Bars that fall 0.02% per bar — should hit 0.3% stop in ~15 bars."""
    return _make_5min_bars(n, start, base, drift=-base * 0.0002)


def _mock_model(proba_value: float = 0.8) -> MagicMock:
    m = MagicMock()
    m.is_trained = True
    m.feature_names = None
    def _predict(X):
        n = len(X)
        return np.ones(n, dtype=int), np.full(n, proba_value)
    m.predict.side_effect = _predict
    return m


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIntradayBacktesterInit:
    def test_imports(self):
        from app.backtesting.intraday_backtest import IntradayBacktester
        bt = IntradayBacktester()
        assert bt.model is None

    def test_constants_match_training(self):
        from app.backtesting.intraday_backtest import HOLD_BARS, FEATURE_BARS
        from app.ml.intraday_training import HOLD_BARS as TRAIN_HOLD_BARS
        assert HOLD_BARS == TRAIN_HOLD_BARS, (
            f"Backtester HOLD_BARS={HOLD_BARS} must match training HOLD_BARS={TRAIN_HOLD_BARS}"
        )
        assert FEATURE_BARS == 12

    def test_no_model_returns_empty(self):
        from app.backtesting.intraday_backtest import IntradayBacktester
        bt = IntradayBacktester(model=None)
        result = bt.run({"AAPL": _make_5min_bars(50, "2024-01-02 09:30")})
        assert result.total_trades == 0


class TestCrosssectionalScoring:
    """Model must be called with all symbols batched per day, not one at a time."""

    def test_batch_predict_called_once_per_day(self):
        from app.backtesting.intraday_backtest import IntradayBacktester, FEATURE_BARS, HOLD_BARS

        model = _mock_model(0.9)
        bt = IntradayBacktester(model=model, top_n=5)

        n_bars = FEATURE_BARS + HOLD_BARS + 5
        syms = {
            "AAPL": _make_5min_bars(n_bars, "2024-01-02 09:30", 100),
            "MSFT": _make_5min_bars(n_bars, "2024-01-02 09:30", 200),
            "NVDA": _make_5min_bars(n_bars, "2024-01-02 09:30", 300),
        }
        bt.run(syms)

        # predict should be called once (all 3 symbols in one batch for that day)
        assert model.predict.call_count == 1
        X_arg = model.predict.call_args[0][0]
        assert X_arg.shape[0] == 3  # all 3 symbols in one call

    def test_top_n_limits_trades_per_day(self):
        from app.backtesting.intraday_backtest import IntradayBacktester, FEATURE_BARS, HOLD_BARS

        model = _mock_model(0.9)
        n_bars = FEATURE_BARS + HOLD_BARS + 5
        syms = {sym: _make_5min_bars(n_bars, "2024-01-02 09:30", 100 + i * 10)
                for i, sym in enumerate(["A", "B", "C", "D", "E", "F", "G"])}

        bt = IntradayBacktester(model=model, top_n=3)
        result = bt.run(syms)
        # max 3 trades per day × 1 day = 3
        assert result.total_trades <= 3


class TestExitLogic:
    def _run_single(self, bars: pd.DataFrame, top_n: int = 5):
        from app.backtesting.intraday_backtest import IntradayBacktester, FEATURE_BARS, HOLD_BARS
        model = _mock_model(0.9)
        n_needed = FEATURE_BARS + HOLD_BARS + 5
        if len(bars) < n_needed:
            extra = _make_5min_bars(n_needed - len(bars), bars.index[-1] + pd.Timedelta("5min"), float(bars["close"].iloc[-1]))
            bars = pd.concat([bars, extra])
        bt = IntradayBacktester(model=model, top_n=top_n)
        return bt.run({"TEST": bars})

    def test_target_hit_exit(self):
        from app.backtesting.intraday_backtest import FEATURE_BARS, TARGET_PCT, HOLD_BARS
        feat = _make_5min_bars(FEATURE_BARS, "2024-01-02 09:30", 100.0, drift=0.001)
        entry_price = float(feat["close"].iloc[-1])
        target = entry_price * (1 + TARGET_PCT)
        # Flat future (no natural drift), but force high above target on bar 5
        future = _make_5min_bars(HOLD_BARS, "2024-01-02 10:30", entry_price, drift=0.0001)
        future.iloc[4, future.columns.get_loc("high")] = target + 0.10
        future.iloc[4, future.columns.get_loc("low")] = entry_price * 0.999
        bars = pd.concat([feat, future])
        result = self._run_single(bars)
        hits = [t for t in result.trades if t.exit_reason == "TARGET"]
        assert len(hits) >= 1

    def test_stop_hit_exit(self):
        from app.backtesting.intraday_backtest import FEATURE_BARS, STOP_PCT, HOLD_BARS
        feat = _make_5min_bars(FEATURE_BARS, "2024-01-02 09:30", 100.0)
        entry_price = 100.0
        stop = entry_price * (1 - STOP_PCT)
        future = _make_5min_bars(HOLD_BARS, "2024-01-02 10:30", 100.0, drift=-0.03)
        future.iloc[3, future.columns.get_loc("low")] = stop - 0.01
        bars = pd.concat([feat, future])
        result = self._run_single(bars)
        hits = [t for t in result.trades if t.exit_reason == "STOP"]
        assert len(hits) >= 1

    def test_time_exit_at_hold_bars(self):
        from app.backtesting.intraday_backtest import FEATURE_BARS, HOLD_BARS
        # Tiny drift (no stop/target hit) → time exit
        bars = _make_5min_bars(FEATURE_BARS + HOLD_BARS + 5, "2024-01-02 09:30", 100.0, drift=0.0001)
        result = self._run_single(bars)
        time_exits = [t for t in result.trades if t.exit_reason == "TIME_EXIT"]
        assert len(time_exits) >= 1

    def test_hold_bars_capped_at_hold_bars(self):
        from app.backtesting.intraday_backtest import FEATURE_BARS, HOLD_BARS
        bars = _make_5min_bars(FEATURE_BARS + HOLD_BARS + 10, "2024-01-02 09:30", 100.0, drift=0.0001)
        result = self._run_single(bars)
        for t in result.trades:
            assert t.hold_bars <= HOLD_BARS


class TestMultiDayMultiSymbol:
    def test_multiple_days_scored(self):
        from app.backtesting.intraday_backtest import IntradayBacktester, FEATURE_BARS, HOLD_BARS

        model = _mock_model(0.9)
        bt = IntradayBacktester(model=model, top_n=2)

        n = FEATURE_BARS + HOLD_BARS + 5
        day1 = _make_5min_bars(n, "2024-01-02 09:30", 100)
        day2 = _make_5min_bars(n, "2024-01-03 09:30", 102)
        bars = pd.concat([day1, day2])

        result = bt.run({"AAPL": bars, "MSFT": _make_5min_bars(n * 2, "2024-01-02 09:30", 200)})
        # 2 days × up to 2 picks = up to 4 trades
        assert result.total_trades > 0
        # predict called once per day = 2 times
        assert model.predict.call_count == 2

    def test_result_has_correct_model_type(self):
        from app.backtesting.intraday_backtest import IntradayBacktester, FEATURE_BARS, HOLD_BARS
        model = _mock_model(0.9)
        bt = IntradayBacktester(model=model, top_n=1)
        n = FEATURE_BARS + HOLD_BARS + 5
        result = bt.run({"X": _make_5min_bars(n, "2024-01-02 09:30", 100)})
        assert result.model_type == "intraday"

    def test_insufficient_bars_skipped(self):
        from app.backtesting.intraday_backtest import IntradayBacktester, FEATURE_BARS, HOLD_BARS
        model = _mock_model(0.9)
        bt = IntradayBacktester(model=model, top_n=5)
        # Only FEATURE_BARS bars — not enough for HOLD_BARS window
        bars = _make_5min_bars(FEATURE_BARS, "2024-01-02 09:30", 100)
        result = bt.run({"X": bars})
        assert result.total_trades == 0
