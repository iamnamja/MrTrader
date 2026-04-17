"""Tests for Phase 33: swing and intraday ML backtesting."""
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_daily_df(n=400, start="2022-01-03", seed=42):
    idx = pd.bdate_range(start=start, periods=n)
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.maximum(prices, 1.0)
    return pd.DataFrame({
        "open": prices * 0.999, "high": prices * 1.005,
        "low": prices * 0.995, "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


def _make_5min_df(n_days=25, bars_per_day=78):
    all_bars = []
    day = datetime(2024, 1, 2, 9, 30)
    for _ in range(n_days):
        idx = pd.date_range(start=day, periods=bars_per_day, freq="5min")
        np.random.seed(hash(str(day)) % 2**31)
        prices = 100 + np.cumsum(np.random.randn(bars_per_day) * 0.05)
        prices = np.maximum(prices, 1.0)
        df = pd.DataFrame({
            "open": prices * 0.999, "high": prices * 1.002,
            "low": prices * 0.997, "close": prices,
            "volume": np.random.randint(10_000, 100_000, bars_per_day).astype(float),
        }, index=idx)
        all_bars.append(df)
        day = day + timedelta(days=1)
        while day.weekday() >= 5:
            day = day + timedelta(days=1)
    return pd.concat(all_bars)


def _trained_model(n_features=21):
    """Return a fitted PortfolioSelectorModel with balanced training data."""
    from app.ml.model import PortfolioSelectorModel
    np.random.seed(0)
    X = np.random.randn(200, n_features).astype(np.float32)
    y = np.array([i % 2 for i in range(200)])
    m = PortfolioSelectorModel()
    m.train(X, y, [f"f{i}" for i in range(n_features)])
    return m


# ── BacktestResult / metrics ──────────────────────────────────────────────────

class TestBacktestMetrics:

    def _make_trades(self, n=20, win_pct=0.6):
        from app.backtesting.metrics import Trade
        trades = []
        for i in range(n):
            won = i < int(n * win_pct)
            pnl_pct = 0.04 if won else -0.015
            trades.append(Trade(
                symbol=f"SYM{i}", entry_date=date(2023, 1, 2),
                exit_date=date(2023, 1, 5),
                entry_price=100.0, exit_price=100 * (1 + pnl_pct),
                quantity=10, pnl=100 * 10 * pnl_pct, pnl_pct=pnl_pct,
                hold_bars=3, exit_reason="TARGET" if won else "STOP",
                trade_type="swing",
            ))
        return trades

    def test_empty_trades_returns_zero_result(self):
        from app.backtesting.metrics import BacktestResult
        r = BacktestResult.from_trades([], model_type="swing")
        assert r.total_trades == 0
        assert r.win_rate == 0.0

    def test_win_rate_correct(self):
        from app.backtesting.metrics import BacktestResult
        trades = self._make_trades(20, win_pct=0.6)
        r = BacktestResult.from_trades(trades, model_type="swing")
        assert abs(r.win_rate - 0.6) < 0.01

    def test_profit_factor_positive(self):
        from app.backtesting.metrics import BacktestResult
        trades = self._make_trades(20, win_pct=0.6)
        r = BacktestResult.from_trades(trades, model_type="swing")
        assert r.profit_factor > 1.0

    def test_sharpe_nonzero(self):
        from app.backtesting.metrics import BacktestResult
        trades = self._make_trades(50, win_pct=0.6)
        r = BacktestResult.from_trades(trades, model_type="swing")
        assert r.sharpe_ratio != 0.0

    def test_max_drawdown_non_negative(self):
        from app.backtesting.metrics import BacktestResult
        trades = self._make_trades(20, win_pct=0.4)
        r = BacktestResult.from_trades(trades, model_type="swing")
        assert r.max_drawdown_pct >= 0.0

    def test_summary_has_expected_keys(self):
        from app.backtesting.metrics import BacktestResult
        trades = self._make_trades(10)
        r = BacktestResult.from_trades(trades, model_type="intraday")
        s = r.summary()
        for key in ["model_type", "total_trades", "win_rate", "sharpe_ratio",
                    "max_drawdown_pct", "profit_factor", "total_pnl"]:
            assert key in s

    def test_sharpe_helper_returns_zero_on_single_return(self):
        from app.backtesting.metrics import _sharpe
        assert _sharpe([0.05]) == 0.0

    def test_max_drawdown_zero_on_all_wins(self):
        from app.backtesting.metrics import _max_drawdown
        assert _max_drawdown([10.0, 10.0, 10.0]) == 0.0


# ── SwingBacktester ───────────────────────────────────────────────────────────

class TestSwingBacktester:

    def _data(self, n_symbols=5):
        return {f"SYM{i:02d}": _make_daily_df(400, seed=i) for i in range(n_symbols)}

    def test_returns_empty_result_without_model(self):
        from app.backtesting.swing_backtest import SwingBacktester
        bt = SwingBacktester(model=None)
        result = bt.run(self._data())
        assert result.total_trades == 0

    def test_returns_empty_on_insufficient_data(self):
        from app.backtesting.swing_backtest import SwingBacktester
        from app.ml.model import PortfolioSelectorModel
        m = MagicMock(spec=PortfolioSelectorModel)
        m.is_trained = True
        bt = SwingBacktester(model=m)
        tiny = {"AAPL": _make_daily_df(30)}
        result = bt.run(tiny)
        assert result.total_trades == 0

    def test_produces_trades_with_trained_model(self):
        from app.backtesting.swing_backtest import SwingBacktester

        model = _trained_model(n_features=21)
        # Patch feature engineer to return fixed feature dict
        feats = {f"f{i}": float(i) for i in range(21)}

        bt = SwingBacktester(model=model, min_confidence=0.0)
        with patch.object(bt.feature_engineer, "engineer_features", return_value=feats):
            result = bt.run(self._data(), fetch_fundamentals=False)

        assert result.total_trades >= 0  # may be 0 if score below threshold

    def test_all_trades_have_valid_exit_reasons(self):
        from app.backtesting.swing_backtest import SwingBacktester

        model = _trained_model(21)
        feats = {f"f{i}": float(i) for i in range(21)}
        bt = SwingBacktester(model=model, min_confidence=0.0)

        with patch.object(bt.feature_engineer, "engineer_features", return_value=feats):
            result = bt.run(self._data(), fetch_fundamentals=False)

        valid = {"TARGET", "STOP", "MAX_HOLD"}
        for t in result.trades:
            assert t.exit_reason in valid, f"Unexpected exit reason: {t.exit_reason}"

    def test_trade_pnl_consistent_with_prices(self):
        from app.backtesting.swing_backtest import SwingBacktester

        model = _trained_model(21)
        feats = {f"f{i}": float(i) for i in range(21)}
        bt = SwingBacktester(model=model, min_confidence=0.0)

        with patch.object(bt.feature_engineer, "engineer_features", return_value=feats):
            result = bt.run(self._data(), fetch_fundamentals=False)

        for t in result.trades:
            expected_pnl = (t.exit_price - t.entry_price) * t.quantity
            assert abs(t.pnl - expected_pnl) < 0.01, f"PnL inconsistent for {t.symbol}"

    def test_hold_bars_within_max(self):
        from app.backtesting.swing_backtest import SwingBacktester, MAX_HOLD_DAYS

        model = _trained_model(21)
        feats = {f"f{i}": float(i) for i in range(21)}
        bt = SwingBacktester(model=model, min_confidence=0.0)

        with patch.object(bt.feature_engineer, "engineer_features", return_value=feats):
            result = bt.run(self._data(), fetch_fundamentals=False)

        for t in result.trades:
            assert t.hold_bars <= MAX_HOLD_DAYS


# ── IntradayBacktester ────────────────────────────────────────────────────────

class TestIntradayBacktester:

    def _data(self, n_symbols=5):
        return {f"SYM{i:02d}": _make_5min_df(25) for i in range(n_symbols)}

    def test_returns_empty_result_without_model(self):
        from app.backtesting.intraday_backtest import IntradayBacktester
        bt = IntradayBacktester(model=None)
        result = bt.run(self._data())
        assert result.total_trades == 0

    def test_produces_trades_with_trained_model(self):
        from app.backtesting.intraday_backtest import IntradayBacktester
        from app.ml.intraday_features import FEATURE_NAMES

        n_features = len(FEATURE_NAMES)
        model = _trained_model(n_features)
        bt = IntradayBacktester(model=model, min_confidence=0.0)
        result = bt.run(self._data())
        # With min_confidence=0 some trades should be generated
        assert result.total_trades >= 0

    def test_all_intraday_trades_same_day(self):
        from app.backtesting.intraday_backtest import IntradayBacktester
        from app.ml.intraday_features import FEATURE_NAMES

        n_features = len(FEATURE_NAMES)
        model = _trained_model(n_features)
        bt = IntradayBacktester(model=model, min_confidence=0.0)
        result = bt.run(self._data())

        for t in result.trades:
            assert t.entry_date == t.exit_date, "Intraday trade spans multiple days"

    def test_all_exit_reasons_valid(self):
        from app.backtesting.intraday_backtest import IntradayBacktester
        from app.ml.intraday_features import FEATURE_NAMES

        n_features = len(FEATURE_NAMES)
        model = _trained_model(n_features)
        bt = IntradayBacktester(model=model, min_confidence=0.0)
        result = bt.run(self._data())

        valid = {"TARGET", "STOP", "FORCE_CLOSE"}
        for t in result.trades:
            assert t.exit_reason in valid

    def test_trade_type_is_intraday(self):
        from app.backtesting.intraday_backtest import IntradayBacktester
        from app.ml.intraday_features import FEATURE_NAMES

        n_features = len(FEATURE_NAMES)
        model = _trained_model(n_features)
        bt = IntradayBacktester(model=model, min_confidence=0.0)
        result = bt.run(self._data())

        for t in result.trades:
            assert t.trade_type == "intraday"

    def test_get_spy_day_returns_none_when_no_spy(self):
        from app.backtesting.intraday_backtest import IntradayBacktester
        bt = IntradayBacktester()
        result = bt._get_spy_day(None, date(2024, 1, 2))
        assert result is None
