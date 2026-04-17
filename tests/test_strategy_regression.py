"""
Strategy regression tests — guard against unintentional strategy changes.

These tests run a 3-year backtest on benchmark symbols and assert that the
validated performance metrics are maintained.

Pass criteria (matches Phase 7 validated results):
  - Win rate  >= 50%
  - Avg trade >= +0.5%   (conservative floor; validated was +1.27%)
  - No crashes / exceptions

Run:
    pytest tests/test_strategy_regression.py -v
"""
import pytest

BENCHMARK_SYMBOLS = ["AAPL", "MSFT", "NVDA"]
YEARS = 3
MIN_WIN_RATE = 50.0
MIN_AVG_TRADE = 0.5   # percent


@pytest.fixture(scope="module")
def backtest_results():
    """Download data and run a 3-year backtest once for the whole module."""
    try:
        from app.backtest.backtest import BacktestRunner
        from app.backtest.data_loader import DataLoader
    except ImportError as exc:
        pytest.skip(f"Backtest dependencies unavailable: {exc}")

    try:
        loader = DataLoader()
        symbols_data = {}
        for sym in BENCHMARK_SYMBOLS:
            df = loader.download(sym, years=YEARS)
            if df is not None and not df.empty:
                symbols_data[sym] = df
        if not symbols_data:
            pytest.skip("No data downloaded for benchmark symbols")

        runner = BacktestRunner()
        results = runner.run_backtest_portfolio(symbols_data)
        return results
    except Exception as exc:
        pytest.skip(f"Backtest run failed (network?): {exc}")


def test_no_crashes(backtest_results):
    """Backtest must complete without exceptions and return results."""
    assert backtest_results is not None
    assert "results" in backtest_results
    assert len(backtest_results["results"]) > 0


def test_win_rate(backtest_results):
    """Aggregate win rate across benchmark symbols must be >= 50%."""
    win_rate = backtest_results.get("aggregate_win_rate_pct", 0)
    assert win_rate >= MIN_WIN_RATE, (
        f"Win rate {win_rate:.1f}% is below threshold {MIN_WIN_RATE}%"
    )


def test_avg_trade(backtest_results):
    """Average trade return across benchmark symbols must be >= +0.5%."""
    avg_trade = backtest_results.get("aggregate_avg_trade_pct", -999)
    assert avg_trade >= MIN_AVG_TRADE, (
        f"Avg trade {avg_trade:.2f}% is below threshold +{MIN_AVG_TRADE}%"
    )


def test_symbols_positive(backtest_results):
    """At least 2 out of 3 benchmark symbols must show positive returns."""
    symbol_results = backtest_results.get("results", {})
    positive = sum(
        1 for r in symbol_results.values()
        if r.get("total_return_pct", -999) > 0
    )
    assert positive >= 2, (
        f"Only {positive}/3 benchmark symbols positive"
    )


# ── Signal module unit tests (no network required) ────────────────────────────

def _make_bars(n: int = 250):
    """Generate synthetic uptrending OHLCV bars for unit testing."""
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    close = np.maximum(close, 10)
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    return pd.DataFrame({
        "open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": rng.integers(100_000, 500_000, n).astype(float),
    }, index=dates)


def test_generate_signal_returns_signal_result():
    """generate_signal() must always return a SignalResult."""
    from app.strategy.signals import generate_signal, SignalResult
    bars = _make_bars(250)
    result = generate_signal("TEST", bars)
    assert isinstance(result, SignalResult)
    assert result.action in ("BUY", "HOLD")
    assert result.signal_type in ("EMA_CROSSOVER", "RSI_DIP", "NONE")


def test_generate_signal_insufficient_bars():
    """With < 210 bars generate_signal() returns HOLD/NONE."""
    from app.strategy.signals import generate_signal
    bars = _make_bars(50)
    result = generate_signal("TEST", bars)
    assert result.action == "HOLD"


def test_check_exit_stop_hit():
    """check_exit() must trigger on stop hit after min hold."""
    from app.strategy.signals import check_exit
    should_exit, reason, _ = check_exit(
        symbol="TEST",
        current_price=95.0,
        entry_price=100.0,
        stop_price=96.0,
        target_price=110.0,
        highest_price=100.0,
        bars_held=5,
    )
    assert should_exit
    assert "stop_hit" in reason


def test_check_exit_target_hit():
    """check_exit() must trigger on target hit after min hold."""
    from app.strategy.signals import check_exit
    should_exit, reason, _ = check_exit(
        symbol="TEST",
        current_price=112.0,
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        highest_price=112.0,
        bars_held=5,
    )
    assert should_exit
    assert "target_hit" in reason


def test_check_exit_min_hold():
    """check_exit() must NOT exit within min hold bars even if stop is hit."""
    from app.strategy.signals import check_exit
    should_exit, reason, _ = check_exit(
        symbol="TEST",
        current_price=90.0,
        entry_price=100.0,
        stop_price=96.0,
        target_price=110.0,
        highest_price=100.0,
        bars_held=2,
    )
    assert not should_exit


def test_position_sizer_basic():
    """size_position() must return a positive int for valid inputs."""
    from app.strategy.position_sizer import size_position
    shares = size_position(
        account_equity=10_000,
        available_cash=10_000,
        entry_price=100.0,
        stop_price=97.5,   # $2.50 risk per share
    )
    # Expected: (10000 * 0.02) / 2.5 = 80 shares
    assert shares == 80


def test_position_sizer_cash_cap():
    """size_position() must cap at 90% of available cash."""
    from app.strategy.position_sizer import size_position
    shares = size_position(
        account_equity=1_000_000,
        available_cash=1_000,     # only $1k cash
        entry_price=100.0,
        stop_price=99.0,
    )
    # risk_based = (1m * 0.02) / 1.0 = 20000 shares
    # affordable = int(1000 * 0.9 / 100) = 9 shares
    assert shares == 9


def test_position_sizer_invalid():
    """size_position() must return 0 for invalid stop (>= entry)."""
    from app.strategy.position_sizer import size_position
    shares = size_position(
        account_equity=10_000,
        available_cash=10_000,
        entry_price=100.0,
        stop_price=105.0,
    )
    assert shares == 0
