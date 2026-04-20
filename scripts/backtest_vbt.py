"""
Vectorbt backtesting utility for rapid signal prototyping.

Usage:
    python scripts/backtest_vbt.py --symbols SPY QQQ --strategy momentum --period 2y
    python scripts/backtest_vbt.py --symbols AAPL MSFT GOOGL --strategy mean_reversion
    python scripts/backtest_vbt.py --symbols SPY --strategy rsi_bands --period 1y

Strategies:
    momentum       — buy when 20d return > 0, sell when < 0
    mean_reversion — buy when RSI < 35, sell when RSI > 65
    rsi_bands      — buy RSI<30 oversold, sell RSI>70 overbought
    dual_ma        — golden/death cross (50d vs 200d EMA)
"""
import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_prices(symbols: list[str], period: str) -> pd.DataFrame:
    df = yf.download(symbols, period=period, auto_adjust=True, progress=False)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    return close.dropna()


def momentum_entries(close: pd.DataFrame, window: int = 20):
    returns = close.pct_change(window)
    entries = returns > 0
    exits = returns < 0
    return entries, exits


def mean_reversion_entries(close: pd.DataFrame, rsi_period: int = 14, oversold: float = 35, overbought: float = 65):
    import pandas_ta as pta
    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    exits = pd.DataFrame(False, index=close.index, columns=close.columns)
    for col in close.columns:
        rsi = pta.rsi(close[col], length=rsi_period)
        entries[col] = rsi < oversold
        exits[col] = rsi > overbought
    return entries, exits


def rsi_bands_entries(close: pd.DataFrame, rsi_period: int = 14):
    return mean_reversion_entries(close, rsi_period, oversold=30, overbought=70)


def dual_ma_entries(close: pd.DataFrame, fast: int = 50, slow: int = 200):
    import pandas_ta as pta
    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    exits = pd.DataFrame(False, index=close.index, columns=close.columns)
    for col in close.columns:
        ema_fast = pta.ema(close[col], length=fast)
        ema_slow = pta.ema(close[col], length=slow)
        cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        cross_dn = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        entries[col] = cross_up
        exits[col] = cross_dn
    return entries, exits


def run_backtest(symbols: list[str], strategy: str, period: str, init_cash: float = 100_000.0):
    import vectorbt as vbt

    print(f"\n{'='*60}")
    print(f"Strategy : {strategy.upper()}")
    print(f"Symbols  : {', '.join(symbols)}")
    print(f"Period   : {period}")
    print(f"Capital  : ${init_cash:,.0f}")
    print(f"{'='*60}\n")

    close = fetch_prices(symbols, period)
    print(f"Data: {len(close)} trading days ({close.index[0].date()} → {close.index[-1].date()})\n")

    strategy_map = {
        "momentum": momentum_entries,
        "mean_reversion": mean_reversion_entries,
        "rsi_bands": rsi_bands_entries,
        "dual_ma": dual_ma_entries,
    }
    if strategy not in strategy_map:
        print(f"Unknown strategy '{strategy}'. Choose from: {list(strategy_map)}")
        sys.exit(1)

    entries, exits = strategy_map[strategy](close)

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=init_cash,
        fees=0.001,       # 0.1% commission
        slippage=0.001,   # 0.1% slippage
        freq="D",
    )

    stats = pf.stats()
    print("── Portfolio Stats ──────────────────────────────────────")
    key_metrics = [
        "Start Value", "End Value", "Total Return [%]",
        "Benchmark Return [%]", "Max Drawdown [%]",
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "Win Rate [%]", "Total Trades", "Profit Factor",
    ]
    for m in key_metrics:
        if m in stats.index:
            val = stats[m]
            if isinstance(val, float):
                print(f"  {m:<30} {val:>10.3f}")
            else:
                print(f"  {m:<30} {str(val):>10}")

    print("\n── Per-Symbol Return ────────────────────────────────────")
    for sym in close.columns:
        try:
            ret = pf[sym].total_return() * 100
            dd = pf[sym].max_drawdown() * 100
            sharpe = pf[sym].sharpe_ratio()
            print(f"  {sym:<8} return={ret:>7.2f}%  max_dd={dd:>6.2f}%  sharpe={sharpe:>6.3f}")
        except Exception:
            pass

    print(f"\n{'='*60}\n")
    return pf


def main():
    parser = argparse.ArgumentParser(description="Vectorbt signal backtester")
    parser.add_argument("--symbols", nargs="+", default=["SPY"], help="Ticker symbols")
    parser.add_argument("--strategy", default="momentum",
                        choices=["momentum", "mean_reversion", "rsi_bands", "dual_ma"])
    parser.add_argument("--period", default="2y", help="yfinance period (1y, 2y, 5y, etc.)")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting capital")
    args = parser.parse_args()

    run_backtest(args.symbols, args.strategy, args.period, args.cash)


if __name__ == "__main__":
    main()
