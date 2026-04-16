"""
MrTrader Backtesting CLI

Usage:
    python scripts/run_backtest.py --symbol AAPL --years 3
    python scripts/run_backtest.py --portfolio --count 20 --years 3
    python scripts/run_backtest.py --portfolio --years 3 > backtest_results.json
"""
import sys
import json
import logging
import argparse

# Ensure project root is on sys.path when run directly
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backtest.data_loader import DataLoader
from app.backtest.backtest import BacktestRunner
from app.utils.constants import SP_100_TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_backtest(symbol: str, years: int = 3) -> dict:
    """Backtest a single symbol."""
    start_date, end_date = DataLoader.get_date_range(years)
    data = DataLoader.download_ohlcv(symbol, start_date, end_date)

    if data.empty:
        logger.error(f"No data for {symbol}")
        return {}

    runner = BacktestRunner(initial_cash=20000)
    return runner.run_backtest(symbol, data)


def run_portfolio_backtest(years: int = 3, symbols: list = None) -> dict:
    """Backtest a portfolio of symbols."""
    if symbols is None:
        symbols = SP_100_TICKERS[:20]

    logger.info(f"Backtesting portfolio of {len(symbols)} symbols ({years} years)...")

    start_date, end_date = DataLoader.get_date_range(years)
    symbols_data = DataLoader.download_multiple(symbols, start_date, end_date)

    if not symbols_data:
        logger.error("No data available")
        return {}

    runner = BacktestRunner(initial_cash=20000)
    summary = runner.run_backtest_portfolio(symbols_data)

    _print_report(summary)
    return summary


def _print_report(summary: dict):
    print("\n" + "=" * 70)
    print("BACKTEST REPORT")
    print("=" * 70)
    print(f"Symbols tested:     {summary['symbols']}  ({summary['symbols_positive']} positive)")
    print(f"Average return:     {summary['average_return_pct']:.2f}%")
    print(f"Aggregate win rate: {summary['aggregate_win_rate_pct']:.1f}%  ({summary['total_closed_trades']} closed trades)")
    print(f"Avg trade return:   {summary['aggregate_avg_trade_pct']:+.2f}%")
    print(f"Avg Sharpe (3+ tr): {summary['average_sharpe_ratio']:.3f}")
    print()
    print(f"{'Symbol':<8} {'Return':>8} {'WinRate':>8} {'AvgTrd':>8} {'MaxDD':>8} {'Trades':>7} {'Benchmark':>10}")
    print("-" * 70)

    for symbol, r in sorted(summary["results"].items()):
        print(
            f"{symbol:<8} "
            f"{r['total_return_pct']:>7.2f}% "
            f"{r['win_rate_pct']:>7.1f}% "
            f"{r['avg_trade_return_pct']:>+7.2f}% "
            f"{r['max_drawdown_pct']:>7.2f}% "
            f"{r['trades_executed']:>7} "
            f"{r['benchmark_return_pct']:>9.2f}%"
        )

    print("=" * 70)

    # Go/No-Go recommendation (realistic criteria for a swing strategy)
    return_ok = summary["average_return_pct"] > 0
    winrate_ok = summary["aggregate_win_rate_pct"] >= 50
    avg_trade_ok = summary["aggregate_avg_trade_pct"] >= 1.0
    majority_positive = summary["symbols_positive"] >= summary["symbols"] * 0.6

    checks = [
        ("Positive avg return", return_ok),
        ("Win rate >= 50%",     winrate_ok, f"{summary['aggregate_win_rate_pct']:.1f}%"),
        ("Avg trade >= +1%",    avg_trade_ok, f"{summary['aggregate_avg_trade_pct']:+.2f}%"),
        ("60%+ symbols positive", majority_positive,
         f"{summary['symbols_positive']}/{summary['symbols']}"),
    ]
    print(f"\nGo/No-Go for Phase 8 (Paper Trading):")
    for item in checks:
        label, ok = item[0], item[1]
        detail = f"  ({item[2]})" if len(item) == 3 else ""
        print(f"  {label+':':<28} {'PASS' if ok else 'FAIL'}{detail}")

    go = all(c[1] for c in checks)
    print(f"\n  Verdict: {'PROCEED to Phase 8 (Paper Trading)' if go else 'REFINE strategy before proceeding'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="MrTrader Backtesting CLI")
    parser.add_argument("--symbol", help="Single symbol to backtest")
    parser.add_argument("--portfolio", action="store_true", help="Backtest portfolio")
    parser.add_argument("--years", type=int, default=3, help="Years of data to test")
    parser.add_argument("--count", type=int, default=20, help="Number of symbols in portfolio")
    parser.add_argument("--json", action="store_true", help="Output JSON (suppresses report table)")
    args = parser.parse_args()

    if args.symbol:
        result = run_single_backtest(args.symbol, years=args.years)
        if args.json or not sys.stdout.isatty():
            print(json.dumps(result, indent=2))
    else:
        symbols = SP_100_TICKERS[: args.count]
        summary = run_portfolio_backtest(years=args.years, symbols=symbols)
        if args.json or not sys.stdout.isatty():
            # Remove nested results for clean JSON when piping
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
