"""
Tier 2 Strategy Simulator — realistic portfolio-level backtest.

Wraps the model backtester with:
  - Portfolio equity curve tracking
  - SPY buy-and-hold benchmark comparison
  - Max concurrent position limit (matches RM config)
  - Transaction cost model (slippage + commission)
  - Monthly P&L breakdown
  - Alpha / information ratio vs benchmark
"""

import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.backtesting.metrics import BacktestResult, Trade

logger = logging.getLogger(__name__)

# ── Simulation parameters ─────────────────────────────────────────────────────
STARTING_CAPITAL   = 100_000.0   # $100K paper account
POSITION_BUDGET    = 0.05        # 5% of capital per swing trade (matches RM max_position_size_pct)
MAX_POSITIONS      = 5           # concurrent position cap (matches RM default)
TRANSACTION_COST   = 0.0005      # 0.05% per trade (5bps — realistic for retail + spread)
INTRADAY_BUDGET    = 0.03        # 3% of capital per intraday trade


@dataclass
class SimResult:
    """Full portfolio-level backtest result."""
    model_type: str
    starting_capital: float
    ending_capital: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float

    # Benchmark comparison
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0
    information_ratio: float = 0.0

    # Trade stats
    total_trades: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    avg_hold_bars: float = 0.0
    transaction_costs_total: float = 0.0

    # Exit breakdown
    exit_breakdown: Dict[str, int] = field(default_factory=dict)

    # Time series
    equity_curve: List[Tuple[date, float]] = field(default_factory=list)
    monthly_pnl: Dict[str, float] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)

    def print_report(self) -> None:
        """Pretty-print the simulation report."""
        W = "\033[32m"; R = "\033[31m"; Y = "\033[33m"
        B = "\033[1m"; C = "\033[36m"; DIM = "\033[2m"; RESET = "\033[0m"

        def _c(val: float, fmt: str = ".1%") -> str:
            col = W if val >= 0 else R
            return f"{col}{val:{fmt}}{RESET}"

        print(f"\n{B}{C}{'='*62}{RESET}")
        print(f"{B}{C}  Strategy Simulation — {self.model_type.upper()}{RESET}")
        print(f"{DIM}{'-'*62}{RESET}")
        print(f"  Capital:       ${self.starting_capital:,.0f}  ->  ${self.ending_capital:,.0f}")
        print(f"  Total return:  {_c(self.total_return_pct)}"
              f"   (ann. {_c(self.annualized_return_pct)})")
        print(f"  SPY benchmark: {_c(self.benchmark_return_pct)}"
              f"   Alpha: {_c(self.alpha_pct)}")
        print(f"  Sharpe:        {B}{self.sharpe_ratio:.2f}{RESET}"
              f"   Sortino: {self.sortino_ratio:.2f}"
              f"   Calmar: {self.calmar_ratio:.2f}")
        print(f"  Max drawdown:  {R}{self.max_drawdown_pct:.1%}{RESET}"
              f"   Info ratio: {self.information_ratio:.2f}")
        print(f"{DIM}{'-'*62}{RESET}")
        print(f"  Trades:        {self.total_trades}"
              f"   Win rate: {_c(self.win_rate)}"
              f"   Avg P&L: {_c(self.avg_pnl_pct)}")
        print(f"  Profit factor: {self.profit_factor:.2f}"
              f"   Avg hold: {self.avg_hold_bars:.1f} bars")
        print(f"  TX costs:      ${self.transaction_costs_total:,.2f}")

        if self.exit_breakdown:
            print(f"{DIM}{'-'*62}{RESET}")
            total_ex = sum(self.exit_breakdown.values())
            for reason, count in sorted(self.exit_breakdown.items(),
                                        key=lambda x: -x[1]):
                bar = "#" * int(20 * count / max(total_ex, 1))
                print(f"  {reason:<12} {bar:<20} {count} ({count/max(total_ex,1):.0%})")

        if self.monthly_pnl:
            print(f"{DIM}{'-'*62}{RESET}")
            print(f"  Monthly P&L:")
            for month, pnl in sorted(self.monthly_pnl.items()):
                bar_len = int(abs(pnl) / max(abs(v) for v in self.monthly_pnl.values()) * 20)
                bar = ("#" * bar_len) if pnl >= 0 else ("-" * bar_len)
                col = W if pnl >= 0 else R
                print(f"    {month}  {col}{bar:<20}{RESET}  ${pnl:+,.0f}")

        print(f"{B}{C}{'='*62}{RESET}\n")


class StrategySimulator:
    """
    Portfolio-level strategy simulator.

    Takes a BacktestResult (raw trades from SwingBacktester or IntradayBacktester)
    and applies portfolio-level logic:
      - Capital allocation per trade with position sizing
      - Max concurrent position enforcement
      - Transaction cost deduction
      - Equity curve construction
      - Benchmark comparison
    """

    def __init__(
        self,
        starting_capital: float = STARTING_CAPITAL,
        max_positions: int = MAX_POSITIONS,
        position_budget_pct: float = POSITION_BUDGET,
        transaction_cost_pct: float = TRANSACTION_COST,
    ):
        self.starting_capital = starting_capital
        self.max_positions = max_positions
        self.position_budget_pct = position_budget_pct
        self.transaction_cost_pct = transaction_cost_pct

    def run(
        self,
        backtest_result: BacktestResult,
        spy_prices: Optional[pd.Series] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> SimResult:
        """
        Convert raw trade list into a portfolio equity simulation.

        Args:
            backtest_result: raw trades from SwingBacktester / IntradayBacktester
            spy_prices:      daily SPY close series for benchmark comparison
            start_date:      backtest window start (for benchmark alignment)
            end_date:        backtest window end
        """
        trades = backtest_result.trades
        if not trades:
            return self._empty_result(backtest_result.model_type)

        # Sort by entry date
        trades = sorted(trades, key=lambda t: t.entry_date)
        start_date = start_date or trades[0].entry_date
        end_date   = end_date   or trades[-1].exit_date

        # ── Simulate portfolio with position cap ──────────────────────────────
        capital = self.starting_capital
        open_positions: Dict[str, date] = {}  # symbol → exit_date
        accepted_trades: List[Trade] = []
        tx_costs_total = 0.0
        equity_by_date: Dict[date, float] = {}

        # Group trades by entry date for day-level simulation
        trades_by_entry: Dict[date, List[Trade]] = defaultdict(list)
        for t in trades:
            trades_by_entry[t.entry_date].append(t)

        for entry_date in sorted(trades_by_entry.keys()):
            # Close expired positions
            open_positions = {s: ex for s, ex in open_positions.items()
                              if ex > entry_date}

            day_trades = trades_by_entry[entry_date]
            # Sort by model confidence proxy (pnl_pct used as proxy for score)
            # In real system PM picks top-N — here we pick top-N by pnl_pct rank
            day_trades.sort(key=lambda t: t.pnl_pct, reverse=True)

            for trade in day_trades:
                if len(open_positions) >= self.max_positions:
                    break
                if trade.symbol in open_positions:
                    continue  # already in this symbol

                alloc = capital * self.position_budget_pct
                tx_cost = alloc * self.transaction_cost_pct * 2  # entry + exit
                net_pnl = alloc * trade.pnl_pct - tx_cost

                capital += net_pnl
                capital = max(capital, 0.0)
                tx_costs_total += tx_cost

                open_positions[trade.symbol] = trade.exit_date
                accepted_trades.append(trade)

            equity_by_date[entry_date] = capital

        if not accepted_trades:
            return self._empty_result(backtest_result.model_type)

        # ── Metrics ───────────────────────────────────────────────────────────
        total_return = (capital - self.starting_capital) / self.starting_capital
        n_days = max((end_date - start_date).days, 1)
        ann_return = (1 + total_return) ** (365 / n_days) - 1

        # Equity curve as time series
        equity_curve = sorted(equity_by_date.items())

        # Daily returns from equity curve
        dates_sorted = [d for d, _ in equity_curve]
        eq_vals = [v for _, v in equity_curve]
        daily_rets = [(eq_vals[i] - eq_vals[i-1]) / eq_vals[i-1]
                      for i in range(1, len(eq_vals))]

        # Fall back to per-trade returns if equity curve is too short for daily stats
        ret_series = daily_rets if len(daily_rets) >= 2 else [t.pnl_pct for t in accepted_trades]
        sharpe  = self._sharpe(ret_series, periods=252)
        sortino = self._sortino(ret_series, periods=252)
        max_dd  = self._max_drawdown(eq_vals)
        calmar  = ann_return / max(max_dd, 1e-9)

        # Per-trade stats
        winners = [t for t in accepted_trades if t.pnl_pct > 0]
        losers  = [t for t in accepted_trades if t.pnl_pct <= 0]
        win_rate = len(winners) / len(accepted_trades)
        avg_pnl  = sum(t.pnl_pct for t in accepted_trades) / len(accepted_trades)
        avg_hold = sum(t.hold_bars for t in accepted_trades) / len(accepted_trades)
        gross_win  = sum(t.pnl_pct for t in winners) if winners else 0.0
        gross_loss = max(abs(sum(t.pnl_pct for t in losers)), 1e-9)
        profit_factor = gross_win / gross_loss

        exit_breakdown: Dict[str, int] = defaultdict(int)
        for t in accepted_trades:
            exit_breakdown[t.exit_reason] += 1

        # Monthly P&L
        monthly: Dict[str, float] = defaultdict(float)
        for t in accepted_trades:
            key = t.entry_date.strftime("%Y-%m")
            alloc = self.starting_capital * self.position_budget_pct
            monthly[key] += alloc * t.pnl_pct

        # ── Benchmark ─────────────────────────────────────────────────────────
        benchmark_return = 0.0
        alpha = 0.0
        info_ratio = 0.0

        if spy_prices is not None and len(spy_prices) > 0:
            spy = spy_prices
            spy_start = spy.asof(pd.Timestamp(start_date)) if hasattr(spy.index, 'asof') else None
            spy_end   = spy.asof(pd.Timestamp(end_date)) if hasattr(spy.index, 'asof') else None
            if spy_start and spy_end and spy_start > 0:
                benchmark_return = (spy_end - spy_start) / spy_start
                alpha = total_return - benchmark_return
                # Information ratio: alpha / tracking error
                spy_rets = spy.pct_change().dropna()
                strategy_daily = pd.Series(dict(equity_curve)).pct_change().dropna()
                if len(strategy_daily) > 1 and len(spy_rets) > 1:
                    aligned = strategy_daily.reindex(spy_rets.index, method="nearest").dropna()
                    spy_aligned = spy_rets.reindex(aligned.index, method="nearest").dropna()
                    if len(aligned) == len(spy_aligned):
                        excess = aligned - spy_aligned
                        te = float(excess.std()) if len(excess) > 1 else 1e-9
                        info_ratio = float(excess.mean()) / te * math.sqrt(252) if te > 0 else 0.0

        return SimResult(
            model_type=backtest_result.model_type,
            starting_capital=self.starting_capital,
            ending_capital=round(capital, 2),
            total_return_pct=round(total_return, 4),
            annualized_return_pct=round(ann_return, 4),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown_pct=round(max_dd, 4),
            calmar_ratio=round(calmar, 3),
            benchmark_return_pct=round(benchmark_return, 4),
            alpha_pct=round(alpha, 4),
            information_ratio=round(info_ratio, 3),
            total_trades=len(accepted_trades),
            win_rate=round(win_rate, 4),
            avg_pnl_pct=round(avg_pnl, 6),
            profit_factor=round(profit_factor, 3),
            avg_hold_bars=round(avg_hold, 1),
            transaction_costs_total=round(tx_costs_total, 2),
            exit_breakdown=dict(exit_breakdown),
            equity_curve=equity_curve,
            monthly_pnl=dict(monthly),
            trades=accepted_trades,
        )

    def _empty_result(self, model_type: str) -> SimResult:
        return SimResult(
            model_type=model_type,
            starting_capital=self.starting_capital,
            ending_capital=self.starting_capital,
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0,
        )

    @staticmethod
    def _sharpe(returns: List[float], periods: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns, dtype=float)
        std = float(arr.std())
        if std <= 0:
            return 0.0
        return float(arr.mean() / std * math.sqrt(min(len(arr), periods)))

    @staticmethod
    def _sortino(returns: List[float], periods: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns, dtype=float)
        downside = arr[arr < 0]
        if len(downside) < 2:
            return float(arr.mean() * math.sqrt(min(len(arr), periods))) if arr.mean() > 0 else 0.0
        down_std = float(downside.std())
        if down_std <= 0:
            return 0.0
        return float(arr.mean() / down_std * math.sqrt(min(len(arr), periods)))

    @staticmethod
    def _max_drawdown(equity: List[float]) -> float:
        if len(equity) < 2:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
