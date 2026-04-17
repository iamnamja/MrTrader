"""
Shared backtest metrics: Sharpe ratio, max drawdown, win rate, avg hold.
"""

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class Trade:
    symbol: str
    entry_date: object        # date or datetime
    exit_date: object
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    hold_bars: int            # bars held (days for swing, 5-min bars for intraday)
    exit_reason: str          # "TARGET", "STOP", "FORCE_CLOSE", "TRAIL"
    trade_type: str = "swing"


@dataclass
class BacktestResult:
    model_type: str           # "swing" or "intraday"
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_hold_bars: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    trades: List[Trade] = field(default_factory=list)

    @classmethod
    def from_trades(cls, trades: List[Trade], model_type: str = "swing") -> "BacktestResult":
        if not trades:
            return cls(model_type=model_type)

        total = len(trades)
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(winners) / total if total else 0.0
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / total
        avg_hold = sum(t.hold_bars for t in trades) / total

        # Sharpe (annualised, assuming daily returns for swing, intra-session for intraday)
        returns = [t.pnl_pct for t in trades]
        sharpe = _sharpe(returns)

        # Max drawdown on cumulative P&L curve
        max_dd = _max_drawdown([t.pnl for t in trades])

        # Profit factor
        gross_win = sum(t.pnl for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1e-9
        profit_factor = gross_win / gross_loss if gross_loss else float("inf")

        return cls(
            model_type=model_type,
            total_trades=total,
            winning_trades=len(winners),
            losing_trades=len(losers),
            total_pnl=round(total_pnl, 2),
            win_rate=round(win_rate, 4),
            avg_pnl_pct=round(avg_pnl_pct, 4),
            avg_hold_bars=round(avg_hold, 1),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd, 4),
            profit_factor=round(profit_factor, 3),
            trades=trades,
        )

    def summary(self) -> dict:
        return {
            "model_type": self.model_type,
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate:.1%}",
            "avg_pnl_pct": f"{self.avg_pnl_pct:.2%}",
            "avg_hold_bars": self.avg_hold_bars,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": f"{self.max_drawdown_pct:.1%}",
            "profit_factor": self.profit_factor,
            "total_pnl": f"${self.total_pnl:,.2f}",
        }


def _sharpe(returns: List[float], periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free = 0)."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 1e-9
    return (mean / std) * math.sqrt(periods_per_year)


def _max_drawdown(pnls: List[float]) -> float:
    """Max drawdown as fraction of peak cumulative P&L (0 = no drawdown)."""
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = (peak - cum) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd
