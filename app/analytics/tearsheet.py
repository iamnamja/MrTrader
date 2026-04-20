"""
Quantstats-powered performance tearsheet from closed trade history.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STARTING_CAPITAL = 100_000.0


def _build_returns(trades) -> pd.Series:
    """Group closed trade P&Ls into a daily returns series."""
    records = []
    for t in trades:
        if t.pnl is None or t.created_at is None:
            continue
        d = t.created_at.date() if hasattr(t.created_at, "date") else t.created_at
        records.append({"date": d, "pnl": float(t.pnl)})

    if not records:
        return pd.Series(dtype=float)

    df = pd.DataFrame(records)
    daily_pnl = df.groupby("date")["pnl"].sum()
    daily_returns = daily_pnl / STARTING_CAPITAL
    daily_returns.index = pd.DatetimeIndex(daily_returns.index)
    daily_returns = daily_returns.sort_index()
    return daily_returns


def _safe(val) -> Any:
    if val is None:
        return None
    try:
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
    except Exception:
        return None


def compute_tearsheet(trades) -> Dict[str, Any]:
    """Return quantstats metrics dict from a list of closed Trade ORM objects."""
    import quantstats as qs

    returns = _build_returns(trades)
    if len(returns) < 2:
        return {"error": "Need at least 2 trading days of closed trades"}

    monthly = returns.resample("ME").sum()

    result: Dict[str, Any] = {
        "trading_days": int(len(returns)),
        "trading_months": int(len(monthly)),
        # Return metrics
        "total_return_pct": _safe(returns.sum() * 100),
        "avg_daily_return_pct": _safe(returns.mean() * 100),
        "best_day_pct": _safe(returns.max() * 100),
        "worst_day_pct": _safe(returns.min() * 100),
        # Risk-adjusted
        "sharpe": _safe(qs.stats.sharpe(returns)),
        "sortino": _safe(qs.stats.sortino(returns)),
        "calmar": _safe(qs.stats.calmar(returns)),
        # Risk
        "max_drawdown_pct": _safe(qs.stats.max_drawdown(returns) * 100),
        "volatility_ann_pct": _safe(qs.stats.volatility(returns) * 100),
        # Win stats
        "win_rate_pct": _safe(qs.stats.win_rate(returns) * 100),
        "profit_factor": _safe(qs.stats.profit_factor(returns)),
        "avg_win_pct": _safe(qs.stats.avg_win(returns) * 100),
        "avg_loss_pct": _safe(qs.stats.avg_loss(returns) * 100),
        # Monthly breakdown
        "monthly_returns": {
            str(k.date()): round(float(v) * 100, 3)
            for k, v in monthly.items()
        },
    }
    return result
