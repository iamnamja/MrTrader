"""
futures_factors.py — Alpha-v10 P1.2: the futures factor zoo (owned Norgate data, $0).

Our `carry_backtest` is already a generic CROSS-SECTIONAL factor engine: it z-scores whatever
signal panel it's given each day, sizes inverse-vol, gross-caps, book-vol-targets, and charges
turnover + (optional) roll cost. So a new factor = a new SIGNAL panel run through that engine
(`xs_factor_backtest` is a semantic alias).

Signals (all PIT — each value at t uses only data <= t):
  * xs_momentum  — cross-sectional 12-1 momentum  P[t-skip]/P[t-lookback]-1   (long winners / short losers)
  * curve_momentum — trend of the carry signal     carry[t] - carry[t-lb]
  * value        — long-horizon (~5y) reversal     -(P[t]/P[t-lb]-1)          (long cheap)
  * skewness     — low realized-skew preference     -rolling_skew(returns)

VERDICT (P1.2, pre-registered, no sign-flipping): on the 76-market liquid universe, **only
xs_momentum survives** (Sharpe ~0.56, corr-to-trend ~0.12, modern-robust, residual-alpha t~1.6).
curve_momentum / value / skewness are dead-or-flat at the economically-motivated sign and are NOT
pursued (flipping the sign to chase a negative Sharpe is the OPT-5 trap). Only xs_momentum is
registered as a sleeve (`futures_xsmom`, P1-FUT-XSMOM).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from app.research import futures_carry as fc


def xs_momentum_signal(prices: pd.DataFrame, *, lookback: int = 252,
                       skip: int = 21) -> pd.DataFrame:
    """12-1 cross-sectional momentum: trailing `lookback`-day return skipping the last `skip`
    days (the standard 1-month skip to avoid short-term reversal). PIT."""
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def curve_momentum_signal(carry: pd.DataFrame, *, lookback: int = 63) -> pd.DataFrame:
    """Trend of the carry signal (its trailing `lookback`-day change). PIT."""
    return carry - carry.shift(lookback)


def value_signal(prices: pd.DataFrame, *, lookback: int = 1260) -> pd.DataFrame:
    """Long-horizon (~5y) reversal: cheap (large negative ~5y return) -> long. PIT."""
    return -(prices / prices.shift(lookback) - 1.0)


def skew_signal(returns: pd.DataFrame, *, window: int = 252,
                min_periods: int = 120) -> pd.DataFrame:
    """Low realized-skewness preference (lottery-avoidance): -rolling skew. PIT."""
    return -returns.rolling(window, min_periods=min_periods).skew()


def xs_factor_backtest(returns: pd.DataFrame, signal: pd.DataFrame,
                       cfg: Optional[fc.CarryConfig] = None,
                       roll_days: Optional[pd.DataFrame] = None) -> pd.Series:
    """Run a cross-sectional factor SIGNAL through the generic XS engine (== carry_backtest)."""
    cfg = cfg or fc.CarryConfig()
    return fc.carry_backtest(returns, signal.reindex_like(returns), cfg, roll_days=roll_days)
