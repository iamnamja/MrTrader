"""
calendar_premia.py — structural CALENDAR / OVERNIGHT premia (Alpha-v7 F1).

Two deep-history, owned-data, trend-orthogonal premia, each a vectorized PIT-safe
single-name (SPY by default) backtest producing a daily NET return series for the
Sleeve Lab. These are CALENDAR-timed, not momentum-timed, so they are mechanically
orthogonal to the live TSMOM trend book — the panel's "most orthogonal / most powered /
cheapest / owned data" F1 bets:

  - turn_of_month_backtest  : long the index only in the turn-of-month window (the last
                              `days_before_end` + first `days_after_start` trading days),
                              flat otherwise. (Ariel 1987; Lakonishok-Smidt 1988.)
  - overnight_premium_backtest : capture the close->open OVERNIGHT return every day
                              (long overnight, flat intraday). (The well-documented
                              overnight-vs-intraday drift; net of realistic round-trip
                              cost it is marginal — an honest, cost-sensitive verdict.)

PIT discipline (mirrors app/strategy/tsmom.py):
  - The position for day t is set by the CALENDAR (known before t opens) or by the prior
    close, so `position[t] * ret[t]` has no look-ahead.
  - Costs are charged on |Δposition| (turnover), in bps of notional.
  - Returns are NET; the index is the trade date.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

ANN = 252


# ──────────────────────────────────────────────────────────────────────────────────
# Shared result
# ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class CalendarPremiaResult:
    """Uniform result for a calendar/overnight premium backtest."""
    label: str
    returns: pd.Series          # daily NET returns, DatetimeIndex
    position: pd.Series         # daily held position in {0, 1} (or fractional)
    sharpe: float
    cagr: float
    ann_vol: float
    n_days: int
    exposure: float             # fraction of days with a non-zero position

    @staticmethod
    def from_returns(label: str, net: pd.Series, position: pd.Series) -> "CalendarPremiaResult":
        net = net.dropna()
        position = position.reindex(net.index).fillna(0.0)
        if net.empty:
            return CalendarPremiaResult(label, net, position, 0.0, 0.0, 0.0, 0, 0.0)
        mu, sd = float(net.mean()), float(net.std())
        sharpe = float(mu / sd * np.sqrt(ANN)) if sd > 0 else 0.0
        ann_vol = float(sd * np.sqrt(ANN))
        growth = float((1.0 + net).prod())
        years = len(net) / ANN
        cagr = float(growth ** (1.0 / years) - 1.0) if years > 0 and growth > 0 else 0.0
        exposure = float((position != 0).mean())
        return CalendarPremiaResult(label, net, position, sharpe, cagr, ann_vol,
                                    int(len(net)), exposure)


def _require_cols(bars: pd.DataFrame, cols: List[str], who: str) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame):
        raise TypeError(f"{who}: bars must be a DataFrame, got {type(bars).__name__}")
    df = bars.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{who}: bars missing required column(s) {missing}; has {list(df.columns)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ──────────────────────────────────────────────────────────────────────────────────
# Turn-of-month
# ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class TurnOfMonthConfig:
    symbol: str = "SPY"
    days_before_end: int = 1     # last N trading days of the month -> long
    days_after_start: int = 3    # first N trading days of the next month -> long
    cost_bps: float = 1.0        # per-side transaction cost in bps of notional
    ann: int = ANN


def turn_of_month_window(index: pd.DatetimeIndex, *, days_before_end: int,
                         days_after_start: int) -> pd.Series:
    """Boolean membership of each trading day in the turn-of-month window, computed
    purely from the trading-day calendar (no price data -> no look-ahead). A day is
    in-window iff it is among the FIRST `days_after_start` or the LAST `days_before_end`
    trading days of its calendar month."""
    idx = pd.DatetimeIndex(index)
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    rank_from_start = ym.groupby(ym).cumcount()                       # 0-based from start
    rank_from_end = ym.groupby(ym).cumcount(ascending=False)         # 0-based from end
    in_first = rank_from_start < days_after_start
    in_last = rank_from_end < days_before_end
    return pd.Series((in_first | in_last).to_numpy(), index=idx, name="in_window")


def turn_of_month_backtest(bars: pd.DataFrame, cfg: TurnOfMonthConfig) -> CalendarPremiaResult:
    df = _require_cols(bars, ["close"], "turn_of_month_backtest")
    close = df["close"].astype(float)
    ret = close.pct_change()
    # position[t] is the calendar membership of day t — known before t opens (PIT-safe).
    position = turn_of_month_window(close.index, days_before_end=cfg.days_before_end,
                                    days_after_start=cfg.days_after_start).astype(float)
    gross = position * ret
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cfg.cost_bps / 1e4)
    net = (gross - cost).rename(f"tom_{cfg.symbol}")
    return CalendarPremiaResult.from_returns(f"tom_{cfg.symbol}", net, position)


# ──────────────────────────────────────────────────────────────────────────────────
# Overnight premium (close -> open)
# ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class OvernightConfig:
    symbol: str = "SPY"
    cost_bps: float = 1.0        # per-SIDE cost; a daily close-buy + open-sell = 2 sides
    ann: int = ANN


def overnight_premium_backtest(bars: pd.DataFrame, cfg: OvernightConfig) -> CalendarPremiaResult:
    """Long the overnight (close[t-1] -> open[t]) session every day, flat intraday.
    Charges a full round-trip (2 * cost_bps) PER DAY — buy at the prior close, sell at
    the open — which is why this premium is so cost-sensitive (the honest part of the
    verdict)."""
    df = _require_cols(bars, ["open", "close"], "overnight_premium_backtest")
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    overnight = open_ / close.shift(1) - 1.0          # close[t-1] -> open[t]; PIT-safe
    round_trip = 2.0 * (cfg.cost_bps / 1e4)           # buy at close, sell at open daily
    net = (overnight - round_trip).rename(f"overnight_{cfg.symbol}")
    position = pd.Series(1.0, index=df.index)          # always long overnight
    return CalendarPremiaResult.from_returns(f"overnight_{cfg.symbol}", net, position)
