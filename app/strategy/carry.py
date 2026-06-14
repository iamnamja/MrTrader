"""
carry.py — CARRY done right (Alpha-v7 F3), small.

The cleanest deep-history carry available on free data: RATES / DURATION carry. The carry
(+ roll-down) of holding duration is positive when the curve is upward-sloping and negative
when inverted; so size a duration position by the term spread (10y − 3m). When the curve is
steep you are PAID to hold duration; when inverted, carry is negative → flat/short.

  spread[t]   = y10[t] − y3m[t]            (yield points, from yfinance ^TNX / ^IRX)
  position[t] = clip(spread / scale_pct, -max, +max)   (continuous; long-short by default)
  carry_ret[t]= position[t-1] * r_duration[t] − cost   (position lagged -> PIT)

Scope (honest): rates duration carry only. FX carry needs foreign short-rate differentials
(not cleanly available free) and commodity carry needs clean futures (we have none) — both
DEFERRED, matching the 2026-06-14 panel ("skip commodity"). Declared `risk_premium` (you are
paid to bear duration/curve risk; it is crisis-correlated by nature, so the worst-regime
backstop is waived — that vulnerability is the premium's nature, judged via Track-B instead).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANN = 252


@dataclass
class RatesCarryConfig:
    duration_etf: str = "IEF"     # intermediate Treasuries (deep, liquid since 2002)
    scale_pct: float = 1.5        # term spread (in %) mapping to a full unit position
    max_pos: float = 1.0
    long_short: bool = True       # allow short duration when the curve is inverted
    cost_bps: float = 1.0         # per unit |Δposition|
    ann: int = ANN


@dataclass
class CarryResult:
    label: str
    returns: pd.Series
    position: pd.Series
    sharpe: float
    cagr: float
    ann_vol: float
    n_days: int
    mean_position: float

    @staticmethod
    def _stats(net: pd.Series):
        net = net.dropna()
        if len(net) < 2:
            return 0.0, 0.0, 0.0
        mu, sd = float(net.mean()), float(net.std())
        sharpe = float(mu / sd * np.sqrt(ANN)) if sd > 0 else 0.0
        growth = float((1.0 + net).prod())
        years = len(net) / ANN
        cagr = float(growth ** (1.0 / years) - 1.0) if years > 0 and growth > 0 else 0.0
        return sharpe, cagr, float(sd * np.sqrt(ANN))


def term_spread(y10: pd.Series, y3m: pd.Series) -> pd.Series:
    """10y − 3m term spread (yield points), aligned on common dates."""
    df = pd.concat([pd.Series(y10).rename("y10"), pd.Series(y3m).rename("y3m")],
                   axis=1, join="inner").dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return (df["y10"] - df["y3m"]).sort_index().rename("term_spread")


def rates_carry_backtest(prices: pd.DataFrame, y10: pd.Series, y3m: pd.Series,
                         cfg: RatesCarryConfig) -> CarryResult:
    """Duration-carry timer: size a position in the duration ETF by the term spread."""
    prices = prices.copy()
    prices.columns = [str(c).upper() for c in prices.columns]
    etf = cfg.duration_etf.upper()
    if etf not in prices.columns:
        raise ValueError(f"prices missing {etf}; has {list(prices.columns)}")
    spread = term_spread(y10, y3m)
    raw = (spread / cfg.scale_pct).clip(-cfg.max_pos, cfg.max_pos)
    if not cfg.long_short:
        raw = raw.clip(lower=0.0)
    ret = prices[etf].astype(float).pct_change()
    aligned = pd.concat([raw.rename("raw"), ret.rename("ret")], axis=1,
                        join="inner").dropna()
    if len(aligned) < 60:
        raise ValueError(f"only {len(aligned)} aligned days for carry backtest")
    position = aligned["raw"].shift(1).fillna(0.0)      # PIT: spread at t -> position t+1
    gross = position * aligned["ret"]
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cfg.cost_bps / 1e4)
    net = (gross - cost).rename(f"rates_carry_{etf}")
    sharpe, cagr, ann_vol = CarryResult._stats(net)
    return CarryResult(label=f"rates_carry_{etf}", returns=net.dropna(),
                       position=position, sharpe=sharpe, cagr=cagr, ann_vol=ann_vol,
                       n_days=int(net.dropna().shape[0]), mean_position=float(position.mean()))
