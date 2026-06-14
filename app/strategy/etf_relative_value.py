"""
etf_relative_value.py — slow ETF RELATIVE-VALUE (Alpha-v7 F2).

Dollar-neutral log-spread MEAN-REVERSION across economically-linked ETF pairs. Unlike the
F1 calendar premia (which were timed SPY beta and failed Track-B), this is genuinely
MARKET-NEUTRAL — long one ETF / short its economic twin — so it has a real shot at
DIVERSIFYING the trend book rather than duplicating its beta.

Per pair (A, B):
  spread[t] = log(A[t]) - log(B[t])                 (1:1 ratio spread; robust, no fitted hedge ratio)
  z[t]      = (spread[t] - rollmean_L) / rollstd_L  (rolling over the trailing L window)
  position  = stateful BAND: enter +1 (long spread = long A / short B) when z < -entry,
              enter -1 when z > +entry, exit to 0 when |z| < exit; HOLD between.
  pair_ret[t] = position[t-1] * (rA[t] - rB[t]) - cost            (position lagged -> PIT)

Slow / low-turnover by construction (long lookback, wide bands). Pairs are pre-registered
and economically motivated (NOT data-mined). The combined sleeve = the equal-weight mean of
the per-pair return streams, declared a `diversifier`.

PIT: the rolling z is computed through day t, but the POSITION is shift(1)-lagged, so the
signal that sets day t's position is known at the close of t-1. No look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

ANN = 252

# Pre-registered, economically-linked pairs (F2-RV). NOT a data-mined search.
DEFAULT_PAIRS: List[Tuple[str, str]] = [
    ("QQQ", "SPY"),   # tech vs broad market
    ("IWM", "SPY"),   # small vs large cap
    ("HYG", "IEF"),   # high-yield credit vs treasuries
    ("TLT", "IEF"),   # long vs intermediate duration
    ("EEM", "EFA"),   # emerging vs developed-ex-US equity
]


@dataclass
class RelativeValueConfig:
    lookback: int = 120          # z-score window (slow)
    entry_z: float = 1.5         # enter when |z| exceeds this
    exit_z: float = 0.5          # exit when |z| falls back inside this
    cost_bps: float = 2.0        # per-LEG cost; a pair trade moves 2 legs
    ann: int = ANN
    pairs: List[Tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_PAIRS))


@dataclass
class RelativeValueResult:
    label: str
    returns: pd.Series           # combined daily NET returns (equal-weight across pairs)
    per_pair: pd.DataFrame       # per-pair NET return streams
    sharpe: float
    cagr: float
    ann_vol: float
    n_days: int
    avg_gross_exposure: float    # mean fraction of pairs holding a position

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


def _band_positions(z: pd.Series, entry: float, exit_: float) -> pd.Series:
    """Stateful mean-reversion band -> position in {-1, 0, +1} from the z-series.
    +1 = long the spread (z below -entry, expect revert UP); -1 = short the spread;
    hold until |z| falls inside `exit_`. NaN z (warmup) -> flat."""
    out = np.zeros(len(z), dtype=float)
    state = 0.0
    zv = z.to_numpy()
    for i, zi in enumerate(zv):
        if not np.isfinite(zi):
            state = 0.0
        elif state == 0.0:
            if zi <= -entry:
                state = 1.0
            elif zi >= entry:
                state = -1.0
        else:  # in a position -> exit when back inside the inner band
            if abs(zi) <= exit_:
                state = 0.0
        out[i] = state
    return pd.Series(out, index=z.index)


def pair_spread_backtest(a: pd.Series, b: pd.Series, cfg: RelativeValueConfig,
                         label: str = "pair") -> pd.Series:
    """One pair's dollar-neutral log-spread mean-reversion NET daily return series."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if (df <= 0).any().any():
        raise ValueError(f"{label}: non-positive prices in spread legs")
    if len(df) <= cfg.lookback + 2:
        raise ValueError(f"{label}: too little history ({len(df)}) for lookback {cfg.lookback}")
    spread = np.log(df["a"]) - np.log(df["b"])
    mu = spread.rolling(cfg.lookback).mean()
    sd = spread.rolling(cfg.lookback).std()
    z = (spread - mu) / sd
    raw_pos = _band_positions(z, cfg.entry_z, cfg.exit_z)
    position = raw_pos.shift(1).fillna(0.0)            # PIT: signal at t-1 -> position at t
    ra = df["a"].pct_change()
    rb = df["b"].pct_change()
    gross = position * (ra - rb)                        # long A / short B (1:1, dollar-neutral)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * 2.0 * (cfg.cost_bps / 1e4)       # two legs traded per unit change
    return (gross - cost).rename(label)


def relative_value_backtest(prices: pd.DataFrame, cfg: RelativeValueConfig
                            ) -> RelativeValueResult:
    """Combine the pre-registered pairs (equal-weight) into one market-neutral sleeve.
    `prices` is a close-price panel containing every symbol in cfg.pairs."""
    prices = prices.copy()
    prices.columns = [str(c).upper() for c in prices.columns]
    streams = {}
    for a_sym, b_sym in cfg.pairs:
        if a_sym not in prices.columns or b_sym not in prices.columns:
            raise ValueError(f"prices missing {a_sym} or {b_sym}; has {list(prices.columns)}")
        lab = f"{a_sym}_{b_sym}"
        streams[lab] = pair_spread_backtest(prices[a_sym], prices[b_sym], cfg, label=lab)
    per_pair = pd.DataFrame(streams).sort_index()
    # equal-weight the pairs on each day (mean over pairs present that day)
    combined = per_pair.mean(axis=1, skipna=True).dropna().rename("etf_relative_value")
    sharpe, cagr, ann_vol = RelativeValueResult._stats(combined)
    # gross exposure proxy: fraction of pair-days holding a position
    active = per_pair.reindex(combined.index).abs() > 0
    avg_gross = float(active.mean(axis=1).mean()) if len(combined) else 0.0
    return RelativeValueResult(
        label="etf_relative_value", returns=combined, per_pair=per_pair,
        sharpe=sharpe, cagr=cagr, ann_vol=ann_vol, n_days=int(len(combined)),
        avg_gross_exposure=avg_gross)
