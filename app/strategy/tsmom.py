"""
Time-Series Momentum (TSMOM / trend-following) sleeve — Alpha-v4 Phase 2.

The crisis-positive, ~uncorrelated diversifier the book needs after Phase 1
established PEAD is a long-biased, market-beta-driven satellite (not standalone
alpha). Classic Moskowitz-Ooi-Pedersen time-series momentum on a small basket of
liquid, multi-asset ETFs: long an instrument when its own trend is up, flat when
down. Crisis-positivity comes from ROTATING into bonds/gold/USD when equities
downtrend (the pragmatic retail "flat/defensive rotation" — no ETF shorting in
the first cut; `allow_short` is an opt-in later lever).

Design notes:
  * Vectorized (pandas), not AgentSimulator — TSMOM is a continuous multi-asset
    rotation, not discrete equity events. Fast + standard + easy to prove PIT.
  * Lookback ENSEMBLE (avoid single-lookback overfit): signal = mean over
    lookbacks of sign(P_t / P_{t-L} - 1), in [-1, 1].
  * Inverse-realized-vol sizing to a per-instrument vol target; per-name and
    total-gross caps; weekly rebalance; one-way ETF transaction costs on turnover.

POINT-IN-TIME (the thing the deep-dive must verify): a weight computed from data
through day t is applied ONLY to the return from t -> t+1. In the backtest the
held-weight matrix is shifted forward one day before multiplying returns
(`held_weights.shift(1) * returns`), so no return is ever earned with a weight
that used that day's (or a future) price. Signals and realized vol use only
data up to and including the decision day.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

ANN = 252


@dataclass
class TSMOMConfig:
    # Multi-asset liquid ETF basket (equity / intl / rates / commodity / USD).
    universe: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP",
    ])
    lookbacks: tuple = (21, 63, 126, 252)   # trading-day trend lookbacks (~1/3/6/12mo)
    vol_lookback: int = 60                  # trailing window for realized vol
    target_vol: float = 0.10               # per-instrument annualized vol target
    rebalance_days: int = 5                 # weekly rebalance
    max_weight: float = 0.25               # per-instrument |weight| cap
    max_gross: float = 1.0                 # total gross exposure cap (sum |weight|)
    allow_short: bool = False              # long-flat first; shorting is opt-in
    vol_floor: float = 0.03                # floor on realized vol (avoid blow-up sizing)
    cost_bps: float = 2.0                  # one-way transaction cost (ETF spread+fee), bps
    ann: int = ANN


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


def tsmom_signals(prices: pd.DataFrame, cfg: TSMOMConfig) -> pd.DataFrame:
    """Ensemble trend signal in [-1, 1] per instrument, PIT.

    signal_i(t) = mean_L sign( P_i,t / P_i,{t-L} - 1 ), averaged over cfg.lookbacks.
    Uses only prices up to and including t (pct_change over L is backward-looking).
    NaN (insufficient history) -> 0 (no position).
    """
    sigs = []
    for L in cfg.lookbacks:
        mom = prices / prices.shift(L) - 1.0   # backward-looking; row t uses t and t-L
        sigs.append(np.sign(mom))
    ens = sum(sigs) / float(len(cfg.lookbacks))
    return ens.fillna(0.0)


def realized_vol(prices: pd.DataFrame, cfg: TSMOMConfig) -> pd.DataFrame:
    """Trailing annualized realized vol per instrument, PIT (window ends at t)."""
    rets = _daily_returns(prices)
    rv = rets.rolling(cfg.vol_lookback, min_periods=max(cfg.vol_lookback // 2, 10)).std()
    return (rv * np.sqrt(cfg.ann)).clip(lower=cfg.vol_floor)


def tsmom_weights(prices: pd.DataFrame, cfg: TSMOMConfig) -> pd.DataFrame:
    """Target weight per instrument per day, PIT (data through t only).

    w_i(t) = signal_i(t) * (target_vol / realized_vol_i(t)), long-flat unless
    allow_short, then clipped to +/-max_weight and scaled so sum|w| <= max_gross.
    """
    sig = tsmom_signals(prices, cfg)
    rv = realized_vol(prices, cfg)
    w = sig * (cfg.target_vol / rv)
    if not cfg.allow_short:
        w = w.clip(lower=0.0)                # long-flat: drop negative-trend shorts
    w = w.clip(lower=-cfg.max_weight, upper=cfg.max_weight)
    # Scale each row down (never up) so total gross <= max_gross.
    gross = w.abs().sum(axis=1)
    scale = (cfg.max_gross / gross).clip(upper=1.0).fillna(1.0)
    w = w.mul(scale, axis=0)
    return w.fillna(0.0)


@dataclass
class TSMOMResult:
    returns: pd.Series                       # daily net portfolio returns
    weights: pd.DataFrame                     # held weights (post-rebalance, per day)
    gross: pd.Series                          # daily gross exposure
    turnover: pd.Series                       # per-day one-way turnover
    sharpe: float
    cagr: float
    ann_vol: float
    max_drawdown: float
    calmar: float
    avg_gross: float
    ann_turnover: float

    def summary(self) -> dict:
        return {
            "sharpe": self.sharpe, "cagr": self.cagr, "ann_vol": self.ann_vol,
            "max_drawdown": self.max_drawdown, "calmar": self.calmar,
            "avg_gross": self.avg_gross, "ann_turnover": self.ann_turnover,
            "n_days": int(len(self.returns)),
        }


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def tsmom_backtest(prices: pd.DataFrame, cfg: TSMOMConfig | None = None) -> TSMOMResult:
    """Vectorized, PIT-safe TSMOM backtest. Returns the net daily return series.

    Weekly rebalance: target weights are recomputed every cfg.rebalance_days and
    HELD between rebalances. The held-weight matrix is shifted forward one day
    before multiplying the day's returns, so a weight set using data through day t
    is applied only to the t -> t+1 return (no look-ahead). Costs are charged on
    rebalance-day turnover.
    """
    cfg = cfg or TSMOMConfig()
    prices = prices[[c for c in cfg.universe if c in prices.columns]].sort_index()
    rets = _daily_returns(prices)
    target = tsmom_weights(prices, cfg)

    # Hold target weights between rebalances: only rows at rebalance steps update;
    # forward-fill the rest. Rebalance grid is anchored on the integer position so
    # it is deterministic and independent of calendar gaps.
    n = len(target)
    is_rebal = (np.arange(n) % cfg.rebalance_days == 0)
    held = target.where(pd.Series(is_rebal, index=target.index), other=np.nan).ffill().fillna(0.0)

    # Turnover at each rebalance = sum|w_t - w_{t-1}| (one-way). Cost charged same day.
    dw = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    cost = dw * (cfg.cost_bps / 1e4)

    # PIT: weight known at end of t-1 earns the t-1 -> t return. shift(1) enforces it.
    gross_ret = (held.shift(1) * rets).sum(axis=1)
    # Cost alignment: the trade that establishes held[t] is paid at the t close but
    # that weight first EARNS on day t+1 (returns use held.shift(1)). So shift the
    # cost forward one day to match — otherwise cost leads its own P&L by a day and
    # the row-0 initial-entry cost is orphaned by dropna() (free first entry). With
    # cost.shift(1) the full initial-entry turnover lands on the first live day.
    net_ret = (gross_ret - cost.shift(1)).dropna()

    eq = _equity_curve(net_ret)
    ann_vol = float(net_ret.std() * np.sqrt(cfg.ann)) if net_ret.std() > 0 else 0.0
    sharpe = float(net_ret.mean() / net_ret.std() * np.sqrt(cfg.ann)) if net_ret.std() > 0 else 0.0
    yrs = max(len(net_ret) / cfg.ann, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if len(eq) else 0.0
    mdd = _max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    gross_series = held.abs().sum(axis=1).reindex(net_ret.index)
    ann_turnover = float(dw.reindex(net_ret.index).sum() / yrs)

    return TSMOMResult(
        returns=net_ret, weights=held.reindex(net_ret.index), gross=gross_series,
        turnover=dw.reindex(net_ret.index), sharpe=sharpe, cagr=cagr, ann_vol=ann_vol,
        max_drawdown=mdd, calmar=calmar, avg_gross=float(gross_series.mean()),
        ann_turnover=ann_turnover,
    )
