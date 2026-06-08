"""
Short-term cross-sectional reversal sleeve — Alpha-v4 P4 (3rd uncorrelated premium).

The natural complement to the TSMOM trend sleeve: TREND is long-horizon momentum;
REVERSAL is short-horizon mean-reversion (Lehmann 1990 / Lo-MacKinlay). Cross-sectional,
dollar-neutral: each day, long the recent LOSERS and short the recent WINNERS over a short
lookback (skipping the most recent day to avoid bid-ask bounce / 1-day microstructure
reversal). Market-neutral by construction → low SPY beta → diversifying.

It mirrors `app/strategy/tsmom.py`'s PIT discipline EXACTLY — weights known at end of
t-1 earn the t-1->t return (`held.shift(1) * rets`), costs charged the day the new weight
first earns (`cost.shift(1)`) — so a weight never sees its own forward return. The only
structural difference: weights are dollar-neutral (sum=0) L/S instead of long-flat.

Short-term reversal is famously CHEAP-to-find but EXPENSIVE-to-trade (turnover-heavy,
concentrated in illiquid names). So this sleeve (a) restricts to the top-N liquid names by
trailing dollar volume each day, and (b) is validated under a punitive one-way cost sweep
— the cost is the make-or-break, exactly as the killed intraday sleeve taught us.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

ANN = 252


@dataclass
class ReversalConfig:
    lookback: int = 5            # reversal horizon (trading days)
    skip: int = 1               # skip the most-recent `skip` day(s) (microstructure bounce)
    rebalance_days: int = 1      # daily (reversal is short-horizon); sweep higher to cut cost
    liquidity_top_n: int = 500   # restrict to the top-N names by trailing dollar volume
    adv_lookback: int = 20       # trailing window for the dollar-volume liquidity rank
    min_names: int = 30          # need >= this many eligible names that day, else flat
    winsor: float = 3.0          # cross-sectional z winsor (caps single-name concentration)
    cost_bps: float = 10.0       # one-way transaction cost (single-name reversal is dear)
    ann: int = ANN


def _xs_demeaned_l1_weights(score: pd.DataFrame, cfg: ReversalConfig,
                            min_names: int) -> pd.DataFrame:
    """Cross-sectional winsorized z of `score` per row -> dollar-neutral (sum=0),
    gross-normalized (sum|w|=1) weights. Rows with < min_names eligible -> all zero."""
    mu = score.mean(axis=1)
    sd = score.std(axis=1)
    z = score.sub(mu, axis=0).div(sd, axis=0).clip(-cfg.winsor, cfg.winsor)
    # Re-center after winsor so the book is exactly dollar-neutral (sum=0).
    z = z.sub(z.mean(axis=1), axis=0)
    gross = z.abs().sum(axis=1)
    w = z.div(gross.replace(0.0, np.nan), axis=0)
    # Kill sparse rows (too few names to form a meaningful cross-section).
    n_ok = score.notna().sum(axis=1)
    w = w.where(n_ok >= min_names)
    return w.fillna(0.0)


def reversal_weights(prices: pd.DataFrame, volumes: pd.DataFrame,
                     cfg: ReversalConfig,
                     eligible: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Dollar-neutral cross-sectional reversal weights per day, PIT (data through t only).

    Signal = the lookback-window return ending `skip` days before t (uses only past prices);
    reversal weight ∝ -signal (long losers / short winners). Restricted to the top-N liquid
    names by trailing dollar volume, optionally intersected with a point-in-time membership
    mask `eligible` (True where the name was an index member on that date). Long-only is NOT
    applied — this is an L/S book.
    """
    prices = prices.sort_index()
    volumes = volumes.reindex_like(prices)

    # PIT signal: return over [t-skip-lookback, t-skip]; shift keeps it strictly historical.
    sig = prices.shift(cfg.skip) / prices.shift(cfg.skip + cfg.lookback) - 1.0
    rev = -sig  # reversal: most-negative past return -> largest long weight

    # Liquidity gate: top-N by trailing dollar volume (PIT — rolling, ends at t).
    dollar_vol = (prices * volumes).rolling(cfg.adv_lookback,
                                            min_periods=max(cfg.adv_lookback // 2, 5)).mean()
    liq_rank = dollar_vol.rank(axis=1, ascending=False)
    elig = rev.notna() & (liq_rank <= cfg.liquidity_top_n)
    if eligible is not None:
        elig &= eligible.reindex_like(rev).fillna(False)

    rev_e = rev.where(elig)
    return _xs_demeaned_l1_weights(rev_e, cfg, cfg.min_names)


@dataclass
class ReversalResult:
    returns: pd.Series
    weights: pd.DataFrame
    gross: pd.Series
    turnover: pd.Series
    sharpe: float
    cagr: float
    ann_vol: float
    max_drawdown: float
    calmar: float
    avg_gross: float
    ann_turnover: float
    avg_n_long: float
    avg_n_short: float

    def summary(self) -> dict:
        return {
            "sharpe": self.sharpe, "cagr": self.cagr, "ann_vol": self.ann_vol,
            "max_drawdown": self.max_drawdown, "calmar": self.calmar,
            "avg_gross": self.avg_gross, "ann_turnover": self.ann_turnover,
            "avg_n_long": self.avg_n_long, "avg_n_short": self.avg_n_short,
            "n_days": int(len(self.returns)),
        }


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def reversal_backtest(prices: pd.DataFrame, volumes: pd.DataFrame,
                      cfg: ReversalConfig | None = None,
                      eligible: Optional[pd.DataFrame] = None) -> ReversalResult:
    """Vectorized, PIT-safe dollar-neutral reversal backtest. Same weight->return->cost
    convention as tsmom_backtest: held weights are shifted forward one day before
    multiplying the day's returns, costs are charged on rebalance-day turnover shifted to
    the day the new weight first earns. Returns the net daily return series + diagnostics."""
    cfg = cfg or ReversalConfig()
    prices = prices.sort_index()
    rets = prices.pct_change()
    target = reversal_weights(prices, volumes, cfg, eligible=eligible)

    n = len(target)
    is_rebal = (np.arange(n) % cfg.rebalance_days == 0)
    held = target.where(pd.Series(is_rebal, index=target.index), other=np.nan).ffill().fillna(0.0)

    dw = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    cost = dw * (cfg.cost_bps / 1e4)
    gross_ret = (held.shift(1) * rets).sum(axis=1)
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
    held_live = held.reindex(net_ret.index)
    avg_n_long = float((held_live > 0).sum(axis=1).mean())
    avg_n_short = float((held_live < 0).sum(axis=1).mean())

    return ReversalResult(
        returns=net_ret, weights=held_live, gross=gross_series,
        turnover=dw.reindex(net_ret.index), sharpe=sharpe, cagr=cagr, ann_vol=ann_vol,
        max_drawdown=mdd, calmar=calmar, avg_gross=float(gross_series.mean()),
        ann_turnover=ann_turnover, avg_n_long=avg_n_long, avg_n_short=avg_n_short,
    )
