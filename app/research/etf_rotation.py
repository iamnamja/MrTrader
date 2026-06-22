"""
etf_rotation.py — Option A: cross-sectional relative-strength ETF rotation (a swing sleeve).

The live trend sleeve is ABSOLUTE time-series momentum (long-flat per asset). This is the distinct
cousin the panel/Min asked for: a CROSS-SECTIONAL relative-strength ROTATION — each rebalance, rank
the ETF universe by 12-1 momentum and hold the TOP-K strongest, inverse-vol weighted, with an
Antonacci DUAL-MOMENTUM filter (only hold a winner if its OWN absolute momentum is also positive,
else that slot goes to cash) so the book de-risks in broad downturns.

The CONFIG IS PRE-REGISTERED, standard, NOT swept (12-1 momentum, top-K = top third, monthly
rebalance, dual-momentum on) — the same OPT-5 discipline as the futures factor zoo: we judge the
canonical construction, not a tuned one. Whether it EARNS its place is decided by Track-B (does it
add residual-α on top of the live trend book?) — relative momentum is correlated to trend, so
redundancy is the null hypothesis the appraisal must reject.

Returns are daily, PIT (signal at t earns at t+1 via `held.shift(1)`), turnover-costed. Report-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RotationConfig:
    lookback: int = 252        # 12-month formation window
    skip: int = 21             # skip the most recent month (avoid 1-month reversal) → "12-1"
    top_k: int = 4             # hold the top-K by relative momentum (~top third of 11 sectors)
    vol_lookback: int = 60     # trailing window for inverse-vol weighting
    vol_floor: float = 0.03    # floor annualized vol before inverting (guards 1/0 on a flat name)
    rebalance_days: int = 21   # monthly rotation (sector RS is a slow signal)
    weight: str = "inverse_vol"  # "inverse_vol" | "equal"
    dual_momentum: bool = True   # hold a winner only if its OWN absolute momentum > 0, else cash
    cost_bps: float = 5.0      # per-side turnover cost (liquid ETFs)
    ann: int = 252


@dataclass
class RotationResult:
    returns: pd.Series
    sharpe: float
    ann_vol: float
    max_drawdown: float
    avg_n_held: float          # average number of ETFs held (≤ top_k; lower in downturns w/ dual-mom)
    cash_fraction: float       # fraction of days fully in cash (dual-momentum defensive)


def rotation_signal(prices: pd.DataFrame, cfg: RotationConfig) -> pd.DataFrame:
    """12-1 momentum per ETF: price[t-skip] / price[t-lookback] - 1. PIT — uses only past prices."""
    return prices.shift(cfg.skip) / prices.shift(cfg.lookback) - 1.0


def _inverse_vol_weights(rets: pd.DataFrame, cfg: RotationConfig) -> pd.DataFrame:
    rv = rets.rolling(cfg.vol_lookback,
                      min_periods=max(cfg.vol_lookback // 2, 10)).std() * np.sqrt(cfg.ann)
    # floor vol before inverting so a flat/zero-vol name can't become a 1/0=inf weight that
    # (via renormalisation) would silently flush the whole row to cash. NaN (vol unknown) is
    # left NaN → handled by the selection mask.
    rv = rv.where(rv.isna(), rv.clip(lower=cfg.vol_floor))
    return 1.0 / rv


def rotation_target_weights(prices: pd.DataFrame, cfg: RotationConfig) -> pd.DataFrame:
    """Daily TARGET weights (pre-rebalance): top-K by 12-1 momentum, dual-momentum filtered,
    inverse-vol (or equal) weighted, renormalised over the selected to sum to 1 (all-zero = cash).
    PIT — momentum + vol use only past prices; this is what the rebalance schedule samples."""
    rets = prices.pct_change()
    mom = rotation_signal(prices, cfg)
    # rank descending within each row (1 = strongest); NaN momentum → not rankable → not selected.
    rank = mom.rank(axis=1, ascending=False, method="first")
    selected = rank <= cfg.top_k
    if cfg.dual_momentum:
        selected = selected & (mom > 0.0)        # absolute filter: only hold genuine uptrends
    selected = selected.fillna(False)
    if cfg.weight == "equal":
        raw = selected.astype(float)
    else:
        ivw = _inverse_vol_weights(rets, cfg)
        raw = ivw.where(selected, other=0.0).fillna(0.0)
    wsum = raw.sum(axis=1)
    return raw.div(wsum.where(wsum > 0, np.nan), axis=0).fillna(0.0)   # cash (all-zero) on no pick


def rotation_backtest(prices: pd.DataFrame, cfg: RotationConfig = RotationConfig()) -> RotationResult:
    """Cross-sectional RS rotation: each day rank by 12-1 momentum, mark the top-K (dual-momentum
    filtered) as target holdings, weight them inverse-vol (or equal), rebalance monthly, shift(1)
    so weights earn next day, charge turnover cost. Returns the daily NET return series + stats."""
    prices = prices.sort_index()
    rets = prices.pct_change()
    target = rotation_target_weights(prices, cfg)

    # --- monthly rebalance, hold between, PIT earn (held.shift(1)), turnover cost ---
    n = len(prices)
    is_rebal = pd.Series(np.arange(n) % cfg.rebalance_days == 0, index=prices.index)
    held = target.where(is_rebal, other=np.nan).ffill().fillna(0.0)
    gross = (held.shift(1) * rets).sum(axis=1)
    dw = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    cost = dw * (cfg.cost_bps / 1e4)
    net = (gross - cost.shift(1)).dropna()
    # trim the leading warmup (no momentum yet → fully in cash → 0 returns) so stats reflect trading
    net = net.loc[net.ne(0).cumsum() > 0].rename("sector_rotation")

    if len(net) < 2 or net.std() == 0:
        return RotationResult(net, 0.0, 0.0, 0.0, 0.0, 1.0)
    sharpe = float(net.mean() / net.std() * np.sqrt(cfg.ann))
    ann_vol = float(net.std() * np.sqrt(cfg.ann))
    eq = (1.0 + net).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min())
    held_active = held.reindex(net.index)
    avg_n_held = float((held_active > 0).sum(axis=1).mean())
    cash_fraction = float((held_active.abs().sum(axis=1) == 0).mean())
    return RotationResult(net, sharpe, ann_vol, mdd, avg_n_held, cash_fraction)


# canonical pre-registered US-equity sector universe (11 SPDR sectors; XLRE 2015 / XLC 2018 join
# late — the rotation naturally excludes them until they have history)
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
