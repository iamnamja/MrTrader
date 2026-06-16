"""
short_interest_governor.py — aggregate SHORT-INTEREST de-risk overlay (Alpha-v8 G2).

Thesis (Rapach–Ringgenberg–Tu 2016): the market-wide aggregate short interest is one of the
strongest known predictors of FUTURE market returns — informed short sellers, in aggregate,
time the market, so a HIGH aggregate short-interest index (SII) predicts LOW forward returns.
Used here as a SLOW, PREDICTIVE de-risk overlay (cut the book's exposure when shorts are
crowded), a different axis from the fast VIX governor and the credit-trend governor.

Data: `app/data/short_interest_provider.py` (Polygon/FINRA, per-security bi-monthly short
interest, PIT `knowable_date` = settlement + 10 bdays). POWER CAVEAT: the cache starts
2017-12-29 (~202 bi-monthly obs, ~8y) — it MISSES the GFC/2011; only ~3 in-window crises
(2018-Q4, COVID-2020, 2022). Bi-monthly cadence → power-limited; judged accordingly.

PIT: the SII is a TRAILING/expanding z of log(aggregate SI) over the last `window` settlement
observations (all knowable). The bi-monthly multiplier is indexed by `knowable_date` and
ffill-ed onto daily trading days, so each day uses the most-recent SII that was already public
(knowable_date ≤ day) — no look-ahead (the 10-bday publication lag is already in knowable_date).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ShortInterestGovernorConfig:
    window: int = 24          # trailing settlement-obs window for the SII z (~1y bi-monthly)
    min_obs: int = 12         # min trailing obs before emitting a z
    z_threshold: float = 1.0  # de-risk when SII z > threshold (crowded shorts = bearish)
    derisk_to: float = 0.5    # exposure multiplier while shorts are crowded


def aggregate_short_interest(store: pd.DataFrame) -> pd.Series:
    """Market-level aggregate short interest (shares summed across the universe) per settlement
    date, INDEXED BY `knowable_date` (the first public day) so it is directly PIT-applicable.
    Survivorship-safe (the store retains delisted names)."""
    need = {"settlement_date", "knowable_date", "short_interest"}
    if not need.issubset(store.columns):
        raise ValueError(f"store missing {need - set(store.columns)}")
    g = (store.groupby("settlement_date")
              .agg(si=("short_interest", "sum"), kd=("knowable_date", "max"))
              .reset_index())
    s = pd.Series(g["si"].to_numpy(dtype=float), index=pd.to_datetime(g["kd"]))
    return s[s > 0].sort_index().rename("aggregate_si")


def short_interest_index(agg_si: pd.Series, window: int = 24, min_obs: int = 12) -> pd.Series:
    """PIT-safe Short Interest Index: trailing z of log(aggregate SI) over the last `window`
    observations (inclusive — all knowable). The trailing mean absorbs the secular uptrend in
    aggregate SI (the RRT detrend); a high z = aggregate shorts crowded vs their recent norm."""
    ls = np.log(agg_si.astype(float))
    mu = ls.rolling(window, min_periods=min_obs).mean()
    sd = ls.rolling(window, min_periods=min_obs).std()
    return ((ls - mu) / sd).rename("sii").replace([np.inf, -np.inf], np.nan)


def _bimonthly_multiplier(store: pd.DataFrame, cfg: ShortInterestGovernorConfig) -> pd.Series:
    agg = aggregate_short_interest(store)
    sii = short_interest_index(agg, cfg.window, cfg.min_obs).dropna()
    if sii.empty:
        return pd.Series(dtype=float)
    stressed = sii > cfg.z_threshold
    return pd.Series(np.where(stressed, cfg.derisk_to, 1.0), index=sii.index, name="si_mult")


def si_multiplier(store: pd.DataFrame, cfg: ShortInterestGovernorConfig, *,
                  daily_index: Optional[pd.DatetimeIndex] = None) -> pd.Series:
    """AS-APPLIED daily exposure multiplier from the aggregate-SI signal. The bi-monthly
    multiplier (indexed by knowable_date) is forward-filled onto daily trading days — each day
    uses the most-recent SII already public (knowable_date ≤ day). PIT-safe; no extra shift."""
    if not (0.0 <= cfg.derisk_to <= 1.0):
        raise ValueError("derisk_to must be in [0, 1]")
    bim = _bimonthly_multiplier(store, cfg)
    if bim.empty:
        return pd.Series(dtype=float)
    if daily_index is None:
        daily_index = pd.bdate_range(bim.index.min(), bim.index.max())
    daily_index = pd.DatetimeIndex(daily_index)
    # union so ffill sees the knowable_date stamps, then restrict to the requested daily grid
    daily = bim.reindex(bim.index.union(daily_index)).ffill().reindex(daily_index).dropna()
    return daily.rename("si_mult")


def live_si_multiplier(store: pd.DataFrame, as_of, cfg: ShortInterestGovernorConfig
                       ) -> Optional[float]:
    """Live scalar for `as_of` from the most-recent KNOWABLE SII. None if too little knowable
    history (caller fail-safes to 1.0)."""
    if not (0.0 <= cfg.derisk_to <= 1.0):
        raise ValueError("derisk_to must be in [0, 1]")
    as_of = pd.Timestamp(as_of)
    bim = _bimonthly_multiplier(store, cfg)
    bim = bim[bim.index <= as_of]
    if bim.empty:
        return None
    return float(bim.iloc[-1])
