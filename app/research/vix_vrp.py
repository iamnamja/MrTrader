"""
vix_vrp.py — Alpha-v10 P3.1: the variance-risk-premium harvested via the VIX-futures curve.

The VRP was earlier PARKED/killed (Alpha-v5) — but on too-short options data and judged as an
*alpha* (it's a risk premium). The 2nd external panel flagged this as a likely wrong kill and
suggested the cheap, owned-data re-test: **short the front VIX future when the curve is in
contango** (capturing the roll-down as the future decays toward the lower spot), **flat when
backwardated** — gated by our existing crash-governor signal (VIX < VIX3M). No options, no NBBO.

Why this is the right framing:
- The short-VIX-future P&L = -1 * (back-adjusted VIX-future continuous return); in contango the
  continuous drifts DOWN, so the short harvests the VRP roll-down.
- The crash-governor gate is essential: naive short-vol loses 50-90% in a spike. Gating to FLAT
  on backwardation caps the damage (verified: Feb-2018 −4.4%, COVID-2020 −4.8%, vs catastrophe).
- Judged on Track-B as a RISK PREMIUM (it is short crash risk), not on an alpha floor.

PIT: the gate at t (known after t's close) sets the position for t+1 (`pos = -gate.shift(1)`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VixVRPConfig:
    target_vol: float = 0.12        # book vol target (VIX futures are ~80% ann vol raw)
    vol_lookback: int = 60
    max_leverage: float = 3.0
    ann: int = 252
    cost_bps: float = 5.0           # per-side cost on VIX-future turnover (wide bid/ask); charged on
    #                                 gate flips (full -1<->0) AND daily re-leveraging, like carry


def contango_gate(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """Boolean: curve in contango (VIX < VIX3M) -> calm -> short-vol ON. Aligned on common dates."""
    vt = pd.concat([vix.rename("v"), vix3m.rename("v3")], axis=1, join="inner").dropna()
    return (vt["v"] < vt["v3"]).rename("contango")


def vix_vrp_returns(vx_returns: pd.Series, vix: pd.Series, vix3m: pd.Series,
                    cfg: VixVRPConfig = VixVRPConfig()) -> pd.Series:
    """Gated short-VIX-future daily NET returns. `vx_returns` = the VIX-future continuous
    (difference-back-adjusted) return; short = -vx_returns, ON only in contango, vol-targeted."""
    gate = contango_gate(vix, vix3m).reindex(vx_returns.index).ffill().astype(float)
    pos = (-1.0 * gate).shift(1)                        # short vol in contango; PIT (gate t -> pos t+1)
    gross = (pos * vx_returns).dropna()
    bvol = gross.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).std() * np.sqrt(cfg.ann)
    lev = (cfg.target_vol / bvol).clip(upper=cfg.max_leverage).shift(1)
    # Net of transaction cost: VIX futures have a wide bid/ask, and this strategy turns over on every
    # gate flip (-1<->0) AND every day via the vol-target re-leveraging. Charging zero (the old code)
    # over-stated the Sharpe vs the cost-charged carry/xsmom sleeves it's compared against at go-live.
    levered = (pos * lev)
    ret = (levered * vx_returns)
    turnover = levered.diff().abs()
    turnover = turnover.fillna(levered.abs())           # charge entry on the first active day too
    cost = turnover * (cfg.cost_bps / 1e4)
    return (ret - cost).dropna().rename("vix_vrp")
