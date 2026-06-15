"""
credit_curve_governor.py — CREDIT / YIELD-CURVE de-risk overlays (Alpha-v8 G1).

Two daily, deep-history, owned-data stress signals that trim the book's exposure BEFORE/AROUND
equity stress — slower, more fundamental cousins of the (fast, reactive) VIX-term crash governor.
Each produces an AS-APPLIED `[derisk_to, 1.0]` multiplier (signal at close[t] → applied t+1 via
`.shift(1)`), so it composes with the VIX governor through `sleeve_lab.compose_overlays`.

  - credit_multiplier:  HYG/IEF total-return ratio (high-yield vs duration-matched Treasuries).
                        When the ratio is BELOW its trailing `lookback` MA (by `band`), high-yield
                        is underperforming = credit spreads widening = risk-off → de-risk.
  - curve_multiplier:   10y − 3m term spread (^TNX − ^IRX). When the curve is inverted (spread <
                        `threshold`), recession/stress risk is elevated → modest de-risk.

PIT: signals use only settled closes; the returned series is `.shift(1)` so the latest settled
close governs the NEXT session (matches the VIX governor convention). Live helpers
(`live_*_multiplier`) take recent settled closes and return the current scalar (None if too thin
→ caller fail-safes to 1.0). Single source of truth: the live scalar reuses the backtest rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────────
# Credit overlay (HYG/IEF)
# ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class CreditGovernorConfig:
    # CANONICAL config = the Alpha-v8 G1 candidate (validated: marginal-to-governor dSharpe
    # +0.064 / dCalmar +0.030, all-3-crises, both-halves stable). The originally pre-registered
    # L=60/band=0 trigger FAILED (fired ~37% of days = a slow trend filter, not a stress signal;
    # Calmar-negative marginal to the governor). The selective trigger (fire only when HY is >2%
    # below a 120d MA = meaningful credit deterioration, ~18% of days) is the principled fix.
    lookback: int = 120           # trailing MA window for the HYG/IEF ratio
    band: float = 0.02            # de-risk when ratio < (1 - band) * MA (2% below = stress, not noise)
    derisk_to: float = 0.5        # exposure multiplier while credit-stressed
    confirm_days: int = 1         # consecutive stressed closes required (debounce)


def _credit_stressed(hyg: pd.Series, ief: pd.Series, cfg: CreditGovernorConfig) -> pd.Series:
    """Boolean 'credit-stressed' per settled close: HYG/IEF ratio below (1-band)×trailing MA,
    confirmed over `confirm_days`. Index-aligned to the common HYG/IEF dates."""
    df = pd.concat([pd.Series(hyg).rename("hyg"), pd.Series(ief).rename("ief")],
                   axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("hyg and ief share no common dates")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    ratio = (df["hyg"] / df["ief"])
    ma = ratio.rolling(cfg.lookback).mean()
    stressed = ratio < (1.0 - cfg.band) * ma          # NaN during warmup -> False below
    stressed = stressed.fillna(False)
    if cfg.confirm_days > 1:
        stressed = (stressed.rolling(cfg.confirm_days).sum() >= cfg.confirm_days).fillna(False)
    return stressed


def credit_multiplier(hyg: pd.Series, ief: pd.Series, cfg: CreditGovernorConfig) -> pd.Series:
    """AS-APPLIED daily exposure multiplier from the HYG/IEF credit signal (shift(1) PIT)."""
    if not (0.0 <= cfg.derisk_to <= 1.0):
        raise ValueError("derisk_to must be in [0, 1]")
    stressed = _credit_stressed(hyg, ief, cfg)
    raw = pd.Series(np.where(stressed, cfg.derisk_to, 1.0), index=stressed.index, name="credit")
    return raw.shift(1).dropna()


def live_credit_multiplier(hyg_recent: pd.Series, ief_recent: pd.Series,
                           cfg: CreditGovernorConfig) -> Optional[float]:
    """Live scalar for the UPCOMING session from the most recent SETTLED closes (no shift — the
    latest settled close governs next session; matches the backtest's shift(1)). None if too
    little data (caller fail-safes to 1.0)."""
    if not (0.0 <= cfg.derisk_to <= 1.0):
        raise ValueError("derisk_to must be in [0, 1]")
    df = pd.concat([pd.Series(hyg_recent).rename("hyg"), pd.Series(ief_recent).rename("ief")],
                   axis=1, join="inner").dropna()
    need = cfg.lookback + max(0, cfg.confirm_days - 1)
    if len(df) < need:
        return None
    stressed = _credit_stressed(df["hyg"], df["ief"], cfg)
    if stressed.empty:
        return None
    return cfg.derisk_to if bool(stressed.iloc[-1]) else 1.0


# ──────────────────────────────────────────────────────────────────────────────────
# Curve overlay (10y - 3m)
# ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class CurveGovernorConfig:
    threshold: float = 0.0        # de-risk when (y10 - y3m) < threshold (0 = inversion)
    derisk_to: float = 0.75       # milder de-risk (curve is a slow regime flag, not tactical)
    confirm_days: int = 5         # require a sustained inversion (debounce, slow signal)


def _curve_stressed(y10: pd.Series, y3m: pd.Series, cfg: CurveGovernorConfig) -> pd.Series:
    df = pd.concat([pd.Series(y10).rename("y10"), pd.Series(y3m).rename("y3m")],
                   axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("y10 and y3m share no common dates")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    stressed = (df["y10"] - df["y3m"]) < cfg.threshold
    if cfg.confirm_days > 1:
        stressed = (stressed.rolling(cfg.confirm_days).sum() >= cfg.confirm_days).fillna(False)
    return stressed.fillna(False)


def curve_multiplier(y10: pd.Series, y3m: pd.Series, cfg: CurveGovernorConfig) -> pd.Series:
    """AS-APPLIED daily exposure multiplier from the yield-curve inversion signal (shift(1))."""
    if not (0.0 <= cfg.derisk_to <= 1.0):
        raise ValueError("derisk_to must be in [0, 1]")
    stressed = _curve_stressed(y10, y3m, cfg)
    raw = pd.Series(np.where(stressed, cfg.derisk_to, 1.0), index=stressed.index, name="curve")
    return raw.shift(1).dropna()


def live_curve_multiplier(y10_recent: pd.Series, y3m_recent: pd.Series,
                          cfg: CurveGovernorConfig) -> Optional[float]:
    if not (0.0 <= cfg.derisk_to <= 1.0):
        raise ValueError("derisk_to must be in [0, 1]")
    df = pd.concat([pd.Series(y10_recent).rename("y10"), pd.Series(y3m_recent).rename("y3m")],
                   axis=1, join="inner").dropna()
    if len(df) < max(1, cfg.confirm_days):
        return None
    stressed = _curve_stressed(df["y10"], df["y3m"], cfg)
    if stressed.empty:
        return None
    return cfg.derisk_to if bool(stressed.iloc[-1]) else 1.0
