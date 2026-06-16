"""
crash_governor.py — VIX term-structure CRASH GOVERNOR (Alpha-v7 F1b).

A book-modifying OVERLAY (not an additive sleeve): when the VIX term structure inverts
(front-month VIX above 3-month VIX3M = backwardation = acute stress), de-risk the book's
exposure for the next session; otherwise hold full exposure. The classic VIX
term-structure regime signal (contango = calm, backwardation = stress).

Produces the AS-APPLIED daily exposure multiplier for `sleeve_lab.Overlay` / `evaluate_overlay`:
the signal is read at the close of day t (vix[t], vix3m[t]) and applied to the book's
day t+1 return — so the returned series is already shifted by one day (PIT-safe).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VixTermGovernorConfig:
    derisk_to: float = 0.5        # exposure multiplier while stressed (0 = fully flat)
    ratio_threshold: float = 1.0  # vix/vix3m > threshold => backwardation => de-risk
    confirm_days: int = 1         # require N consecutive inverted closes (debounce)


def vix_term_multiplier(vix: pd.Series, vix3m: pd.Series,
                        cfg: VixTermGovernorConfig) -> pd.Series:
    """AS-APPLIED daily exposure multiplier from the VIX/VIX3M term-structure signal.

    stressed[t]   = (vix[t] / vix3m[t]) > ratio_threshold   (optionally confirmed over
                    `confirm_days` consecutive closes)
    multiplier[t] = derisk_to if stressed[t] else 1.0   (decided at close t)
    returned      = multiplier.shift(1)   (applied to the book's NEXT-day return -> PIT)
    """
    if not (0.0 <= cfg.derisk_to <= 1.0):
        raise ValueError("derisk_to must be in [0, 1]")
    v = pd.Series(vix).astype(float)
    v3 = pd.Series(vix3m).astype(float)
    df = pd.concat([v.rename("vix"), v3.rename("vix3m")], axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("vix and vix3m share no common dates")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    ratio = df["vix"] / df["vix3m"]
    stressed = ratio > cfg.ratio_threshold
    if cfg.confirm_days > 1:
        # require the inversion to hold for `confirm_days` consecutive closes (debounce)
        stressed = (stressed.rolling(cfg.confirm_days).sum() >= cfg.confirm_days).fillna(False)
    raw = pd.Series(np.where(stressed, cfg.derisk_to, 1.0), index=df.index, name="mult")
    # PIT: signal at close[t] governs day t+1 -> shift forward by one trading day.
    return raw.shift(1).dropna()


def live_governor_multiplier(vix_recent: pd.Series, vix3m_recent: pd.Series,
                             cfg: VixTermGovernorConfig) -> Optional[float]:
    """The exposure multiplier to apply to the UPCOMING session, from the most recent
    settled VIX / VIX3M closes — the LIVE counterpart of `vix_term_multiplier` (shared
    stress rule, single source of truth). Returns `derisk_to` iff the term structure is
    inverted (vix/vix3m > ratio_threshold) on the last `confirm_days` settled closes, else
    1.0. Returns None when there is too little data (the caller MUST fail-safe to 1.0).

    No shift here on purpose: the latest SETTLED close governs the next session, which IS
    the t->t+1 PIT mapping that `vix_term_multiplier` achieves via `.shift(1)` over a full
    backtest series — so the live scalar and the backtest series agree by construction."""
    if not (0.0 <= cfg.derisk_to <= 1.0):
        raise ValueError("derisk_to must be in [0, 1]")
    df = pd.concat([pd.Series(vix_recent).rename("vix"),
                    pd.Series(vix3m_recent).rename("vix3m")], axis=1, join="inner").dropna()
    df = df[df["vix3m"] > 0]
    n = max(1, int(cfg.confirm_days))
    if len(df) < n:
        return None
    ratio = (df["vix"] / df["vix3m"]).iloc[-n:]
    stressed = bool((ratio > cfg.ratio_threshold).all())
    return cfg.derisk_to if stressed else 1.0
