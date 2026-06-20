"""
futures_roll.py — Alpha-v10 P0.2: explicit roll-transaction modeling for futures sleeves.

The carry/trend backtests earn returns from the difference-back-adjusted continuous series
(`ΔCCB/unadj_prev`), which CORRECTLY includes the economic roll yield (the back-adjustment
removes only the artificial price gap, not the realized roll-down P&L). So the roll *yield* is
already in the returns — subtracting it again would DOUBLE-COUNT the very premium carry harvests
(the panel's explicit warning).

What is MISSING is the **transaction cost of physically rolling**: at each roll you close the
expiring front and open the next (a calendar-spread round-trip) even when the target *weight* is
unchanged — so the `|Δweight|` turnover cost charges nothing for it. This module supplies the
roll-day schedule so the backtests can add that transaction cost (and ONLY that).

`roll_schedule(market)` = the days the front contract changes (scheduled-expiry front + roll
buffer, identical to the carry front-selection). Reads the local Norgate mirror only.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from app.data import norgate_provider as ng
from app.research.futures_carry import _scheduled_expiry, ROLL_BUFFER_DAYS


def roll_schedule(market: str) -> pd.Series:
    """Boolean Series (indexed by date): True on days the front contract rolls to the next.
    Front = nearest scheduled-expiry contract more than ROLL_BUFFER_DAYS from expiry (the same
    contract the position effectively holds), so a change of front = a roll."""
    df = ng.load_contracts(market)[["date", "contract", "close"]].copy()
    df = df[df["close"] > 0]
    if df.empty:
        return pd.Series(dtype=bool, name=market)
    df["date"] = pd.to_datetime(df["date"])
    df["exp"] = df["contract"].map(_scheduled_expiry)
    df = df.dropna(subset=["exp"])
    df = df[df["exp"] > df["date"] + pd.Timedelta(days=ROLL_BUFFER_DAYS)]
    if df.empty:
        return pd.Series(dtype=bool, name=market)
    df = df.sort_values(["date", "exp"])
    front = df.groupby("date").first()["contract"]
    roll = (front != front.shift(1)) & front.shift(1).notna()
    return roll.sort_index().rename(market)


def roll_days_panel(markets: Sequence[str],
                    index: Optional[pd.Index] = None) -> pd.DataFrame:
    """Boolean roll-day panel (dates × markets). Reindexed to `index` (False off-grid) if given."""
    cols = {}
    for m in markets:
        try:
            s = roll_schedule(m)
        except FileNotFoundError:
            continue
        if not s.empty:
            cols[m] = s
    if not cols:
        return pd.DataFrame(index=index if index is not None else None)
    panel = pd.DataFrame(cols).fillna(False)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    if index is not None:
        panel = panel.reindex(pd.to_datetime(index)).fillna(False)
    return panel.astype(bool)
