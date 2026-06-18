"""
futures_carry.py — Alpha-v9 P4-2: futures CARRY from the term structure.

Carry (Koijen-Moskowitz-Pedersen-Vrugt 2018) is the return a futures position earns if the
curve is unchanged — the annualized slope between the nearest two contracts:

    carry_t = (P_front - P_next) / P_next  *  365.25 / (expiry_next - expiry_front in days)

Backwardation (front > next) -> positive carry (you roll UP the curve as the front rolls to
spot). Contango -> negative carry. Computed from the OWN-mirror individual contracts; each
contract's expiry is taken as the LAST date in its price series (its last trading day) — a
clean, data-driven proxy. PIT: carry_t uses only day-t closes, so it informs the t+1 position.

The carry SLEEVE is CROSS-SECTIONAL (the standard global carry factor): each rebalance,
cross-sectionally demean + standardize carry across markets, size inverse-vol, long high /
short low carry (dollar-neutral-ish), book-vol-target the whole thing. Reuses the winsorized
true-return panel from futures_data (no NDU dependency — reads the parquet mirror).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from app.data import norgate_provider as ng
from app.research import futures_data as fd

ANN = 252
ROLL_BUFFER_DAYS = 5       # don't use a front contract within this many days of expiry (roll noise)
MIN_DT_YEARS = 0.04        # need a real expiry gap (~2wk) between front/next
CARRY_CLIP = 2.0           # winsorize absurd carry (illiquid stub contracts)
MIN_XS_WIDTH = 5           # min markets with carry on a day to trade the cross-section
STALE_LIMIT = 10           # max trading days to forward-fill a stale carry value (~2 weeks)

# Futures month codes -> month number (the contract symbol encodes the SCHEDULED delivery
# month, which is ex-ante known — used for the expiry instead of the realized last trade date,
# which both leaked hindsight near rolls AND went stale at the data edge).
_MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


def _scheduled_expiry(contract: str):
    """Parse 'ES-2026U' -> Timestamp(2026-09-15). Day-15 proxy: a constant within-month
    offset cancels in the front/next gap. Returns NaT if unparseable."""
    try:
        tag = str(contract).rsplit("-", 1)[1]
        yr, mc = int(tag[:4]), tag[4:5]
        mo = _MONTH_CODE.get(mc.upper())
        return pd.Timestamp(year=yr, month=mo, day=15) if mo else pd.NaT
    except Exception:
        return pd.NaT


def carry_series(market: str) -> pd.Series:
    """Daily annualized carry from the term structure (nearest two LIVE contracts), using the
    SCHEDULED (ex-ante) expiry from the contract code — not the realized last-trade date."""
    df = ng.load_contracts(market)[["date", "contract", "close"]].copy()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0.0]                              # >0 (negative close would flip sign)
    if df.empty:
        return pd.Series(dtype=float, name=market)
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = df["contract"].map(_scheduled_expiry)
    df = df.dropna(subset=["expiry"])
    # front must be > ROLL_BUFFER beyond t (scheduled, ex-ante) -> no hindsight, no stale edge
    df = df[df["expiry"] > df["date"] + pd.Timedelta(days=ROLL_BUFFER_DAYS)]
    if df.empty:
        return pd.Series(dtype=float, name=market)
    df = df.sort_values(["date", "expiry"])
    df["rank"] = df.groupby("date")["expiry"].rank(method="first")
    front = df[df["rank"] == 1].set_index("date")
    nxt = df[df["rank"] == 2].set_index("date")
    j = front[["close", "expiry"]].join(nxt[["close", "expiry"]], lsuffix="_f", rsuffix="_n",
                                        how="inner")
    if j.empty:
        return pd.Series(dtype=float, name=market)
    dt_years = (j["expiry_n"] - j["expiry_f"]).dt.days / 365.25
    dt_years = dt_years.where(dt_years > MIN_DT_YEARS)
    carry = (j["close_f"] - j["close_n"]) / j["close_n"] / dt_years
    carry = carry.replace([np.inf, -np.inf], np.nan).dropna()
    return carry.clip(-CARRY_CLIP, CARRY_CLIP).sort_index().rename(market)


def carry_panel(markets: Optional[Sequence[str]] = None) -> pd.DataFrame:
    markets = list(markets) if markets is not None else fd.liquid_universe()
    cols = {}
    for m in markets:
        try:
            s = carry_series(m)
        except FileNotFoundError:
            continue
        if len(s) >= 60:
            cols[m] = s
    if not cols:
        raise RuntimeError("carry_panel: no markets produced carry")
    return pd.DataFrame(cols).sort_index()


@dataclass
class CarryConfig:
    vol_lookback: int = 60
    target_vol: float = 0.10           # per-instrument vol target (inverse-vol sizing)
    vol_floor: float = 0.03
    rebalance_days: int = 5
    max_weight: float = 0.10
    max_gross: float = 5.0
    cost_bps: float = 3.0
    book_vol_target: float = 0.12
    book_vol_max_leverage: float = 4.0
    min_xs_width: int = MIN_XS_WIDTH    # min markets with carry on a day to trade the cross-section
    ann: int = ANN


def carry_backtest(returns: pd.DataFrame, carry: pd.DataFrame,
                   cfg: CarryConfig = CarryConfig()) -> pd.Series:
    """Cross-sectional carry book daily NET returns. weight_i ∝ (cross-sectional carry
    z-score) × inverse-vol, clipped, gross-capped, book-vol-targeted; PIT (carry at t →
    position t+1); cost on |Δweight|. `returns` = winsorized true returns panel."""
    rets = returns.sort_index()
    cz_raw = carry.reindex(rets.index).ffill(limit=STALE_LIMIT)   # cap stale-carry contamination
    # cross-sectional z-score of carry each day (demean + standardize across markets)
    mu = cz_raw.mean(axis=1)
    sd = cz_raw.std(axis=1)
    sd = sd.where(sd > 1e-8)                              # tolerance guard (std is never exactly 0)
    cz = cz_raw.sub(mu, axis=0).div(sd, axis=0).clip(-3, 3)
    # don't trade a too-thin cross-section (a 1-2 market day = a bet on noise, not a factor)
    n_valid = cz_raw.notna().sum(axis=1)
    cz = cz.where(n_valid >= cfg.min_xs_width)

    # inverse-vol scaling per instrument (PIT trailing vol)
    inst_vol = (rets.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).std()
                * np.sqrt(cfg.ann)).clip(lower=cfg.vol_floor)
    raw_w = cz * (cfg.target_vol / inst_vol)
    raw_w = raw_w.clip(-cfg.max_weight, cfg.max_weight)

    # rebalance only every rebalance_days (hold between); PIT shift signal by 1 day
    keep = (np.arange(len(raw_w)) % cfg.rebalance_days) == 0
    w = raw_w.where(pd.Series(keep, index=raw_w.index), np.nan).ffill().shift(1)

    # gross cap
    gross = w.abs().sum(axis=1).replace(0.0, np.nan)
    scale = (cfg.max_gross / gross).clip(upper=1.0).fillna(0.0)
    w = w.mul(scale, axis=0).fillna(0.0)

    gross_ret = (w * rets).sum(axis=1)
    # book-vol-target overlay (PIT: scale_t uses book returns through t-1)
    bvol = (gross_ret.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).std()
            * np.sqrt(cfg.ann))
    blev = (cfg.book_vol_target / bvol).clip(upper=cfg.book_vol_max_leverage).shift(1)
    net_gross = gross_ret * blev.fillna(0.0)

    # cost on the ACTUAL traded (levered) weights W = w*blev, so daily re-leveraging turnover
    # from the vol overlay is charged too (charging |Δw|*blev understates it by ~half).
    blev_f = blev.fillna(0.0)
    W = w.mul(blev_f, axis=0)
    turnover = W.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cfg.cost_bps / 1e4)
    return (net_gross - cost).dropna().rename("futures_carry")
