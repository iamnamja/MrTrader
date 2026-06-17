"""
overnight_intraday.py — Alpha-v9 P3-3: overnight vs intraday return decomposition.

The equity risk premium is well documented to accrue almost entirely OVERNIGHT
(close[t-1] -> open[t]); the intraday session (open[t] -> close[t]) historically
contributes ~zero or negative. Both legs are tradeable with MOC/MOO orders (no tick
data). The catch — and the whole question P3-3 answers — is COST: capturing the
overnight premium means a full round-trip EVERY day (buy at the close, sell at the
open), so realistic round-trip costs can erase it.

This module:
  * decompose(bars)              -> per-day {overnight, intraday, close_to_close}, with the
                                    exact reconciliation (1+overnight)(1+intraday)=1+cc.
  * decompose_universe(...)      -> equal-weight-of-symbols leg return streams + a cost-grid
                                    sweep of net Sharpe (the "cost cliff").
  * overnight_intraday_verdict() -> a PRE-REGISTERED net-of-cost PASS/KILL verdict.

Pre-registered criterion (frozen — registration_id P3-3-OVERNIGHT-INTRADAY):
  On the equal-weight liquid-ETF universe, the OVERNIGHT leg NET of the realistic
  round-trip cost (REALISTIC_COST_BPS_PER_SIDE per side -> 2x/day) must show
    (1) net annualized Sharpe >= PAPER_SR_FLOOR (0.30, the Ruler-v2 PAPER floor), AND
    (2) net CAGR > 0, AND
    (3) net overnight Sharpe > net intraday Sharpe (the premium genuinely lives overnight,
        not just being market beta split arbitrarily).
  All three -> PASS (worth a formal Sleeve-Lab/live-paper track); else KILL cleanly.

PIT discipline: overnight[t] uses close[t-1] (known) and open[t]; intraday[t] uses same-day
open[t]/close[t] — neither looks ahead. Costs are charged as a daily round-trip in bps.
Bars should be consistently split/dividend-adjusted (O and C on the same basis) so the legs
reconcile; on an ex-dividend day the overnight leg absorbs the gap (a minor, known wrinkle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

ANN = 252

# ── frozen pre-registration constants ──
REGISTRATION_ID = "P3-3-OVERNIGHT-INTRADAY"
REALISTIC_COST_BPS_PER_SIDE = 1.0          # liquid ETF MOC/MOO; round-trip = 2 bps/day
PAPER_SR_FLOOR = 0.30                       # matches the Ruler-v2 PAPER point-SR floor
DEFAULT_COST_GRID = (0.0, 0.5, 1.0, 2.0, 5.0)   # bps per SIDE
# Liquid US-equity ETFs we could actually trade MOC/MOO (subset of the live trend universe).
DEFAULT_UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "DIA")


@dataclass(frozen=True)
class LegStats:
    label: str
    n_days: int
    gross_sharpe: float
    gross_cagr: float
    ann_vol: float
    net_sharpe: float          # at cost_bps_per_side
    net_cagr: float
    cost_bps_per_side: float


@dataclass(frozen=True)
class OvernightIntradayVerdict:
    verdict: str               # "PASS" | "KILL"
    reason: str
    registration_id: str
    realistic_cost_bps_per_side: float
    paper_sr_floor: float
    overnight: LegStats
    intraday: LegStats
    cost_cliff: Dict[float, float]          # cost_bps_per_side -> net overnight Sharpe
    per_symbol: Dict[str, Dict[str, float]]
    universe: List[str] = field(default_factory=list)


def _norm_bars(bars: pd.DataFrame, who: str) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame):
        raise TypeError(f"{who}: bars must be a DataFrame, got {type(bars).__name__}")
    df = bars.copy()
    df.columns = [str(c).lower() for c in df.columns]
    for c in ("open", "close"):
        if c not in df.columns:
            raise ValueError(f"{who}: bars missing '{c}'; has {list(df.columns)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def decompose(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-day {overnight, intraday, close_to_close} GROSS returns (no costs).
    overnight[t] = open[t]/close[t-1]-1 ; intraday[t] = close[t]/open[t]-1.
    The first row (no prior close) is dropped."""
    df = _norm_bars(bars, "decompose")
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    out = pd.DataFrame({
        "overnight": open_ / close.shift(1) - 1.0,
        "intraday": close / open_ - 1.0,
        "close_to_close": close / close.shift(1) - 1.0,
    })
    return out.dropna(subset=["overnight", "close_to_close"])


def _metrics(net: pd.Series, ann: int = ANN) -> Dict[str, float]:
    net = net.dropna()
    if net.empty:
        return {"sharpe": 0.0, "cagr": 0.0, "ann_vol": 0.0, "n_days": 0}
    mu, sd = float(net.mean()), float(net.std())
    sharpe = float(mu / sd * np.sqrt(ann)) if sd > 0 else 0.0
    ann_vol = float(sd * np.sqrt(ann))
    growth = float((1.0 + net).prod())
    years = len(net) / ann
    cagr = float(growth ** (1.0 / years) - 1.0) if years > 0 and growth > 0 else 0.0
    return {"sharpe": sharpe, "cagr": cagr, "ann_vol": ann_vol, "n_days": int(len(net))}


def _round_trip(cost_bps_per_side: float) -> float:
    """Daily round-trip cost (fraction): a leg buys then sells once per day -> 2 sides."""
    return 2.0 * (max(0.0, float(cost_bps_per_side)) / 1e4)


def equal_weight_legs(bars_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight-of-symbols GROSS overnight & intraday daily return streams.
    Each day = the cross-sectional mean over symbols that have a value that day (so a
    symbol with a shorter history simply doesn't contribute until it starts)."""
    on_cols, id_cols = {}, {}
    for sym, bars in bars_by_symbol.items():
        d = decompose(bars)
        if d.empty:
            continue
        on_cols[sym] = d["overnight"]
        id_cols[sym] = d["intraday"]
    if not on_cols:
        raise RuntimeError("equal_weight_legs: no symbol produced a decomposition")
    overnight = pd.DataFrame(on_cols).sort_index().mean(axis=1, skipna=True)
    intraday = pd.DataFrame(id_cols).sort_index().mean(axis=1, skipna=True)
    return pd.DataFrame({"overnight": overnight, "intraday": intraday}).dropna()


def _leg_stats(gross: pd.Series, label: str, cost_bps_per_side: float,
               ann: int = ANN) -> LegStats:
    g = _metrics(gross, ann)
    net = (gross - _round_trip(cost_bps_per_side))
    n = _metrics(net, ann)
    return LegStats(label=label, n_days=g["n_days"],
                    gross_sharpe=g["sharpe"], gross_cagr=g["cagr"], ann_vol=g["ann_vol"],
                    net_sharpe=n["sharpe"], net_cagr=n["cagr"],
                    cost_bps_per_side=float(cost_bps_per_side))


def decompose_universe(bars_by_symbol: Dict[str, pd.DataFrame], *,
                       cost_grid: tuple = DEFAULT_COST_GRID,
                       realistic_cost: float = REALISTIC_COST_BPS_PER_SIDE,
                       ann: int = ANN) -> Dict[str, object]:
    """Equal-weight-universe leg stats at the realistic cost + a cost-grid sweep of the
    net overnight Sharpe (the cost cliff) + per-symbol overnight/intraday net Sharpe."""
    legs = equal_weight_legs(bars_by_symbol)
    overnight = _leg_stats(legs["overnight"], "overnight_universe", realistic_cost, ann)
    intraday = _leg_stats(legs["intraday"], "intraday_universe", realistic_cost, ann)
    cost_cliff = {float(c): _metrics(legs["overnight"] - _round_trip(c), ann)["sharpe"]
                  for c in cost_grid}
    per_symbol: Dict[str, Dict[str, float]] = {}
    for sym, bars in bars_by_symbol.items():
        d = decompose(bars)
        if d.empty:
            continue
        on = _metrics(d["overnight"] - _round_trip(realistic_cost), ann)
        idy = _metrics(d["intraday"] - _round_trip(realistic_cost), ann)
        per_symbol[sym] = {"overnight_net_sharpe": on["sharpe"],
                           "intraday_net_sharpe": idy["sharpe"],
                           "overnight_gross_cagr": _metrics(d["overnight"], ann)["cagr"],
                           "n_days": on["n_days"]}
    return {"overnight": overnight, "intraday": intraday,
            "cost_cliff": cost_cliff, "per_symbol": per_symbol,
            "universe": list(bars_by_symbol.keys())}


def overnight_intraday_verdict(bars_by_symbol: Dict[str, pd.DataFrame], *,
                               cost_grid: tuple = DEFAULT_COST_GRID,
                               realistic_cost: float = REALISTIC_COST_BPS_PER_SIDE,
                               paper_sr_floor: float = PAPER_SR_FLOOR,
                               ann: int = ANN) -> OvernightIntradayVerdict:
    """Run the frozen pre-registered net-of-cost test and return PASS/KILL."""
    d = decompose_universe(bars_by_symbol, cost_grid=cost_grid,
                           realistic_cost=realistic_cost, ann=ann)
    on: LegStats = d["overnight"]
    idy: LegStats = d["intraday"]

    c1 = on.net_sharpe >= paper_sr_floor
    c2 = on.net_cagr > 0.0
    c3 = on.net_sharpe > idy.net_sharpe
    if c1 and c2 and c3:
        verdict = "PASS"
        reason = (f"overnight net Sharpe {on.net_sharpe:.2f} >= {paper_sr_floor:.2f}, "
                  f"net CAGR {on.net_cagr:+.2%} > 0, and beats intraday "
                  f"({idy.net_sharpe:.2f}) at {realistic_cost:.1f}bps/side")
    else:
        verdict = "KILL"
        fails = []
        if not c1:
            fails.append(f"net Sharpe {on.net_sharpe:.2f} < floor {paper_sr_floor:.2f}")
        if not c2:
            fails.append(f"net CAGR {on.net_cagr:+.2%} <= 0")
        if not c3:
            fails.append(f"does not beat intraday ({on.net_sharpe:.2f} <= {idy.net_sharpe:.2f})")
        reason = (f"overnight premium does not clear realistic cost "
                  f"({realistic_cost:.1f}bps/side): " + "; ".join(fails))

    return OvernightIntradayVerdict(
        verdict=verdict, reason=reason, registration_id=REGISTRATION_ID,
        realistic_cost_bps_per_side=float(realistic_cost), paper_sr_floor=float(paper_sr_floor),
        overnight=on, intraday=idy, cost_cliff=d["cost_cliff"],
        per_symbol=d["per_symbol"], universe=d["universe"])
