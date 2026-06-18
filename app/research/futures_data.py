"""
futures_data.py — Alpha-v9 P4-2: correct return panel + liquid universe for Norgate futures.

THE correctness issue for continuous futures (deep-dived 2026-06-18):
  * Norgate `&MKT_CCB` is DIFFERENCE (Panama) back-adjusted — 27/105 markets have NEGATIVE
    or ZERO back-adjusted close over long history, so `pct_change(CCB)` is GARBAGE (sign
    flips / inf). The correct daily return is the back-adjusted POINT change divided by the
    prior UNADJUSTED (actual) price level:  r_t = (CCB_t - CCB_{t-1}) / Unadj_{t-1}.
    (This is why we mirror BOTH series.)
  * Near-zero denominator blowup: when the actual price collapses toward zero (e.g. WTI/CL
    on 2020-04-20 went negative), Unadj_{t-1} is tiny and r_t explodes (CL shows a -306%
    day). We WINSORIZE daily returns to +/- RETURN_CAP (default 0.5 = 50%/day, already far
    beyond any real futures move) to neutralize this representation artifact.

To feed the existing TSMOM engine (which does pct_change internally) a CORRECT series, we
build a SYNTHETIC positive price = 100 * cumprod(1 + r): its pct_change is exactly r, and it
never goes negative, so all momentum/vol math is well-defined.

Universe (objective, no hand-picking → no selection bias):
  * exclude MICRO duplicates (M-prefixed when the full-size base exists, e.g. MES->ES)
  * require min history (min_days) + recent liquidity (median volume)
  * exclude ultra-low-vol STIRs (ann vol < MIN_ANN_VOL) — 3M SOFR/SONIA etc. would lever
    up absurdly under inverse-vol sizing and are a different instrument class.
Reads ONLY the local parquet mirror (no NDU dependency).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.data import norgate_provider as ng

RETURN_CAP = 0.5            # winsorize daily returns to +/-50% (near-zero-denominator guard)
MIN_DAYS = 2500            # ~10y of history for a robust walk-forward
MIN_MED_VOL = 5000         # recent median daily contract volume
MIN_ANN_VOL = 0.02         # exclude ultra-low-vol STIRs (3M SOFR/SONIA ~ <1% ann vol)
ANN = 252


def true_returns(market: str, *, cap: float = RETURN_CAP) -> pd.Series:
    """Correct daily futures return r_t = (CCB_t - CCB_{t-1}) / Unadj_{t-1}, winsorized to
    +/- cap. Reads the local mirror (back-adjusted + unadjusted)."""
    cb = ng.load_continuous(market, price_type="backadjusted")["close"].astype(float)
    ua = ng.load_continuous(market, price_type="unadjusted")["close"].astype(float)
    df = pd.DataFrame({"cb": cb, "ua": ua}).dropna().sort_index()
    r = df["cb"].diff() / df["ua"].shift(1)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if cap is not None:
        r = r.clip(-abs(cap), abs(cap))
    return r.rename(market)


def _ann_vol(market: str) -> float:
    r = true_returns(market)
    return float(r.std() * np.sqrt(ANN)) if len(r) > 30 else 0.0


def _is_micro(market: str, all_markets: set) -> bool:
    """A micro/mini duplicate: 'M'-prefixed with the full-size base also present (MES->ES,
    MNQ->NQ, MGC->GC, MCL->CL, M2K->...). Conservative: only when the base exists."""
    return len(market) >= 2 and market[0] == "M" and market[1:] in all_markets


def liquid_universe(*, min_days: int = MIN_DAYS, min_med_vol: float = MIN_MED_VOL,
                    min_ann_vol: float = MIN_ANN_VOL, exclude_micros: bool = True) -> List[str]:
    """Objective liquid, deep-history, non-micro, non-STIR futures universe (from the mirror)."""
    all_m = {f[:-8] for f in os.listdir(ng.CONTINUOUS_DIR) if f.endswith(".parquet")}
    keep = []
    for m in sorted(all_m):
        if exclude_micros and _is_micro(m, all_m):
            continue
        d = ng.load_continuous(m, price_type="unadjusted")
        if len(d) < min_days:
            continue
        med_vol = float(d.tail(60)["volume"].median()) if "volume" in d.columns else 0.0
        if med_vol < min_med_vol:
            continue
        if _ann_vol(m) < min_ann_vol:
            continue
        keep.append(m)
    return keep


def synthetic_price_panel(markets: Optional[Sequence[str]] = None, *,
                          cap: float = RETURN_CAP,
                          start: Optional[str] = None) -> pd.DataFrame:
    """Panel of SYNTHETIC positive prices (100*cumprod(1+r)) for the TSMOM engine — one
    column per market, outer-joined on the union of dates (NaN where a market has no data
    yet). Feeding these to tsmom_backtest recovers the correct winsorized returns."""
    markets = list(markets) if markets is not None else liquid_universe()
    cols: Dict[str, pd.Series] = {}
    for m in markets:
        r = true_returns(m, cap=cap)
        if start is not None:
            r = r[r.index >= pd.Timestamp(start)]
        if len(r) < 30:
            continue
        cols[m] = 100.0 * (1.0 + r).cumprod()
    if not cols:
        raise RuntimeError("synthetic_price_panel: no markets produced returns")
    return pd.DataFrame(cols).sort_index()


def returns_panel(markets: Optional[Sequence[str]] = None, *,
                  cap: float = RETURN_CAP) -> pd.DataFrame:
    """Panel of the correct winsorized daily returns (for carry/diagnostics)."""
    markets = list(markets) if markets is not None else liquid_universe()
    cols = {m: true_returns(m, cap=cap) for m in markets}
    return pd.DataFrame({m: s for m, s in cols.items() if len(s) >= 30}).sort_index()
