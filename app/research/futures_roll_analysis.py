"""
futures_roll_analysis.py — Alpha-v10 R1.3: empirically compare roll-timing rules on Norgate history.

Answers "fixed calendar vs FND floor vs dynamic (liquidity) roll?" with DATA instead of live-shadow
accrual. For each historical roll cycle it locates:
  - `liquidity_roll` — the date open interest actually migrated to the next contract (front-by-OI flips).
  - `fixed_roll`     — the backtested rule: 5 days before the SCHEDULED (day-15) expiry.
  - `fnd`            — the First-Notice-Day estimate (futures_roll_policy), physical markets only.
and reports the gaps. Interpretation of `fixed_minus_liq` (days):
  >0 → the fixed rule rolls LATE (holds past where liquidity moved → execution drag; dynamic helps).
  <0 → the fixed rule rolls EARLY (into a still-thinner next contract).
  ≈0 → fixed ≈ liquidity → a fixed calendar rule is fine; dynamic adds little.

Pure analysis over the mirrored data (no live calls, nothing traded).
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from app.data import norgate_provider as ng
from app.live_trading import futures_roll_policy as rp
from app.research.futures_carry import ROLL_BUFFER_DAYS, _scheduled_expiry


def _contract_ym(contract: str):
    """'ES-2026U' -> (2026, 9) via the scheduled-expiry parser; None if unparseable."""
    exp = _scheduled_expiry(contract)
    return (int(exp.year), int(exp.month)) if pd.notna(exp) else None


def liquidity_front(market: str, metric: str = "open_interest", *, nearest_k: int = 4) -> pd.Series:
    """date -> the front contract by `metric` (open_interest default; 'volume' cross-check). Restricted
    to the `nearest_k` un-expired contracts by SCHEDULED expiry so a far-dated OI hump (e.g. crude's
    December calendar contracts) can't masquerade as the front — the roll is front→next, not front→far.
    A change of this front is a liquidity roll."""
    df = ng.load_contracts(market)[["date", "contract", metric]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["exp"] = df["contract"].map(_scheduled_expiry)
    df = df.dropna(subset=["exp"])
    df = df[(df["exp"] > df["date"]) & (df[metric] > 0)]
    if df.empty:
        return pd.Series(dtype=object, name=market)
    df = df.sort_values(["date", "exp"])
    df = df[df.groupby("date")["exp"].rank(method="first") <= nearest_k]   # nearest-K by expiry only
    idx = df.groupby("date")[metric].idxmax()
    return df.loc[idx].set_index("date")["contract"].sort_index().rename(market)


def roll_timing_comparison(market: str, root: Optional[str] = None, *,
                           metric: str = "open_interest", last_n: int = 24) -> pd.DataFrame:
    """Per-roll-cycle table: liquidity_roll vs fixed_roll vs fnd (+ signed day gaps). Last `last_n`
    cycles (most recent). `root` (defaults to `market`) drives the FND settlement class."""
    root = (root or market).upper()
    front = liquidity_front(market, metric)
    rows = []
    prevc = None
    for d, c in front.items():
        if prevc is not None and c != prevc:
            ym = _contract_ym(prevc)                        # we ROLLED OUT of prevc on date d
            if ym:
                y, m = ym
                sched = pd.Timestamp(year=y, month=m, day=15).date()
                fixed = (pd.Timestamp(sched) - pd.Timedelta(days=ROLL_BUFFER_DAYS)).date()
                fnd = rp.first_notice_day_estimate(y, m, root)
                liq = d.date()
                rows.append({
                    "cycle": prevc, "settlement": rp.settlement(root), "liquidity_roll": liq,
                    "fixed_roll": fixed, "fnd": fnd,
                    "fixed_minus_liq": (fixed - liq).days,
                    "fnd_minus_liq": ((fnd - liq).days if fnd else None),
                })
        prevc = c
    return pd.DataFrame(rows).tail(last_n).reset_index(drop=True)


def summarize(markets: List[str], *, metric: str = "open_interest", last_n: int = 24) -> pd.DataFrame:
    """One row per market: median gaps + settlement class, over the last `last_n` liquidity rolls.
    `median_fixed_minus_liq` is the headline: how many days the fixed rule lags liquidity migration."""
    out = []
    for m in markets:
        try:
            t = roll_timing_comparison(m, metric=metric, last_n=last_n)
        except FileNotFoundError:
            continue
        if t.empty:
            out.append({"market": m, "n_cycles": 0})
            continue
        fnd_gaps = t["fnd_minus_liq"].dropna()
        out.append({
            "market": m, "settlement": t["settlement"].iloc[0], "n_cycles": len(t),
            "median_fixed_minus_liq": float(t["fixed_minus_liq"].median()),
            "median_abs_fixed_minus_liq": float(t["fixed_minus_liq"].abs().median()),
            "median_fnd_minus_liq": (float(fnd_gaps.median()) if len(fnd_gaps) else None),
        })
    return pd.DataFrame(out)
