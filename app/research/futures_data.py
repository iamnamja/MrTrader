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
VOL_FLOOR_EXCL = 0.005     # backstop: drop pathological near-zero-vol series
ANN = 252

# Short-term interest-rate (STIR) futures — a different instrument class (price ~100-yield,
# tiny vol) that would lever up absurdly under inverse-vol sizing. Excluded by NAME (robust),
# not by a raw vol threshold (which wrongly cut real low-vol BONDS like the 2-Year T-Note).
_STIR_NAME_KEYS = ("sofr", "sonia", "euribor", "eurodollar", "fed fund", "federal funds",
                   "bank bill", "short sterling", "90 day", "90-day", "3 month", "3-month",
                   "1 month", "1-month", "overnight rate", "cash rate")


def _market_names() -> Dict[str, str]:
    """Per-market display names from the mirror metadata ({} if absent)."""
    if not os.path.exists(ng.MARKETS_META):
        return {}
    try:
        md = pd.read_parquet(ng.MARKETS_META)
        return {str(r["market"]): str(r.get("name") or "") for _, r in md.iterrows()}
    except Exception:
        return {}


def _is_stir(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _STIR_NAME_KEYS)


def _is_micro_name(name: str) -> bool:
    """Micro/mini DUPLICATE of a full-size contract — by name. 'micro' anywhere, or 'mini'
    that is NOT 'e-mini' (E-mini ES/NQ/RTY are the STANDARD liquid contracts we KEEP)."""
    n = name.lower()
    return ("micro" in n) or ("mini" in n and "e-mini" not in n)


def true_returns(market: str, *, cap: float = RETURN_CAP) -> pd.Series:
    """Correct daily futures return r_t = (CCB_t - CCB_{t-1}) / Unadj_{t-1}, winsorized to
    +/- cap. Reads the local mirror (back-adjusted + unadjusted).

    NEGATIVE-DENOMINATOR GUARD (hardening 2026-06-18): when the prior UNADJUSTED price is
    <= 0 (e.g. WTI/CL on 2020-04-21, prior actual price -37.63), dividing a real point move
    by a negative level FLIPS the sign of the return — and the result lands INSIDE the +/-cap
    band, so winsorization can't catch it. We therefore NaN-out (and drop) any day whose
    prior unadjusted level is non-positive; those days are untradeable artifacts anyway."""
    cb = ng.load_continuous(market, price_type="backadjusted")["close"].astype(float)
    ua = ng.load_continuous(market, price_type="unadjusted")["close"].astype(float)
    df = pd.DataFrame({"cb": cb, "ua": ua}).dropna().sort_index()
    den = df["ua"].shift(1)
    r = df["cb"].diff() / den.where(den > 0)        # non-positive prior price -> NaN -> dropped
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
                    exclude_micros: bool = True, exclude_stirs: bool = True) -> List[str]:
    """Objective liquid, deep-history, non-micro, non-STIR futures universe (from the mirror).

    Micros/STIRs are classified by NAME (mirror metadata) when available — robust to ticker
    quirks (M2K/MHI escape the 'M'-prefix heuristic; the old ann-vol STIR cut wrongly removed
    real low-vol bonds like the 2-Year T-Note). Falls back to the 'M'-prefix heuristic for
    micros if names are unavailable.

    NOTE (known limitation — survivorship/selection): membership uses CURRENT (last-60d)
    liquidity + full-history length, then is applied to all (incl. OOS) folds. For major
    futures this bias is mild (they were liquid throughout) and the carry edge is independently
    confirmed in the modern regime where the universe is legitimately liquid; see the
    `as_of`-quantification in scripts/run_futures_research.py / DECISIONS 2026-06-18."""
    all_m = {f[:-8] for f in os.listdir(ng.CONTINUOUS_DIR) if f.endswith(".parquet")}
    names = _market_names()
    keep = []
    for m in sorted(all_m):
        nm = names.get(m, "")
        if exclude_micros and (_is_micro_name(nm) if nm else _is_micro(m, all_m)):
            continue
        if exclude_stirs and nm and _is_stir(nm):
            continue
        d = ng.load_continuous(m, price_type="unadjusted")
        if len(d) < min_days:
            continue
        med_vol = float(d.tail(60)["volume"].median()) if "volume" in d.columns else 0.0
        if med_vol < min_med_vol:
            continue
        if _ann_vol(m) < VOL_FLOOR_EXCL:        # backstop only (STIRs handled by name)
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
