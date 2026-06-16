"""P1-3 — Credit-overlay SHADOW monitor (report-only; touches no live trading code).

The credit governor (HYG/IEF de-risk overlay, `pm.credit_governor_enabled`, default OFF) is a
deterministic function of settled HYG/IEF closes, which `data/macro/macro_history.parquet`
accumulates daily. So we can faithfully reconstruct what it WOULD have done over any window —
no need to deploy a logger into the live cycle. This script reconstructs, over a shadow window:

  * the credit overlay's daily as-applied multiplier (HYG/IEF vs trailing MA),
  * the LIVE VIX crash-governor multiplier (for the marginal comparison), and
  * the trend book's return under three policies — none / VIX-only (live today) / VIX+credit —
    and reports the credit overlay's MARGINAL effect plus its fire episodes.

Used by the scheduled P1-3 check-in. `python -m scripts.shadow_credit_governor [--start YYYY-MM-DD]`.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from app.data.macro_history import load_macro_history
from app.strategy.credit_curve_governor import CreditGovernorConfig, credit_multiplier
from app.strategy.crash_governor import VixTermGovernorConfig, vix_term_multiplier
from scripts.walkforward.sleeves import live_trend_book_returns

ANN = 252
OVERLAY_FLOOR = 0.25  # matches trend_sleeve._OVERLAY_DERISK_FLOOR

# Shadow window opened when P1-3 began (Alpha-v9 Phase 1).
SHADOW_START = date(2026, 6, 16)


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[["date", col]].dropna().copy()
    s.index = pd.to_datetime(s["date"])
    return s[col].astype(float).sort_index()


def _stats(r: pd.Series) -> dict:
    if len(r) < 2 or r.std(ddof=1) == 0:
        return dict(cum=float((1 + r).prod() - 1), vol=float("nan"),
                    sharpe=float("nan"), mdd=float("nan"), n=len(r))
    mu_a, sd_a = r.mean() * ANN, r.std(ddof=1) * np.sqrt(ANN)
    eq = (1 + r).cumprod()
    return dict(cum=float(eq.iloc[-1] - 1), vol=float(sd_a),
                sharpe=float(mu_a / sd_a), mdd=float((eq / eq.cummax() - 1).min()),
                n=len(r))


def run(start: date, end: date | None = None) -> dict:
    end = end or datetime.now(timezone.utc).date()
    mh = load_macro_history()
    if mh.empty:
        raise RuntimeError("macro_history.parquet is empty — nothing to shadow")

    hyg, ief = _series(mh, "hyg"), _series(mh, "ief")
    vix, vix3m = _series(mh, "vix"), _series(mh, "vix3m")

    # Reconstruct as-applied (shift(1) PIT) multipliers over ALL available history, then slice.
    credit = credit_multiplier(hyg, ief, CreditGovernorConfig())
    vixm = vix_term_multiplier(vix, vix3m, VixTermGovernorConfig())

    r = live_trend_book_returns(start=date(2007, 1, 1))
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    idx = r.index[(r.index >= lo) & (r.index <= hi)]
    r = r.loc[idx]
    credit = credit.reindex(idx).fillna(1.0)
    vixm = vixm.reindex(idx).fillna(1.0)
    combined = np.maximum(OVERLAY_FLOOR, vixm * credit)

    none_r = r
    vix_only = r * vixm
    vix_credit = r * combined

    fired = credit < 1.0
    n_fired = int(fired.sum())
    print("=" * 74)
    print(f"CREDIT-OVERLAY SHADOW  window {start} -> {end}  ({len(r)} trading days)")
    print("=" * 74)
    if len(r) == 0:
        print("  No trend-book trading days in the window yet — check back later.")
        return {"n_days": 0}

    print(f"  credit overlay fired on {n_fired}/{len(r)} days ({n_fired/len(r)*100:.0f}%)")
    print(f"  VIX governor (live)  fired on {int((vixm < 1.0).sum())}/{len(r)} days")
    if n_fired:
        ep = r.index[fired]
        print(f"  credit-stress dates: {ep[0].date()} .. {ep[-1].date()}")
        ratio = (hyg / ief).reindex(idx)
        for d in ep[:10]:
            print(f"     {d.date()}  HYG/IEF={ratio.get(d, float('nan')):.4f}  -> x{credit.get(d):.2f}")
        if n_fired > 10:
            print(f"     ... (+{n_fired-10} more)")

    print()
    print(f"  {'policy':>22} | {'cum ret':>9} | {'ann vol':>8} | {'Sharpe':>7} | {'maxDD':>7}")
    print("  " + "-" * 64)
    for label, rr in [("none (no overlay)", none_r),
                      ("VIX-only (LIVE today)", vix_only),
                      ("VIX+credit (SHADOW)", vix_credit)]:
        s = _stats(rr)
        print(f"  {label:>22} | {s['cum']*100:>+8.2f}% | {s['vol']*100:>7.2f}% | "
              f"{s['sharpe']:>7.3f} | {s['mdd']*100:>6.1f}%")

    # Marginal credit effect (vs the live VIX-only policy) over the window.
    marg = (vix_credit - vix_only)
    print()
    print(f"  marginal credit effect over window: {marg.sum()*100:+.3f}% cumulative "
          f"({'help' if marg.sum() >= 0 else 'drag'})")
    if n_fired == 0:
        print("  VERDICT: credit overlay INERT this window (no stress) — no evidence yet; keep shadowing.")
    return {"n_days": len(r), "n_fired": n_fired,
            "marginal_cum": float(marg.sum()),
            "vix_only_sharpe": _stats(vix_only)["sharpe"],
            "vix_credit_sharpe": _stats(vix_credit)["sharpe"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=SHADOW_START.isoformat())
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    start = date.fromisoformat(a.start)
    end = date.fromisoformat(a.end) if a.end else None
    run(start, end)


if __name__ == "__main__":
    main()
