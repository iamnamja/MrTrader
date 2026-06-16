"""P1-2 — Kelly / vol-target sizing analysis for the trend sleeve.

The live trend book's `live_trend_book_returns()` series is the book at 100% GROSS
(TSMOMConfig.max_gross=1.0). `pm.trend_allocation_pct` (today 0.25) scales that gross
into NAV, so live trend return_t = alloc * r_t. Choosing the allocation == choosing the
leverage applied to this unit series. This script reports Kelly + vol-target allocations
from the REAL deep-history series. Report-only; changes nothing.
"""
from __future__ import annotations

from datetime import date
import numpy as np

from scripts.walkforward.sleeves import live_trend_book_returns

ANN = 252


def stats(r):
    mu_d, sd_d = float(r.mean()), float(r.std(ddof=1))
    mu_a, sd_a = mu_d * ANN, sd_d * np.sqrt(ANN)
    sharpe = mu_a / sd_a if sd_a else float("nan")
    eq = (1.0 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min())
    cagr = float(eq.iloc[-1] ** (ANN / len(r)) - 1.0)
    return dict(mu_a=mu_a, sd_a=sd_a, sharpe=sharpe, mdd=mdd, cagr=cagr,
                mu_d=mu_d, sd_d=sd_d, n=len(r))


def levered_mdd(r, k):
    """Max drawdown of the daily-rebalanced book at leverage k on the unit series."""
    eq = (1.0 + k * r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def main():
    r = live_trend_book_returns(start=date(2007, 1, 1))
    s = stats(r)
    print("=" * 72)
    print("TREND BOOK @ 100% GROSS (the unit series; alloc=1.0)")
    print("=" * 72)
    print(f"  window         : {r.index[0].date()} -> {r.index[-1].date()}  ({s['n']} days, ~{s['n']/ANN:.1f}y)")
    print(f"  annualized mean: {s['mu_a']*100:+.2f}%")
    print(f"  annualized vol : {s['sd_a']*100:.2f}%")
    print(f"  Sharpe         : {s['sharpe']:.3f}")
    print(f"  CAGR           : {s['cagr']*100:+.2f}%")
    print(f"  max drawdown   : {s['mdd']*100:.2f}%")

    # Kelly fraction on the unit series: f* = mu/var (per-period == annualized ratio).
    kelly = s['mu_a'] / (s['sd_a'] ** 2)
    print()
    print("=" * 72)
    print("KELLY (continuous, on the trend unit series)")
    print("=" * 72)
    print(f"  full Kelly  f* = mu/sigma^2 = {kelly:.2f}x gross  ({kelly*100:.0f}% of NAV)")
    print(f"  half Kelly       = {kelly/2:.2f}x gross  ({kelly/2*100:.0f}% of NAV)")
    print(f"  quarter Kelly    = {kelly/4:.2f}x gross  ({kelly/4*100:.0f}% of NAV)")
    print("  (full Kelly on a low-vol high-Sharpe book is enormous + assumes the")
    print("   in-sample Sharpe is the true forward Sharpe -> NOT actionable directly.)")

    print()
    print("=" * 72)
    print("VOLATILITY-TARGET ALLOCATIONS  (alloc = target_vol / 100%-gross vol)")
    print("=" * 72)
    print(f"  {'target vol':>10} | {'alloc':>7} | {'realized trend vol':>18} | {'~CAGR contrib':>13} | {'~maxDD':>8}")
    print("  " + "-" * 66)
    for V in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
        alloc = V / s['sd_a']
        mdd_k = levered_mdd(r, alloc)
        cagr_k = alloc * s['mu_a']  # approx (small-return linear)
        flag = "  <- 80% schema cap exceeded" if alloc > 0.80 else ""
        print(f"  {V*100:>9.0f}% | {alloc*100:>6.1f}% | {alloc*s['sd_a']*100:>17.2f}% | {cagr_k*100:>12.2f}% | {mdd_k*100:>7.1f}%{flag}")

    print()
    print("=" * 72)
    print("CURRENT vs CANDIDATES (standalone trend contribution to the book)")
    print("=" * 72)
    for label, alloc in [("current (25%)", 0.25), ("40% (old)", 0.40),
                         ("50%", 0.50), ("60%", 0.60), ("80% (cap)", 0.80)]:
        print(f"  {label:>14}: vol {alloc*s['sd_a']*100:5.2f}%  | CAGR ~{alloc*s['mu_a']*100:+5.2f}%  | maxDD {levered_mdd(r, alloc)*100:6.1f}%")


if __name__ == "__main__":
    main()
