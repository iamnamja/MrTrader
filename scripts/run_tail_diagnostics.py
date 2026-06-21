"""
run_tail_diagnostics.py — Alpha-v10 GL-1 runner: tail / co-crash diagnostics for the multi-premia
book (trend, futures carry, futures xs-momentum, VIX-VRP). Decides VRP in/out + defensive-sleeve.

    PYTHONPATH=. venv/Scripts/python scripts/run_tail_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd


def _spy_returns() -> pd.Series:
    """SPY daily returns from the local daily cache (covers the ~2007+ sleeve overlap)."""
    p = ROOT / "data" / "cache" / "daily" / "SPY.parquet"
    df = pd.read_parquet(p)
    close = df["close"] if "close" in df.columns else df.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.sort_index().pct_change(fill_method=None).dropna().rename("spy")


def main() -> None:
    from scripts.walkforward.sleeves import build_sleeve, live_trend_book_returns
    from app.research import tail_diagnostics as td

    print("Building the four premia sleeve return series...")
    trend = live_trend_book_returns().rename("trend")
    carry = build_sleeve("futures_carry").returns.rename("carry")
    xsmom = build_sleeve("futures_xsmom").returns.rename("xsmom")
    vrp = build_sleeve("vix_vrp").returns.rename("vrp")
    sleeves = pd.concat([trend, carry, xsmom, vrp], axis=1, join="inner").dropna()
    spy = _spy_returns()
    print(f"  aligned sleeves: {len(sleeves)} days "
          f"({sleeves.index[0].date()} -> {sleeves.index[-1].date()}); SPY {len(spy)} days.")

    res = td.run_tail_diagnostics(sleeves, spy, vrp_col="vrp")

    print("\n=== UNCONDITIONAL vs STRESS-CONDITIONAL CORRELATION ===")
    print(f"  unconditional avg pairwise corr       {res.uncond_corr_avg:+.2f}")
    print(f"  SPY worst-5% avg corr  full-history    {res.spy_exceedance[0.05]:+.2f}   "
          f"(> 0.60 = 'one bet')")
    print(f"  SPY worst-5% avg corr  post-2015       {res.spy_exceedance_post2015[0.05]:+.2f}")
    print(f"  SPY worst-10% avg corr full / post     {res.spy_exceedance[0.10]:+.2f} / "
          f"{res.spy_exceedance_post2015[0.10]:+.2f}")
    print(f"  book worst-5% avg corr (biased-low)    {res.book_exceedance[0.05]:+.2f}   "
          f"(report-only; collider-biased)")

    print("\n=== DOWN-vs-UP BETA vs SPY (down>up = negative convexity) ===")
    for c, b in res.sleeve_beta.items():
        print(f"  {c:8s} down {b['down_beta']:+.2f}  up {b['up_beta']:+.2f}  "
              f"asym {b['asymmetry']:+.2f}")
    bw, bo = res.book_with_vrp_beta, res.book_without_vrp_beta
    p, lo, hi = res.core_asymmetry_ci
    print(f"  book(+VRP) down {bw['down_beta']:+.2f}  up {bw['up_beta']:+.2f}  asym {bw['asymmetry']:+.2f}")
    print(f"  book(-VRP) down {bo['down_beta']:+.2f}  up {bo['up_beta']:+.2f}  asym {p:+.2f} "
          f"[90% CI {lo:+.2f}, {hi:+.2f}]")
    print(f"  VRP deepens the book's loss in {res.vrp_crises_worse}/{res.vrp_crises_total} crises "
          f"(book-delta, weight-honest).")

    print("\n=== CRISIS-WINDOW CUMULATIVE RETURNS (per sleeve) ===")
    if not res.crisis_table.empty:
        ct = res.crisis_table.copy()
        for c in [c for c in ct.columns if c != "days"]:
            ct[c] = (ct[c] * 100).round(1)
        print(ct.to_string())

    print("\n=== VERDICT ===")
    for note in res.verdict_notes:
        print(f"  - {note}")
    print(f"\n  one_bet={res.one_bet}  vrp_worsens_tail={res.vrp_worsens_tail}  "
          f"defensive_sleeve_needed={res.defensive_sleeve_needed}")


if __name__ == "__main__":
    main()
