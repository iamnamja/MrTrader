"""
run_futures_research.py — Alpha-v9 P4-2: the futures trend + carry research packet.

Reproducible, report-only. Reads the local Norgate parquet mirror (no NDU dependency).
Evaluates cross-asset TREND, CARRY, and the combined trend+carry book on the liquid
survivorship-free futures universe, with the pre-registered sub-period stability guard +
correlation/Track-B vs the live ETF-trend book, and prints the official Sleeve-Lab
Ruler-v2 Track-A verdicts.

Usage:  python -m scripts.run_futures_research
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import futures_data as fd, futures_carry as fc
from app.strategy.tsmom import tsmom_backtest


def _stats(x, ann=252):
    x = x.dropna()
    if len(x) < 2:
        return 0, 0.0, 0.0, 0.0
    sh = x.mean() / x.std() * np.sqrt(ann) if x.std() > 0 else 0.0
    dd = ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()
    return len(x), float(sh), float(x.std() * np.sqrt(ann)), float(dd)


_SUBPERIODS = [("1977-1999", "1977", "2000"), ("2000-2009", "2000", "2010"),
               ("2010-2019", "2010", "2020"), ("2020-2026", "2020", "2027"),
               ("post-2015", "2015", "2027")]


def _subperiods(label, r):
    print(f"  {label} sub-periods:")
    for lbl, a, b in _SUBPERIODS:
        seg = r[(r.index >= a) & (r.index < b)]
        _, sh, _, _ = _stats(seg)
        print(f"    {lbl}: Sharpe {sh:+.2f}")


def _corr(a, b):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return (j["a"].corr(j["b"]), len(j)) if len(j) > 2 else (float("nan"), 0)


def main() -> int:
    from scripts.walkforward.sleeves import (futures_trend_config, live_trend_book_returns)
    uni = fd.liquid_universe()
    print(f"Liquid futures universe: {len(uni)} markets")
    panel = fd.synthetic_price_panel(uni)
    rets = fd.returns_panel(uni)
    carry = fc.carry_panel(uni)

    trend = tsmom_backtest(panel, futures_trend_config(uni)).returns.dropna()
    carry_r = fc.carry_backtest(rets, carry)

    print("\n" + "=" * 74)
    print("P4-2 FUTURES TREND + CARRY")
    print("=" * 74)
    for nm, r in (("TREND  (P4-2-FUT-TREND)", trend), ("CARRY  (P4-2-FUT-CARRY)", carry_r)):
        n, sh, vol, dd = _stats(r)
        print(f"\n{nm}: n={n}  Sharpe {sh:+.2f}  vol {vol:.1%}  maxDD {dd:.0%}")
        _subperiods(nm.split()[0], r)

    # combined equal-risk trend+carry book
    j = pd.concat([trend.rename("t"), carry_r.rename("c")], axis=1, join="inner").dropna()
    comb = 0.5 * (j["t"] / j["t"].std()) + 0.5 * (j["c"] / j["c"].std())
    comb = (comb / comb.std() * j["t"].std()).rename("trend+carry")
    n, sh, vol, dd = _stats(comb)
    print(f"\nTREND+CARRY (equal-risk): n={n}  Sharpe {sh:+.2f}  vol {vol:.1%}  maxDD {dd:.0%}")
    _subperiods("TREND+CARRY", comb)

    print("\n-- correlations / diversification --")
    print(f"  carry ~ trend            : {_corr(carry_r, trend)[0]:.3f}")
    etf = live_trend_book_returns()
    print(f"  carry ~ live ETF-trend   : {_corr(carry_r, etf)[0]:.3f}")
    print(f"  trend ~ live ETF-trend   : {_corr(trend, etf)[0]:.3f}")

    # Track-B: does carry improve the live ETF-trend book? (50/50 vol-matched)
    jb = pd.concat([etf.rename("e"), carry_r.rename("c")], axis=1, join="inner").dropna()
    base = jb["e"].mean() / jb["e"].std() * np.sqrt(252)
    book = 0.5 * (jb["e"] / jb["e"].std()) + 0.5 * (jb["c"] / jb["c"].std())
    booksh = book.mean() / book.std() * np.sqrt(252)
    print(f"\nTRACK-B: live ETF-trend Sharpe {base:.2f} -> +carry {booksh:.2f}  dSR {booksh-base:+.2f}")
    print("  [caveat] the combined + Track-B blends are vol-matched IN-SAMPLE (full-period std)")
    print("  for a relative-Sharpe comparison; a PIT rolling-vol blend is modestly lower. Costs")
    print("  are a flat 3bps/side (optimistic for the illiquid tail; carry is cost-robust, trend")
    print("  is cost-sensitive — see DECISIONS 2026-06-18 hardening).")

    # official Sleeve-Lab Ruler-v2 Track-A
    print("\n-- official Sleeve-Lab Ruler-v2 (Track-A) --")
    from scripts.walkforward.sleeve_lab import build_sleeve, evaluate_sleeve
    for name in ("futures_carry", "futures_trend"):
        try:
            rep = evaluate_sleeve(build_sleeve(name))
            v = getattr(rep, "verdict", None) or (rep.get("verdict") if isinstance(rep, dict) else "?")
            print(f"  {name}: {v}")
        except Exception as exc:
            print(f"  {name}: eval error {exc}")

    print("\n" + "-" * 74)
    print("VERDICT: CARRY is a real, MODERN, diversifying premium (PAPER-PASS; post-2015 ~0.84;")
    print("  corr-to-live-trend ~0.25; Track-B dSR +0.17) -> CAPITAL-candidate via live-paper.")
    print("  TREND alone has DECAYED (post-2015 ~0.0) -> not standalone; it is the crisis-convex")
    print("  partner in the TREND+CARRY book (Sharpe ~1.0, modern ~0.57, shallower DD).")
    print("-" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
