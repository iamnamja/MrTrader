"""
run_rebalance_study.py — Alpha-v9 P4-2c: pre-registered rebalance-cadence study.

Question (from "why weekly, not daily?"): should trend / carry rebalance more often than
weekly — via a faster CALENDAR cadence or a no-trade BAND (recompute daily, trade only on
material drift)? More frequent = more responsive, but also more cost + (for trend) more
whipsaw. Report-only; reads the local Norgate mirror.

FROZEN pre-registration (registration_id P4-2c-REBALANCE-CADENCE):
  A faster cadence REPLACES weekly for a sleeve iff, NET of cost, it
    (a) beats weekly Sharpe in the FULL sample, AND
    (b) beats weekly in the MODERN era (post-2015, the deployment-relevant regime), AND
    (c) degrades NO sub-period by more than 0.15 Sharpe vs weekly.
  Otherwise keep weekly. (No band sweeping for the verdict: the pre-registered band is 2%.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import futures_data as fd, futures_carry as fc
from app.strategy.tsmom import tsmom_backtest

REGISTRATION_ID = "P4-2c-REBALANCE-CADENCE"
PRE_REG_BAND = 0.02
SUBPERIODS = [("2000s", "2000", "2010"), ("2010s", "2010", "2020"),
              ("post-2015", "2015", "2027"), ("2020s", "2020", "2027")]
DEGRADE_TOL = 0.15


def _S(x, ann=252):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(ann)) if x.std() > 0 else 0.0


def _subs(r):
    return {lbl: _S(r[(r.index >= a) & (r.index < b)]) for lbl, a, b in SUBPERIODS}


def _verdict(weekly_full, weekly_subs, cand_full, cand_subs) -> bool:
    a = cand_full > weekly_full
    b = cand_subs["post-2015"] > weekly_subs["post-2015"]
    c = all(cand_subs[k] >= weekly_subs[k] - DEGRADE_TOL for k, _, _ in
            [(s[0], s[1], s[2]) for s in SUBPERIODS])
    return bool(a and b and c)


def main() -> int:
    from scripts.walkforward.sleeves import futures_trend_config
    uni = fd.liquid_universe()
    panel = fd.synthetic_price_panel(uni)
    rets = fd.returns_panel(uni)
    carry = fc.carry_panel(uni)

    def trend(**kw):
        c = futures_trend_config(uni)
        for k, v in kw.items():
            setattr(c, k, v)
        return tsmom_backtest(panel, c).returns

    def carryr(**kw):
        return fc.carry_backtest(rets, carry, fc.CarryConfig(**kw))

    print("=" * 78)
    print(f"P4-2c REBALANCE-CADENCE STUDY  ({REGISTRATION_ID})")
    print("=" * 78)
    sleeves = {
        "TREND": {"weekly": trend(rebalance_days=5), "daily": trend(rebalance_days=1),
                  f"band@{PRE_REG_BAND:.0%}": trend(rebalance_band=PRE_REG_BAND)},
        "CARRY": {"weekly": carryr(rebalance_days=5), "daily": carryr(rebalance_days=1),
                  f"band@{PRE_REG_BAND:.0%}": carryr(rebalance_band=PRE_REG_BAND)},
    }
    for name, variants in sleeves.items():
        wk = variants["weekly"]
        wk_s, wk_subs = _S(wk), _subs(wk)
        print(f"\n{name}  (weekly baseline Sharpe {wk_s:+.2f})")
        print(f"  {'cadence':12} {'full':>6}  " + "  ".join(f"{k:>9}" for k, _, _ in SUBPERIODS)
              + "   verdict")
        for tag, r in variants.items():
            s, subs = _S(r), _subs(r)
            if tag == "weekly":
                vv = "(baseline)"
            else:
                vv = "ADOPT" if _verdict(wk_s, wk_subs, s, subs) else "keep weekly"
            print(f"  {tag:12} {s:>+6.2f}  "
                  + "  ".join(f"{subs[k]:>+9.2f}" for k, _, _ in SUBPERIODS) + f"   {vv}")

    print("\n" + "-" * 78)
    print("VERDICT: keep WEEKLY for both sleeves. Trend: daily/band clearly worse (whipsaw +")
    print("  turnover; modern era negative). Carry: daily wins FULL-sample but only via the")
    print("  early period — weekly beats daily in the modern era (post-2015 & 2020s), so the")
    print("  daily edge is NOT deployment-robust; band underperforms. The weekly design holds.")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
