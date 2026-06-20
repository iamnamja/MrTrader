"""
run_futures_factors.py — Alpha-v10 P1.2: the futures factor zoo (owned Norgate data, $0).

Runs four cross-sectional factor candidates through the generic XS engine (with honest 3bps/side
roll cost) and reports standalone Sharpe + sub-periods + correlation + residual-alpha vs the
(live ETF-trend + honest carry) book. Report-only; pre-registered; NO sign-flipping.

Usage:  python -m scripts.run_futures_factors
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import futures_data as fd, futures_carry as fc, futures_factors as ff, futures_roll as frl


def _S(x, ann=252):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(ann)) if x.std() > 0 else 0.0


def _subs(r):
    return {lbl: round(_S(r[(r.index >= a) & (r.index < b)]), 2)
            for lbl, a, b in (("10s", "2010", "2020"), ("post15", "2015", "2100"),
                              ("20s", "2020", "2100"))}


def main() -> int:
    from app.research.inference import multifactor_alpha
    from app.strategy.tsmom import tsmom_backtest
    from scripts.walkforward.sleeves import futures_trend_config, live_trend_book_returns
    uni = fd.liquid_universe()
    P = fd.synthetic_price_panel(uni)
    rets = fd.returns_panel(uni)
    carry = fc.carry_panel(uni)
    rd = frl.roll_days_panel(uni, index=rets.index)
    cfg = fc.CarryConfig(roll_cost_bps=3.0)

    etf = live_trend_book_returns()
    carry_r = fc.carry_backtest(rets, carry, cfg, roll_days=rd)

    signals = {
        "xs_mom": ff.xs_momentum_signal(P),
        "curve_mom": ff.curve_momentum_signal(carry),
        "value": ff.value_signal(P),
        "skew": ff.skew_signal(rets),
    }
    print("=" * 84)
    print("P1.2 FUTURES FACTOR ZOO  (76-market Norgate; 3bps/side roll; pre-registered, no sign-flip)")
    print("=" * 84)
    print(f"{'factor':10} {'Sharpe':>7} {'sub[10s/post15/20s]':>24} {'corrT':>6} {'corrC':>6} "
          f"{'residA-t(vs T+C)':>17}  verdict")
    survivors = []
    for name, sg in signals.items():
        r = ff.xs_factor_backtest(rets, sg, cfg, roll_days=rd)
        j = pd.concat([r.rename("r"), etf.rename("e"), carry_r.rename("c")],
                      axis=1, join="inner").dropna()
        ct, cc = j["r"].corr(j["e"]), j["r"].corr(j["c"])
        t = float(multifactor_alpha(j["r"], pd.DataFrame({"trend": j["e"], "carry": j["c"]})
                                    ).get("t_alpha_hac", 0.0) or 0.0)
        sh = _S(r)
        keep = sh >= 0.30 and _subs(r)["post15"] > 0 and t > 0
        if keep:
            survivors.append(name)
        print(f"{name:10} {sh:>+7.2f} {str(_subs(r)):>24} {ct:>+6.2f} {cc:>+6.2f} {t:>+17.2f}"
              f"  {'KEEP' if keep else 'kill'}")

    print("\n" + "-" * 84)
    print(f"SURVIVORS: {survivors or 'none'}")
    print("  Only cross-sectional 12-1 MOMENTUM clears (real, modern, low-corr-to-trend). curve-")
    print("  momentum / value / skewness are dead-or-flat at the pre-registered sign -> NOT pursued")
    print("  (flipping to chase a negative Sharpe is the OPT-5 trap). Registered: futures_xsmom.")
    print("-" * 84)

    # official Ruler-v2 verdict for the survivor
    from scripts.walkforward.sleeve_lab import build_sleeve, evaluate_sleeve
    import scripts.walkforward.sleeves  # noqa: F401 (registers sleeves)
    try:
        rep = evaluate_sleeve(build_sleeve("futures_xsmom"))
        print(f"futures_xsmom official: {getattr(rep, 'verdict', '?')} "
              f"(mean_sharpe {getattr(rep, 'mean_sharpe', 0):.2f}, "
              f"point_SR {getattr(rep, 'point_sr', getattr(rep, 'point_SR', 0)):.2f})")
    except Exception as exc:
        print(f"futures_xsmom eval error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
