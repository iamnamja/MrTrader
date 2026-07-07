"""R1.3 before-live gate: does the carry+xSMOM futures book survive on the 16 IBKR-tradeable markets
(vs the validated 76-market universe)? Reuses the EXACT book construction + Track-B machinery so the
numbers are directly comparable to GL-0 (full-book Track-B t = 2.611)."""
import numpy as np
import pandas as pd

from app.live_trading import instrument_master as im
from app.research import futures_carry as fc
from app.research import futures_data as fd
from app.research import futures_factors as ff
from app.research import futures_roll as frl
from app.research.null_zoo import track_b_stat
from scripts.walkforward.sleeves import live_trend_book_returns

ANN = 252


def _sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ANN)) if len(r) > 20 and r.std() > 0 else float("nan")


def _book(uni, base):
    rets = fd.returns_panel(uni)
    carry = fc.carry_panel(uni)
    mom = ff.xs_momentum_signal(fd.synthetic_price_panel(uni)).reindex_like(rets)
    roll_days = frl.roll_days_panel(uni, index=rets.index)
    cfg = fc.CarryConfig(roll_cost_bps=3.0)
    carry_r = fc.carry_backtest(rets, carry, cfg, roll_days=roll_days)
    xsmom_r = ff.xs_factor_backtest(rets, mom, cfg, roll_days=roll_days)
    j = pd.concat([carry_r.rename("c"), xsmom_r.rename("x")], axis=1, join="inner").dropna()
    book = (0.5 * j["c"] + 0.5 * j["x"]).rename("book")
    out = {}
    for nm, r in (("book", book), ("carry", carry_r), ("xsmom", xsmom_r)):
        t, rs = track_b_stat(r, base)
        out[nm] = (_sharpe(r), t, rs)
    return out, len(uni)


def main():
    base = live_trend_book_returns()
    full = fd.liquid_universe()
    roots16 = sorted({str(inst.root).upper() for inst in im.futures_instruments().values()})
    ibkr16 = [m for m in full if str(m).upper() in roots16]

    print(f"Full liquid universe: {len(full)} markets")
    print(f"IBKR-tradeable (16 roots): {len(ibkr16)} present in the mirror -> {ibkr16}")
    missing = [r for r in roots16 if r not in {m.upper() for m in full}]
    if missing:
        print(f"  (roots not in the liquid universe: {missing})")

    res_full, nf = _book(full, base)
    res_16, n16 = _book(ibkr16, base)

    print("\n" + "=" * 70)
    print(f"{'sleeve':<8} | {'FULL-'+str(nf):^22} | {'IBKR-'+str(n16):^22}")
    print(f"{'':8} | {'Sharpe':>7} {'TrkB-t':>7} {'rSR':>6} | {'Sharpe':>7} {'TrkB-t':>7} {'rSR':>6}")
    print("-" * 70)
    for k in ("book", "carry", "xsmom"):
        sf, tf, rf = res_full[k]
        s6, t6, r6 = res_16[k]
        print(f"{k:<8} | {sf:>7.2f} {tf:>7.2f} {rf:>6.2f} | {s6:>7.2f} {t6:>7.2f} {r6:>6.2f}")
    print("=" * 70)
    bt_full, bt_16 = res_full["book"][1], res_16["book"][1]
    print(f"\nBook Track-B t: full {bt_full:.2f}  ->  IBKR-16 {bt_16:.2f}  "
          f"(GL-0 full-book reference = 2.611)")
    print("VERDICT:", "SURVIVES (t>2 still significant)" if bt_16 > 2.0
          else ("MARGINAL (1<t<2)" if bt_16 > 1.0 else "DOES NOT SURVIVE (t<1) — breadth kills it"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
