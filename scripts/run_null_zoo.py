"""
run_null_zoo.py — Alpha-v10 GL-0 runner: the selection-aware null-strategy zoo + look-ahead audit
for the futures book (carry + xs-momentum). Decides BASKET_REAL / CARRY_ONLY / RESIDUE.

    PYTHONPATH=. venv/Scripts/python scripts/run_null_zoo.py            # 1000 nulls (default)
    PYTHONPATH=. venv/Scripts/python scripts/run_null_zoo.py --n 5000   # tighter tails
    PYTHONPATH=. venv/Scripts/python scripts/run_null_zoo.py --quick    # 200 nulls (smoke)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of null replications")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="200 nulls (smoke test)")
    args = ap.parse_args()
    n = 200 if args.quick else args.n

    from app.research import (futures_data as fd, futures_carry as fc,
                              futures_factors as ff, futures_roll as frl, null_zoo as nz)
    from scripts.walkforward.sleeves import live_trend_book_returns

    print("Loading the exact panels the real futures book is built from...")
    uni = fd.liquid_universe()
    returns = fd.returns_panel(uni)
    prices = fd.synthetic_price_panel(uni)
    carry = fc.carry_panel(uni)
    mom_signal = ff.xs_momentum_signal(prices)
    roll_days = frl.roll_days_panel(uni, index=returns.index)
    base = live_trend_book_returns()
    print(f"  {len(uni)} markets, {len(returns)} days; live-trend base {len(base)} days.")

    print("\n=== LOOK-AHEAD AUDIT (engine + signal future-blindness) ===")
    audit = nz.look_ahead_audit(returns, carry, mom_signal, roll_days, prices=prices)
    for k in ("carry_engine_future_blind", "xsmom_engine_future_blind",
              "xsmom_signal_future_blind", "pit_clean"):
        print(f"  {k:32s} {audit[k]}")
    for note in audit["notes"]:
        print(f"  - {note}")

    print(f"\n=== NULL-STRATEGY ZOO  (n={n}, seed={args.seed}) ===")
    res = nz.run_null_zoo(returns, carry, mom_signal, roll_days, base,
                          prices=prices, n_nulls=n, seed=args.seed)
    print(f"  observed Track-B t vs live trend:  book={res.t_obs_book:.2f}  "
          f"carry={res.t_obs_carry:.2f}  xsmom={res.t_obs_xsmom:.2f}  "
          f"(book resid-Sharpe {res.resid_sharpe_book:.2f})")
    print(f"  NULL-BOOKS p (primary)          {res.null_book_p:.3f}   "
          f"[95th-pct null t {res.null_book_p95:.2f}, 99th {res.null_book_p99:.2f}]")
    print(f"  NULL-BOOKS p (circular-shift)   {res.null_book_p_shift:.3f}")
    print(f"  carry single-factor null p      {res.carry_null_p:.3f}   "
          f"[95th-pct {res.carry_null_p95:.2f}]")
    print(f"  xs-momentum max-of-6 null p     {res.xsmom_maxof6_p:.3f}   "
          f"[95th-pct {res.xsmom_maxof6_p95:.2f}]")
    print(f"  Deflated Sharpe  N=10 {res.dsr_n10:.3f}  N=20 {res.dsr_n20:.3f}  "
          f"N=30 {res.dsr_n30:.3f}  (bar > 0.95)")
    print("\n  --- interpretation ---")
    for note in res.notes:
        print(f"  - {note}")
    print(f"\n  VERDICT: {res.verdict}")
    print("    BASKET_REAL = size carry + xs-momentum book | "
          "CARRY_ONLY = size carry, drop xs-momentum | RESIDUE = no futures book")


if __name__ == "__main__":
    main()
