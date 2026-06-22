"""
run_multistrat_eval.py — Phase A: evaluate the live + candidate strategies AS ONE BOOK.

Wires the live ETF-trend book + the paper-candidate futures factors (carry, xs-momentum) into the
unified combined-book walk-forward evaluator and prints the holistic report: book-level CPCV
(raw + drawdown-governed), per-sleeve attribution + leave-one-out Track-B, GL-1 cross-strategy
tail, and the fold-in union book. Report-only.

    PYTHONPATH=. venv/Scripts/python scripts/run_multistrat_eval.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def _fmt_bookwf(b) -> str:
    return (f"meanSR={b.mean_sharpe:+.3f}  path-t={b.path_sharpe_tstat:+.2f}  "
            f"%pos={b.pct_positive:.0%}  p5SR={b.p5_sharpe:+.3f}  calmar={b.calmar:+.2f}  "
            f"DSR(N={'fam'})p={b.dsr_family_p:.3f}  PAPER={'PASS' if b.paper_passed else 'fail'}  "
            f"[{b.n_obs} obs, {b.n_folds} folds]")


def main() -> None:
    from scripts.walkforward.sleeves import live_trend_book_returns, build_sleeve
    from scripts.walkforward.sleeves import fetch_universe_closes
    from app.research.multistrat_eval import run_multistrat_eval

    print("=" * 88)
    print("  Phase A — combined-book multi-strategy evaluation (trend + carry + xsmom)")
    print("=" * 88)

    print("[build] live ETF-trend book…")
    trend = live_trend_book_returns()
    print("[build] futures carry…")
    carry = build_sleeve("futures_carry").returns
    print("[build] futures xs-momentum…")
    xsmom = build_sleeve("futures_xsmom").returns
    print("[build] SPY (tail conditioner)…")
    spy = fetch_universe_closes(["SPY"])["SPY"]

    sleeves = {"trend": trend, "carry": carry, "xsmom": xsmom}
    ctypes = {"trend": "risk_premium", "carry": "risk_premium", "xsmom": "diversifier"}

    rep = run_multistrat_eval(sleeves, spy=spy, scheme="vol", component_types=ctypes,
                              apply_governor=True, run_tail=True)

    print("\n" + "=" * 88)
    print(f"  BOOK  ({rep.window_start} -> {rep.window_end}, {rep.n_days} days, scheme={rep.scheme}, "
          f"family N={rep.n_families})")
    print("=" * 88)
    print("  RAW      :", _fmt_bookwf(rep.book_raw))
    if rep.book_governed:
        print("  GOVERNED :", _fmt_bookwf(rep.book_governed))

    print("\n  PER-SLEEVE (standalone SR | avg weight | leave-one-out Track-B):")
    for s in rep.sleeves:
        tb = (f"appraisal-IR={s.track_b_appraisal_ir:+.3f} t_alpha={s.track_b_t_alpha_hac:+.2f} "
              f"{'PASS' if s.track_b_passed else ('fail' if s.track_b_passed is not None else 'n/a')}")
        print(f"    {s.label:8s} SR={s.standalone_sharpe:+.3f}  w={s.avg_weight:.2f}  {tb}")

    if rep.tail:
        t = rep.tail
        print("\n  TAIL (GL-1):")
        print(f"    uncond avg corr={t['uncond_corr_avg']:.3f}  SPY-worst-5% exc-corr="
              f"{t['spy_exceedance'].get(0.05, float('nan')):.3f} (post-2015 "
              f"{t['spy_exceedance_post2015'].get(0.05, float('nan')):.3f})  one_bet={t['one_bet']}")

    if rep.union:
        u = rep.union
        print(f"\n  UNION (fold-in): {u['n_days']} days SR={u['sharpe']:+.3f} "
              f"(+{u['uses_extra_days']} days beyond the {u['common_window_n_days']}-day common book)")
    for n in rep.notes:
        print("  note:", n)


if __name__ == "__main__":
    main()
