"""
run_sector_rotation_eval.py — Option A: evaluate the sector-rotation swing sleeve.

Three views, report-only:
  1. STANDALONE — does the rotation sleeve pass the two-track gate on its own?
  2. TRACK-B vs the LIVE trend book — does it add residual-α, or is it redundant (the null)?
  3. COMBINED BOOK (Phase A) — trend + rotation as one book (CPCV + governor + tail + attribution).

    PYTHONPATH=. venv/Scripts/python scripts/run_sector_rotation_eval.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def main() -> None:
    from scripts.walkforward.sleeves import build_sleeve, live_trend_book_returns, fetch_universe_closes
    from scripts.walkforward.sleeve_lab import evaluate_sleeve, format_sleeve_report
    from app.research.multistrat_eval import run_multistrat_eval

    print("=" * 88)
    print("  Option A — sector-rotation swing sleeve evaluation")
    print("=" * 88)

    print("[build] sector_rotation sleeve…")
    sleeve = build_sleeve("sector_rotation")
    r = sleeve.returns
    print(f"  window {r.index.min().date()} -> {r.index.max().date()} ({len(r)} days)")

    print("[build] live ETF-trend book (Track-B base)…")
    trend = live_trend_book_returns()

    print("\n--- 1+2. STANDALONE gate + TRACK-B vs live trend ---")
    rep = evaluate_sleeve(sleeve, base_book_returns=trend)
    try:
        print(format_sleeve_report(rep))
    except Exception:
        print(f"  mean_sharpe={rep.mean_sharpe:+.3f} path-t={rep.path_sharpe_tstat:+.2f} "
              f"paper={'PASS' if rep.paper_passed else 'fail'}")
        if rep.track_b is not None:
            print(f"  TRACK-B vs trend: appraisal-IR={rep.track_b.appraisal_ir:+.3f} "
                  f"t_alpha={rep.track_b.t_alpha_hac:+.2f} -> "
                  f"{'ADDS (PASS)' if rep.track_b.passed else 'REDUNDANT (fail)'}")

    print("\n--- 3. COMBINED BOOK (Phase A): trend + sector_rotation ---")
    spy = fetch_universe_closes(["SPY"])["SPY"]
    mrep = run_multistrat_eval({"trend": trend, "rotation": r}, spy=spy, scheme="vol",
                               component_types={"trend": "risk_premium", "rotation": "diversifier"},
                               apply_governor=True, run_tail=False)
    b = mrep.book_raw
    print(f"  BOOK ({mrep.window_start}->{mrep.window_end}, {mrep.n_days}d): meanSR={b.mean_sharpe:+.3f} "
          f"path-t={b.path_sharpe_tstat:+.2f} %pos={b.pct_positive:.0%} "
          f"DSR(fam)p={b.dsr_family_p:.3f} PAPER={'PASS' if b.paper_passed else 'fail'}")
    if mrep.book_governed:
        g = mrep.book_governed
        print(f"  GOVERNED: meanSR={g.mean_sharpe:+.3f} path-t={g.path_sharpe_tstat:+.2f}")
    for s in mrep.sleeves:
        print(f"    {s.label:9s} SR={s.standalone_sharpe:+.3f} w={s.avg_weight:.2f} "
              f"trackB-IR={s.track_b_appraisal_ir:+.3f}")

    print("\n  -> Verdict hinges on Track-B: if rotation is REDUNDANT to trend (the null), it does "
          "not earn a place; only a real residual-alpha adds it as a swing diversifier.")


if __name__ == "__main__":
    main()
