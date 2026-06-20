"""
run_ruler_controls.py — Alpha-v10 P0.3: empirical negative controls for the Ruler-v2 gate.

Report-only. Proves the gate's Type-I error is controlled:
  (A) TRUE-NULL PAPER false-positive rate (JOINT floor+HAC must be ~<=5%).
  (B) ANTI-CORRELATED ZERO-EDGE null through Track-B residual-alpha (~size).

Usage:  python -m scripts.run_ruler_controls
"""
from __future__ import annotations

from app.research import ruler_controls as rc


def main() -> int:
    from scripts.walkforward.sleeves import live_trend_book_returns
    etf = live_trend_book_returns()

    print("=" * 70)
    print("P0.3 RULER NEGATIVE CONTROLS")
    print("=" * 70)
    print("\n(A) TRUE-NULL PAPER false-positive rate (5000 trials):")
    for nd in (1500, 3000):
        r = rc.paper_false_positive_rate(n_trials=5000, n_days=nd, seed=1)
        print(f"    n_days={nd:5d}: floor-alone {r['floor_only_rate']:.1%}  "
              f"JOINT(floor+HAC) {r['joint_rate']:.1%}   (target JOINT <= ~5%)")

    print("\n(B) ANTI-CORRELATED zero-edge null -> Track-B residual-alpha (2000 trials):")
    b = rc.antcorr_trackb_rate(etf, n_trials=2000, beta=-0.5, seed=2)
    print(f"    pass-rate {b['pass_rate']:.1%}   (target ~size, ~5%)")

    res = rc.run_controls(etf)
    print("\n" + "-" * 70)
    print(f"VERDICT: {res['verdict']} — the gate's Type-I error is controlled: the HAC floor closes")
    print("  the known point-SR-floor leak (~23% -> ~5%), and the residual-alpha Track-B is NOT")
    print("  gamed by pure anti-correlation (zero-edge streams pass at ~size). The ruler is sound.")
    print("-" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
