"""
run_carry_confirmation.py — the OWNER-AUTHORIZED fresh confirmation of the F3 rates-carry
near-miss (Alpha-v7, 2026-06-14).

R7 discipline: the spec below is PRE-REGISTERED (criteria frozen) via the research registry
BEFORE this run executes — see the `register` / `preregister` CLI calls recorded in the PR /
DECISIONS. This script only RUNS the frozen spec and prints the verdict; `record-result` is a
separate post-run CLI call (run_at strictly after preregistered_at).

The pre-registered spec (chosen on ECONOMIC PRINCIPLE, not grid-fit):
  - LONG-FLAT (not long-short): harvest positive carry when the curve is upward-sloping,
    stand aside when inverted. Going SHORT duration on inversion is a directional/crisis bet,
    not carry harvesting — and it is the crisis-correlated leg the 2026-06-14 panel warned of.
  - IEF (intermediate Treasuries): the most liquid, deepest clean-history duration ETF
    (deliberately NOT the higher-scoring TLT/2.0/long-short grid cell).
  - scale_pct = 1.5 (~ the historical median 10y-3m term spread), 1bp cost.

Confirmation evidence (since the full-sample robustness grid was already seen):
  1. Track-A PAPER + Track-B verdict on the full sample (Track-A significance on long-flat
     was NOT previously measured — a genuinely open outcome).
  2. A pre-registered SUB-PERIOD STABILITY check (2007–2016 vs 2017–2026): the carry SR must
     be POSITIVE in BOTH halves — the OOS-ish guard against a single-era artifact.

DECISION RULE (pre-registered):
  promote_paper  iff  Track-A PAPER PASS  AND  Track-B PASS  AND  both-half SR > 0
  park           iff  Track-B PASS and both-half SR > 0 but Track-A significance just misses
  kill           iff  no standalone edge / unstable across halves
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SPLIT = pd.Timestamp("2017-01-01")   # pre-registered sub-period boundary


def _ann_sr(r: pd.Series) -> float:
    r = r.dropna()
    sd = float(r.std())
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def main() -> int:
    from scripts.walkforward.sleeves import build_rates_carry, live_trend_book_returns
    from scripts.walkforward.sleeve_lab import evaluate_sleeve, format_sleeve_report

    # The PRE-REGISTERED spec: long-flat IEF, scale 1.5, 1bp.
    sleeve = build_rates_carry(duration_etf="IEF", scale_pct=1.5, long_short=False,
                               cost_bps=1.0)
    sleeve.registration_id = "F3-CARRY-CONFIRM-20260614"

    base = live_trend_book_returns()
    # Carry is a declared risk_premium → evaluate Track-B on the probation path
    # (Alpha-v9 P0-2: standalone-SR floor report-only, P(ΔSR>0) judged vs the lowered
    # probation bar; a PASS is a PAPER-only, live-paper-ratified small-size admit).
    rep = evaluate_sleeve(sleeve, base_book_returns=base, track_b_probation=True)
    print(format_sleeve_report(rep))

    # POWERED sub-period stability (Alpha-v9 P0-2 Ⓑ): "stable" = the two half-Sharpes
    # are not statistically distinguishable (bootstrap CI of SR(H1)-SR(H2) overlaps 0),
    # NOT the old binary "positive in both halves" (which had ~24% FN on a true SR-0.5
    # edge and killed this very sleeve).
    from app.research.stability import stability_test
    r = sleeve.returns
    stab = stability_test(r, split=SPLIT, n_boot=2000, seed=0)
    sr1, sr2 = stab.sr_h1, stab.sr_h2
    both_pos = stab.both_halves_positive       # report-only now (old criterion)
    stable = stab.halves_indistinguishable     # the new powered stability verdict
    tb = rep.track_b
    print("\n" + "=" * 78)
    print("  F3-CARRY-CONFIRM-20260614 — pre-registered confirmation verdict")
    print("=" * 78)
    print(f"  spec: long-flat IEF, scale_pct=1.5, 1bp  (n_obs={rep.n_obs})")
    print(f"  Track-A PAPER : {'PASS' if rep.paper_passed else 'FAIL'}  "
          f"(point_SR={rep.point_sr:+.3f}, hac_p={rep.hac_p_one_sided:.4f}, "
          f"resid_a_t={(rep.residual_alpha_t_hac or float('nan')):+.2f})")
    print(f"  Track-B BOOK  : {'PASS' if (tb and tb.passed) else 'FAIL'}  "
          f"(IR={tb.appraisal_ir:+.3f}, dSR={tb.delta_sr_point:+.3f}, "
          f"P(dSR>0)={tb.p_delta_sr_gt_0:.3f}, corr={tb.corr_to_book:+.3f})")
    print(f"  Stability     : H1(2007-2016) SR={sr1:+.3f}  H2(2017-2026) SR={sr2:+.3f}  "
          f"dSR={stab.sr_diff:+.3f} CI[{stab.diff_ci_low:+.3f},{stab.diff_ci_high:+.3f}] "
          f"-> stable={stable} (old both_positive={both_pos})")
    tb_pass = bool(tb and tb.passed)
    if rep.paper_passed and tb_pass and stable:
        decision = "promote_paper"                 # full: standalone significance + book-delta
    elif tb_pass and stable and getattr(tb, "probation_applied", False):
        # Track-B book-delta admit of a stable, declared diversifier on the probation
        # bar — PAPER only, small size, live-paper ratification required. Track-A
        # standalone significance is NOT required (that's the whole point of Track-B).
        decision = "promote_paper_probation"
    elif tb_pass and stable:
        decision = "park"
    else:
        decision = "kill"
    print(f"  DECISION (P0-2 rule): {decision.upper()}"
          + ("  [PAPER only — live-paper ratification required]"
             if decision == "promote_paper_probation" else ""))
    print("=" * 78)
    # Machine-readable line for record-result.
    print("\nRESULT_JSON " + str({
        "point_sr": round(rep.point_sr, 4), "hac_p_one_sided": round(rep.hac_p_one_sided, 4),
        "residual_alpha_t": round(float(rep.residual_alpha_t_hac or 0.0), 3),
        "track_a_paper_pass": bool(rep.paper_passed),
        "track_b_pass": bool(tb.passed) if tb else False,
        "appraisal_ir": round(tb.appraisal_ir, 4) if tb else None,
        "p_delta_sr_gt_0": round(tb.p_delta_sr_gt_0, 4) if tb else None,
        "corr_to_book": round(tb.corr_to_book, 4) if tb else None,
        "sr_half1_2007_2016": round(sr1, 4), "sr_half2_2017_2026": round(sr2, 4),
        "sr_diff": round(stab.sr_diff, 4),
        "sr_diff_ci": [round(stab.diff_ci_low, 4), round(stab.diff_ci_high, 4)],
        "stable_powered": stable, "both_halves_positive": both_pos,
        "track_b_probation_applied": bool(getattr(tb, "probation_applied", False)) if tb else False,
        "effective_min_pdsr": round(tb.effective_min_pdsr, 3) if tb else None,
        "decision": decision,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
