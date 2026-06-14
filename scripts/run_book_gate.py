"""
run_book_gate.py - real application of the Track B (book-delta) gate.

Base book = PEAD (the live alpha satellite); candidate = TSMOM trend (the
crisis-diversifier Track B was built for). The FIRST run (2026-06-11, budget
0.10, hypothesis TRACKB-TSMOM-VS-PEAD-20260611) answered the OPEN CALIBRATION
QUESTION recorded in docs/living/DECISIONS.md (2026-06-10): TSMOM improved
the book on every metric but missed Delta-Sharpe (+0.0885 vs 0.10) at the
10% budget. The owner-approved REGISTERED amendment (2026-06-11) raised
TRACKB_MAX_RISK_BUDGET 0.10 -> 0.25 (see the comment block in
app/ml/retrain_config.py); this runner now RE-RUNS the gate at the amended
budget. It only OBSERVES - it changes no constant.

Mechanics:
  1. rets = _sleeve_returns() (reused from scripts/run_book_allocator.py:
     PEAD R1K cached full-window pass + TSMOM ETF backtest, inner-joined).
  2. criteria = BookGateCriteria.from_retrain_config() (frozen; the budget
     amendment was registered 2026-06-11T05:00Z BEFORE this re-run).
  3. result = book_delta_gate(pead, trend) at the default (max) risk budget.
  4. Print format_report(result); dump the result dict to
     logs/track_b_tsmom_vs_pead_<as_of>.json (--as-of supplies the date so
     the filename never reads the wall clock).
  5. Best-effort: record the re-run in the research registry
     (data/research_registry.db) as the REGISTERED RE-TEST (R4 path)
     TRACKB-TSMOM-VS-PEAD-20260611-AMEND25 (parent = the original
     hypothesis, cooling_off_until = the amendment registration instant)
     with decision='park' regardless of PASS/FAIL - book inclusion is
     owner-gated, never auto-promoted. A registry hiccup (e.g. duplicate id
     on a re-run) NEVER crashes the gate run; it is reported and the verdict
     still stands.
  6. --sweep: budget-transparency curve. Runs the gate at each budget in
     SWEEP_BUDGETS (all <= the registered 0.25 cap) and prints an ASCII
     table (budget | dSharpe | verdict | failed criteria) so the amended
     0.25 is auditable against the whole curve, not a magic number.
     Sweep mode is REPORT-ONLY: no JSON dump, no registry write.

Usage:
  python -m scripts.run_book_gate --as-of 2026-06-11
  python -m scripts.run_book_gate --as-of 2026-06-11 --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

# REGISTERED RE-TEST of the original first run (registry R4 path): a NEW
# hypothesis_id with parent_id = the original and a cooling_off_until that the
# run must strictly post-date.
HYPOTHESIS_ID = "TRACKB-TSMOM-VS-PEAD-20260611-AMEND25"
PARENT_HYPOTHESIS_ID = "TRACKB-TSMOM-VS-PEAD-20260611"
# The owner-approved budget amendment (0.10 -> 0.25) was registered
# 2026-06-11T05:00Z (retrain_config TRACKB_MAX_RISK_BUDGET comment +
# DECISIONS.md 2026-06-11), BEFORE this re-run - legitimate pre-registration.
# The same instant is the re-test's cooling_off_until: R4 requires the run to
# EXECUTE strictly after it, i.e. after the amendment was on the books.
PREREGISTERED_AT = "2026-06-11T05:00:00+00:00"
COOLING_OFF_UNTIL = "2026-06-11T05:00:00+00:00"
MECHANISM = (
    "Track B book-delta RE-TEST under the registered budget amendment "
    "(0.10 -> 0.25): does TSMOM (crisis-diversifier) improve the PEAD book "
    "at a 25% risk budget?"
)

# --sweep budget grid: the transparency curve around the amended cap. Every
# value is <= TRACKB_MAX_RISK_BUDGET (0.25) so the gate's input validation
# accepts it; 0.10 reproduces the original run's budget for lineage.
SWEEP_BUDGETS = (0.05, 0.10, 0.125, 0.15, 0.20, 0.25)


def _to_jsonable(obj):
    """Plain-Python (json.dumps-able) deep copy: numpy scalars -> int/float/
    bool, timestamps -> ISO strings, tuples -> lists."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _record_in_registry(result_dict: dict, criteria_dict: dict, run_at: str) -> None:
    """Dogfood the research registry's RE-TEST path (R4): new hypothesis_id,
    parent_id = the original first run, cooling_off_until = the amendment
    registration instant (strictly before run_at). Best-effort; never crashes
    the run. decision='park' regardless of PASS/FAIL: Track B inclusion is
    owner-gated."""
    from app.research.registry import RegistryIntegrityError, ResearchRegistry

    reg = ResearchRegistry()
    try:
        reg.register(
            hypothesis_id=HYPOTHESIS_ID,
            family="trend",
            label="confirmatory",
            mechanism=MECHANISM,
            parent_id=PARENT_HYPOTHESIS_ID,
            cooling_off_until=COOLING_OFF_UNTIL,
        )
        reg.preregister(
            HYPOTHESIS_ID,
            acceptance_criteria=criteria_dict,
            preregistered_at=PREREGISTERED_AT,
        )
    except RegistryIntegrityError as exc:
        # Re-run: the row already exists (R1) or is already preregistered (R5).
        # Skip registration and try to record on the existing row.
        print(f"[registry] register/preregister skipped (existing row): {exc}")

    row = reg.record_result(
        HYPOTHESIS_ID, run_at=run_at, result=result_dict, decision="park"
    )
    print(f"[registry] recorded {HYPOTHESIS_ID} "
          f"(parent={row['parent_id']}, re-test accepted by R4) at {reg.db_path} "
          f"(run_at={row['run_at']}, decision={row['decision']})")


def _evaluate(base, candidate, *, mode, candidate_risk_budget=None,
              candidate_label="tsmom_trend"):
    """Dispatch on TRACKB_MODE. Returns a uniform tuple regardless of gate:
    (result, report_str, criteria_dict, delta_label, delta_value, passed, failed).

    - mode="ruler_v2"  -> track_b_appraisal.appraise_track_b (budget-invariant appraisal
      IR + block-bootstrap P(dSR>0); the trend candidate declares component_type=
      "risk_premium" so its worst-regime backstop is waived).
    - mode="book_delta" (legacy) -> book_gate.book_delta_gate (budget-dependent dSharpe).
    Both results expose .to_dict()/.passed/.failed_criteria."""
    if mode == "ruler_v2":
        from scripts.walkforward.track_b_appraisal import (
            TrackBAppraisalCriteria, appraise_track_b, format_report,
        )
        crit = TrackBAppraisalCriteria.from_retrain_config()
        res = appraise_track_b(
            base, candidate, component_type="risk_premium", criteria=crit,
            candidate_risk_budget=candidate_risk_budget, candidate_label=candidate_label)
        return (res, format_report(res), crit.to_dict(),
                "P(dSR>0)", res.p_delta_sr_gt_0, res.passed, res.failed_criteria)
    from scripts.walkforward.book_gate import (
        BookGateCriteria, book_delta_gate, format_report,
    )
    crit = BookGateCriteria.from_retrain_config()
    res = book_delta_gate(
        base, candidate, criteria=crit,
        candidate_risk_budget=candidate_risk_budget, candidate_label=candidate_label)
    return (res, format_report(res), crit.to_dict(),
            "dSharpe", res.sharpe_delta, res.passed, res.failed_criteria)


def _run_sweep(rets, mode) -> None:
    """Budget-transparency curve: the gate at every SWEEP_BUDGETS value, as an ASCII
    table (budget | delta-metric | verdict | failed criteria). Report-only. The
    delta-metric is dSharpe under book_delta, P(dSR>0) under ruler_v2 (note the ruler_v2
    appraisal IR is budget-INVARIANT by design; the sweep shows the budget-dependent
    significance side). All budgets are <= the registered cap so input validation accepts them."""
    bar = "-" * 78
    print()
    print(f"  TRACK B BUDGET SWEEP (mode={mode}) - delta-metric + verdict vs risk budget")
    print(bar)
    _, _, _, dlabel, _, _, _ = _evaluate(
        rets["pead"], rets["trend"], mode=mode, candidate_risk_budget=SWEEP_BUDGETS[0])
    print(f"  {'budget':>8} | {dlabel:>9} | {'verdict':>7} | failed criteria")
    print(bar)
    for b in SWEEP_BUDGETS:
        _, _, _, _, dval, passed, failed = _evaluate(
            rets["pead"], rets["trend"], mode=mode, candidate_risk_budget=b)
        failed_s = ", ".join(failed) if failed else "-"
        print(f"  {b:>8.3f} | {dval:>+9.4f} | "
              f"{'PASS' if passed else 'FAIL':>7} | {failed_s}")
    print(bar)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Track B book-delta gate: TSMOM trend vs the PEAD book")
    ap.add_argument("--as-of", required=True, metavar="YYYY-MM-DD",
                    help="run date used for the JSON dump filename")
    ap.add_argument("--sweep", action="store_true",
                    help="report-only budget-transparency curve over "
                         f"{SWEEP_BUDGETS} (no JSON dump, no registry write)")
    args = ap.parse_args()

    from scripts.run_book_allocator import _sleeve_returns
    from app.ml.retrain_config import TRACKB_MODE

    print(f"[gate] TRACKB_MODE={TRACKB_MODE!r} "
          f"({'Ruler-v2 budget-invariant appraisal' if TRACKB_MODE == 'ruler_v2' else 'legacy book-delta'})")

    rets = _sleeve_returns()
    print(f"[gate] sleeve overlap: {rets.index[0].date()} -> "
          f"{rets.index[-1].date()} ({len(rets)} days)")

    if args.sweep:
        _run_sweep(rets, TRACKB_MODE)
        return 0

    result, report, criteria_dict, _, _, _, _ = _evaluate(
        rets["pead"], rets["trend"], mode=TRACKB_MODE)
    print(f"[gate] frozen criteria: {criteria_dict}")
    print()
    print(report)

    result_dict = _to_jsonable(result.to_dict())
    out_path = ROOT / "logs" / f"track_b_tsmom_vs_pead_{args.as_of}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_dict, indent=2, sort_keys=True))
    print(f"[gate] result dict written to {out_path}")

    # Registry recording is best-effort: a hiccup never crashes the gate run.
    run_at = datetime.now(timezone.utc).isoformat()
    try:
        _record_in_registry(result_dict, _to_jsonable(criteria_dict), run_at)
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all guard
        print(f"[registry] WARNING: recording failed (gate verdict unaffected): "
              f"{type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
