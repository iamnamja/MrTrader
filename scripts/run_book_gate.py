"""
run_book_gate.py - FIRST real application of the Track B (book-delta) gate.

Base book = PEAD (the live alpha satellite); candidate = TSMOM trend (the
crisis-diversifier Track B was built for). Answers the OPEN CALIBRATION
QUESTION recorded in docs/living/DECISIONS.md (2026-06-10): at the 10% risk
budget, does Delta-Sharpe >= 0.10 structurally reject TSMOM (SR ~ 0.71,
corr ~ +0.25 to PEAD)? This runner only OBSERVES - it changes no constant.

Mechanics:
  1. rets = _sleeve_returns() (reused from scripts/run_book_allocator.py:
     PEAD R1K cached full-window pass + TSMOM ETF backtest, inner-joined).
  2. criteria = BookGateCriteria.from_retrain_config() (frozen, registered
     2026-06-10 BEFORE this run).
  3. result = book_delta_gate(pead, trend) at the default (max) risk budget.
  4. Print format_report(result); dump the result dict to
     logs/track_b_tsmom_vs_pead_<as_of>.json (--as-of supplies the date so
     the filename never reads the wall clock).
  5. Best-effort: record the run in the research registry
     (data/research_registry.db) as confirmatory hypothesis
     TRACKB-TSMOM-VS-PEAD-20260611 with decision='park' regardless of
     PASS/FAIL - book inclusion is owner-gated, never auto-promoted.
     A registry hiccup (e.g. duplicate id on a re-run) NEVER crashes the
     gate run; it is reported and the verdict still stands.

Usage: python -m scripts.run_book_gate --as-of 2026-06-11
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

HYPOTHESIS_ID = "TRACKB-TSMOM-VS-PEAD-20260611"
# Genuinely registered 2026-06-10 (retrain_config TRACKB_* constants + the
# DECISIONS.md entry), BEFORE this first run - legitimate pre-registration.
PREREGISTERED_AT = "2026-06-10T00:00:00+00:00"
MECHANISM = (
    "Track B book-delta: does TSMOM (crisis-diversifier) improve the PEAD "
    "book at a 10% risk budget?"
)


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
    """Dogfood the research registry (best-effort; never crashes the run).
    decision='park' regardless of PASS/FAIL: Track B inclusion is owner-gated."""
    from app.research.registry import RegistryIntegrityError, ResearchRegistry

    reg = ResearchRegistry()
    try:
        reg.register(
            hypothesis_id=HYPOTHESIS_ID,
            family="trend",
            label="confirmatory",
            mechanism=MECHANISM,
            parent_id=None,
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
    print(f"[registry] recorded {HYPOTHESIS_ID} at {reg.db_path} "
          f"(run_at={row['run_at']}, decision={row['decision']})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Track B book-delta gate: TSMOM trend vs the PEAD book")
    ap.add_argument("--as-of", required=True, metavar="YYYY-MM-DD",
                    help="run date used for the JSON dump filename")
    args = ap.parse_args()

    from scripts.run_book_allocator import _sleeve_returns
    from scripts.walkforward.book_gate import (
        BookGateCriteria, book_delta_gate, format_report,
    )

    criteria = BookGateCriteria.from_retrain_config()
    print(f"[gate] frozen criteria (registered 2026-06-10): {criteria.to_dict()}")

    rets = _sleeve_returns()
    print(f"[gate] sleeve overlap: {rets.index[0].date()} -> "
          f"{rets.index[-1].date()} ({len(rets)} days)")

    result = book_delta_gate(
        rets["pead"], rets["trend"],
        criteria=criteria, candidate_label="tsmom_trend",
    )
    print()
    print(format_report(result))

    result_dict = _to_jsonable(result.to_dict())
    out_path = ROOT / "logs" / f"track_b_tsmom_vs_pead_{args.as_of}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_dict, indent=2, sort_keys=True))
    print(f"[gate] result dict written to {out_path}")

    # Registry recording is best-effort: a hiccup never crashes the gate run.
    run_at = datetime.now(timezone.utc).isoformat()
    try:
        _record_in_registry(result_dict, _to_jsonable(criteria.to_dict()), run_at)
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all guard
        print(f"[registry] WARNING: recording failed (gate verdict unaffected): "
              f"{type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
