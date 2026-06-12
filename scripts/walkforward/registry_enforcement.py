"""
registry_enforcement.py — shared --hypothesis-id enforcement for the run scripts.

Blueprint Phase 0 (docs/reference/NEXT_PHASE_BLUEPRINT_2026-06.md, "Research
registry"): the `run_pead_*` / `run_*_cpcv` scripts gain `--hypothesis-id`
(warn-only for 2 weeks, then required for any run labeled confirmatory;
`preregistered_at < run_at` asserted). This module is the SINGLE implementation
all nine run scripts share, so the semantics cannot drift between them.

Contract (each run script):

    ap = argparse.ArgumentParser(...)
    add_arguments(ap)                       # --hypothesis-id / --exploratory
    args = ap.parse_args()
    run = begin_run(args.hypothesis_id, exploratory=args.exploratory)
    ... expensive fetch + CPCV ...
    if run is not None:
        run.record({...compact result summary...}, decision=None)

begin_run() FAILS FAST — RegistryEnforcementError BEFORE any data is fetched —
on the violations the registry would otherwise only reject HOURS later at
record_result() time:
  - a hypothesis_id that is not registered (register/preregister first);
  - a confirmatory/live_confirm hypothesis that was never preregistered
    (criteria must be frozen BEFORE the run; registry rule R2);
  - a missing hypothesis_id on/after GRACE_UNTIL without --exploratory
    (grace window over: an unlabeled run is treated as confirmatory).
During the grace window a missing hypothesis_id only WARNS (blueprint:
"warn-only for 2 weeks"). Exploratory runs (--exploratory) never need an id —
exploratory runs are unlimited and can inspire, but can never promote (R3).
Passing BOTH --hypothesis-id and --exploratory is contradictory and fails
fast: an exploratory run must never record under (and so consume) a real id.

HypothesisRun.record() is BEST-EFFORT by design: a completed multi-hour CPCV
must never be lost to a registry hiccup (e.g. R4 one-shot already recorded on
an accidental re-run). Failures are printed as [registry] warnings and
swallowed — mirrors scripts/run_book_gate.py. The run_at stamped on the row is
the instant begin_run() was called (the run START), so for any honest run
preregistered_at < run_at < the recording instant, and run_at is the time the
run actually EXECUTED (what R2/R4 compare against), not when the result
landed.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from typing import Any

# Blueprint Phase 0: "--hypothesis-id (warn-only for 2 weeks, then required for
# any run labeled confirmatory)". Enforcement shipped 2026-06-11, so the 2-week
# warn-only window covers runs dated strictly BEFORE 2026-06-25; from that day
# on a run without --hypothesis-id and without --exploratory refuses to start.
GRACE_UNTIL = date(2026, 6, 25)

# Labels whose runs must have frozen criteria BEFORE executing (registry R2).
_PREREG_REQUIRED_LABELS = ("confirmatory", "live_confirm")


class RegistryEnforcementError(RuntimeError):
    """A pre-run registry-enforcement rule was violated (fail fast — raised
    BEFORE the expensive fetch/run, never after)."""


def _to_jsonable(obj: Any) -> Any:
    """Plain-Python (json.dumps-able) deep copy: numpy scalars -> int/float/
    bool, timestamps -> ISO strings, tuples -> lists. Mirrors run_book_gate."""
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


class HypothesisRun:
    """Handle for a registry-enforced run: carries the hypothesis_id and the
    run-START timestamp; record() writes the outcome best-effort."""

    def __init__(self, hypothesis_id: str, run_at: datetime):
        self.hypothesis_id = hypothesis_id
        self.run_at = run_at

    def record(self, result: dict, decision: str | None = None) -> bool:
        """Record the run's outcome in the research registry. BEST-EFFORT:
        any failure (RegistryIntegrityError incl. R4-already-recorded, a locked
        DB, anything) is printed and swallowed — a completed multi-hour CPCV
        must never be lost to a registry hiccup. Returns True iff recorded.

        decision defaults to None deliberately: promotion is owner-gated and
        never auto-decided from a single run's metrics.
        """
        try:
            from app.research.registry import ResearchRegistry
            reg = ResearchRegistry()
            row = reg.record_result(
                self.hypothesis_id,
                run_at=self.run_at.isoformat(),
                result=_to_jsonable(result),
                decision=decision,
            )
            print(f"[registry] recorded {self.hypothesis_id} at {reg.db_path} "
                  f"(run_at={row['run_at']}, decision={row['decision']})")
            return True
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all guard
            print(f"[registry] WARNING: result NOT recorded for "
                  f"{self.hypothesis_id} (run verdict unaffected): "
                  f"{type(exc).__name__}: {exc}")
            return False


def begin_run(
    hypothesis_id: str | None,
    *,
    run_at: datetime | None = None,
    as_of: date | None = None,
    exploratory: bool = False,
) -> HypothesisRun | None:
    """Enforce the registry contract BEFORE an expensive run starts.

    Call this first thing in main(), before any data fetch. Returns a
    HypothesisRun (record the outcome on it after the gate verdict) or None
    (nothing to record: exploratory / grace-window run without an id).

    run_at defaults to now-UTC captured HERE (the run start) — it is the
    timestamp R2/R4 compare against, so it must reflect when the run executed.
    as_of defaults to today and exists only so the grace-window check is
    injectable in tests.

    Raises RegistryEnforcementError (fail fast) when:
      - hypothesis_id and exploratory are BOTH given (contradictory: an
        exploratory run must never carry a real hypothesis id — it would
        record under it and consume the R4 one-shot);
      - hypothesis_id is given but not registered;
      - the row is confirmatory/live_confirm but has no preregistered_at
        (criteria were never frozen — the run's result could never be
        recorded anyway, so refuse to start);
      - hypothesis_id is None, exploratory=False and as_of >= GRACE_UNTIL.
    """
    if run_at is None:
        run_at = datetime.now(timezone.utc)
    if as_of is None:
        as_of = date.today()

    # Contradictory flags FIRST (before any registry lookup): an exploratory
    # run is UNRECORDED by definition, but a supplied id would be recorded
    # against — a user trusting --exploratory would burn the R4 one-shot.
    if hypothesis_id is not None and exploratory:
        raise RegistryEnforcementError(
            f"--exploratory cannot be combined with --hypothesis-id "
            f"({hypothesis_id!r}): an exploratory run must not carry a real "
            f"hypothesis id (it would record under it and consume its R4 "
            f"one-shot). Drop --exploratory to run the hypothesis, or drop "
            f"--hypothesis-id for an unrecorded dev run."
        )

    if hypothesis_id is not None:
        from app.research.registry import ResearchRegistry
        row = ResearchRegistry().get(hypothesis_id)
        if row is None:
            raise RegistryEnforcementError(
                f"hypothesis_id={hypothesis_id!r} is not registered. Register "
                f"(and for confirmatory runs, preregister) it first - e.g. "
                f"`python scripts/registry.py register {hypothesis_id} "
                f"--label confirmatory ...` then `preregister`. Unregistered "
                f"runs do not exist for this program."
            )
        label = row.get("label")
        if label in _PREREG_REQUIRED_LABELS and row.get("preregistered_at") is None:
            raise RegistryEnforcementError(
                f"hypothesis_id={hypothesis_id!r} is label={label!r} but was "
                f"never preregistered. Confirmatory criteria must be frozen "
                f"BEFORE the run (R2) - preregister first; refusing to start "
                f"a run whose result could never be recorded."
            )
        # Fail fast on a hypothesis that already has a result (registry R4 one-shot):
        # an accidental re-run's record() is guaranteed to fail and be swallowed, so
        # the multi-hour CPCV would be burned for nothing. Refuse to start instead;
        # the sanctioned path is a registered re-test (new id, parent_id, cooling-off).
        if (row.get("run_at") is not None or row.get("result_json") is not None
                or row.get("decision") is not None):
            raise RegistryEnforcementError(
                f"hypothesis_id={hypothesis_id!r} already has a recorded result "
                f"(run_at={row.get('run_at')}, decision={row.get('decision')}). A "
                f"hypothesis gets exactly one run (R4) — its result could never be "
                f"recorded. Register a re-test (new id, parent_id={hypothesis_id!r}, "
                f"cooling_off_until) to run again."
            )
        # Fail fast when this run START is not strictly after the frozen criteria
        # (registry R2 ordering) — record() would reject it after the fact anyway.
        prereg = row.get("preregistered_at")
        if label in _PREREG_REQUIRED_LABELS and prereg is not None:
            prereg_dt = datetime.fromisoformat(str(prereg))
            if prereg_dt.tzinfo is None:
                prereg_dt = prereg_dt.replace(tzinfo=timezone.utc)
            if prereg_dt >= run_at:
                raise RegistryEnforcementError(
                    f"hypothesis_id={hypothesis_id!r}: run start run_at="
                    f"{run_at.isoformat()} is not strictly after preregistered_at="
                    f"{prereg}. Criteria must be frozen BEFORE the run executes "
                    f"(R2); the run cannot start until after pre-registration."
                )
        print(f"[registry] run enforced under {hypothesis_id} "
              f"(label={label}, run_at={run_at.isoformat()})")
        return HypothesisRun(hypothesis_id, run_at)

    if exploratory:
        # Exploratory runs are unlimited (blueprint): nothing to enforce or record.
        return None

    if as_of < GRACE_UNTIL:
        print(f"[registry] WARNING: no --hypothesis-id given. This is allowed "
              f"during the warn-only window but will be REQUIRED from "
              f"{GRACE_UNTIL.isoformat()} (pass --exploratory for a dev run, "
              f"or register + preregister a hypothesis and pass its id).")
        return None

    raise RegistryEnforcementError(
        f"--hypothesis-id is required from {GRACE_UNTIL.isoformat()} (the "
        f"2-week warn-only window is over). Confirmatory runs must be "
        f"pre-registered: register + preregister a hypothesis and pass "
        f"--hypothesis-id, or pass --exploratory for an unrecorded dev run."
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the shared --hypothesis-id / --exploratory flags to a run
    script's parser. All-optional: omitting both preserves each script's
    existing env-var-driven behavior (warn-only until GRACE_UNTIL)."""
    parser.add_argument(
        "--hypothesis-id", default=None, metavar="HYP-ID",
        help="registered hypothesis id for this run (research registry); the "
             "result is recorded against it best-effort after the gate verdict",
    )
    parser.add_argument(
        "--exploratory", action="store_true",
        help="unrecorded exploratory/dev run (no hypothesis id required; "
             "exploratory runs are unlimited and can never promote)",
    )
