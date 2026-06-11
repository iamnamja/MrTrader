"""
preregister_event_hypotheses.py - freeze H1/H2/H3 in the research registry.

Phase-3 prep (blueprint NEXT_PHASE_BLUEPRINT_2026-06.md: "Three pre-registered
confirmatory hypotheses (registered in Phase 0's registry BEFORE running)").
This script register()+preregister()s the three confirmatory event hypotheses
with FROZEN acceptance criteria — the scientific contract for the Phase-3
event-panel runs. The criteria below are the bar; the runs (PR3 /
run_event_panel_inference.py) get exactly ONE confirmatory shot each (R4).

  H1 — PEAD base, event-level re-adjudication of the LIVE edge.
  H2 — implied-move reaction_ratio as a CONTINUOUS feature (settles OPT-5).
  H3 — PEAD v2 monotonic scorecard (options-conditioned continuous score).

Idempotent and safe to re-run: an already-registered id (R1) or an
already-preregistered row (R5) is skipped with a notice, mirroring
scripts/run_book_gate.py. Criteria are IMMUTABLE once preregistered — editing
this file and re-running cannot move the goalposts (the registry refuses);
a revision requires a NEW hypothesis_id with parent_id set (re-test path).

Usage:
  python -m scripts.preregister_event_hypotheses
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

# The instant the criteria were frozen (2026-06-11 PR1). Every confirmatory
# run must EXECUTE strictly after this (R2: preregistered_at < run_at).
PREREGISTERED_AT = "2026-06-11T12:00:00+00:00"

# H3's feature list is frozen here ONCE and referenced from both params and
# acceptance_criteria so the two can never drift.
H3_FEATURES = [
    "sue", "revision_momentum", "announce_gap_vs_vol", "reaction_ratio",
    "iv_runup_t10_t1", "cpiv_pre", "skew_25d_pre", "opt_volume_z_pre",
    "post_iv_retention_t1",
]

# ── The three frozen hypotheses (blueprint Phase 3 design, C3/C5/C6) ─────────
HYPOTHESES = [
    {
        "hypothesis_id": "H1-PEAD-EVENTLEVEL-20260611",
        "label": "confirmatory",
        "family": "pead",
        "universe": "russell1000",
        "window": "2019-2026 equity",
        "folds": "event-level (announce_date x firm clustered)",
        "params": {"component_type": "alpha", "H_label": "H1"},
        "acceptance_criteria": {
            "mechanism": (
                "Event-level re-adjudication of the LIVE PEAD edge (replaces "
                "the 8-fold path t-stat shown unable to separate PEAD from "
                "noise)."),
            "instrument": (
                "Two-way cluster-robust OLS, clusters = (announce_date, firm), "
                "Cameron-Gelbach-Miller SEs; quarter-cluster block bootstrap "
                "retained as the conservative bound."),
            "population": (
                "Current PEAD-qualified event population, R1K, equity panel "
                "2019->2026 (equity-only hedged returns; no options columns "
                "required)."),
            "target": (
                "SPY-hedged forward event returns, horizons {5,10,20} trading "
                "days; PRIMARY = 10d."),
            "test": (
                "H0: mean hedged event return = 0; one-sided alternative "
                "mean > 0, day x firm clustered."),
            "decision_rule": {
                "p_lt_0.05": (
                    "PEAD graduates from waiver-paper to an HONEST Track-A "
                    "paper pass; event-sparsity regime waiver retired for "
                    "PEAD."),
                "p_gt_0.15": (
                    "PEAD demoted -> live book becomes trend-plus-cash "
                    "(TSMOM trend, Track-B-validated, is the capital base)."),
                "0.05_to_0.15": (
                    "Inconclusive: PEAD stays at telemetry size; no live "
                    "posture change."),
            },
            "robustness_reported_not_redeciding": [
                "leave-one-quarter-out",
                "leave-one-sector-out",
                "leave-top-10-events-out",
                "decile monotonicity of hedged returns vs pead_score_v1",
            ],
            "caps": [
                "no single quarter > 40% of P&L",
                "no single name > 15%",
                "survives gapper-slippage 50bps stress",
            ],
            "shots": (
                "one confirmatory run (R4); re-test requires a registered "
                "cooling-off."),
        },
    },
    {
        "hypothesis_id": "H2-IMPLIEDMOVE-CONTINUOUS-20260611",
        "label": "confirmatory",
        "family": "options_signal",
        "universe": "russell1000 (options-covered)",
        "window": "2022-06->2026-06",
        "folds": "event-level (announce_date x firm clustered)",
        "params": {"component_type": "filter", "H_label": "H2"},
        "acceptance_criteria": {
            "mechanism": (
                "Settle OPT-5 PROPERLY as a CONTINUOUS feature (no binary "
                "thresholds - the OPT-5 trap is explicitly forbidden)."),
            "feature": (
                "reaction_ratio = |announce-day move| / pre-event implied "
                "move (continuous)."),
            "panel": "FULL options-covered panel 2022-06->2026-06.",
            "hypothesis": (
                "reaction_ratio has a NEGATIVE monotonic relationship with "
                "hedged forward drift (under-reaction proxy: small realized "
                "vs implied -> more residual drift)."),
            "instrument": (
                "Two-way clustered OLS coefficient on reaction_ratio (sign + "
                "significance) + decile monotonicity of hedged forward drift "
                "across reaction_ratio deciles."),
            "target": (
                "SPY-hedged forward drift, horizons {5,10,20}; PRIMARY = "
                "10d."),
            "decision_rule": {
                "confirmed": (
                    "coefficient NEGATIVE AND day-clustered t <= -2 AND "
                    "monotonic deciles -> implied-move signal confirmed as a "
                    "continuous feature; eligible as a PEAD v2 input / sizing "
                    "tilt (paper telemetry behind replay-diff)."),
                "not_confirmed": (
                    "not significant OR non-monotonic OR wrong sign -> "
                    "logged, PARKED, NO threshold-hunting (the OPT-5 FRAGILE "
                    "verdict stands)."),
            },
            "shots": "one confirmatory run (R4).",
        },
    },
    {
        "hypothesis_id": "H3-PEADV2-SCORECARD-20260611",
        "label": "confirmatory",
        "family": "pead",
        "universe": ("russell1000 (options-covered subset for options cols; "
                     "equity cols on 2019+ extension)"),
        "window": ("train 2022-24, validate 2025-26 (sacred holdout "
                   "2026-11-09 untouched)"),
        "folds": "event-level (announce_date x firm clustered)",
        "params": {
            "component_type": "alpha",
            "model_class": ("regularized monotonic scorecard (logistic/linear "
                            "or shallow GAM bins) - explicitly NOT XGBoost"),
            "features": H3_FEATURES,
            "H_label": "H3",
        },
        "acceptance_criteria": {
            "mechanism": (
                "PEAD v2 - a continuous, options-conditioned event score."),
            "model_class": (
                "Regularized MONOTONIC scorecard (logistic/linear or shallow "
                "GAM bins); explicitly NOT XGBoost (small-cluster overfit "
                "warning)."),
            "feature_list_frozen": H3_FEATURES,
            "protocol": (
                "Train 2022-24, validate 2025-26; sacred holdout (2026-11-09) "
                "untouched."),
            "target": (
                "Top-vs-bottom-decile hedged forward spread; PRIMARY = 10d."),
            "decision_rule": {
                "pass": (
                    "top-minus-bottom decile hedged spread > 0 AND "
                    "day-clustered t >= 2 AND monotonic deciles on validation "
                    "-> PEAD v2 to paper telemetry behind replay-diff, "
                    "evidence-tiered sizing."),
                "fail": (
                    "any leg fails -> logged, PARKED, no threshold-hunting."),
            },
            "caps": [
                "no single quarter > 40% P&L",
                "no single name > 15%",
                ("survives 50bps gapper stress + the fill_quality empirical "
                 "distribution"),
            ],
            "shots": "one confirmatory run (R4).",
        },
    },
]


def _head_commit() -> str | None:
    """Current HEAD short sha (the code the criteria were frozen against)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def preregister_all(registry=None) -> list[str]:
    """register()+preregister() all three hypotheses. Idempotent: R1/R5 on a
    re-run are reported and skipped (the frozen rows are left untouched).
    Returns the hypothesis_ids verified present AND preregistered."""
    from app.research.registry import RegistryIntegrityError, ResearchRegistry

    reg = registry if registry is not None else ResearchRegistry()
    commit = _head_commit()
    verified: list[str] = []

    for h in HYPOTHESES:
        hid = h["hypothesis_id"]
        criteria = h["acceptance_criteria"]
        try:
            reg.register(
                hid,
                label=h["label"],
                family=h["family"],
                params=h["params"],
                universe=h["universe"],
                window=h["window"],
                folds=h["folds"],
                code_commit=commit,
                mechanism=criteria["mechanism"],
            )
            print(f"[registry] registered   {hid} (label={h['label']}, "
                  f"family={h['family']}, code_commit={commit})")
        except RegistryIntegrityError:
            # Idempotent skip ONLY when the row genuinely already exists (R1 on a
            # re-run). Any other integrity error (bad label/params/parent on a
            # row that is NOT present) is a real defect — re-raise it rather than
            # masquerade it as a benign "already registered".
            if reg.get(hid) is None:
                raise
            print(f"[registry] register     {hid}: already registered (R1) - skipped")
        try:
            reg.preregister(
                hid,
                acceptance_criteria=criteria,
                preregistered_at=PREREGISTERED_AT,
            )
            print(f"[registry] preregistered {hid} at {PREREGISTERED_AT} "
                  f"(criteria FROZEN)")
        except RegistryIntegrityError:
            # R5: already preregistered — criteria are immutable; skip.
            print(f"[registry] preregister  {hid}: already preregistered (R5) "
                  f"- criteria unchanged")

        row = reg.get(hid)
        if row is not None and row.get("preregistered_at") is not None:
            verified.append(hid)
    return verified


def main() -> int:
    from app.research.registry import ResearchRegistry

    reg = ResearchRegistry()
    print(f"[registry] db: {reg.db_path}")
    verified = preregister_all(reg)

    print()
    print("=" * 78)
    print("  PHASE-3 CONFIRMATORY HYPOTHESES - pre-registration summary")
    print("=" * 78)
    for h in HYPOTHESES:
        hid = h["hypothesis_id"]
        row = reg.get(hid)
        status = ("PREREGISTERED" if row and row.get("preregistered_at")
                  else "MISSING/NOT-PREREGISTERED")
        prereg = row.get("preregistered_at") if row else None
        print(f"  {hid:<38} {status}  (preregistered_at={prereg})")
    print("=" * 78)

    if len(verified) != len(HYPOTHESES):
        print("  ERROR: not all hypotheses are preregistered - inspect the "
              "registry above.")
        return 1
    print(f"  All {len(verified)} hypotheses frozen. Each gets ONE "
          f"confirmatory run (R4); runs must execute after "
          f"{PREREGISTERED_AT} (R2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
