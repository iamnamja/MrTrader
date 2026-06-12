"""
preregister_options_xs_features.py - freeze H4a-H4e in the research registry.

Phase-4 prep (blueprint NEXT_PHASE_BLUEPRINT_2026-06.md: "options-as-signal" -
the cross-sectional EQUITY-sleeve adjudication of five options-derived features
on the R1K options-quality-qualified universe). This script register()+
preregister()s the FIVE confirmatory cross-sectional hypotheses with FROZEN
acceptance criteria - the scientific contract for the Phase-4 L/S runs. Each
gets exactly ONE confirmatory shot (R4); runs must EXECUTE strictly after
PREREGISTERED_AT (R2).

The line is a SIMPLE-decile-sort test, on purpose: the kill rule (below) is
"simple decile sorts show nothing net of costs -> CLOSE the line; do NOT
escalate to ML combinations" - this is NOT a revival of the dead XS-ML.

Idempotent and safe to re-run: an already-registered id (R1) or an already-
preregistered row (R5) is skipped with a notice, mirroring
scripts/preregister_event_hypotheses.py. Criteria are IMMUTABLE once
preregistered - editing this file and re-running cannot move the goalposts
(the registry refuses); a revision requires a NEW hypothesis_id with parent_id.

Usage:
  python -m scripts.preregister_options_xs_features
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

# The instant the criteria were frozen (2026-06-12 PR4a). Every confirmatory
# run must EXECUTE strictly after this (R2: preregistered_at < run_at).
PREREGISTERED_AT = "2026-06-12T12:00:00+00:00"

# ── The shared acceptance template (the SAME instrument / decision_rule / shots /
# caps for every Hn; only mechanism + direction + feature differ). Frozen ONCE
# here and spread into each hypothesis so the five can never drift apart. ─────
_SHARED = {
    "instrument": (
        "weekly dollar-neutral L/S EQUITY sleeve on R1K options-quality-"
        "qualified names; per-feature DECILE sort (before any multivariate "
        "combination); panel inference with WEEKS as clusters (t-stat on the "
        "weekly L/S long-short spread) + decile monotonicity; multi-factor "
        "residual alpha (ETF set SPY/IWM-SPY/MTUM-SPY/VLUE-SPY/VIXY) must be "
        "positive; equity costs (10bps + spread stress); CPCV via an "
        "EventEdge-style L/S adapter as the robustness backstop."
    ),
    "decision_rule": {
        "pass": (
            "week-clustered t>=2 in the hypothesized direction AND monotonic "
            "deciles AND positive cost-net multi-factor residual alpha"
        ),
        "kill": (
            "simple decile sorts show nothing net of costs -> CLOSE the line; "
            "do NOT escalate to ML combinations (this is NOT a revival of the "
            "dead XS-ML)"
        ),
    },
    "shots": "one confirmatory run (R4).",
    "caps": (
        "4y window = one regime; paper time is the extension; expect post-"
        "publication crowding decay."
    ),
}

# ── The five frozen hypotheses (blueprint Phase 4, options-as-signal) ─────────
# Each Hn's `direction` is the FROZEN sign hypothesis; `feature` is the column
# in data/options_features.parquet (app/data/options_features.py) it sorts on.
HYPOTHESES = [
    {
        "hypothesis_id": "H4a-OPTIONS-CPIV-20260612",
        "mechanism": (
            "Cremers-Weinbaum call-put matched-delta IV spread as a cross-"
            "sectional EQUITY signal (information at equity cost)."
        ),
        "direction": (
            "HIGH cpiv -> POSITIVE forward equity return; L/S = long high-CPIV "
            "/ short low-CPIV; expected coefficient POSITIVE."
        ),
        "params": {"component_type": "alpha", "feature": "cpiv_matched_delta",
                   "H_label": "H4a"},
    },
    {
        "hypothesis_id": "H4b-OPTIONS-SKEW-20260612",
        "mechanism": "Xing-Zhang-Zhao 25-delta put skew.",
        "direction": (
            "STEEP skew -> NEGATIVE forward return; L/S = short steep-skew / "
            "long flat-skew; expected coefficient NEGATIVE."
        ),
        "params": {"component_type": "alpha", "feature": "skew_25d_put",
                   "H_label": "H4b"},
    },
    {
        "hypothesis_id": "H4c-OPTIONS-OSRATIO-20260612",
        "mechanism": (
            "Roll-Schwartz-Subrahmanyam option/stock volume (O/S), put-"
            "heaviness-conditioned."
        ),
        "direction": (
            "HIGH O/S with put-heavy flow -> NEGATIVE forward return; L/S = "
            "short high-put-O/S / long low; expected coefficient NEGATIVE."
        ),
        "params": {"component_type": "alpha", "feature": "opt_share_volume_ratio",
                   "conditioned_on": "put_call_volume_ratio", "H_label": "H4c"},
    },
    {
        "hypothesis_id": "H4d-OPTIONS-TERMSLOPE-20260612",
        "mechanism": "ATM IV term-structure slope (60d-30d).",
        "direction": (
            "BACKWARDATION (negative slope, front IV elevated) -> near-term "
            "stress -> NEGATIVE forward return; L/S = long contango (high "
            "slope) / short backwardation (low slope); expected coefficient on "
            "term_slope POSITIVE."
        ),
        "params": {"component_type": "alpha", "feature": "term_slope_30_60",
                   "H_label": "H4d"},
    },
    {
        "hypothesis_id": "H4e-OPTIONS-IVRV-20260612",
        "mechanism": (
            "IV/RV richness as an EQUITY residual-return predictor (NOT a vol "
            "trade); low-vol-anomaly grounding."
        ),
        "direction": (
            "HIGH IV/RV (rich implied) -> NEGATIVE forward return; L/S = short "
            "high-IV/RV / long low; expected coefficient NEGATIVE."
        ),
        "params": {"component_type": "alpha", "feature": "iv_rv_20d_ratio",
                   "H_label": "H4e"},
    },
]

# Static metadata shared by all five rows.
_FAMILY = "options_signal"
_LABEL = "confirmatory"
_UNIVERSE = "russell1000 (options-quality-qualified)"
_WINDOW = "2022-06->2026-06 (4y options-greeks store; one regime)"
_FOLDS = "weekly L/S panel, WEEKS as clusters + CPCV L/S adapter backstop"


def _acceptance_criteria(h: dict) -> dict:
    """Assemble the full frozen acceptance_criteria for one hypothesis from the
    shared template + its mechanism/direction. Putting the rich detail here (and
    only component_type/feature in params) matches preregister_event_hypotheses."""
    return {
        "mechanism": h["mechanism"],
        "feature": h["params"]["feature"],
        "direction": h["direction"],
        "instrument": _SHARED["instrument"],
        "decision_rule": _SHARED["decision_rule"],
        "shots": _SHARED["shots"],
        "caps": _SHARED["caps"],
    }


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
    """register()+preregister() all five hypotheses. Idempotent: R1/R5 on a
    re-run are reported and skipped (the frozen rows are left untouched).
    Returns the hypothesis_ids verified present AND preregistered."""
    from app.research.registry import RegistryIntegrityError, ResearchRegistry

    reg = registry if registry is not None else ResearchRegistry()
    commit = _head_commit()
    verified: list[str] = []

    for h in HYPOTHESES:
        hid = h["hypothesis_id"]
        criteria = _acceptance_criteria(h)
        try:
            reg.register(
                hid,
                label=_LABEL,
                family=_FAMILY,
                params=h["params"],
                universe=_UNIVERSE,
                window=_WINDOW,
                folds=_FOLDS,
                code_commit=commit,
                mechanism=criteria["mechanism"],
            )
            print(f"[registry] registered   {hid} (label={_LABEL}, "
                  f"family={_FAMILY}, code_commit={commit})")
        except RegistryIntegrityError:
            # Idempotent skip ONLY when the row genuinely already exists (R1 on a
            # re-run). Any other integrity error (bad label/params on a row that
            # is NOT present) is a real defect — re-raise rather than mask it.
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
    print("  PHASE-4 OPTIONS-XS CONFIRMATORY HYPOTHESES - pre-registration summary")
    print("=" * 78)
    for h in HYPOTHESES:
        hid = h["hypothesis_id"]
        row = reg.get(hid)
        status = ("PREREGISTERED" if row and row.get("preregistered_at")
                  else "MISSING/NOT-PREREGISTERED")
        prereg = row.get("preregistered_at") if row else None
        feat = h["params"]["feature"]
        print(f"  {hid:<34} {status}  (feature={feat}, preregistered_at={prereg})")
    print("=" * 78)

    if len(verified) != len(HYPOTHESES):
        print("  ERROR: not all hypotheses are preregistered - inspect the "
              "registry above.")
        return 1
    print(f"  All {len(verified)} hypotheses frozen. Each gets ONE confirmatory "
          f"run (R4); runs must execute after {PREREGISTERED_AT} (R2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
