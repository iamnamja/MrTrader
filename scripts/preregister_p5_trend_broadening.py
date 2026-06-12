"""
preregister_p5_trend_broadening.py — freeze the ONE Alpha-v6 P5 trend-broadening
confirmatory hypothesis (Track B).

P5 broadens the validated TSMOM sleeve (app/strategy/tsmom.py, +0.71 standalone
Sharpe over 19y) on the 19y window where t≥2 is REACHABLE (t ≈ Sharpe·√years).
The spec is FROZEN here BEFORE the run — a SINGLE broadened configuration, NOT a
sweep (no universe/lever grid search — that is the OPT-5 overfitting trap).

The three broadening levers (owner-approved 2026-06-12): (1) MORE LEGS — the 10-ETF
core + 6 liquid diversifiers (credit / short-rates / silver / Europe / Japan);
(2) LONG-SHORT — allow_short=True (capture downtrends, more crisis-positive);
(3) BOOK-LEVEL VOL TARGETING — a 10% annualized book-vol overlay.

Acceptance (FROZEN, "beat the book + significant"): the broadened sleeve PASSES iff
its 19y standalone Sharpe is significant (t ≈ Sharpe·√years ≥ 2.0) AND it DOMINATES
the current 10-ETF live sleeve as a book contributor (higher Sharpe AND maxDD not
materially worse). Otherwise PARK (the current sleeve stays; no lever-hunting).

ONE confirmatory shot (R4); the run must EXECUTE strictly after PREREGISTERED_AT.
Usage: python -m scripts.preregister_p5_trend_broadening
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

HYPOTHESIS_ID = "P5-TRENDBROADEN-20260612"
PREREGISTERED_AT = "2026-06-12T16:00:00+00:00"

# The 10-ETF live core (the current sleeve / the baseline to beat).
CORE_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP"]
# +6 liquid diversifiers, all with >=2007 history (credit / short-rates / silver /
# Europe / Japan). Pre-committed, NOT swept.
ADDED_LEGS = ["HYG", "LQD", "SHY", "SLV", "VGK", "EWJ"]
BROADENED_UNIVERSE = CORE_UNIVERSE + ADDED_LEGS

# The FROZEN broadened TSMOMConfig (the exact spec the confirmatory run uses).
BROADENED_SPEC = {
    "universe": BROADENED_UNIVERSE,
    "lookbacks": [21, 63, 126, 252],
    "vol_lookback": 60,
    "target_vol": 0.10,
    "rebalance_days": 5,
    "max_weight": 0.25,
    "max_gross": 3.0,            # long-short headroom; the book-vol overlay is the risk control
    "allow_short": True,
    "vol_floor": 0.03,
    "cost_bps": 2.0,
    "book_vol_target": 0.10,     # 10% annualized book vol
    "book_vol_max_leverage": 2.0,
}

ACCEPTANCE = {
    "mechanism": (
        "Broaden the validated TSMOM trend sleeve on 19y where t>=2 is reachable: "
        "more legs (16-ETF multi-asset) + long-short + a 10% book-vol overlay."
    ),
    "instrument": (
        "tsmom_backtest on the FROZEN BROADENED_SPEC over ~2007->now (19y), vs the "
        "current 10-ETF live sleeve as the baseline. Standalone annualized Sharpe + "
        "its t-stat (t ~= Sharpe*sqrt(years)) + per-regime/crisis behavior; the same "
        "metrics for the 10-ETF baseline on the identical window."
    ),
    "decision_rule": {
        "pass": (
            "broadened 19y Sharpe t>=2 AND broadened DOMINATES the 10-ETF baseline "
            "(Sharpe higher AND maxDD not materially worse) -> the broadened sleeve "
            "graduates as the candidate live trend sleeve (deployment STILL gated on "
            "the live trend replay-diff being boring; owner-executed)."
        ),
        "park": (
            "not significant OR does not beat the baseline -> PARK; the current "
            "10-ETF sleeve stays; NO lever-hunting / universe sweep (OPT-5 trap)."
        ),
    },
    "frozen_spec": BROADENED_SPEC,
    "baseline": {"universe": CORE_UNIVERSE, "note": "current live sleeve, long-flat, "
                 "per-instrument vol target, no book-vol overlay"},
    "window": "~2007-01-01 -> now (19y; Track-B power window)",
    "shots": "one confirmatory run (R4).",
    "caps": "single frozen spec; no sweep; deployment owner-gated on live fidelity.",
}


def _head_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def preregister(registry=None) -> str:
    from app.research.registry import RegistryIntegrityError, ResearchRegistry
    reg = registry or ResearchRegistry()
    commit = _head_commit()
    try:
        reg.register(HYPOTHESIS_ID, label="confirmatory", family="trend_broadening",
                     params={"component_type": "diversifier", "track": "B",
                             "spec": BROADENED_SPEC},
                     universe="16-ETF multi-asset (10 core + 6 diversifiers)",
                     window=ACCEPTANCE["window"], folds="19y standalone + baseline compare",
                     code_commit=commit, mechanism=ACCEPTANCE["mechanism"])
        print(f"[registry] registered   {HYPOTHESIS_ID} (code_commit={commit})")
    except RegistryIntegrityError:
        if reg.get(HYPOTHESIS_ID) is None:
            raise
        print(f"[registry] register     {HYPOTHESIS_ID}: already registered - skipped")
    try:
        reg.preregister(HYPOTHESIS_ID, acceptance_criteria=ACCEPTANCE,
                        preregistered_at=PREREGISTERED_AT)
        print(f"[registry] preregistered {HYPOTHESIS_ID} at {PREREGISTERED_AT} (FROZEN)")
    except RegistryIntegrityError:
        print(f"[registry] preregister  {HYPOTHESIS_ID}: already preregistered - unchanged")
    return HYPOTHESIS_ID


def main() -> int:
    from app.research.registry import ResearchRegistry
    reg = ResearchRegistry()
    print(f"[registry] db: {reg.db_path}")
    preregister(reg)
    row = reg.get(HYPOTHESIS_ID)
    ok = row is not None and row.get("preregistered_at") is not None
    print("=" * 70)
    print(f"  P5 trend-broadening {'FROZEN' if ok else 'NOT preregistered'} "
          f"(feature spec: {len(BROADENED_UNIVERSE)} ETFs, long-short, "
          f"book-vol {BROADENED_SPEC['book_vol_target']:.0%})")
    print(f"  ONE confirmatory run (R4); run_at must be after {PREREGISTERED_AT}.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
