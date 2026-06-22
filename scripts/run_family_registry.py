"""
run_family_registry.py — Alpha-v10 P0.5: print the strategy-family registry + the program's
true N_TRIALS (family-level) + the research degrees-of-freedom log.

This is the auditable replacement for the hardcoded "~20 families" estimate that fed the GL-0
deflated-Sharpe cross-check. The enumerated count here is what `null_zoo`'s parametric DSR now uses.

    PYTHONPATH=. venv/Scripts/python scripts/run_family_registry.py
"""
from __future__ import annotations

from app.research import family_registry as fr


def main() -> None:
    print("=" * 84)
    print("  Alpha-v10 P0.5 — Strategy-Family Registry (the program's true N_TRIALS)")
    print("=" * 84)

    cur_ac = None
    for f in sorted(fr.FAMILIES, key=lambda x: (x.asset_class, x.id)):
        if f.asset_class != cur_ac:
            cur_ac = f.asset_class
            print(f"\n[{cur_ac}]")
        print(f"  {f.status:15s} {f.name:42s} - {f.verdict}")

    excluded = [f.id for f in fr.FAMILIES if not f.counts_as_trial]
    print("\n" + "-" * 84)
    print("  COUNT BY STATUS:", fr.count_by_status())
    print(f"  FAMILY-LEVEL TRIAL COUNT (N_TRIALS) = {fr.family_trial_count()}"
          f"  (of {len(fr.FAMILIES)} registry entries; excluded as infra/ensemble: {excluded})")
    print("-" * 84)

    print("\n  RESEARCH DEGREES-OF-FREEDOM (within-family search burden):")
    for d in fr.DEGREES_OF_FREEDOM:
        print(f"   - {d}")

    print("\n  -> null_zoo's parametric Deflated-Sharpe cross-check now deflates at this enumerated")
    print("     N (was a hardcoded ~20). The PRIMARY GL-0 test remains the empirical max-stat null.")


if __name__ == "__main__":
    main()
