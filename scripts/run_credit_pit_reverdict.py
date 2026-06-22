"""
run_credit_pit_reverdict.py — Alpha-v10 P0.4: PIT re-verdict for the credit de-risk overlay.

The 2nd 5-LLM panel (Alpha-v10) bundled the credit overlay with carry under "kill in-sample
vol-matching → the +0.17 carry dSR AND the +0.064 credit overlay likely shrink under PIT vol."
P0.2 confirmed that for CARRY (its in-sample +0.17 dSR was a vol-match artifact → ~0 under PIT).
This script tests the same hypothesis for the CREDIT overlay — and refutes it:

  • The credit overlay is a TIME-VARYING de-risk MULTIPLIER applied to the trend book
    (`m[t]·base[t]`). Sharpe is scale-invariant, so a constant vol change can't move it — only
    the overlay's TIMING can. Its dSharpe therefore does NOT depend on vol-matching at all.
  • The baseline trend book it is measured against is PIT — trailing-window signals/realized-vol
    and `held.shift(1)` earning (no look-ahead). (`book_vol_target` is OFF on the live trend book,
    and where it IS used elsewhere it is itself PIT/non-circular — so no in-sample vol enters.)

So there is no in-sample vol-matching in the credit overlay's evaluation to remove. Running the
same G1 confirmatory harness (`run_credit_curve`) on current data reproduces +0.064 marginal
dSharpe — the number is PIT-robust. The binding caveat is the one DECISIONS 2026-06-14 already
logged: the L=120/band=0.02 trigger is POST-HOC (the pre-registered L=60/band=0 failed) →
multiplicity, not vol. Verdict: status UNCHANGED (marginal tail-insurance CANDIDATE, flag OFF);
the PIT-vol question is now closed.

    PYTHONPATH=. venv/Scripts/python scripts/run_credit_pit_reverdict.py
"""
from __future__ import annotations

from dataclasses import dataclass

# The marginal-dSharpe floor below which we'd call the +0.064 "shrunk to noise" (the panel's
# hypothesis). The carry analogue collapsed to ~0.00; we treat <0.02 marginal dSharpe as shrunk.
SHRUNK_DSHARPE = 0.02


@dataclass(frozen=True)
class CreditPitVerdict:
    marginal_dsharpe: float
    marginal_dcalmar: float
    marginal_dmaxdd: float
    h1_dmaxdd: float
    h2_dmaxdd: float
    crises_helped: int
    crises_total: int

    @property
    def pit_robust(self) -> bool:
        """True if the marginal benefit SURVIVES (did NOT shrink to noise) — i.e. the panel's
        'shrinks under PIT vol' hypothesis is REFUTED. Requires the marginal dSharpe to clear the
        shrunk floor AND the overlay to still reduce tail risk (dMaxDD >= 0)."""
        return self.marginal_dsharpe >= SHRUNK_DSHARPE and self.marginal_dmaxdd >= 0.0

    @property
    def both_halves_tail_positive(self) -> bool:
        return self.h1_dmaxdd > 0 and self.h2_dmaxdd > 0

    @property
    def verdict(self) -> str:
        if not self.pit_robust:
            return "SHRUNK_KILL"          # panel hypothesis confirmed → kill
        return "PIT_ROBUST_CANDIDATE"     # survives PIT; remains a marginal candidate (flag-off)

    def summary(self) -> str:
        return (
            f"credit overlay marginal-vs-VIX-governor: dSharpe={self.marginal_dsharpe:+.4f} "
            f"dCalmar={self.marginal_dcalmar:+.4f} dMaxDD={self.marginal_dmaxdd:+.4f} | "
            f"both-halves dMaxDD H1={self.h1_dmaxdd:+.4f} H2={self.h2_dmaxdd:+.4f} | "
            f"crises helped {self.crises_helped}/{self.crises_total} | "
            f"PIT-robust={self.pit_robust} -> {self.verdict}")


def build_verdict_from_report(marginal, h1_dmaxdd: float = 0.0,
                              h2_dmaxdd: float = 0.0) -> CreditPitVerdict:
    """Build the P0.4 verdict from an OverlayReport (marginal-vs-governor) + the both-halves
    dMaxDD split. Pure — unit-testable without re-running the full backtest."""
    crisis = marginal.crisis or {}
    helped = sum(1 for d in crisis.values() if d.get("dd_improve", 0.0) > 0)
    return CreditPitVerdict(
        marginal_dsharpe=float(marginal.d_sharpe),
        marginal_dcalmar=float(marginal.d_calmar),
        marginal_dmaxdd=float(marginal.d_max_dd),
        h1_dmaxdd=float(h1_dmaxdd), h2_dmaxdd=float(h2_dmaxdd),
        crises_helped=helped, crises_total=len(crisis))


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    from scripts.walkforward.sleeves import run_credit_curve

    print("=" * 78)
    print("  Alpha-v10 P0.4 — credit de-risk overlay: PIT re-verdict")
    print("=" * 78)
    out = run_credit_curve()
    cg = out["credit_governor"]
    v = build_verdict_from_report(cg["marginal"], cg["h1_d_max_dd"], cg["h2_d_max_dd"])

    print("\n" + "=" * 78)
    print("  P0.4 VERDICT")
    print("=" * 78)
    print("  " + v.summary())
    print()
    if v.verdict == "PIT_ROBUST_CANDIDATE":
        print("  -> PIT-vol hypothesis REFUTED: the +0.064 is NOT a vol-match artifact (unlike")
        print("    carry). The overlay is a scale-invariant multiplier on an already-PIT book.")
        print("    Status UNCHANGED: marginal tail-insurance CANDIDATE, flag OFF. Binding caveat")
        print("    is the POST-HOC L=120/band=0.02 trigger (multiplicity), not vol — see")
        print("    DECISIONS 2026-06-14 (G1). The PIT-vol question is now closed.")
    else:
        print("  -> PIT-vol hypothesis CONFIRMED: the marginal benefit shrank to noise -> KILL.")


if __name__ == "__main__":
    main()
