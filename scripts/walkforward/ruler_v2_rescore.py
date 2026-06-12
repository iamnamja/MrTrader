"""
ruler_v2_rescore.py — re-score the kill ledger under Ruler v2 (Alpha-v7 Phase B,
Phase 5). STRICTLY REPORT-ONLY.

The Alpha-v6 confirmatory program killed/parked every hypothesis (H1 DEMOTE, H2
NOT_CONFIRMED, H3 BLOCKED, H4a–e KILL, P5 PARK) under the SIGNIFICANCE gate. Ruler v2
deliberately moves the PAPER bar from significance to PLAUSIBILITY — so the natural
question is: would any of those kills be REVIVED to a paper slot under the new ruler,
and would any prior pass be DEMOTED? This module answers that as a comparison TABLE.

⚠️ HARD CONTRACT — this module:
  • NEVER flips GATE_MODE / TRACKB_MODE,
  • NEVER writes the research registry or any verdict,
  • NEVER promotes, demotes, or re-tests anything live,
  • only TABULATES what `ruler_v2.evaluate` WOULD say for each result vs the recorded
    significance verdict.
A flip in this table is a CANDIDATE for the owner's review (a sanctioned, R4-logged
re-test is a separate, human-initiated step) — it is NOT an action. PURE: arrays in,
dataclasses out, no I/O, no config mutation.

CAPITAL is reported too, but a re-scored kill almost never has a live-paper record, so
its CAPITAL column will read FAIL on `live_paper_present` by construction (Ruler v2's
posterior is P(SR>0 | backtest AND live paper)). The MEANINGFUL column is PAPER.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from app.research import ruler_v2

# Flip classification on the PAPER tier (the meaningful revival question).
FLIP_REVIVED = "REVIVED_PAPER"        # significance FAIL → ruler_v2 PAPER PASS
FLIP_DEMOTED = "DEMOTED_PAPER"        # significance PASS → ruler_v2 PAPER FAIL
FLIP_UNCHANGED_PASS = "UNCHANGED_PASS"
FLIP_UNCHANGED_FAIL = "UNCHANGED_FAIL"


@dataclass
class RescoreRow:
    label: str
    declared_class: str
    n_obs: int
    n_folds: int
    # Recorded SIGNIFICANCE verdict (the gate that produced the kill ledger).
    sig_paper_pass: bool
    sig_capital_pass: bool
    # Ruler-v2 verdict (report-only re-score).
    rv2_paper_pass: bool
    rv2_capital_pass: bool
    rv2_paper_failed: List[str]
    rv2_capital_failed: List[str]
    # Key Ruler-v2 statistics (for the owner's eyeball; not actions).
    point_sr: float
    posterior_p_sr_gt_0: float
    bootstrap_p_sr_gt_0: float
    residual_alpha_t_hac: Optional[float]
    paper_flip: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_paper_flip(sig_pass: bool, rv2_pass: bool) -> str:
    if sig_pass and rv2_pass:
        return FLIP_UNCHANGED_PASS
    if sig_pass and not rv2_pass:
        return FLIP_DEMOTED
    if not sig_pass and rv2_pass:
        return FLIP_REVIVED
    return FLIP_UNCHANGED_FAIL


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def rescore_result(result, *, label: str, declared_class: str = "",
                   n_trials: Optional[int] = None,
                   live_paper: Optional[dict] = None,
                   regime_waiver_approved: bool = False,
                   notes: str = "") -> RescoreRow:
    """Re-score ONE CPCVResult under Ruler v2, alongside its recorded significance
    verdict. REPORT-ONLY — no flips, no writes. The result must carry the Phase-2
    `oos_returns_dated` field for the Ruler-v2 inference to run (else its PAPER point-SR
    floor fails closed and the row is honestly a FAIL, flagged in notes)."""
    # Recorded SIGNIFICANCE verdict (call the significance path directly so the answer
    # does not depend on the ambient GATE_MODE).
    sig_paper = bool(_safe(lambda: result.significance_gate_passed(tier="paper"), False))
    sig_capital = bool(_safe(
        lambda: result.significance_gate_passed(tier="capital"), False))

    # Ruler-v2 re-score.
    rv2_paper = bool(ruler_v2.gate_passed(
        result, tier="paper", n_trials=n_trials,
        regime_waiver_approved=regime_waiver_approved))
    rv2_capital = bool(ruler_v2.gate_passed(
        result, tier="capital", n_trials=n_trials, live_paper=live_paper,
        regime_waiver_approved=regime_waiver_approved))
    pd_detail = ruler_v2.evaluate(result, tier="paper", n_trials=n_trials,
                                  regime_waiver_approved=regime_waiver_approved)
    cap_detail = ruler_v2.evaluate(result, tier="capital", n_trials=n_trials,
                                   live_paper=live_paper,
                                   regime_waiver_approved=regime_waiver_approved)
    paper_failed = [k for k, (_v, ok) in pd_detail.items()
                    if not ok and k not in ruler_v2.INFORMATIONAL_KEYS]
    cap_failed = [k for k, (_v, ok) in cap_detail.items()
                  if not ok and k not in ruler_v2.INFORMATIONAL_KEYS]

    point_sr = float(pd_detail.get("point_sr_floor", (float("nan"), False))[0])
    post_p = float(cap_detail.get("posterior_p_sr_gt_0", (float("nan"), False))[0])
    boot_p = float(cap_detail.get("bootstrap_p_sr_gt_0", (float("nan"), False))[0])
    ra_t = getattr(result, "residual_alpha_t_hac", None)

    n_obs = len(getattr(result, "oos_returns_dated", []) or [])
    auto_notes = notes
    if n_obs == 0:
        auto_notes = (auto_notes + "; " if auto_notes else "") + \
            "no oos_returns_dated — Ruler-v2 inference fails closed (legacy result?)"

    return RescoreRow(
        label=label, declared_class=declared_class, n_obs=n_obs,
        n_folds=int(getattr(result, "n_folds", 0) or 0),
        sig_paper_pass=sig_paper, sig_capital_pass=sig_capital,
        rv2_paper_pass=rv2_paper, rv2_capital_pass=rv2_capital,
        rv2_paper_failed=paper_failed, rv2_capital_failed=cap_failed,
        point_sr=point_sr, posterior_p_sr_gt_0=post_p,
        bootstrap_p_sr_gt_0=boot_p, residual_alpha_t_hac=ra_t,
        paper_flip=_classify_paper_flip(sig_paper, rv2_paper), notes=auto_notes)


def rescore_table(items: List[Tuple], *, n_trials: Optional[int] = None) -> List[RescoreRow]:
    """Re-score many results. `items` is an iterable of (label, result) or
    (label, declared_class, result). REPORT-ONLY."""
    rows: List[RescoreRow] = []
    for it in items:
        if len(it) == 3:
            label, declared_class, result = it
        else:
            (label, result), declared_class = it, ""
        rows.append(rescore_result(result, label=label,
                                   declared_class=declared_class, n_trials=n_trials))
    return rows


def summarize_flips(rows: List[RescoreRow]) -> Dict[str, int]:
    """Count PAPER-tier flips across a re-score table. The owner reviews REVIVED_PAPER
    rows (Type-II recoveries the new ruler proposes) and DEMOTED_PAPER rows."""
    out = {FLIP_REVIVED: 0, FLIP_DEMOTED: 0,
           FLIP_UNCHANGED_PASS: 0, FLIP_UNCHANGED_FAIL: 0}
    for r in rows:
        out[r.paper_flip] = out.get(r.paper_flip, 0) + 1
    return out


def format_rescore_table(rows: List[RescoreRow]) -> str:
    """Human-readable OC-style table. REPORT-ONLY framing baked into the header."""
    counts = summarize_flips(rows)
    lines = [
        "Ruler-v2 kill-ledger RE-SCORE (REPORT-ONLY — no flips, owner adjudicates)",
        "=" * 78,
        f"{'label':<26}{'class':<14}{'sigP':>5}{'rv2P':>5}{'flip':>16}{'ptSR':>7}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"{r.label[:25]:<26}{r.declared_class[:13]:<14}"
            f"{('Y' if r.sig_paper_pass else 'n'):>5}"
            f"{('Y' if r.rv2_paper_pass else 'n'):>5}"
            f"{r.paper_flip:>16}{r.point_sr:>7.2f}")
    lines += [
        "-" * 78,
        f"REVIVED(paper): {counts[FLIP_REVIVED]}   DEMOTED(paper): {counts[FLIP_DEMOTED]}"
        f"   unchanged pass/fail: {counts[FLIP_UNCHANGED_PASS]}/{counts[FLIP_UNCHANGED_FAIL]}",
        "REVIVED rows are CANDIDATES for an owner-initiated, R4-logged re-test — "
        "NOT promotions.",
    ]
    return "\n".join(lines)
