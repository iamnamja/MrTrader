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

import copy
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from app.research import ruler_v2

# Flip classification on the PAPER tier (the meaningful revival question).
FLIP_REVIVED = "REVIVED_PAPER"        # significance FAIL → ruler_v2 PAPER PASS
FLIP_DEMOTED = "DEMOTED_PAPER"        # significance PASS → ruler_v2 PAPER FAIL
FLIP_UNCHANGED_PASS = "UNCHANGED_PASS"
FLIP_UNCHANGED_FAIL = "UNCHANGED_FAIL"
FLIP_ERROR_SIG = "ERROR_SIG"          # significance verdict could not be computed


@dataclass
class RescoreRow:
    label: str
    declared_class: str
    n_obs: int
    n_folds: int
    # Recorded SIGNIFICANCE verdict (the gate that produced the kill ledger). None
    # when it could not be scored (the row is then ERROR_SIG, never a flip).
    sig_paper_pass: Optional[bool]
    sig_capital_pass: Optional[bool]
    # Recorded significance discriminators — shown so the owner can see WHY the two
    # rulers disagree (path-Sharpe distribution vs the pooled OOS series).
    path_sharpe_tstat: float
    mean_sharpe: float
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


def _classify_paper_flip(sig_pass: Optional[bool], rv2_pass: bool) -> str:
    if sig_pass is None:               # significance verdict unavailable → never a flip
        return FLIP_ERROR_SIG
    if sig_pass and rv2_pass:
        return FLIP_UNCHANGED_PASS
    if sig_pass and not rv2_pass:
        return FLIP_DEMOTED
    if not sig_pass and rv2_pass:
        return FLIP_REVIVED
    return FLIP_UNCHANGED_FAIL


def _safe_stat(result, attr: str) -> float:
    """A recorded scalar stat (property) that may raise on a malformed result → nan."""
    try:
        return float(getattr(result, attr))
    except Exception:
        return float("nan")


def _score_sig(result, tier: str) -> Tuple[Optional[bool], str]:
    """Recorded significance verdict for a tier. Returns (verdict_or_None, err). An
    exception is NOT silently a FAIL (that could manufacture a spurious REVIVED) — it
    yields None + the error text, and the row becomes ERROR_SIG, never a flip."""
    try:
        return bool(result.significance_gate_passed(tier=tier)), ""
    except Exception as e:             # noqa: BLE001 — report, don't fabricate a verdict
        return None, f"{type(e).__name__}: {e}"


def rescore_result(result, *, label: str, declared_class: str = "",
                   n_trials: Optional[int] = None,
                   live_paper: Optional[dict] = None,
                   regime_waiver_approved: bool = False,
                   notes: str = "") -> RescoreRow:
    """Re-score ONE CPCVResult under Ruler v2, alongside its recorded significance
    verdict. REPORT-ONLY — no flips, no writes. The result must carry the Phase-2
    `oos_returns_dated` field for the Ruler-v2 inference to run (else its PAPER point-SR
    floor fails closed and the row is honestly a FAIL, flagged in notes).

    The caller's result is NEVER mutated — both gates set `requires_human_review_flag`
    on the regime-waiver path, so the re-score operates on a deep copy (PURE contract)."""
    result = copy.deepcopy(result)     # never touch the caller's ledger object
    # Recorded SIGNIFICANCE verdict (call the significance path directly so the answer
    # does not depend on the ambient GATE_MODE). An exception → None (ERROR_SIG), never
    # a silent FAIL that could read as a spurious REVIVED.
    sig_paper, sig_err = _score_sig(result, "paper")
    sig_capital, _ = _score_sig(result, "capital")

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
    if sig_err:
        auto_notes = (auto_notes + "; " if auto_notes else "") + \
            f"significance score raised ({sig_err}) — ERROR_SIG, not a flip"
    if n_obs == 0:
        auto_notes = (auto_notes + "; " if auto_notes else "") + \
            ("no oos_returns_dated — Ruler-v2 fails closed; a recorded-PASS legacy "
             "result shows as DEMOTED only because the OOS series is missing, "
             "NOT a real demotion")

    return RescoreRow(
        label=label, declared_class=declared_class, n_obs=n_obs,
        n_folds=int(getattr(result, "n_folds", 0) or 0),
        sig_paper_pass=sig_paper, sig_capital_pass=sig_capital,
        path_sharpe_tstat=_safe_stat(result, "path_sharpe_tstat"),
        mean_sharpe=_safe_stat(result, "mean_sharpe"),
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
           FLIP_UNCHANGED_PASS: 0, FLIP_UNCHANGED_FAIL: 0, FLIP_ERROR_SIG: 0}
    for r in rows:
        out[r.paper_flip] = out.get(r.paper_flip, 0) + 1
    return out


def format_rescore_table(rows: List[RescoreRow]) -> str:
    """Human-readable OC-style table. REPORT-ONLY framing baked into the header."""
    counts = summarize_flips(rows)
    lines = [
        "Ruler-v2 kill-ledger RE-SCORE (REPORT-ONLY — no flips, owner adjudicates)",
        "=" * 88,
        f"{'label':<24}{'class':<13}{'sigP':>5}{'rv2P':>5}{'flip':>16}"
        f"{'ptSR':>7}{'pathT':>7}{'meanSR':>8}",
        "-" * 88,
    ]

    def _yn(v):
        return "?" if v is None else ("Y" if v else "n")

    for r in rows:
        lines.append(
            f"{r.label[:23]:<24}{r.declared_class[:12]:<13}"
            f"{_yn(r.sig_paper_pass):>5}{_yn(r.rv2_paper_pass):>5}"
            f"{r.paper_flip:>16}{r.point_sr:>7.2f}"
            f"{r.path_sharpe_tstat:>7.2f}{r.mean_sharpe:>8.2f}")
    lines += [
        "-" * 88,
        f"REVIVED(paper): {counts[FLIP_REVIVED]}   DEMOTED(paper): {counts[FLIP_DEMOTED]}"
        f"   unchanged P/F: {counts[FLIP_UNCHANGED_PASS]}/{counts[FLIP_UNCHANGED_FAIL]}"
        f"   error: {counts[FLIP_ERROR_SIG]}",
        "REVIVED rows are CANDIDATES for an owner-initiated, R4-logged re-test — "
        "NOT promotions.",
        "A flip can be DEFINITIONAL (the rulers measure different objects: significance "
        "scores the path-Sharpe distribution [pathT/meanSR], Ruler-v2 the pooled OOS "
        "series [ptSR]) — inspect those columns before treating a REVIVED as a real "
        "Type-II recovery.",
    ]
    return "\n".join(lines)
