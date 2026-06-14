"""
gate_calibration.py - Phase 0 (Alpha-v6): gate operating-characteristic (OC) calibration.

WHY (NEXT_PHASE_BLUEPRINT_2026-06.md, Phase 0 / consensus C1+C11): the production
significance gate (t>=2.0 on N_eff = n_folds CPCV folds of <=4y data) may be a
Type-II machine - t ~= SR*sqrt(years) means a TRUE Sharpe-0.5-0.7 edge fails more
often than it passes. No KILL verdict is trustworthy until the gate's operating
characteristics are MEASURED by pushing KNOWN-real (positive) and KNOWN-null
(negative) control strategies through the EXACT production gate
(CPCVResult.gate_passed under GATE_MODE='significance').

DESIGN
======
Every control is scored by the real pipeline: a strategy adapter -> run_cpcv()
(scripts/walkforward/cpcv.py) -> CPCVResult.gate_passed(tier='paper'|'capital').
Nothing in this module reimplements the gate or CPCV.

Controls (full mode):
  POSITIVE (a well-calibrated gate should pass the genuine ones):
    tsmom_4y      positive_alpha   prior ~0.71  TSMOM ETF sleeve (app/strategy/tsmom.py,
                                                the live trend sleeve, +0.71 over 19y)
                                                restricted to the last 4y window.
                                                THE decisive control.
    tsmom_19y     positive_alpha   prior ~0.71  same sleeve on the full 19y window -
                                                the fully-powered "should pass" anchor.
    xmom_12_1     positive_alpha   prior ~0.50  classic 12-1 cross-sectional momentum
                                                decile L/S, R1K PIT members, monthly
                                                rebalance, 10bps one-way (minimal
                                                PIT-correct implementation below; no
                                                existing reusable builder was found).
    pead_baseline known_marginal   prior ~0.58  the validated PEAD config (unbiased CPCV
                                                +0.578, t=1.81) - byte-identical reuse of
                                                scripts/run_pead_cpcv.py (PEADStrategy +
                                                build_pead_scorer, k=8, total_years=6).
    spy_buyhold   positive_beta    prior ~0.60  SPY buy-and-hold, 4y. Pure BETA, not
                                                alpha - included only to characterize the
                                                gate on a positive-return beta series.
                                                EXCLUDED from the recalibration rule.
  NEGATIVE:
    random_balanced_seed_1..5  null  prior 0.0  TRUE zero-SR nulls: direction-balanced
                                                random books (long random-top / short
                                                random-bottom, net ~ 0; seeds 1403+i)
                                                through EventEdgeStrategy +
                                                AgentSimulator (the same factor-scorer
                                                short path run_qualityshort_cpcv.py
                                                uses). These are the nulls the
                                                pre-registered rule's null-pass-rate
                                                condition evaluates. A calibrated gate
                                                fails >=95% of them. CAVEAT: all 5
                                                seeds share one market window, so
                                                their pass/fail outcomes share a
                                                common factor (not 5 independent
                                                draws).
    random_seed_1..5  null_beta    prior n/a    long-only seeded random books (seeds
                                                1303+i): alpha 0, Sharpe ~= beta share.
                                                A random LONG book on the (survivorship
                                                -biased, current-membership) R1K list
                                                carries market beta + drift, so in a
                                                bull window it can legitimately clear
                                                mean_sharpe/t-stat - that is BETA, not
                                                a gate false-positive on alpha. Kept as
                                                a labeled beta-loaded diagnostic,
                                                EXCLUDED from the recalibration rule;
                                                interpret any pass via the residual-
                                                alpha t column (rA_t ~= 0 => beta).
    leaky_tplus1      leaky        prior n/a    deliberate look-ahead scorer (ranks names
                                                by the FUTURE open->close[t+2] return that
                                                AgentSimulator's scorer contract makes
                                                visible). Expected to PASS the significance
                                                gate and TRIP the implausibility ceiling
                                                (SHARPE_IMPLAUSIBILITY_CEILING=3.0). NOTE:
                                                the ceiling is enforced by the PROMOTION
                                                RUNNER via requires_human_review(), not by
                                                gate_passed() - this harness records the
                                                flag explicitly (implausibility_flag).

Series controls (tsmom/xmom/spy) are continuous daily-return strategies, not
discrete-trade event books, so per-fold profit factor / Calmar / win-rate are
computed from DAILY returns (documented semantic mapping; the per-criterion gate
detail in each row makes any backstop-driven failure attributable). All signals
are strictly backward-looking, so computing the full-window return series once
and slicing CPCV test folds out of it is PIT-correct.

BLOCKER-1 (PF metric-mapping): the gate's MIN_PROFIT_FACTOR=1.10 backstop is
calibrated for PER-TRADE returns, but series controls have no discrete trades -
their PF is computed on DAILY returns, where PF ~= 1 + 0.157*SR_annual for a
normal series. A TRUE SR-0.5 series therefore has daily PF ~= 1.08 and FAILS the
PF backstop even when the significance core is perfect; tsmom_4y (SR~0.71,
daily PF ~= 1.12) sits on the knife-edge. To keep this artifact out of the
headline, every row also records `significance_core_pass` (tstat + pct_positive
+ p5_sharpe + mean_sharpe ONLY, derived from the recorded paper_detail), the OC
table prints BOTH the full-gate and significance-core positive pass-rates, and
series rows that pass the core but fail ONLY on PF/Calmar are tagged
"SERIES PF-MAPPING ARTIFACT (not a significance failure)".

Smoke mode (--smoke): fully synthetic GBM data, zero network / zero DB, reduced
geometry (k=4, ~320 trading days, 10 symbols). It proves the WIRING end-to-end
(adapter -> run_cpcv -> production gate -> OC table -> JSON); smoke verdicts are
NOT calibration evidence. pead_baseline in smoke substitutes a placeholder scorer
(PEADScorer needs the FMP earnings store) - wiring only, flagged in row notes.

OC TABLE / ARTIFACT SCHEMA v2 (CalibrationRow):
  control_id, declared_class (positive_alpha|positive_beta|known_marginal|null|
  null_beta|leaky), true_sr_prior, prior_note, window_start/end, n_folds,
  n_paths_evaluated, mean_sharpe, std_sharpe, tstat (path_sharpe_tstat),
  pct_positive, p5_sharpe, dsr_p (report-only under significance mode),
  worst_regime_sharpe, paper_pass (FULL gate), significance_core_pass (tstat+
  pct_positive+p5_sharpe+mean_sharpe ONLY, from the recorded paper_detail),
  capital_pass, implausibility_flag, regime_waived, residual_alpha_t (CAPM/HAC
  SPY-hedged alpha t from CPCVResult.residual_alpha_t_hac - the alpha-vs-beta
  lens on every row), paper_detail (per-criterion (value, ok) from
  significance_gate_detail), failed_paper_criteria, control_kind, run_failed
  (True = control crashed; EXCLUDED from all aggregates and the rule), run_at
  (per-control run timestamp for merge recency), runtime_sec, smoke, notes.
Aggregates (run_failed rows excluded from ALL of them): positive-control PAPER
pass-rate (full gate) AND positive-control significance-core pass-rate, null
(balanced) PAPER pass-rate, beta-loaded null diagnostic pass-rate, leaky
implausibility detection count. Failed rows are listed in a separate
"FAILED RUNS - rerun before interpreting" section.
Artifact: logs/gate_calibration_{YYYYMMDD}.json (date from --as-of). Smoke runs
write logs/gate_calibration_{YYYYMMDD}_smoke.json so a wiring check can NEVER
clobber a multi-hour full artifact. Full runs READ-MERGE-UPSERT rows by
control_id into the existing dated artifact (newest row per control_id wins,
run_at records recency), so staged --only runs accumulate into one complete
table. The recalibration recommendation is only evaluated when the in-scope
control set is COMPLETE (all in-scope positives + all nulls present, none
failed); otherwise the artifact records verdict "INCOMPLETE - partial control
set".

PRE-REGISTRATION INTEGRITY: the recalibration rule below is fixed BEFORE any
control was run. This module only EVALUATES and REPORTS the rule - it is
incapable of editing thresholds (it never writes to app/ml/retrain_config.py and
recalibration_recommendation() is a pure function of already-computed rows).

FULL RUN (multi-hour; run controls separately if preferred):
  python -m scripts.walkforward.gate_calibration --as-of 2026-06-10
  python -m scripts.walkforward.gate_calibration --as-of 2026-06-10 --only tsmom_4y,spy_buyhold
Approx runtimes: tsmom_4y/tsmom_19y/spy_buyhold ~1-3 min each; xmom_12_1 ~5 min
on shared data; shared R1K yfinance fetch ~30-60 min (once, reused by xmom +
random_* + leaky); each random/leaky control ~1-3 h (AgentSimulator over 4y R1K);
pead_baseline ~3-6 h (own 6y fetch + FMP earnings lookups, exactly as
run_pead_cpcv.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import date as _date
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANN = 252
_SYNTHETIC_SYMS = {"^VIX", "VIX", "SPY"}

# -- Production-parity CPCV geometry (matches scripts/run_pead_cpcv.py) ---------
FULL_N_FOLDS = 8
FULL_N_PATHS = 2
FULL_PURGE_DAYS = 10
FULL_EMBARGO_DAYS = 10
FULL_WINDOW_YEARS = 4
TSMOM_LONG_YEARS = 19

# -- Smoke geometry (synthetic, fast, network-free) -----------------------------
SMOKE_N_FOLDS = 4
SMOKE_N_PATHS = 2
SMOKE_TRADING_DAYS = 320
SMOKE_N_SYMBOLS = 10
SMOKE_DATA_SEED = 7

LEAK_HORIZON_DAYS = 2          # leaky scorer looks at open[t] -> close[t+2]
LEAKY_MAX_HOLD_BARS = 3
RANDOM_SCORER_PICKS = 5
XMOM_REBALANCE_DAYS = 21       # monthly
XMOM_COST_BPS = 10.0           # one-way equity cost on turnover
XMOM_DECILE = 0.10

# ==============================================================================
# PRE-REGISTERED RECALIBRATION RULE - fixed 2026-06-10, BEFORE any control ran.
# ==============================================================================
RECAL_POSITIVE_CLASSES = ("positive_alpha", "known_marginal")
RECAL_PRIOR_LO = 0.50
RECAL_PRIOR_HI = 0.80
RECAL_MIN_POSITIVE_PASS_RATE = 0.50   # trigger condition (A): positives pass < 50%
RECAL_MAX_NULL_PASS_RATE = 0.20       # trigger condition (B) AND constraint (ii)
RECAL_TARGET_POSITIVE_PASS = 2.0 / 3.0
RECAL_TSTAT_FLOOR = 1.50
RECAL_TSTAT_STEP = 0.05

# BLOCKER-1: the significance-CORE criteria (what the gate's significance tier is
# really about). PF / Calmar / worst-regime are BACKSTOPS calibrated for per-trade
# books; on daily-return series controls the PF backstop is a metric-mapping
# artifact, so the core pass-rate is reported alongside the full-gate pass-rate.
SIGNIFICANCE_CORE_CRITERIA = ("tstat", "pct_positive", "p5_sharpe", "mean_sharpe")
BACKSTOP_CRITERIA = ("avg_profit_factor", "avg_calmar", "worst_regime_sharpe")

PREREGISTERED_RECALIBRATION_RULE = """\
PRE-REGISTERED RECALIBRATION RULE (v1, registered 2026-06-10 BEFORE any control ran):
IF  (A) the PAPER-tier pass-rate of positive controls with declared_class in
        {positive_alpha, known_marginal} AND true_sr_prior in [0.50, 0.80] is < 50%,
AND (B) the PAPER-tier pass-rate of null controls (declared_class == 'null') is <= 20%,
THEN recommend lowering PAPER_GATE_MIN_TSTAT to t*, the LARGEST value on the grid
     {current, current-0.05, ..., 1.50} (floor 1.50) such that, re-evaluating ONLY the
     t-stat criterion (every other recorded per-criterion outcome held fixed),
     (i) >= 2/3 of the in-scope positive controls pass PAPER and (ii) the null
     PAPER pass-rate stays <= 20%. If no grid value satisfies both, report
     'no admissible t* on grid' and recommend NO change (the binding failure is not
     the t-stat criterion).
ELSE recommend NO change. If TSMOM-on-4y passes unchanged, log that the power
critique is overstated and leave thresholds alone.
AMENDMENT v1.1 (2026-06-10, still BEFORE any control ran): "null controls
(declared_class == 'null')" means the direction-balanced random books
(random_balanced_seed_1..5, long random-top / short random-bottom, net ~ 0 -
TRUE zero-SR nulls). The long-only random books are beta-loaded (alpha 0,
Sharpe ~= beta share on a survivorship-biased current-R1K list), are
reclassified declared_class='null_beta', and sit OUTSIDE the rule as a labeled
diagnostic. FAIL-SAFE: the rule is NOT evaluated (verdict INCOMPLETE) if any
in-scope positive or null control failed to run, or if the in-scope control set
is incomplete - it can only ever under-recommend loosening.
This harness only REPORTS the recommendation; it NEVER mutates
app/ml/retrain_config.py. A human applies any change via a follow-up PR."""


# -- Gate config snapshot (read from retrain_config, never hardcoded) -----------
@dataclass(frozen=True)
class GateConfigSnapshot:
    gate_mode: str
    paper_min_tstat: float
    paper_min_pct_positive: float
    paper_min_p5_sharpe: float
    paper_min_mean_sharpe: float
    capital_min_tstat: float
    capital_min_n_folds: int
    capital_min_mean_sharpe: float
    n_trials_tested: int
    sharpe_implausibility_ceiling: float
    sacred_holdout_start: str

    @classmethod
    def from_retrain_config(cls) -> "GateConfigSnapshot":
        import app.ml.retrain_config as rc
        return cls(
            gate_mode=rc.GATE_MODE,
            paper_min_tstat=rc.PAPER_GATE_MIN_TSTAT,
            paper_min_pct_positive=rc.PAPER_GATE_MIN_PCT_POSITIVE,
            paper_min_p5_sharpe=rc.PAPER_GATE_MIN_P5_SHARPE,
            paper_min_mean_sharpe=rc.PAPER_GATE_MIN_MEAN_SHARPE,
            capital_min_tstat=rc.CAPITAL_GATE_MIN_TSTAT,
            capital_min_n_folds=rc.CAPITAL_GATE_MIN_N_FOLDS,
            capital_min_mean_sharpe=rc.CAPITAL_GATE_MIN_MEAN_SHARPE,
            n_trials_tested=rc.N_TRIALS_TESTED,
            sharpe_implausibility_ceiling=rc.SHARPE_IMPLAUSIBILITY_CEILING,
            sacred_holdout_start=rc.SACRED_HOLDOUT_START,
        )


# -- Row / recommendation dataclasses -------------------------------------------
@dataclass
class CalibrationRow:
    control_id: str
    declared_class: str    # positive_alpha|positive_beta|known_marginal|null|null_beta|leaky
    true_sr_prior: Optional[float]
    prior_note: str
    window_start: str
    window_end: str
    n_folds: int
    n_paths_evaluated: int
    mean_sharpe: float
    std_sharpe: float
    tstat: float
    pct_positive: float
    p5_sharpe: float
    dsr_p: float
    worst_regime_sharpe: Optional[float]
    paper_pass: bool
    capital_pass: bool
    implausibility_flag: bool
    regime_waived: bool
    paper_detail: Dict[str, tuple] = field(default_factory=dict)
    failed_paper_criteria: List[str] = field(default_factory=list)
    # BLOCKER-1: pass/fail on the significance CORE only (tstat, pct_positive,
    # p5_sharpe, mean_sharpe - derived from the recorded paper_detail, never a
    # re-run). None when the detail is missing/incomplete.
    significance_core_pass: Optional[bool] = None
    # MAJOR-4: CAPM/HAC SPY-hedged alpha t-stat (CPCVResult.residual_alpha_t_hac).
    # rA_t ~= 0 on a passing null_beta row => the pass is BETA, not an alpha
    # false-positive. None when SPY/returns unavailable or < 30 aligned obs.
    residual_alpha_t: Optional[float] = None
    # ControlSpec.kind ("series_tsmom"|"series_spy"|"series_xmom"|"event"|"pead")
    # so the PF-mapping artifact tag does not depend on control_id parsing.
    control_kind: str = ""
    # Alpha-v7 R4: the Ruler-v2 verdict on the SAME result, REPORT-ONLY (computed via
    # ruler_v2.evaluate, NOT the GATE_MODE dispatch, so the significance scoring + the
    # pre-registered recalibration rule are untouched). The R4 check: positive controls
    # should rv2_paper_pass=True; TRUE nulls should rv2_paper_pass=False. None if the
    # result lacked oos_returns_dated (ruler_v2 inference couldn't run). capital is
    # backtest-only here (no live paper) so it fails closed by construction.
    rv2_paper_pass: Optional[bool] = None
    rv2_capital_pass: Optional[bool] = None
    rv2_paper_failed: List[str] = field(default_factory=list)
    # MAJOR-2: True when the control crashed (or NIT-8: its gate detail was
    # empty/divergent). run_failed rows are EXCLUDED from every aggregate and
    # from the recalibration rule, and force an INCOMPLETE verdict.
    run_failed: bool = False
    # MAJOR-3: per-control run timestamp (ISO) for merge-upsert recency.
    run_at: str = ""
    runtime_sec: float = 0.0
    smoke: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuples -> lists for stable JSON round-trip
        d["paper_detail"] = {k: [v[0], bool(v[1])] for k, v in self.paper_detail.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationRow":
        """Reconstruct a row from its artifact dict (MAJOR-3 merge-upsert).
        Unknown keys are ignored so older/newer schema rows still load."""
        from dataclasses import fields as _dc_fields
        d = dict(d)
        d["paper_detail"] = {k: (v[0], bool(v[1]))
                             for k, v in (d.get("paper_detail") or {}).items()}
        known = {f.name for f in _dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RecalRecommendation:
    triggered: bool
    positive_pass_rate: Optional[float]
    null_pass_rate: Optional[float]
    n_positive_in_scope: int
    n_null: int
    recommended_paper_min_tstat: Optional[float]
    rationale: str
    # Machine-readable outcome: "RECOMMEND_LOWER_TSTAT" | "NO_CHANGE" |
    # "NO_ADMISSIBLE_TSTAR" | "INSUFFICIENT_CONTROLS" |
    # "INCOMPLETE - N controls failed to run; rerun before interpreting" |
    # "INCOMPLETE - partial control set" | "SMOKE - rule not evaluated".
    verdict: str = ""
    rule_text: str = PREREGISTERED_RECALIBRATION_RULE

    def to_dict(self) -> dict:
        return asdict(self)


# -- Pure derivations from the recorded gate detail ------------------------------
def significance_core_pass_from_detail(paper_detail: Dict[str, tuple]) -> Optional[bool]:
    """BLOCKER-1: does the recorded PAPER detail pass the significance CORE only
    (tstat, pct_positive, p5_sharpe, mean_sharpe vs the live retrain_config
    thresholds, as already evaluated by significance_gate_detail)? Derived from
    the RECORDED detail - never a re-run. Returns None when the detail is empty
    or missing any core criterion (NIT-8: such rows must not be silently scored)."""
    if not paper_detail:
        return None
    if any(c not in paper_detail for c in SIGNIFICANCE_CORE_CRITERIA):
        return None
    return all(bool(paper_detail[c][1]) for c in SIGNIFICANCE_CORE_CRITERIA)


def is_series_pf_mapping_artifact(row: CalibrationRow) -> bool:
    """BLOCKER-1 tag: a SERIES-class row (daily-return PF semantics) that passes
    the significance core but fails the full PAPER gate ONLY on the PF and/or
    Calmar backstops. That failure is the daily-vs-per-trade PF metric mapping,
    NOT a significance Type-II - tagged explicitly in the OC table."""
    if row.run_failed or not str(row.control_kind).startswith("series"):
        return False
    if row.paper_pass or row.significance_core_pass is not True:
        return False
    failed = set(row.failed_paper_criteria)
    return bool(failed) and failed <= {"avg_profit_factor", "avg_calmar"}


# -- Pure recalibration evaluation (REPORT-ONLY; mutates nothing) ---------------
def _paper_pass_at_tstat(row: CalibrationRow, t_star: float) -> bool:
    """Re-evaluate the recorded PAPER gate detail with ONLY the t-stat threshold
    replaced by t_star. Every other criterion keeps its recorded ok flag. Pure."""
    if not row.paper_detail:
        return False
    for crit, (val, ok) in row.paper_detail.items():
        if crit == "requires_human_review":
            continue  # informational flag, excluded from the gate AND (mirrors cpcv.py)
        if crit == "tstat":
            if not (float(row.tstat) >= t_star):
                return False
        elif not ok:
            return False
    return True


def rule_required_control_ids() -> set:
    """Control ids the pre-registered rule needs for a COMPLETE evaluation
    (MAJOR-3): every in-scope positive (positive_alpha/known_marginal with prior
    in [LO, HI]) plus every TRUE null (declared_class == 'null')."""
    req = set()
    for cid, spec in CONTROLS.items():
        if (spec.declared_class in RECAL_POSITIVE_CLASSES
                and spec.true_sr_prior is not None
                and RECAL_PRIOR_LO <= spec.true_sr_prior <= RECAL_PRIOR_HI):
            req.add(cid)
        elif spec.declared_class == "null":
            req.add(cid)
    return req


def recalibration_recommendation(rows: List[CalibrationRow],
                                 cfg: GateConfigSnapshot,
                                 required_control_ids: Optional[set] = None,
                                 ) -> RecalRecommendation:
    """Evaluate the PRE-REGISTERED rule against an OC table. PURE: reads rows +
    a config snapshot, returns a recommendation. Never touches retrain_config.

    FAIL-SAFE (MAJOR-2/MAJOR-3): REFUSES to evaluate (verdict INCOMPLETE, no
    loosening recommendation) when any in-scope positive or null control has
    run_failed=True, or - when required_control_ids is given - when any required
    control is missing. It can only ever under-recommend loosening."""
    pos_all = [r for r in rows
               if r.declared_class in RECAL_POSITIVE_CLASSES
               and r.true_sr_prior is not None
               and RECAL_PRIOR_LO <= r.true_sr_prior <= RECAL_PRIOR_HI]
    nulls_all = [r for r in rows if r.declared_class == "null"]
    in_scope = [r for r in pos_all if not r.run_failed]
    nulls = [r for r in nulls_all if not r.run_failed]
    # MAJOR-2: any crashed in-scope positive or null => the calibration evidence
    # is incomplete; refusing is the only fail-safe option.
    failed = [r for r in pos_all + nulls_all if r.run_failed]
    if failed:
        ids = ", ".join(sorted(r.control_id for r in failed))
        return RecalRecommendation(
            triggered=False, positive_pass_rate=None, null_pass_rate=None,
            n_positive_in_scope=len(in_scope), n_null=len(nulls),
            recommended_paper_min_tstat=None,
            verdict=(f"INCOMPLETE - {len(failed)} controls failed to run; "
                     "rerun before interpreting"),
            rationale=(f"rule NOT evaluated: {len(failed)} in-scope control(s) "
                       f"FAILED TO RUN ({ids}). A crashed control is missing "
                       "evidence, not a gate failure - rerun it before "
                       "interpreting the OC table or the rule."),
        )
    # MAJOR-3: only evaluate on a COMPLETE in-scope control set.
    if required_control_ids is not None:
        present = {r.control_id for r in rows if not r.run_failed}
        missing = sorted(set(required_control_ids) - present)
        if missing:
            return RecalRecommendation(
                triggered=False, positive_pass_rate=None, null_pass_rate=None,
                n_positive_in_scope=len(in_scope), n_null=len(nulls),
                recommended_paper_min_tstat=None,
                verdict="INCOMPLETE - partial control set",
                rationale=("rule NOT evaluated: in-scope control set incomplete "
                           f"(missing: {', '.join(missing)}). Run the remaining "
                           "controls (--only) - they merge-upsert into the dated "
                           "artifact - then re-evaluate."),
            )
    if not in_scope or not nulls:
        return RecalRecommendation(
            triggered=False, positive_pass_rate=None, null_pass_rate=None,
            n_positive_in_scope=len(in_scope), n_null=len(nulls),
            recommended_paper_min_tstat=None,
            verdict="INSUFFICIENT_CONTROLS",
            rationale=("insufficient controls to evaluate the rule "
                       f"(positives in scope={len(in_scope)}, nulls={len(nulls)}) - "
                       "run the full control suite"),
        )
    pos_rate = float(np.mean([r.paper_pass for r in in_scope]))
    null_rate = float(np.mean([r.paper_pass for r in nulls]))
    cond_a = pos_rate < RECAL_MIN_POSITIVE_PASS_RATE
    cond_b = null_rate <= RECAL_MAX_NULL_PASS_RATE
    if not (cond_a and cond_b):
        why = []
        if not cond_a:
            why.append(f"positive pass-rate {pos_rate:.0%} >= {RECAL_MIN_POSITIVE_PASS_RATE:.0%}"
                       " (gate not under-powered on positives)")
        if not cond_b:
            why.append(f"null pass-rate {null_rate:.0%} > {RECAL_MAX_NULL_PASS_RATE:.0%}"
                       " (gate is not specific enough to relax)")
        return RecalRecommendation(
            triggered=False, positive_pass_rate=pos_rate, null_pass_rate=null_rate,
            n_positive_in_scope=len(in_scope), n_null=len(nulls),
            recommended_paper_min_tstat=None,
            verdict="NO_CHANGE",
            rationale="rule NOT triggered: " + "; ".join(why),
        )
    # Triggered: search the descending grid for the LARGEST admissible t*.
    # NIT-9a: guard the degenerate grid - if the live PAPER threshold is already
    # at/below the pre-registered floor there is no admissible t* by construction.
    if cfg.paper_min_tstat < RECAL_TSTAT_FLOOR:
        return RecalRecommendation(
            triggered=True, positive_pass_rate=pos_rate, null_pass_rate=null_rate,
            n_positive_in_scope=len(in_scope), n_null=len(nulls),
            recommended_paper_min_tstat=None,
            verdict="NO_ADMISSIBLE_TSTAR",
            rationale=(f"rule TRIGGERED but the t* grid is EMPTY: live "
                       f"PAPER_GATE_MIN_TSTAT {cfg.paper_min_tstat:.2f} is already "
                       f"below the pre-registered floor {RECAL_TSTAT_FLOOR:.2f} - "
                       "no admissible t* on grid; recommend NO change."),
        )
    n_steps = int(round((cfg.paper_min_tstat - RECAL_TSTAT_FLOOR) / RECAL_TSTAT_STEP))
    grid = [round(cfg.paper_min_tstat - i * RECAL_TSTAT_STEP, 2) for i in range(n_steps + 1)]
    for t_star in grid:
        pos_at = float(np.mean([_paper_pass_at_tstat(r, t_star) for r in in_scope]))
        null_at = float(np.mean([_paper_pass_at_tstat(r, t_star) for r in nulls]))
        if pos_at >= RECAL_TARGET_POSITIVE_PASS and null_at <= RECAL_MAX_NULL_PASS_RATE:
            return RecalRecommendation(
                triggered=True, positive_pass_rate=pos_rate, null_pass_rate=null_rate,
                n_positive_in_scope=len(in_scope), n_null=len(nulls),
                recommended_paper_min_tstat=t_star,
                verdict="RECOMMEND_LOWER_TSTAT",
                rationale=(f"rule TRIGGERED (positives {pos_rate:.0%} < "
                           f"{RECAL_MIN_POSITIVE_PASS_RATE:.0%}, nulls {null_rate:.0%} <= "
                           f"{RECAL_MAX_NULL_PASS_RATE:.0%}). RECOMMEND (report-only): lower "
                           f"PAPER_GATE_MIN_TSTAT {cfg.paper_min_tstat:.2f} -> {t_star:.2f} "
                           f"(at t*={t_star:.2f}: positives pass {pos_at:.0%}, nulls "
                           f"{null_at:.0%}). A HUMAN applies this in retrain_config.py via a "
                           "follow-up PR."),
            )
    return RecalRecommendation(
        triggered=True, positive_pass_rate=pos_rate, null_pass_rate=null_rate,
        n_positive_in_scope=len(in_scope), n_null=len(nulls),
        recommended_paper_min_tstat=None,
        verdict="NO_ADMISSIBLE_TSTAR",
        rationale=("rule TRIGGERED but NO admissible t* on the grid down to "
                   f"{RECAL_TSTAT_FLOOR:.2f} - the binding PAPER failure is not the t-stat "
                   "criterion (see failed_paper_criteria per row). Recommend NO change; "
                   "investigate the binding criterion instead."),
    )


def ruler_v2_r4_summary(rows: List[CalibrationRow]) -> dict:
    """Alpha-v7 R4: does Ruler-v2's PAPER plausibility tier clear real POSITIVES and
    reject TRUE NULLS? REPORT-ONLY. Reads the rv2_paper_pass column (None rows — no
    oos_returns_dated, or run_failed — are excluded). 'clean' iff every scored positive
    control PASSES and every scored true-null FAILS Ruler-v2 PAPER. The owner-ratified
    pre-flip gate: do not flip GATE_MODE live until this is clean on the real controls."""
    scored = [r for r in rows if not r.run_failed and r.rv2_paper_pass is not None]
    positives = [r for r in scored if r.declared_class == "positive_alpha"]
    nulls = [r for r in scored if r.declared_class == "null"]
    leaky = [r for r in scored if r.declared_class == "leaky"]
    pos_pass = [r.control_id for r in positives if r.rv2_paper_pass]
    pos_fail = [r.control_id for r in positives if not r.rv2_paper_pass]
    null_pass = [r.control_id for r in nulls if r.rv2_paper_pass]   # BAD if non-empty
    null_fail = [r.control_id for r in nulls if not r.rv2_paper_pass]
    clean = (len(positives) > 0 and not pos_fail and len(nulls) > 0 and not null_pass)
    return {
        "clean": bool(clean),
        "n_positives": len(positives), "positives_passed": pos_pass,
        "positives_FAILED_should_pass": pos_fail,
        "n_nulls": len(nulls), "nulls_failed_correctly": null_fail,
        "nulls_PASSED_should_fail": null_pass,
        "leaky": {r.control_id: r.rv2_paper_pass for r in leaky},
        "verdict": ("R4 CLEAN - positives clear, nulls dead"
                    if clean else
                    "R4 NOT CLEAN - review positives_FAILED / nulls_PASSED"),
    }


def smoke_recommendation() -> RecalRecommendation:
    """MINOR-5: under --smoke the pre-registered rule is NOT evaluated - smoke
    rows are synthetic wiring checks (smoke pead_baseline is a relabeled random
    scorer), so the rule must not appear to trigger (or pass) on smoke data."""
    return RecalRecommendation(
        triggered=False, positive_pass_rate=None, null_pass_rate=None,
        n_positive_in_scope=0, n_null=0, recommended_paper_min_tstat=None,
        verdict="SMOKE - rule not evaluated",
        rationale=("SMOKE - rule not evaluated: smoke rows are synthetic wiring "
                   "checks (pead_baseline substitutes a relabeled random scorer); "
                   "the pre-registered rule is only evaluated on FULL-mode rows."),
    )


# ==============================================================================
# Strategy adapters
# ==============================================================================
class SeriesReturnStrategy:
    """run_cpcv adapter for a precomputed PIT daily net-return series.

    For strategies whose signal is strictly backward-looking (TSMOM, 12-1 momentum,
    buy-and-hold) the full-window return series is computed once and CPCV test
    folds are sliced out of it - PIT-correct because no return at date d uses any
    information after d. Rules-based: is_trained=False / trained_through=date.min,
    so run_cpcv bypasses the BUG-23 overlap guard (full fold coverage), exactly
    like EventEdgeStrategy.

    Semantic mapping for fold metrics (documented; series have no discrete trades):
    profit_factor / win_rate are computed from DAILY returns; trades := n_obs.
    """

    is_trained = False

    def __init__(self, control_id: str, returns: pd.Series, *,
                 spy_prices: Optional[pd.Series] = None,
                 regime_map: Optional[dict] = None):
        self.model_type = control_id
        r = returns.dropna().sort_index()
        r.index = pd.to_datetime(r.index)
        self.returns = r
        self.spy_prices = spy_prices
        self._global_regime_map = regime_map or {}
        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.all_days_sorted = [d.date() if hasattr(d, "date") else d for d in r.index]
        self.model = type("_NoModel", (), {"trained_through": _date.min})()
        self.allow_in_sample = False
        self.per_fold_retrain = False

    def fetch_data(self, start=None, end=None) -> None:  # series already provided
        return None

    def run_fold(self, fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        from scripts.walkforward.gates import (
            FoldResult, compute_profit_factor, compute_calmar, compute_k_ratio, fold_years,
        )
        from scripts.walkforward.regime import compute_regime_sharpes as _crs

        d = np.array(self.all_days_sorted)
        mask = (d >= te_start) & (d <= te_end)
        r = self.returns.iloc[np.flatnonzero(mask)]
        n_obs = int(len(r))
        if n_obs < 2:
            return FoldResult(
                fold=fold_idx, train_start=tr_start, train_end=tr_end,
                test_start=te_start, test_end=te_end, trades=0, win_rate=0.0,
                sharpe=0.0, max_drawdown=0.0, total_return=0.0, stop_exit_rate=0.0,
            )
        vals = r.to_numpy(dtype=float)
        std = float(np.std(vals, ddof=1))
        sharpe = float(np.mean(vals) / std * np.sqrt(ANN)) if std > 0 else 0.0
        eq_vals = np.cumprod(1.0 + vals)
        total_ret = float(eq_vals[-1] - 1.0)
        peak = np.maximum.accumulate(eq_vals)
        max_dd = float(abs((eq_vals / peak - 1.0).min()))
        win_rate = float(np.mean(vals > 0))
        # Equity curve anchored one day before the first return so the curve's
        # diffs reproduce the return series exactly (n_obs consistent).
        dates = [ts.date() for ts in r.index]
        anchor = dates[0] - timedelta(days=1)
        equity_curve = [(anchor, 1.0)] + list(zip(dates, eq_vals.tolist()))
        _obs: dict = {}
        regime_sharpes = _crs(equity_curve, te_start, te_end,
                              regime_map=self._global_regime_map, obs_counts=_obs)
        years = fold_years(te_start, te_end)
        return FoldResult(
            fold=fold_idx, train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=n_obs,                 # proxy: one observation-day per "trade"
            win_rate=win_rate,
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_return=total_ret,
            stop_exit_rate=0.0,
            profit_factor=compute_profit_factor(list(vals)),   # daily-return PF
            calmar_ratio=compute_calmar(total_ret, max_dd, years, daily_returns=list(vals)),
            k_ratio=compute_k_ratio(equity_curve),
            n_obs=n_obs,
            regime_sharpes=regime_sharpes,
            regime_obs_counts=_obs,
            daily_returns_dated=list(zip(dates, vals.tolist())),
        )


def _make_synthetic_bars(n_days: int, end: _date, *, seed: int, drift: float = 0.0004,
                         vol: float = 0.02, base: float = 100.0,
                         constant: Optional[float] = None) -> pd.DataFrame:
    """Daily OHLCV GBM bars on business days ending at `end` (deterministic)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n_days)
    if constant is not None:
        closes = np.full(n_days, constant)
    else:
        rets = rng.normal(drift, vol, size=n_days)
        closes = base * np.cumprod(1.0 + rets)
    opens = np.empty_like(closes)
    opens[0] = closes[0]
    opens[1:] = closes[:-1] * (1.0 + rng.normal(0.0, 0.002, size=n_days - 1))
    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) * 1.004,
        "low": np.minimum(opens, closes) * 0.996,
        "close": closes,
        "volume": np.full(n_days, 1e6),
    }, index=idx)


def _synthetic_symbols_data(symbols: List[str], n_days: int, end: _date,
                            data_seed: int = SMOKE_DATA_SEED) -> Dict[str, pd.DataFrame]:
    out = {sym: _make_synthetic_bars(n_days, end, seed=data_seed * 1000 + i)
           for i, sym in enumerate(symbols)}
    out["SPY"] = _make_synthetic_bars(n_days, end, seed=data_seed * 1000 + 999,
                                      drift=0.0004, vol=0.011)
    out["^VIX"] = _make_synthetic_bars(n_days, end, seed=data_seed * 1000 + 998,
                                       constant=15.0)
    return out


def _smoke_symbols() -> List[str]:
    return [f"SYN{i:02d}" for i in range(SMOKE_N_SYMBOLS)]


# Imported lazily so module import stays light for the pure functions/tests.
def _event_edge_cls():
    from scripts.walkforward.event_edge import EventEdgeStrategy
    return EventEdgeStrategy


class SyntheticEventStrategy:
    """Smoke/test variant of EventEdgeStrategy: synthetic bars, no network, no DB.

    Built dynamically as a subclass so this module imports without pulling the
    full event_edge/AgentSimulator stack until a control actually runs.
    """

    def __new__(cls, scorer, symbols, *, n_days: int, as_of: _date,
                data_seed: int = SMOKE_DATA_SEED, **kw):
        Base = _event_edge_cls()

        class _Synth(Base):
            download_vix = False

            def fetch_data(self, start=None, end=None) -> None:
                self.symbols_data = _synthetic_symbols_data(
                    [s for s in self.symbols if s not in _SYNTHETIC_SYMS],
                    n_days, as_of, data_seed=data_seed)
                self.spy_prices = self.symbols_data["SPY"]["close"]
                self.all_days_sorted = sorted({
                    d.date() for d in self.symbols_data["SPY"].index})
                # Single-bucket NEUTRAL map: keeps the regime backstop ACTIVE
                # (worst_regime_sharpe == overall) without network access.
                self._global_regime_map = {d: "NEUTRAL" for d in self.all_days_sorted}

            def _fold_universe(self, tr_start, te_start) -> set:
                return set(self.symbols)  # no DB in smoke mode

        return _Synth(scorer, symbols, **kw)


# -- Scorers (negative controls) ------------------------------------------------
def make_random_scorer(seed: int, n_picks: int = RANDOM_SCORER_PICKS):
    """Null control: random long signals. Deterministic per (seed, day) so results
    are reproducible regardless of fold/call order."""
    def scorer(day, symbols_data, vix_history=None):
        rng = np.random.default_rng((int(seed), int(day.toordinal())))
        syms = sorted(s for s in symbols_data if s not in _SYNTHETIC_SYMS)
        if not syms:
            return []
        k = min(n_picks, len(syms))
        picks = rng.choice(syms, size=k, replace=False)
        return [(str(s), float(rng.uniform(0.5, 1.0)), "long") for s in picks]
    return scorer


def make_balanced_random_scorer(seed: int, n_per_side: int = RANDOM_SCORER_PICKS):
    """TRUE zero-SR null (MAJOR-4): direction-balanced random book - random longs
    vs random shorts, net ~ 0, so market beta and survivorship drift cancel in
    expectation. Uses the SAME AgentSimulator factor-scorer short path as
    run_qualityshort_cpcv.py: 3-tuples (sym, -conf, "short"); the simulator sizes
    on abs(confidence). Long/short entries are INTERLEAVED so a binding
    max-open-positions cap cannot systematically fill longs first (which would
    re-introduce a net-long beta bias). Deterministic per (seed, day)."""
    def scorer(day, symbols_data, vix_history=None):
        rng = np.random.default_rng((int(seed), int(day.toordinal())))
        syms = sorted(s for s in symbols_data if s not in _SYNTHETIC_SYMS)
        k = min(n_per_side, len(syms) // 2)
        if k <= 0:
            return []
        picks = rng.choice(syms, size=2 * k, replace=False)
        out = []
        for i in range(k):
            out.append((str(picks[2 * i]), float(rng.uniform(0.5, 1.0)), "long"))
            out.append((str(picks[2 * i + 1]), -float(rng.uniform(0.5, 1.0)), "short"))
        return out
    return scorer


def make_leaky_scorer(horizon: int = LEAK_HORIZON_DAYS, n_picks: int = RANDOM_SCORER_PICKS):
    """DELIBERATE look-ahead control: ranks names by the FUTURE return from day
    t's open to close[t+horizon]. AgentSimulator passes the scorer the full fold
    dataframes (the contract trusts the scorer to stay PIT), and entries fill at
    day t's open - so this is a genuine leak. Expected: PASSES the significance
    gate, TRIPS SHARPE_IMPLAUSIBILITY_CEILING (requires_human_review)."""
    def scorer(day, symbols_data, vix_history=None):
        scored = []
        for sym, df in symbols_data.items():
            if sym in _SYNTHETIC_SYMS or df is None or len(df) == 0:
                continue
            d_arr = df.index.date if hasattr(df.index, "date") else np.array(df.index)
            pos = int(np.searchsorted(d_arr, day))
            if pos >= len(df) or d_arr[pos] != day:
                continue
            fpos = pos + horizon
            if fpos >= len(df):
                continue
            o = float(df["open"].iloc[pos])
            c = float(df["close"].iloc[fpos])
            if o <= 0:
                continue
            scored.append((sym, c / o - 1.0))
        scored.sort(key=lambda x: -x[1])
        top = [s for s in scored[:n_picks] if s[1] > 0]
        n = len(top)
        return [(sym, 1.0 - 0.4 * i / max(n, 1), "long") for i, (sym, _) in enumerate(top)]
    return scorer


# ==============================================================================
# Series builders (positive controls)
# ==============================================================================
def _tsmom_series(as_of: _date, years: int, smoke: bool):
    """TSMOM net daily returns over [as_of - years, as_of]. Reuses the live sleeve
    (app/strategy/tsmom.py tsmom_backtest) - signals are backward-looking, so the
    full-history run sliced to the window is PIT-correct."""
    from app.strategy.tsmom import TSMOMConfig, tsmom_backtest
    cfg = TSMOMConfig()
    window_start = as_of - timedelta(days=int(years * 365.25))
    if smoke:
        n = SMOKE_TRADING_DAYS + 320  # lookback warmup for 252d signal
        prices = pd.DataFrame({
            sym: _make_synthetic_bars(n, as_of, seed=SMOKE_DATA_SEED * 100 + i)["close"]
            for i, sym in enumerate(cfg.universe)})
        window_start = as_of - timedelta(days=int(SMOKE_TRADING_DAYS * 7 / 5))
    else:
        from scripts.run_tsmom import _fetch_etf_prices
        fetch_start = window_start - timedelta(days=420)
        prices = _fetch_etf_prices(list(cfg.universe), start=fetch_start.isoformat())
        prices = prices[np.array([ts.date() <= as_of for ts in prices.index])]
    res = tsmom_backtest(prices, cfg)
    r = res.returns
    r = r[np.array([ts.date() >= window_start for ts in r.index])]
    spy = prices["SPY"] if "SPY" in prices.columns else None
    return r, spy


def _spy_buyhold_series(as_of: _date, years: int, smoke: bool):
    window_start = as_of - timedelta(days=int(years * 365.25))
    if smoke:
        close = _make_synthetic_bars(SMOKE_TRADING_DAYS + 1, as_of,
                                     seed=SMOKE_DATA_SEED * 100 + 50,
                                     drift=0.0005, vol=0.011)["close"]
    else:
        from scripts.run_tsmom import _fetch_etf_prices
        prices = _fetch_etf_prices([], start=window_start.isoformat())  # always adds SPY
        close = prices["SPY"]
        close = close[np.array([ts.date() <= as_of for ts in close.index])]
    r = close.pct_change().dropna()
    r = r[np.array([ts.date() >= window_start for ts in r.index])]
    return r, close


def xmom_12_1_returns(closes: pd.DataFrame, window_start: _date, window_end: _date, *,
                      pit_filter: bool = True, reb_days: int = XMOM_REBALANCE_DAYS,
                      cost_bps: float = XMOM_COST_BPS,
                      decile: float = XMOM_DECILE, min_names: int = 20) -> pd.Series:
    """Minimal PIT-correct 12-1 cross-sectional momentum decile L/S daily returns.

    Formation at each rebalance date t: mom = P[t-21]/P[t-252] - 1 (strictly
    backward). Long top decile / short bottom decile, each leg gross 1.0, equal
    weight, held reb_days; weights earn from t+1 (held.shift(1)); cost_bps one-way
    on turnover, charged when the weight first earns (mirrors tsmom_backtest's
    cost alignment). PIT membership via app.data.universe_history.pit_union per
    rebalance date when pit_filter=True (falls back with a warning if the DB is
    unavailable). KNOWN LIMITATION (same as the production event harness):
    yfinance lacks delisted names, so the short leg misses delisted losers -
    residual survivorship risk, flagged in the row notes."""
    closes = closes.sort_index()
    rets = closes.pct_change()
    mom = closes.shift(21) / closes.shift(252) - 1.0
    idx = closes.index
    dates = np.array([ts.date() for ts in idx])
    in_win = np.flatnonzero((dates >= window_start) & (dates <= window_end))
    if len(in_win) < 30:
        return pd.Series(dtype=float)
    target = pd.DataFrame(np.nan, index=idx, columns=closes.columns)
    members_fn = None
    if pit_filter:
        try:
            from app.data.universe_history import pit_union
            members_fn = pit_union
        except Exception as exc:  # pragma: no cover - env-dependent
            logger.warning("xmom: pit_union unavailable (%s) - no PIT member filter", exc)
    for i in range(int(in_win[0]), int(in_win[-1]) + 1, reb_days):
        row = mom.iloc[i].dropna()
        row = row.drop(labels=[s for s in _SYNTHETIC_SYMS if s in row.index], errors="ignore")
        if members_fn is not None:
            try:
                members = set(members_fn("russell1000", dates[i], dates[i]))
                row = row[row.index.isin(members)]
            except Exception as exc:
                logger.warning("xmom: pit_union failed at %s (%s) - using all names",
                               dates[i], exc)
        n = len(row)
        if n < min_names:
            continue
        k = max(1, int(round(n * decile)))
        ranked = row.sort_values()
        w = pd.Series(0.0, index=closes.columns)
        w[ranked.index[-k:]] = 1.0 / k
        w[ranked.index[:k]] = -1.0 / k
        target.iloc[i] = w
    held = target.ffill().fillna(0.0)
    dw = held.diff().abs().sum(axis=1)
    dw.iloc[0] = held.iloc[0].abs().sum()
    cost = dw * (cost_bps / 1e4)
    gross_ret = (held.shift(1) * rets).sum(axis=1)
    net = (gross_ret - cost.shift(1)).iloc[1:]
    net = net[np.array([ts.date() >= window_start for ts in net.index])]
    net = net[np.array([ts.date() <= window_end for ts in net.index])]
    return net.fillna(0.0)


# ==============================================================================
# Control registry
# ==============================================================================
@dataclass(frozen=True)
class ControlSpec:
    control_id: str
    declared_class: str
    true_sr_prior: Optional[float]
    prior_note: str
    kind: str                       # "series_tsmom"|"series_spy"|"series_xmom"|"event"|"pead"
    seed: Optional[int] = None
    years: int = FULL_WINDOW_YEARS
    # Alpha-v7 R4: the sleeve's component_type for the Ruler-v2 regime waiver. A crisis-
    # diversifier (tsmom) declares "risk_premium" so Ruler-v2 PAPER waives its worst-
    # regime floor (it fails its worst regime BY DESIGN); alpha/null controls stay
    # subject to the floor. Default "" = no waiver.
    component_type: str = ""


def _build_controls() -> Dict[str, ControlSpec]:
    c: Dict[str, ControlSpec] = {}
    c["tsmom_4y"] = ControlSpec(
        "tsmom_4y", "positive_alpha", 0.71,
        "live trend sleeve, +0.71 standalone over 19y incl. every crisis; 4y window - "
        "THE decisive Type-II control", "series_tsmom", years=FULL_WINDOW_YEARS,
        component_type="risk_premium")   # R4: crisis-diversifier → regime waived in PAPER
    c["tsmom_19y"] = ControlSpec(
        "tsmom_19y", "positive_alpha", 0.71,
        "same sleeve, full 19y window - fully-powered should-pass anchor",
        "series_tsmom", years=TSMOM_LONG_YEARS, component_type="risk_premium")
    c["xmom_12_1"] = ControlSpec(
        "xmom_12_1", "known_marginal", 0.50,
        "Jegadeesh-Titman 12-1 XS momentum decile L/S. RECLASSIFIED 2026-06-13 "
        "positive_alpha->known_marginal (owner-approved): post-2010 it is attenuated to "
        "insignificance (realized meanSR 0.17, t 0.77; BOTH the significance core AND "
        "Ruler-v2 reject it), and cross-sectional momentum was ruled dead 2026-06-03 "
        "(DECISIONS). It is no longer a clean should-pass positive; a gate rejecting it "
        "is CORRECT, so it must not count as a Ruler-v2 Type-II in the R4 check.",
        "series_xmom", years=FULL_WINDOW_YEARS)
    c["pead_baseline"] = ControlSpec(
        "pead_baseline", "known_marginal", 0.578,
        "validated PEAD config: unbiased CPCV +0.578, t=1.81 (known-marginal)", "pead",
        years=6)
    c["spy_buyhold"] = ControlSpec(
        "spy_buyhold", "positive_beta", 0.60,
        "pure beta, NOT alpha - positive-return sanity floor; excluded from the "
        "recalibration rule", "series_spy", years=FULL_WINDOW_YEARS)
    for i in range(1, 6):
        # MAJOR-4: TRUE zero-SR nulls - the ones the pre-registered rule's
        # null-pass-rate condition uses (declared_class == "null").
        c[f"random_balanced_seed_{i}"] = ControlSpec(
            f"random_balanced_seed_{i}", "null", 0.0,
            f"direction-balanced random book (seed {1403 + i}): long random-top / "
            "short random-bottom, net ~ 0 - TRUE zero-SR null; a calibrated gate "
            "fails >=95%. Caveat: all 5 seeds share one market window (common "
            "factor)", "event", seed=1403 + i)
    for i in range(1, 6):
        # MAJOR-4: long-only randoms are NOT Sharpe-0 nulls - a random LONG book
        # on the survivorship-biased current-R1K list carries market beta +
        # drift. Kept as a labeled beta-loaded diagnostic OUTSIDE the rule.
        c[f"random_seed_{i}"] = ControlSpec(
            f"random_seed_{i}", "null_beta", None,
            f"long-only random book (seed {1303 + i}): alpha 0, Sharpe ~= beta "
            "share (beta + survivorship drift on current-R1K list) - beta-loaded "
            "null DIAGNOSTIC, excluded from the recalibration rule; interpret any "
            "pass via residual-alpha t (rA_t ~= 0 => beta, not an alpha "
            "false-positive)", "event", seed=1303 + i)
    c["leaky_tplus1"] = ControlSpec(
        "leaky_tplus1", "leaky", None,
        f"deliberate look-ahead (open[t] -> close[t+{LEAK_HORIZON_DAYS}]); expected to "
        "pass significance and trip SHARPE_IMPLAUSIBILITY_CEILING", "event")
    return c


CONTROLS: Dict[str, ControlSpec] = _build_controls()


# -- Shared full-mode event data (one yfinance R1K fetch, reused) ---------------
def _fetch_shared_event_data(as_of: _date, years: int = FULL_WINDOW_YEARS) -> dict:
    from app.utils.constants import RUSSELL_1000_TICKERS
    EventEdgeStrategy = _event_edge_cls()
    window_start = as_of - timedelta(days=int(years * 365.25))
    fetch_start = window_start - timedelta(days=420)  # xmom 12-1 formation lookback
    probe = EventEdgeStrategy(scorer=None, symbols=list(RUSSELL_1000_TICKERS))
    probe.fetch_data(datetime.combine(fetch_start, datetime.min.time()),
                     datetime.combine(as_of, datetime.min.time()))
    return {
        "symbols_data": probe.symbols_data,
        "spy_prices": probe.spy_prices,
        "regime_map": probe._global_regime_map,
        "window_start": window_start,
        "as_of": as_of,
    }


def _result_to_row(spec: ControlSpec, result, *, window_start, window_end,
                   runtime_sec: float, smoke: bool, notes: str) -> CalibrationRow:
    from scripts.walkforward.gates import deflated_sharpe_ratio, N_TRIALS_TESTED
    # Calibration ALWAYS scores the significance columns via the significance gate
    # DIRECTLY (not result.gate_detail/gate_passed, which DISPATCH on GATE_MODE) — so
    # the OC table's significance columns + the recalibration rule are correct
    # regardless of the live GATE_MODE (incl. after the flip to "ruler_v2"). The
    # Ruler-v2 columns are computed separately below, also directly.
    paper_detail = result.significance_gate_detail(tier="paper")
    paper_pass = result.significance_gate_passed(tier="paper")
    capital_pass = result.significance_gate_passed(tier="capital")
    _, dsr_p = deflated_sharpe_ratio(result.mean_sharpe, N_TRIALS_TESTED,
                                     result._dsr_n_obs())
    failed = [k for k, (_v, ok) in paper_detail.items() if not ok]
    # Alpha-v7 R4: Ruler-v2 verdict on the SAME result — REPORT-ONLY, computed
    # EXPLICITLY (not via the GATE_MODE dispatch) so the significance scoring above and
    # the pre-registered recalibration rule are untouched. None when the result has no
    # oos_returns_dated (e.g. a smoke/legacy result) so it isn't silently scored.
    rv2_paper_pass = rv2_capital_pass = None
    rv2_paper_failed: List[str] = []
    if getattr(result, "oos_returns_dated", None):
        from app.research import ruler_v2 as _rv2
        # R4: declare the sleeve's component_type so Ruler-v2 PAPER applies the
        # diversifier regime waiver (a crisis-diversifier fails its worst regime by
        # design). Pure-additive on the result; only the dark ruler_v2 path reads it.
        result.component_type = spec.component_type
        rv2_paper_pass = bool(_rv2.gate_passed(result, tier="paper"))
        rv2_capital_pass = bool(_rv2.gate_passed(result, tier="capital"))
        _rv2_detail = _rv2.evaluate(result, tier="paper")
        rv2_paper_failed = [k for k, (_v, ok) in _rv2_detail.items()
                            if not ok and k not in _rv2.INFORMATIONAL_KEYS]
    # BLOCKER-1: significance-core pass derived from the RECORDED detail.
    core_pass = significance_core_pass_from_detail(
        {k: (v, bool(ok)) for k, (v, ok) in paper_detail.items()})
    # NIT-8: assert the recorded booleans are consistent with the recorded detail.
    # significance_gate_passed() has early-returns (in_sample_override,
    # REQUIRE_TRUE_WF_FOR_PROMOTION, n_paths floor) that significance_gate_detail()
    # does not fully mirror - if they diverge (or the detail is empty/incomplete,
    # e.g. a non-significance GATE_MODE early-return), the row must NOT be
    # silently scored: mark it run_failed and exclude it from all aggregates.
    run_failed = False
    consistency_notes: List[str] = []
    if core_pass is None:
        run_failed = True
        consistency_notes.append(
            "DETAIL INCOMPLETE: paper_detail empty or missing significance-core "
            "criteria (gate early-return / non-significance GATE_MODE?) - row "
            "NOT scored, excluded from all aggregates")
    else:
        detail_and = all(bool(ok) for k, (_v, ok) in paper_detail.items()
                         if k != "requires_human_review")
        if bool(paper_pass) != detail_and:
            run_failed = True
            consistency_notes.append(
                "DETAIL DIVERGENCE: gate_passed() != AND(paper_detail) (latent "
                "REQUIRE_TRUE_WF_FOR_PROMOTION / in_sample_override early-return) "
                "- row NOT scored, excluded from all aggregates")
    if consistency_notes:
        notes = "; ".join(consistency_notes + ([notes] if notes else []))
    return CalibrationRow(
        control_id=spec.control_id,
        declared_class=spec.declared_class,
        true_sr_prior=spec.true_sr_prior,
        prior_note=spec.prior_note,
        window_start=str(window_start),
        window_end=str(window_end),
        n_folds=result.n_folds,
        n_paths_evaluated=len(result.path_sharpes),
        mean_sharpe=round(result.mean_sharpe, 4),
        std_sharpe=round(result.std_sharpe, 4),
        tstat=round(result.path_sharpe_tstat, 4),
        pct_positive=round(result.pct_positive, 4),
        p5_sharpe=round(result.p5_sharpe, 4),
        dsr_p=round(dsr_p, 4),
        worst_regime_sharpe=(None if result.worst_regime_sharpe is None
                             else round(result.worst_regime_sharpe, 4)),
        paper_pass=bool(paper_pass),
        capital_pass=bool(capital_pass),
        implausibility_flag=bool(result.requires_human_review()),
        regime_waived=bool(result.requires_human_review_flag),
        paper_detail={k: (v, bool(ok)) for k, (v, ok) in paper_detail.items()},
        failed_paper_criteria=failed,
        significance_core_pass=core_pass,
        residual_alpha_t=(None if result.residual_alpha_t_hac is None
                          else round(float(result.residual_alpha_t_hac), 4)),
        control_kind=spec.kind,
        rv2_paper_pass=rv2_paper_pass,
        rv2_capital_pass=rv2_capital_pass,
        rv2_paper_failed=rv2_paper_failed,
        run_failed=run_failed,
        run_at=datetime.now().isoformat(timespec="seconds"),
        runtime_sec=round(runtime_sec, 1),
        smoke=smoke,
        notes=notes,
    )


def run_control(control_id: str, *, as_of: _date, smoke: bool = False,
                shared_event_data: Optional[dict] = None) -> CalibrationRow:
    """Run one control through the production gate; return its OC row."""
    from scripts.walkforward.cpcv import run_cpcv
    if control_id not in CONTROLS:
        raise KeyError(f"unknown control_id '{control_id}'; known: {sorted(CONTROLS)}")
    spec = CONTROLS[control_id]
    t0 = time.time()
    n_folds = SMOKE_N_FOLDS if smoke else FULL_N_FOLDS
    n_paths = SMOKE_N_PATHS if smoke else FULL_N_PATHS
    notes_extra: List[str] = []
    if smoke:
        # MINOR-7: explicit per-row smoke caveats where smoke differs from full.
        notes_extra.append("SMOKE: synthetic data - wiring check only, NOT calibration "
                           "evidence; smoke geometry k=4/n_paths=2 (full: k=8)")
        if spec.kind == "series_tsmom":
            notes_extra.append("smoke: tsmom_4y==tsmom_19y synthetic (years param "
                               "ignored - same GBM window/seed)")

    if spec.kind in ("series_tsmom", "series_spy", "series_xmom"):
        if spec.kind == "series_tsmom":
            r, spy = _tsmom_series(as_of, spec.years, smoke)
        elif spec.kind == "series_spy":
            r, spy = _spy_buyhold_series(as_of, spec.years, smoke)
        else:
            if smoke:
                syms = _smoke_symbols()
                closes = pd.DataFrame({
                    s: _make_synthetic_bars(SMOKE_TRADING_DAYS + 320, as_of,
                                            seed=SMOKE_DATA_SEED * 1000 + i)["close"]
                    for i, s in enumerate(syms)})
                spy = None
                window_start = as_of - timedelta(days=int(SMOKE_TRADING_DAYS * 7 / 5))
                r = xmom_12_1_returns(closes, window_start, as_of, pit_filter=False,
                                      min_names=4)
            else:
                shared = shared_event_data or _fetch_shared_event_data(as_of, spec.years)
                closes = pd.DataFrame({
                    s: df["close"] for s, df in shared["symbols_data"].items()
                    if s != "^VIX"})
                spy = shared["spy_prices"]
                window_start = shared["window_start"]
                r = xmom_12_1_returns(closes, window_start, as_of, pit_filter=True)
                notes_extra.append("residual survivorship risk: yfinance lacks delisted "
                                   "names (same limitation as the production event "
                                   "harness)")
        if r.empty or len(r) < 30:
            raise RuntimeError(f"{control_id}: empty/too-short return series ({len(r)})")
        if float(r.std()) <= 0.0:
            raise RuntimeError(f"{control_id}: degenerate (zero-variance) return series "
                               "- builder produced no positions")
        if smoke:
            regime_map = {ts.date(): "NEUTRAL" for ts in r.index}
        else:
            from scripts.walkforward.regime import load_regime_map
            regime_map = load_regime_map(r.index[0].date(), r.index[-1].date())
        strategy = SeriesReturnStrategy(control_id, r, spy_prices=spy,
                                        regime_map=regime_map)
        result = run_cpcv(strategy, purge_days=FULL_PURGE_DAYS,
                          embargo_days=FULL_EMBARGO_DAYS, n_folds=n_folds,
                          n_paths=n_paths, total_years=None)
        notes_extra.append("series control: PF/win-rate from daily returns; "
                           "trades := n_obs")
        win_lo, win_hi = strategy.all_days_sorted[0], strategy.all_days_sorted[-1]

    elif spec.kind == "event":
        if spec.control_id == "leaky_tplus1":
            scorer = make_leaky_scorer()
        elif spec.control_id.startswith("random_balanced"):
            scorer = make_balanced_random_scorer(spec.seed)
            notes_extra.append("direction-balanced (long+short, net ~ 0) TRUE "
                               "zero-SR null; shorts via the factor-scorer short "
                               "path (same as run_qualityshort_cpcv.py)")
        else:
            scorer = make_random_scorer(spec.seed)
            notes_extra.append("BETA-LOADED null (long-only): alpha 0, Sharpe ~= "
                               "beta share - a pass here is attributable via "
                               "residual_alpha_t, not a gate false-positive on "
                               "alpha; excluded from the recalibration rule")
        hold_kw = ({"max_hold_bars_override": LEAKY_MAX_HOLD_BARS}
                   if spec.control_id == "leaky_tplus1" else {})
        if smoke:
            strategy = SyntheticEventStrategy(
                scorer, _smoke_symbols(), n_days=SMOKE_TRADING_DAYS, as_of=as_of,
                model_type=control_id, **hold_kw)
            strategy.fetch_data()
        else:
            from app.utils.constants import RUSSELL_1000_TICKERS
            EventEdgeStrategy = _event_edge_cls()
            strategy = EventEdgeStrategy(scorer, list(RUSSELL_1000_TICKERS),
                                         model_type=control_id, **hold_kw)
            shared = shared_event_data or _fetch_shared_event_data(as_of)
            strategy.symbols_data = shared["symbols_data"]
            strategy.spy_prices = shared["spy_prices"]
            strategy._global_regime_map = shared["regime_map"]
            strategy.all_days_sorted = [
                d for d in sorted({dd.date() for df in shared["symbols_data"].values()
                                   for dd in df.index})
                if shared["window_start"] <= d <= as_of]
        result = run_cpcv(strategy, purge_days=FULL_PURGE_DAYS,
                          embargo_days=FULL_EMBARGO_DAYS, n_folds=n_folds,
                          n_paths=n_paths, total_years=None)
        if spec.control_id == "leaky_tplus1":
            notes_extra.append("ceiling check: SHARPE_IMPLAUSIBILITY_CEILING is enforced "
                               "by the promotion runner (requires_human_review), not by "
                               "gate_passed - recorded here as implausibility_flag")
        win_lo, win_hi = strategy.all_days_sorted[0], strategy.all_days_sorted[-1]

    elif spec.kind == "pead":
        if smoke:
            # PEADScorer needs the FMP earnings store (DB) - full mode only.
            strategy = SyntheticEventStrategy(
                make_random_scorer(1303), _smoke_symbols(),
                n_days=SMOKE_TRADING_DAYS, as_of=as_of, model_type=control_id)
            strategy.fetch_data()
            result = run_cpcv(strategy, purge_days=FULL_PURGE_DAYS,
                              embargo_days=FULL_EMBARGO_DAYS, n_folds=n_folds,
                              n_paths=n_paths, total_years=None)
            notes_extra.append("SMOKE substitution: PEADScorer requires the FMP earnings "
                               "store; placeholder scorer exercises the wiring only")
            win_lo, win_hi = strategy.all_days_sorted[0], strategy.all_days_sorted[-1]
        else:
            # Byte-identical reuse of the production PEAD CPCV run.
            from scripts.run_pead_cpcv import (
                PEADStrategy, build_pead_scorer, CPCV_K, CPCV_PATHS, TOTAL_YEARS,
            )
            from app.utils.constants import RUSSELL_1000_TICKERS
            strategy = PEADStrategy(scorer=build_pead_scorer(),
                                    symbols=list(RUSSELL_1000_TICKERS),
                                    transaction_cost_pct=0.0005)
            end_all = datetime.now()
            start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
            strategy.fetch_data(start_all, end_all)
            result = run_cpcv(strategy, purge_days=10, embargo_days=10,
                              n_folds=CPCV_K, n_paths=CPCV_PATHS,
                              total_years=TOTAL_YEARS)
            notes_extra.append("production parity: window anchored to retrain_as_of() "
                               "by run_cpcv (total_years mode) - --as-of not applied")
            win_lo, win_hi = start_all.date(), end_all.date()
    else:  # pragma: no cover
        raise ValueError(f"unknown control kind {spec.kind}")

    row = _result_to_row(spec, result, window_start=win_lo, window_end=win_hi,
                         runtime_sec=time.time() - t0, smoke=smoke,
                         notes="; ".join(notes_extra))
    logger.info("control %s done in %.1fs: meanSR=%+.3f t=%.2f paper=%s capital=%s impl=%s",
                row.control_id, row.runtime_sec, row.mean_sharpe, row.tstat,
                row.paper_pass, row.capital_pass, row.implausibility_flag)
    return row


def run_all_controls(*, as_of: _date, smoke: bool = False,
                     only: Optional[List[str]] = None) -> List[CalibrationRow]:
    ids = list(CONTROLS) if not only else list(only)
    unknown = [i for i in ids if i not in CONTROLS]
    if unknown:
        raise KeyError(f"unknown control id(s): {unknown}; known: {sorted(CONTROLS)}")
    shared = None
    rows: List[CalibrationRow] = []
    for cid in ids:
        spec = CONTROLS[cid]
        needs_shared = (not smoke) and (
            spec.kind == "event"
            or (spec.kind == "series_xmom" and spec.years == FULL_WINDOW_YEARS))
        if needs_shared and shared is None:
            logger.info("fetching shared R1K event data (once; reused by xmom + "
                        "random_* + leaky)...")
            shared = _fetch_shared_event_data(as_of)
        try:
            rows.append(run_control(cid, as_of=as_of, smoke=smoke,
                                    shared_event_data=shared))
        except Exception as exc:
            logger.error("control %s FAILED: %s", cid, exc, exc_info=True)
            # MAJOR-2: run_failed=True marks this row unambiguously as MISSING
            # EVIDENCE (not a gate failure). It is excluded from every aggregate
            # and forces the recalibration rule to an INCOMPLETE verdict.
            rows.append(CalibrationRow(
                control_id=cid, declared_class=spec.declared_class,
                true_sr_prior=spec.true_sr_prior, prior_note=spec.prior_note,
                window_start="", window_end="", n_folds=0, n_paths_evaluated=0,
                mean_sharpe=0.0, std_sharpe=0.0, tstat=0.0, pct_positive=0.0,
                p5_sharpe=0.0, dsr_p=0.0, worst_regime_sharpe=None,
                paper_pass=False, capital_pass=False, implausibility_flag=False,
                regime_waived=False, significance_core_pass=None,
                control_kind=spec.kind, run_failed=True,
                run_at=datetime.now().isoformat(timespec="seconds"),
                smoke=smoke, notes=f"RUN FAILED: {exc}"))
    return rows


# -- OC table / artifact --------------------------------------------------------
def build_oc_table(rows: List[CalibrationRow], *,
                   recommendation: Optional[RecalRecommendation] = None,
                   as_of: str = "", smoke: bool = False) -> str:
    """Deterministic ASCII operating-characteristic table.

    MAJOR-2: run_failed rows are EXCLUDED from the scored listing and from every
    aggregate; they appear only in the 'FAILED RUNS - rerun before interpreting'
    section. BLOCKER-1: both the full-gate AND significance-core positive
    pass-rates are printed, and series rows failing only on PF/Calmar while
    passing the core are tagged as PF-mapping artifacts."""
    width = 132
    lines: List[str] = []
    bar = "=" * width
    lines.append(bar)
    mode = "SMOKE (synthetic - wiring only)" if smoke else "FULL"
    lines.append(f"  GATE CALIBRATION - OPERATING CHARACTERISTICS  (as_of={as_of}, "
                 f"mode={mode})")
    lines.append(bar)
    hdr = (f"{'control_id':<22} {'class':<15} {'prior':>6} {'meanSR':>8} {'tstat':>7} "
           f"{'%pos':>6} {'P5':>7} {'dsr_p':>6} {'rA_t':>6} {'nf':>3} {'np':>3} "
           f"{'CORE':>5} {'PAPER':>6} {'CAPITAL':>8} {'IMPL':>5}  failed-paper-criteria")
    lines.append(hdr)
    lines.append("-" * width)
    ok_rows = [r for r in rows if not r.run_failed]
    failed_rows = [r for r in rows if r.run_failed]
    for r in ok_rows:
        prior = "n/a" if r.true_sr_prior is None else f"{r.true_sr_prior:.2f}"
        ra = "n/a" if r.residual_alpha_t is None else f"{r.residual_alpha_t:+.2f}"
        core = ("n/a" if r.significance_core_pass is None
                else "PASS" if r.significance_core_pass else "fail")
        tail = ",".join(r.failed_paper_criteria) if r.failed_paper_criteria else "-"
        if is_series_pf_mapping_artifact(r):
            tail += "  [SERIES PF-MAPPING ARTIFACT (not a significance failure)]"
        lines.append(
            f"{r.control_id:<22} {r.declared_class:<15} {prior:>6} "
            f"{r.mean_sharpe:>+8.3f} {r.tstat:>7.2f} {r.pct_positive:>6.1%} "
            f"{r.p5_sharpe:>+7.3f} {r.dsr_p:>6.3f} {ra:>6} {r.n_folds:>3d} "
            f"{r.n_paths_evaluated:>3d} {core:>5} "
            f"{('PASS' if r.paper_pass else 'fail'):>6} "
            f"{('PASS' if r.capital_pass else 'fail'):>8} "
            f"{('YES' if r.implausibility_flag else '-'):>5}  {tail}")
    lines.append("-" * width)
    positives = [r for r in ok_rows if r.declared_class in RECAL_POSITIVE_CLASSES]
    betas = [r for r in ok_rows if r.declared_class == "positive_beta"]
    nulls = [r for r in ok_rows if r.declared_class == "null"]
    beta_nulls = [r for r in ok_rows if r.declared_class == "null_beta"]
    leaks = [r for r in ok_rows if r.declared_class == "leaky"]
    if positives:
        n_pass = sum(r.paper_pass for r in positives)
        n_core = sum(r.significance_core_pass is True for r in positives)
        n_artifact = sum(is_series_pf_mapping_artifact(r) for r in positives)
        lines.append(f"  AGGREGATE positive-control PAPER pass-rate (FULL gate; "
                     f"positive_alpha+known_marginal): {n_pass}/{len(positives)} "
                     f"({n_pass / len(positives):.0%})")
        lines.append(f"  AGGREGATE positive-control significance-core pass-rate "
                     f"(tstat/pct_pos/p5/mean ONLY - excludes PF/Calmar/regime "
                     f"backstops): {n_core}/{len(positives)} "
                     f"({n_core / len(positives):.0%})")
        if n_core > n_pass:
            lines.append(f"    core > full: {n_core - n_pass} failure(s) are "
                         f"backstop artifacts, not significance Type-II "
                         f"({n_artifact} tagged SERIES PF-MAPPING ARTIFACT)")
    if betas:
        n_pass = sum(r.paper_pass for r in betas)
        lines.append(f"  AGGREGATE positive_beta PAPER pass-rate (informational): "
                     f"{n_pass}/{len(betas)} ({n_pass / len(betas):.0%})")
    if nulls:
        n_pass = sum(r.paper_pass for r in nulls)
        lines.append(f"  AGGREGATE null PAPER pass-rate (direction-balanced TRUE "
                     f"zero-SR nulls): {n_pass}/{len(nulls)} "
                     f"({n_pass / len(nulls):.0%})  (calibrated gate: <= 5% expected)")
    if beta_nulls:
        n_pass = sum(r.paper_pass for r in beta_nulls)
        lines.append(f"  AGGREGATE beta-loaded null PAPER pass-rate (long-only "
                     f"randoms; alpha 0, Sharpe ~= beta share - DIAGNOSTIC, "
                     f"excluded from the rule; attribute passes via rA_t): "
                     f"{n_pass}/{len(beta_nulls)} ({n_pass / len(beta_nulls):.0%})")
    if leaks:
        n_flag = sum(r.implausibility_flag for r in leaks)
        lines.append(f"  LEAKY controls flagged implausible "
                     f"(Sharpe > ceiling): {n_flag}/{len(leaks)}")
    # MINOR-6: the CAPITAL column is structurally unreachable at this geometry.
    lines.append("  NOTE: CAPITAL unreachable here - fold count is below the "
                 "CAPITAL n_folds floor (CAPITAL_GATE_MIN_N_FOLDS=10; full=8, "
                 "smoke=4) - the CAPITAL column is expected to fail for every "
                 "control at this geometry.")
    if failed_rows:
        lines.append("-" * width)
        lines.append("  FAILED RUNS - rerun before interpreting (EXCLUDED from "
                     "all aggregates and from the recalibration rule):")
        for r in failed_rows:
            lines.append(f"    {r.control_id} ({r.declared_class}): {r.notes}")
    if recommendation is not None:
        lines.append("-" * width)
        lines.append("  PRE-REGISTERED RECALIBRATION RULE (report-only; never "
                     "auto-applied):")
        if recommendation.verdict:
            lines.append(f"    VERDICT: {recommendation.verdict}")
        lines.append(f"    {recommendation.rationale}")
    # Alpha-v7 R4: Ruler-v2 PAPER both-ways check (report-only; the pre-flip gate).
    r4 = ruler_v2_r4_summary(rows)
    lines.append(bar)
    lines.append("  RULER-v2 R4 CHECK (report-only - the pre-flip gate for "
                 "GATE_MODE='ruler_v2'):")
    lines.append(f"    VERDICT: {r4['verdict']}")
    lines.append(f"    positives PAPER-pass: {r4['positives_passed']}  "
                 f"(should-pass that FAILED: {r4['positives_FAILED_should_pass']})")
    lines.append(f"    true-nulls PAPER-fail: {r4['nulls_failed_correctly']}  "
                 f"(should-fail that PASSED: {r4['nulls_PASSED_should_fail']})")
    if r4["leaky"]:
        lines.append(f"    leaky (expect fail/ceiling): {r4['leaky']}")
    lines.append(bar)
    return "\n".join(lines)


# Schema v2: + significance_core_pass, residual_alpha_t, control_kind,
# run_failed, run_at on rows; + verdict on the recommendation (BLOCKER-1 /
# MAJOR-2 / MAJOR-3 / MAJOR-4).
# Schema v3 (Alpha-v7 R4): + rv2_paper_pass / rv2_capital_pass / rv2_paper_failed
# (report-only Ruler-v2 verdict per control; never feeds the recalibration rule).
ARTIFACT_SCHEMA_VERSION = 3


def default_artifact_path(as_of: _date, smoke: bool) -> str:
    """MAJOR-3: smoke runs get a distinct _smoke suffix so a wiring check can
    NEVER clobber a multi-hour full artifact."""
    from pathlib import Path
    suffix = "_smoke" if smoke else ""
    return str(Path("logs") / f"gate_calibration_{as_of.strftime('%Y%m%d')}{suffix}.json")


def load_artifact_rows(path: str) -> List[CalibrationRow]:
    """Load rows from an existing FULL artifact for merge-upsert (MAJOR-3).
    Returns [] when the file is missing/unparseable or is a smoke artifact
    (smoke rows must never be merged into full calibration evidence)."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("smoke"):
            logger.warning("existing artifact %s is a SMOKE artifact - not merging", path)
            return []
        return [CalibrationRow.from_dict(d) for d in payload.get("rows", [])]
    except Exception as exc:
        logger.warning("could not load existing artifact %s for merge (%s) - "
                       "starting fresh", path, exc)
        return []


def merge_rows(existing: List[CalibrationRow],
               new: List[CalibrationRow]) -> List[CalibrationRow]:
    """MAJOR-3 merge-upsert by control_id: rows from the current run replace any
    prior row with the same control_id (per-control run_at records recency);
    untouched prior rows are preserved, so staged --only runs accumulate into
    one complete table. Output is ordered by the control registry."""
    by_id: Dict[str, CalibrationRow] = {r.control_id: r for r in existing}
    by_id.update({r.control_id: r for r in new})
    ordered = [by_id[cid] for cid in CONTROLS if cid in by_id]
    ordered += [r for cid, r in by_id.items() if cid not in CONTROLS]
    return ordered


def write_artifact(rows: List[CalibrationRow], recommendation: RecalRecommendation,
                   cfg: GateConfigSnapshot, *, as_of: _date, smoke: bool,
                   out_path: Optional[str] = None) -> str:
    from pathlib import Path
    if out_path is None:
        out_path = default_artifact_path(as_of, smoke)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "smoke": bool(smoke),
        "gate_config": asdict(cfg),
        "preregistered_rule": PREREGISTERED_RECALIBRATION_RULE,
        "rows": [r.to_dict() for r in rows],
        "recommendation": recommendation.to_dict(),
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return str(p)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate calibration harness (Alpha-v6 Phase 0): positive + negative "
                    "controls through the production CPCV significance gate.")
    parser.add_argument("--as-of", type=str, default=None,
                        help="evaluation date YYYY-MM-DD (default: today)")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic data, reduced geometry - wiring check only")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated control ids (default: all)")
    parser.add_argument("--out", type=str, default=None,
                        help="JSON artifact path (default: logs/gate_calibration_"
                             "YYYYMMDD.json; smoke runs default to ..._smoke.json "
                             "and full runs merge-upsert by control_id)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [gate_cal] %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    as_of = _date.fromisoformat(args.as_of) if args.as_of else _date.today()
    from app.ml.retrain_config import assert_no_sacred_holdout
    assert_no_sacred_holdout(as_of, context="gate_calibration")

    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    cfg = GateConfigSnapshot.from_retrain_config()
    if cfg.gate_mode != "significance":
        logger.warning("GATE_MODE=%r (not 'significance') - the OC table will reflect "
                       "the LEGACY gate; calibration targets the significance gate.",
                       cfg.gate_mode)

    rows = run_all_controls(as_of=as_of, smoke=args.smoke, only=only)

    # MAJOR-3: full runs merge-upsert into the existing dated artifact so staged
    # --only runs accumulate; smoke artifacts are separate and never merged.
    out_path = args.out or default_artifact_path(as_of, args.smoke)
    if args.smoke and not out_path.endswith("_smoke.json"):
        # MINOR-A: a smoke run must NEVER clobber a full artifact, even via --out.
        from pathlib import Path as _P
        _p = _P(out_path)
        out_path = str(_p.with_name(_p.stem + "_smoke" + _p.suffix))
    if not args.smoke:
        existing = load_artifact_rows(out_path)
        if existing:
            before = {r.control_id for r in existing}
            rows = merge_rows(existing, rows)
            logger.info("merged %d existing row(s) from %s (upsert by control_id; "
                        "now %d rows)", len(before), out_path, len(rows))

    if args.smoke:
        # MINOR-5: the pre-registered rule is never evaluated on smoke data.
        rec = smoke_recommendation()
    else:
        # MAJOR-2/MAJOR-3: refuses (verdict INCOMPLETE) on failed or missing
        # in-scope controls; evaluated over the MERGED row set.
        rec = recalibration_recommendation(
            rows, cfg, required_control_ids=rule_required_control_ids())

    table = build_oc_table(rows, recommendation=rec, as_of=as_of.isoformat(),
                           smoke=args.smoke)
    print(table)
    path = write_artifact(rows, rec, cfg, as_of=as_of, smoke=args.smoke,
                          out_path=out_path)
    print(f"\n  artifact: {path}")
    if args.smoke:
        print("  NOTE: --smoke results are a wiring check, NOT calibration evidence.")
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    raise SystemExit(main())
