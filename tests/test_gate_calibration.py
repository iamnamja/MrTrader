"""
Tests for the Phase 0 (Alpha-v6) gate-calibration harness
(scripts/walkforward/gate_calibration.py).

Coverage:
  1. recalibration_recommendation - triggers on a synthetic Type-II table
     (positives fail only on t-stat, nulls fail) and recommends the LARGEST
     admissible t* on the pre-registered grid; does NOT trigger when positives
     already pass; reports "no admissible t*" when the binding failure is not
     the t-stat; NEVER mutates retrain_config. MAJOR-2: refuses (INCOMPLETE)
     when any in-scope control run_failed. MAJOR-3: refuses (INCOMPLETE)
     when the required control set is partial. NIT-9a: degenerate-grid guard.
     MINOR-5: smoke replaces the rule with "SMOKE - rule not evaluated".
  2. BLOCKER-1: significance_core_pass derivation from the recorded detail;
     dual aggregates (full-gate AND significance-core) in the OC table; series
     rows passing core but failing only PF/Calmar tagged as PF-mapping
     artifacts (counted in the core aggregate, not the full one).
  3. build_oc_table - contains every control + aggregate lines; deterministic
     ASCII; run_failed rows excluded from aggregates and listed in a separate
     FAILED RUNS section; residual-alpha (rA_t) column on every row; CAPITAL
     n_folds>=10 footnote.
  4. Random null scorers - long-only reproducibility, and the MAJOR-4
     direction-balanced TRUE zero-SR null (interleaved long/short, net ~ 0).
  5. Leaky scorer - provably prefers future winners on synthetic frames (pure),
     and the end-to-end smoke leaky control's Sharpe crosses
     SHARPE_IMPLAUSIBILITY_CEILING through the REAL run_cpcv + production gate.
  6. Artifact JSON - schema v2 round-trip; MAJOR-3 smoke suffix (never
     clobbers a full artifact) and merge-upsert by control_id.
  7. NIT-8: _result_to_row marks rows run_failed on empty/divergent gate detail
     instead of silently scoring them.

No network, no DB: integration tests use --smoke synthetic mode only.
"""
from datetime import date

import json

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward.gate_calibration import (
    CONTROLS,
    CalibrationRow,
    GateConfigSnapshot,
    RECAL_MAX_NULL_PASS_RATE,
    build_oc_table,
    default_artifact_path,
    is_series_pf_mapping_artifact,
    load_artifact_rows,
    make_balanced_random_scorer,
    make_leaky_scorer,
    make_random_scorer,
    merge_rows,
    recalibration_recommendation,
    rule_required_control_ids,
    run_all_controls,
    significance_core_pass_from_detail,
    smoke_recommendation,
    write_artifact,
)

AS_OF = date(2026, 6, 10)


# -- Helpers -------------------------------------------------------------------

def _cfg(paper_min_tstat: float = 2.0) -> GateConfigSnapshot:
    """Fixed snapshot for pure tests (matches the production values by design,
    but constructed locally so the tests do not depend on live config)."""
    return GateConfigSnapshot(
        gate_mode="significance",
        paper_min_tstat=paper_min_tstat, paper_min_pct_positive=0.75,
        paper_min_p5_sharpe=0.0, paper_min_mean_sharpe=0.35,
        capital_min_tstat=2.5, capital_min_n_folds=10,
        capital_min_mean_sharpe=0.45, n_trials_tested=300,
        sharpe_implausibility_ceiling=3.0, sacred_holdout_start="2026-11-09",
    )


def _row(control_id, klass, prior, tstat, *, mean=0.60, pctpos=0.85, p5=0.05,
         pf_ok=True, cal_ok=True, regime_ok=True, control_kind="event",
         residual_alpha_t=None) -> CalibrationRow:
    """Synthetic CalibrationRow whose paper_detail mirrors the significance gate."""
    detail = {
        "n_paths": (15, True),
        "tstat": (tstat, tstat >= 2.0),
        "pct_positive": (pctpos, pctpos >= 0.75),
        "p5_sharpe": (p5, p5 >= 0.0),
        "mean_sharpe": (mean, mean >= 0.35),
        "avg_profit_factor": (1.5 if pf_ok else 0.9, pf_ok),
        "avg_calmar": (0.8 if cal_ok else 0.1, cal_ok),
        "worst_regime_sharpe": (0.1 if regime_ok else -1.0, regime_ok),
    }
    paper_pass = all(ok for _, ok in detail.values())
    return CalibrationRow(
        control_id=control_id, declared_class=klass, true_sr_prior=prior,
        prior_note="synthetic", window_start="2022-06-10", window_end="2026-06-10",
        n_folds=8, n_paths_evaluated=28, mean_sharpe=mean, std_sharpe=0.4,
        tstat=tstat, pct_positive=pctpos, p5_sharpe=p5, dsr_p=0.5,
        worst_regime_sharpe=0.1, paper_pass=paper_pass, capital_pass=False,
        implausibility_flag=mean > 3.0, regime_waived=False, paper_detail=detail,
        failed_paper_criteria=[k for k, (_, ok) in detail.items() if not ok],
        significance_core_pass=significance_core_pass_from_detail(detail),
        residual_alpha_t=residual_alpha_t, control_kind=control_kind,
    )


def _failed_row(control_id, klass, prior, *, note="RUN FAILED: boom") -> CalibrationRow:
    """MAJOR-2 placeholder row for a crashed control (mirrors run_all_controls)."""
    return CalibrationRow(
        control_id=control_id, declared_class=klass, true_sr_prior=prior,
        prior_note="synthetic", window_start="", window_end="", n_folds=0,
        n_paths_evaluated=0, mean_sharpe=0.0, std_sharpe=0.0, tstat=0.0,
        pct_positive=0.0, p5_sharpe=0.0, dsr_p=0.0, worst_regime_sharpe=None,
        paper_pass=False, capital_pass=False, implausibility_flag=False,
        regime_waived=False, significance_core_pass=None, run_failed=True,
        notes=note,
    )


def _type2_table():
    """Positives (prior 0.5-0.8) that fail ONLY the t-stat; nulls that fail hard."""
    rows = [
        _row("tsmom_4y", "positive_alpha", 0.71, 1.72, control_kind="series_tsmom"),
        _row("xmom_12_1", "positive_alpha", 0.50, 1.55, control_kind="series_xmom"),
        _row("pead_baseline", "known_marginal", 0.578, 1.81, control_kind="pead"),
        _row("spy_buyhold", "positive_beta", 0.60, 1.90,
             control_kind="series_spy"),  # beta: out of rule scope
    ]
    for i in range(1, 6):
        rows.append(_row(f"random_balanced_seed_{i}", "null", 0.0, 0.3,
                         mean=0.05, pctpos=0.5, p5=-0.5))
    rows.append(_row("random_seed_1", "null_beta", None, 2.5,
                     residual_alpha_t=0.1))  # beta-loaded diagnostic, passes raw
    rows.append(_row("leaky_tplus1", "leaky", None, 9.0, mean=8.0, pctpos=1.0, p5=5.0))
    return rows


def _complete_table():
    """A table covering EVERY control the pre-registered rule requires
    (rule_required_control_ids): in-scope positives + all 5 balanced nulls."""
    rows = [
        _row("tsmom_4y", "positive_alpha", 0.71, 1.72, control_kind="series_tsmom"),
        _row("tsmom_19y", "positive_alpha", 0.71, 1.72, control_kind="series_tsmom"),
        _row("xmom_12_1", "positive_alpha", 0.50, 1.55, control_kind="series_xmom"),
        _row("pead_baseline", "known_marginal", 0.578, 1.81, control_kind="pead"),
    ]
    for i in range(1, 6):
        rows.append(_row(f"random_balanced_seed_{i}", "null", 0.0, 0.3,
                         mean=0.05, pctpos=0.5, p5=-0.5))
    return rows


# -- 1. Recalibration rule -----------------------------------------------------

class TestRecalibrationRule:
    def test_triggers_on_type2_table_and_picks_largest_admissible_tstar(self):
        rows = _type2_table()
        rec = recalibration_recommendation(rows, _cfg())
        assert rec.triggered is True
        assert rec.positive_pass_rate == 0.0
        assert rec.null_pass_rate == 0.0
        # Grid descends 2.00, 1.95, ... ; at 1.75 only pead (1/3); at 1.70 tsmom+pead
        # (2/3 >= target) -> largest admissible t* is 1.70.
        assert rec.recommended_paper_min_tstat == pytest.approx(1.70)
        assert rec.verdict == "RECOMMEND_LOWER_TSTAT"
        assert "report-only" in rec.rationale or "HUMAN" in rec.rationale

    def test_spy_beta_control_excluded_from_rule_scope(self):
        rows = _type2_table()
        rec = recalibration_recommendation(rows, _cfg())
        assert rec.n_positive_in_scope == 3  # tsmom_4y, xmom_12_1, pead - NOT spy

    def test_null_beta_diagnostic_excluded_from_rule(self):
        """MAJOR-4: the beta-loaded long-only random PASSES the raw gate in the
        synthetic table, but it is class null_beta - outside the rule - so the
        null pass-rate (balanced nulls only) stays 0 and the rule still fires."""
        rows = _type2_table()
        rec = recalibration_recommendation(rows, _cfg())
        assert rec.n_null == 5            # the 5 balanced nulls only
        assert rec.null_pass_rate == 0.0  # null_beta pass not counted
        assert rec.triggered is True

    def test_not_triggered_when_positives_already_pass(self):
        rows = [
            _row("tsmom_4y", "positive_alpha", 0.71, 2.4),
            _row("xmom_12_1", "positive_alpha", 0.50, 2.1),
            _row("pead_baseline", "known_marginal", 0.578, 2.6),
        ]
        for i in range(1, 6):
            rows.append(_row(f"random_balanced_seed_{i}", "null", 0.0, 0.2,
                             mean=0.02, pctpos=0.4, p5=-0.6))
        rec = recalibration_recommendation(rows, _cfg())
        assert rec.triggered is False
        assert rec.recommended_paper_min_tstat is None
        assert rec.verdict == "NO_CHANGE"
        assert "NOT triggered" in rec.rationale

    def test_not_triggered_when_nulls_leak_through(self):
        """If nulls pass too often, the gate must NOT be loosened (condition B)."""
        rows = [
            _row("tsmom_4y", "positive_alpha", 0.71, 1.7),
            _row("xmom_12_1", "positive_alpha", 0.50, 1.6),
            _row("pead_baseline", "known_marginal", 0.578, 1.8),
        ]
        # 3 of 5 nulls pass the paper gate outright (50% > 20% bound).
        for i in range(1, 4):
            rows.append(_row(f"random_balanced_seed_{i}", "null", 0.0, 2.5))
        for i in range(4, 6):
            rows.append(_row(f"random_balanced_seed_{i}", "null", 0.0, 0.2,
                             mean=0.02, pctpos=0.4, p5=-0.6))
        rec = recalibration_recommendation(rows, _cfg())
        assert rec.triggered is False
        assert rec.null_pass_rate is not None
        assert rec.null_pass_rate > RECAL_MAX_NULL_PASS_RATE

    def test_no_admissible_tstar_when_binding_failure_is_not_tstat(self):
        """Positives fail t-stat AND pct_positive: lowering t* can never rescue
        them, so the rule triggers but recommends NO change."""
        rows = [
            _row("tsmom_4y", "positive_alpha", 0.71, 1.7, pctpos=0.6),
            _row("xmom_12_1", "positive_alpha", 0.50, 1.6, pctpos=0.55),
            _row("pead_baseline", "known_marginal", 0.578, 1.8, pctpos=0.5),
        ]
        for i in range(1, 6):
            rows.append(_row(f"random_balanced_seed_{i}", "null", 0.0, 0.2,
                             mean=0.02, pctpos=0.4, p5=-0.6))
        rec = recalibration_recommendation(rows, _cfg())
        assert rec.triggered is True
        assert rec.recommended_paper_min_tstat is None
        assert rec.verdict == "NO_ADMISSIBLE_TSTAR"
        assert "no admissible" in rec.rationale.lower()

    def test_degenerate_grid_guard_min_tstat_below_floor(self):
        """NIT-9a: paper_min_tstat < 1.50 makes the descending grid empty - the
        rule must return 'no admissible t*' explicitly, never crash or loosen."""
        rows = _type2_table()
        rec = recalibration_recommendation(rows, _cfg(paper_min_tstat=1.40))
        assert rec.triggered is True
        assert rec.recommended_paper_min_tstat is None
        assert rec.verdict == "NO_ADMISSIBLE_TSTAR"
        assert "no admissible" in rec.rationale.lower()

    def test_insufficient_controls_is_safe(self):
        rec = recalibration_recommendation([_row("tsmom_4y", "positive_alpha",
                                                 0.71, 1.7)], _cfg())
        assert rec.triggered is False
        assert "insufficient" in rec.rationale

    def test_run_failed_in_scope_positive_forces_incomplete(self):
        """MAJOR-2: a crashed in-scope positive is MISSING EVIDENCE - the rule
        must refuse to evaluate (no loosening recommendation possible)."""
        rows = _type2_table()
        rows[0] = _failed_row("tsmom_4y", "positive_alpha", 0.71)
        rec = recalibration_recommendation(rows, _cfg())
        assert rec.triggered is False
        assert rec.recommended_paper_min_tstat is None
        assert rec.verdict.startswith("INCOMPLETE")
        assert "failed to run" in rec.verdict
        assert "tsmom_4y" in rec.rationale

    def test_run_failed_null_forces_incomplete(self):
        rows = _type2_table()
        rows[5] = _failed_row("random_balanced_seed_2", "null", 0.0)
        rec = recalibration_recommendation(rows, _cfg())
        assert rec.triggered is False
        assert rec.verdict.startswith("INCOMPLETE")
        assert "failed to run" in rec.verdict

    def test_partial_control_set_forces_incomplete(self):
        """MAJOR-3: with required_control_ids, the rule only evaluates on a
        COMPLETE in-scope control set."""
        rows = _complete_table()
        required = rule_required_control_ids()
        # Complete set -> evaluated (and triggers on this Type-II table).
        rec = recalibration_recommendation(rows, _cfg(),
                                           required_control_ids=required)
        assert rec.verdict == "RECOMMEND_LOWER_TSTAT"
        assert rec.recommended_paper_min_tstat == pytest.approx(1.70)
        # Drop one required null -> INCOMPLETE, no recommendation.
        partial = [r for r in rows if r.control_id != "random_balanced_seed_5"]
        rec2 = recalibration_recommendation(partial, _cfg(),
                                            required_control_ids=required)
        assert rec2.triggered is False
        assert rec2.verdict == "INCOMPLETE - partial control set"
        assert "random_balanced_seed_5" in rec2.rationale

    def test_rule_required_control_ids(self):
        req = rule_required_control_ids()
        assert {"tsmom_4y", "tsmom_19y", "xmom_12_1", "pead_baseline"} <= req
        assert {f"random_balanced_seed_{i}" for i in range(1, 6)} <= req
        # Out of scope: beta control, beta-loaded nulls, leaky.
        assert "spy_buyhold" not in req
        assert "random_seed_1" not in req
        assert "leaky_tplus1" not in req

    def test_smoke_recommendation_rule_not_evaluated(self):
        """MINOR-5: smoke runs must never appear to trigger the rule."""
        rec = smoke_recommendation()
        assert rec.triggered is False
        assert rec.recommended_paper_min_tstat is None
        assert rec.verdict == "SMOKE - rule not evaluated"
        table = build_oc_table(_type2_table(), recommendation=rec,
                               as_of="2026-06-10", smoke=True)
        assert "SMOKE - rule not evaluated" in table

    def test_never_mutates_retrain_config(self):
        import app.ml.retrain_config as rc
        before = (rc.PAPER_GATE_MIN_TSTAT, rc.PAPER_GATE_MIN_PCT_POSITIVE,
                  rc.PAPER_GATE_MIN_P5_SHARPE, rc.PAPER_GATE_MIN_MEAN_SHARPE,
                  rc.CAPITAL_GATE_MIN_TSTAT, rc.CAPITAL_GATE_MIN_N_FOLDS,
                  rc.CAPITAL_GATE_MIN_MEAN_SHARPE, rc.GATE_MODE)
        recalibration_recommendation(_type2_table(), _cfg())
        recalibration_recommendation(_type2_table(),
                                     GateConfigSnapshot.from_retrain_config())
        after = (rc.PAPER_GATE_MIN_TSTAT, rc.PAPER_GATE_MIN_PCT_POSITIVE,
                 rc.PAPER_GATE_MIN_P5_SHARPE, rc.PAPER_GATE_MIN_MEAN_SHARPE,
                 rc.CAPITAL_GATE_MIN_TSTAT, rc.CAPITAL_GATE_MIN_N_FOLDS,
                 rc.CAPITAL_GATE_MIN_MEAN_SHARPE, rc.GATE_MODE)
        assert before == after

    def test_rule_text_pre_registered_and_report_only(self):
        """Pre-registration integrity: the rule text is a fixed module constant
        carrying its registration date and the report-only contract."""
        from scripts.walkforward.gate_calibration import (
            PREREGISTERED_RECALIBRATION_RULE,
        )
        assert "registered 2026-06-10" in PREREGISTERED_RECALIBRATION_RULE
        assert "NEVER mutates" in PREREGISTERED_RECALIBRATION_RULE
        assert "1.50" in PREREGISTERED_RECALIBRATION_RULE  # the floor
        # MAJOR-4 amendment (pre-run): balanced books are the rule's nulls.
        assert "AMENDMENT v1.1" in PREREGISTERED_RECALIBRATION_RULE
        assert "null_beta" in PREREGISTERED_RECALIBRATION_RULE


# -- 2. BLOCKER-1: significance core vs full gate --------------------------------

class TestSignificanceCore:
    def test_core_pass_derived_from_recorded_detail(self):
        passing = _row("x", "positive_alpha", 0.7, 2.5)
        assert passing.significance_core_pass is True
        core_only = _row("x", "positive_alpha", 0.7, 2.5, pf_ok=False)
        assert core_only.significance_core_pass is True   # PF is a backstop
        assert core_only.paper_pass is False              # full gate fails
        core_fail = _row("x", "positive_alpha", 0.7, 1.2)
        assert core_fail.significance_core_pass is False

    def test_core_pass_none_on_missing_detail(self):
        assert significance_core_pass_from_detail({}) is None
        assert significance_core_pass_from_detail(
            {"tstat": (2.5, True)}) is None  # incomplete core set

    def test_series_pf_artifact_tagging(self):
        """A series row that passes the core but fails ONLY PF/Calmar is a
        metric-mapping artifact, not a significance failure."""
        art = _row("tsmom_4y", "positive_alpha", 0.71, 2.4, pf_ok=False,
                   control_kind="series_tsmom")
        assert is_series_pf_mapping_artifact(art) is True
        # Event-class row with the same pattern: PF is per-trade there - no tag.
        ev = _row("pead_baseline", "known_marginal", 0.578, 2.4, pf_ok=False,
                  control_kind="pead")
        assert is_series_pf_mapping_artifact(ev) is False
        # Series row failing the core too: genuine significance failure - no tag.
        sig = _row("tsmom_4y", "positive_alpha", 0.71, 1.2, pf_ok=False,
                   control_kind="series_tsmom")
        assert is_series_pf_mapping_artifact(sig) is False
        # Series row passing the full gate: nothing to tag.
        ok = _row("tsmom_4y", "positive_alpha", 0.71, 2.4,
                  control_kind="series_tsmom")
        assert is_series_pf_mapping_artifact(ok) is False

    def test_dual_aggregates_and_artifact_tag_in_table(self):
        """The artifact row counts in the CORE aggregate but not the FULL one,
        and carries the explicit tag in the table body."""
        rows = [
            _row("tsmom_4y", "positive_alpha", 0.71, 2.4, pf_ok=False,
                 control_kind="series_tsmom"),
        ]
        for i in range(1, 6):
            rows.append(_row(f"random_balanced_seed_{i}", "null", 0.0, 0.2,
                             mean=0.02, pctpos=0.4, p5=-0.6))
        table = build_oc_table(rows, as_of="2026-06-10")
        assert "SERIES PF-MAPPING ARTIFACT (not a significance failure)" in table
        assert ("AGGREGATE positive-control PAPER pass-rate (FULL gate; "
                "positive_alpha+known_marginal): 0/1" in table)
        assert "significance-core pass-rate" in table
        assert "1/1 (100%)" in table
        assert "backstop artifacts, not significance Type-II" in table


# -- 3. OC table ---------------------------------------------------------------

class TestOCTable:
    def test_contains_every_control_and_aggregates(self):
        rows = _type2_table()
        table = build_oc_table(rows, as_of="2026-06-10", smoke=False)
        for r in rows:
            assert r.control_id in table
        assert "positive-control PAPER pass-rate" in table
        assert "significance-core pass-rate" in table
        assert "null PAPER pass-rate" in table
        assert "beta-loaded null PAPER pass-rate" in table
        assert "LEAKY controls flagged implausible" in table
        assert "positive_beta PAPER pass-rate" in table

    def test_residual_alpha_column_on_every_row(self):
        rows = _type2_table()
        table = build_oc_table(rows, as_of="2026-06-10")
        assert "rA_t" in table  # header
        # The null_beta row recorded rA_t=0.1 -> printed; others -> n/a.
        assert "+0.10" in table
        assert "n/a" in table

    def test_capital_floor_footnote(self):
        """MINOR-6: CAPITAL is structurally unreachable at k=8."""
        table = build_oc_table(_type2_table(), as_of="2026-06-10")
        assert "CAPITAL unreachable here" in table
        assert "CAPITAL_GATE_MIN_N_FOLDS=10" in table

    def test_run_failed_rows_excluded_from_aggregates_and_listed(self):
        """MAJOR-2: failed rows go to the FAILED RUNS section, never into the
        scored listing or the aggregate denominators."""
        rows = _type2_table()
        rows[0] = _failed_row("tsmom_4y", "positive_alpha", 0.71,
                              note="RUN FAILED: yfinance timeout")
        table = build_oc_table(rows, as_of="2026-06-10")
        assert "FAILED RUNS - rerun before interpreting" in table
        assert "yfinance timeout" in table
        # Denominator drops from 3 in-scope positives to 2 (xmom + pead).
        assert "0/2 (0%)" in table
        assert "0/3" not in table

    def test_no_failed_section_when_all_ran(self):
        table = build_oc_table(_type2_table(), as_of="2026-06-10")
        assert "FAILED RUNS" not in table

    def test_deterministic_and_ascii(self):
        rows = _type2_table()
        t1 = build_oc_table(rows, as_of="2026-06-10")
        t2 = build_oc_table(rows, as_of="2026-06-10")
        assert t1 == t2
        t1.encode("ascii")  # raises on any non-ASCII char (Windows cp1252 console)

    def test_includes_recommendation_when_given(self):
        rows = _type2_table()
        rec = recalibration_recommendation(rows, _cfg())
        table = build_oc_table(rows, recommendation=rec, as_of="2026-06-10")
        assert "PRE-REGISTERED RECALIBRATION RULE" in table
        assert "VERDICT: RECOMMEND_LOWER_TSTAT" in table
        assert "1.70" in table


# -- 4. Null scorers -------------------------------------------------------------

class TestRandomScorer:
    def _data(self):
        idx = pd.bdate_range("2024-01-02", periods=30)
        df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                           "close": 100.0, "volume": 1e6}, index=idx)
        return {f"S{i}": df for i in range(12)}

    def test_same_seed_same_day_identical(self):
        data = self._data()
        s1, s2 = make_random_scorer(1304), make_random_scorer(1304)
        d = date(2024, 2, 1)
        assert s1(d, data) == s2(d, data)
        # Repeated calls on the same instance are also stable (per-(seed,day) RNG,
        # independent of call order - fold ordering cannot change results).
        assert s1(d, data) == s1(d, data)

    def test_different_seed_differs(self):
        data = self._data()
        s1, s2 = make_random_scorer(1304), make_random_scorer(1305)
        days = [date(2024, 2, d) for d in range(1, 10)]
        assert any(s1(d, data) != s2(d, data) for d in days)

    def test_returns_long_triples(self):
        out = make_random_scorer(1304)(date(2024, 2, 1), self._data())
        assert out and all(len(t) == 3 and t[2] == "long" and 0 < t[1] <= 1.0
                           for t in out)


class TestBalancedRandomScorer:
    """MAJOR-4: the TRUE zero-SR null - direction-balanced, net ~ 0."""

    def _data(self):
        idx = pd.bdate_range("2024-01-02", periods=30)
        df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                           "close": 100.0, "volume": 1e6}, index=idx)
        d = {f"S{i}": df for i in range(12)}
        d["SPY"] = df  # synthetic input symbol, must be excluded
        return d

    def test_balanced_long_short_and_interleaved(self):
        out = make_balanced_random_scorer(1404)(date(2024, 2, 1), self._data())
        longs = [t for t in out if t[2] == "long"]
        shorts = [t for t in out if t[2] == "short"]
        assert len(longs) == len(shorts) > 0              # net ~ 0 by construction
        # Interleaved so a position cap cannot fill longs first (net-long bias).
        assert [t[2] for t in out[:4]] == ["long", "short", "long", "short"]
        # qualityshort convention: shorts carry negative confidence; sim sizes
        # on abs(conf) in (0, 1].
        assert all(t[1] < 0 and 0 < abs(t[1]) <= 1.0 for t in shorts)
        assert all(0 < t[1] <= 1.0 for t in longs)
        # No symbol on both sides; SPY never traded.
        syms = [t[0] for t in out]
        assert len(set(syms)) == len(syms)
        assert "SPY" not in syms

    def test_deterministic_per_seed_day(self):
        data = self._data()
        d = date(2024, 2, 1)
        s1, s2 = make_balanced_random_scorer(1404), make_balanced_random_scorer(1404)
        assert s1(d, data) == s2(d, data)
        assert s1(d, data) != make_balanced_random_scorer(1405)(d, data)

    def test_too_few_symbols_returns_empty(self):
        idx = pd.bdate_range("2024-01-02", periods=5)
        df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                           "close": 100.0, "volume": 1e6}, index=idx)
        assert make_balanced_random_scorer(1404)(date(2024, 1, 8), {"A": df}) == []


# -- 5. Leaky scorer -----------------------------------------------------------

class TestLeakyScorer:
    def test_prefers_future_winners(self):
        idx = pd.bdate_range("2024-01-02", periods=10)

        def df(path):
            arr = np.array(path, dtype=float)
            return pd.DataFrame({"open": arr, "high": arr * 1.01, "low": arr * 0.99,
                                 "close": arr, "volume": 1e6}, index=idx)

        data = {
            "WIN": df([100, 100, 110, 120, 130, 140, 150, 160, 170, 180]),
            "FLAT": df([100] * 10),
            "LOSE": df([100, 100, 90, 80, 70, 60, 50, 40, 30, 20]),
            "SPY": df([100] * 10),
        }
        out = make_leaky_scorer(horizon=2)(idx[1].date(), data)
        syms = [s for s, _, _ in out]
        assert syms[0] == "WIN"          # ranked by FUTURE return
        assert "LOSE" not in syms        # negative future return filtered
        assert "SPY" not in syms         # synthetic input symbol excluded

    def test_no_future_bars_no_signal(self):
        idx = pd.bdate_range("2024-01-02", periods=5)
        arr = np.linspace(100, 120, 5)
        data = {"A": pd.DataFrame({"open": arr, "high": arr, "low": arr,
                                   "close": arr, "volume": 1e6}, index=idx)}
        # Scoring on the last day: no t+2 bar exists -> nothing to leak.
        assert make_leaky_scorer(horizon=2)(idx[-1].date(), data) == []


# -- 6. End-to-end smoke (synthetic; REAL run_cpcv + REAL production gate) -----

@pytest.fixture(scope="module")
def smoke_rows():
    return run_all_controls(as_of=AS_OF, smoke=True,
                            only=["random_seed_1", "random_balanced_seed_1",
                                  "leaky_tplus1"])


class TestSmokeEndToEnd:
    def test_leaky_crosses_implausibility_ceiling(self, smoke_rows):
        from app.ml.retrain_config import SHARPE_IMPLAUSIBILITY_CEILING
        leaky = next(r for r in smoke_rows if r.control_id == "leaky_tplus1")
        assert leaky.notes and "wiring" in leaky.notes
        assert leaky.mean_sharpe > SHARPE_IMPLAUSIBILITY_CEILING
        assert leaky.implausibility_flag is True
        # The leak is strong enough to clear the significance gate - proving the
        # gate alone cannot catch it; the ceiling flag is what catches it.
        assert leaky.paper_pass is True
        assert leaky.significance_core_pass is True
        assert leaky.run_failed is False

    def test_random_null_fails_paper_gate(self, smoke_rows):
        rnd = next(r for r in smoke_rows if r.control_id == "random_seed_1")
        assert rnd.paper_pass is False
        assert rnd.failed_paper_criteria  # attribution recorded
        assert rnd.n_paths_evaluated > 0  # paths actually ran
        assert rnd.declared_class == "null_beta"  # MAJOR-4 relabel

    def test_balanced_null_runs_through_short_path(self, smoke_rows):
        """MAJOR-4: the direction-balanced null actually runs (shorts via the
        factor-scorer short path) and produces a scoreable row."""
        bal = next(r for r in smoke_rows
                   if r.control_id == "random_balanced_seed_1")
        assert bal.run_failed is False
        assert bal.declared_class == "null"
        assert bal.n_paths_evaluated > 0
        assert bal.significance_core_pass is not None
        assert "direction-balanced" in bal.notes

    def test_rows_run_through_real_cpcv_geometry(self, smoke_rows):
        for r in smoke_rows:
            assert r.n_folds == 4 and r.smoke is True
            assert r.window_start and r.window_end
            assert r.run_at  # MAJOR-3: per-control run timestamp recorded
            # MINOR-7: smoke geometry flagged explicitly per row.
            assert "k=4" in r.notes

    def test_artifact_roundtrip(self, smoke_rows, tmp_path):
        cfg = GateConfigSnapshot.from_retrain_config()
        rec = recalibration_recommendation(smoke_rows, cfg)
        out = tmp_path / "gate_calibration_test.json"
        path = write_artifact(smoke_rows, rec, cfg, as_of=AS_OF, smoke=True,
                              out_path=str(out))
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["schema_version"] == 2
        assert payload["as_of"] == "2026-06-10"
        assert payload["smoke"] is True
        assert payload["gate_config"]["paper_min_tstat"] == cfg.paper_min_tstat
        assert "PRE-REGISTERED" in payload["preregistered_rule"]
        assert len(payload["rows"]) == len(smoke_rows)
        for row_d, row in zip(payload["rows"], smoke_rows):
            assert row_d["control_id"] == row.control_id
            assert row_d["paper_pass"] == row.paper_pass
            assert row_d["significance_core_pass"] == row.significance_core_pass
            assert row_d["run_failed"] == row.run_failed
            assert "residual_alpha_t" in row_d
            assert set(row_d["paper_detail"]) == set(row.paper_detail)
        assert payload["recommendation"]["triggered"] in (True, False)
        assert "verdict" in payload["recommendation"]


# -- 7. MAJOR-3: artifact path / merge-upsert ------------------------------------

class TestArtifactMerge:
    def test_smoke_suffix_never_clobbers_full_artifact(self):
        full = default_artifact_path(AS_OF, smoke=False)
        smoke = default_artifact_path(AS_OF, smoke=True)
        assert full != smoke
        assert full.endswith("gate_calibration_20260610.json")
        assert smoke.endswith("gate_calibration_20260610_smoke.json")

    def test_merge_upsert_by_control_id(self, tmp_path):
        cfg = _cfg()
        p = tmp_path / "gate_calibration_20260610.json"
        r_old = _row("tsmom_4y", "positive_alpha", 0.71, 1.72,
                     control_kind="series_tsmom")
        rec = recalibration_recommendation([r_old], cfg)
        write_artifact([r_old], rec, cfg, as_of=AS_OF, smoke=False,
                       out_path=str(p))
        # Second staged run: re-runs tsmom_4y (new tstat) and adds xmom.
        loaded = load_artifact_rows(str(p))
        assert [r.control_id for r in loaded] == ["tsmom_4y"]
        assert loaded[0].paper_detail == r_old.paper_detail  # tuple round-trip
        r_new = _row("tsmom_4y", "positive_alpha", 0.71, 2.40,
                     control_kind="series_tsmom")
        r_x = _row("xmom_12_1", "positive_alpha", 0.50, 1.55,
                   control_kind="series_xmom")
        merged = merge_rows(loaded, [r_new, r_x])
        assert [r.control_id for r in merged] == ["tsmom_4y", "xmom_12_1"]
        assert next(r for r in merged
                    if r.control_id == "tsmom_4y").tstat == pytest.approx(2.40)

    def test_smoke_artifact_never_merged_into_full(self, tmp_path):
        cfg = _cfg()
        p = tmp_path / "gate_calibration_20260610.json"
        write_artifact([_row("tsmom_4y", "positive_alpha", 0.71, 1.72)],
                       smoke_recommendation(), cfg, as_of=AS_OF, smoke=True,
                       out_path=str(p))
        assert load_artifact_rows(str(p)) == []  # smoke rows are not evidence

    def test_load_missing_or_corrupt_artifact_is_safe(self, tmp_path):
        assert load_artifact_rows(str(tmp_path / "nope.json")) == []
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_artifact_rows(str(bad)) == []


# -- 8. NIT-8: row-build consistency guard ---------------------------------------

class _StubResult:
    """Minimal CPCVResult stand-in for _result_to_row consistency tests."""

    def __init__(self, detail, paper_pass=None):
        self._detail = detail
        self._paper = paper_pass
        self.mean_sharpe = 1.0
        self.std_sharpe = 0.2
        self.path_sharpe_tstat = 2.5
        self.pct_positive = 1.0
        self.p5_sharpe = 0.5
        self.worst_regime_sharpe = 0.1
        self.n_folds = 4
        self.path_sharpes = [1.0] * 6
        self.residual_alpha_t_hac = 0.42
        self.requires_human_review_flag = False

    def gate_detail(self, tier="paper"):
        return dict(self._detail)

    def gate_passed(self, tier="paper"):
        if tier == "capital":
            return False
        if self._paper is not None:
            return self._paper
        return all(ok for k, (_v, ok) in self._detail.items()
                   if k != "requires_human_review")

    def requires_human_review(self):
        return False

    def _dsr_n_obs(self):
        return 500


def _full_detail():
    return {
        "n_paths": (6, True), "tstat": (2.5, True), "pct_positive": (1.0, True),
        "p5_sharpe": (0.5, True), "mean_sharpe": (1.0, True),
        "avg_profit_factor": (1.5, True), "avg_calmar": (0.8, True),
        "worst_regime_sharpe": (0.1, True),
    }


class TestRowBuildConsistency:
    def _build(self, result):
        from scripts.walkforward.gate_calibration import _result_to_row
        return _result_to_row(CONTROLS["tsmom_4y"], result,
                              window_start="2022-06-10", window_end="2026-06-10",
                              runtime_sec=1.0, smoke=True, notes="")

    def test_consistent_row_is_scored(self):
        row = self._build(_StubResult(_full_detail()))
        assert row.run_failed is False
        assert row.paper_pass is True
        assert row.significance_core_pass is True
        assert row.residual_alpha_t == pytest.approx(0.42)
        assert row.control_kind == "series_tsmom"

    def test_empty_detail_marks_run_failed(self):
        """Early-returned/empty gate detail must not be silently scored."""
        row = self._build(_StubResult({}, paper_pass=False))
        assert row.run_failed is True
        assert row.significance_core_pass is None
        assert "DETAIL INCOMPLETE" in row.notes

    def test_pass_detail_divergence_marks_run_failed(self):
        """gate_passed()=False with an all-ok detail = the latent
        REQUIRE_TRUE_WF_FOR_PROMOTION / in_sample_override early-return surface."""
        row = self._build(_StubResult(_full_detail(), paper_pass=False))
        assert row.run_failed is True
        assert "DETAIL DIVERGENCE" in row.notes

    def test_regime_waived_passing_row_not_failed(self):
        """NIT-5/NIT-8 (highest-risk surface): a regime-waived PASSING control
        (e.g. pead_baseline via the event-sparsity waiver) carries the
        INFORMATIONAL requires_human_review key (ok=False) alongside a
        worst_regime_sharpe of None. The consistency guard must EXCLUDE that
        informational key from AND(detail), so the row is NOT falsely marked
        run_failed - otherwise the decisive PEAD evidence silently vanishes."""
        detail = _full_detail()
        detail["worst_regime_sharpe"] = (None, True)        # waived -> ok
        detail["requires_human_review"] = (True, False)     # informational, ok=False
        row = self._build(_StubResult(detail, paper_pass=True))
        assert row.run_failed is False
        assert row.paper_pass is True
        assert row.significance_core_pass is True


# -- Registry sanity -----------------------------------------------------------

class TestControlRegistry:
    def test_blueprint_controls_present(self):
        expected = {"tsmom_4y", "tsmom_19y", "xmom_12_1", "pead_baseline",
                    "spy_buyhold", "leaky_tplus1"} | {
                        f"random_seed_{i}" for i in range(1, 6)} | {
                        f"random_balanced_seed_{i}" for i in range(1, 6)}
        assert expected == set(CONTROLS)

    def test_declared_classes_and_priors(self):
        assert CONTROLS["tsmom_4y"].declared_class == "positive_alpha"
        assert CONTROLS["tsmom_4y"].true_sr_prior == pytest.approx(0.71)
        assert CONTROLS["pead_baseline"].declared_class == "known_marginal"
        assert CONTROLS["spy_buyhold"].declared_class == "positive_beta"
        assert CONTROLS["leaky_tplus1"].true_sr_prior is None
        for i in range(1, 6):
            # MAJOR-4: balanced books are the TRUE zero-SR nulls (rule scope).
            bal = CONTROLS[f"random_balanced_seed_{i}"]
            assert bal.declared_class == "null"
            assert bal.true_sr_prior == 0.0
            assert bal.seed == 1403 + i
            # Long-only randoms: beta-loaded diagnostic outside the rule.
            spec = CONTROLS[f"random_seed_{i}"]
            assert spec.declared_class == "null_beta"
            assert spec.true_sr_prior is None
            assert "beta" in spec.prior_note.lower()
            assert spec.seed == 1303 + i
