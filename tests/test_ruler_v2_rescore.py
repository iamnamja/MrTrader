"""
Tests for the Ruler-v2 kill-ledger RE-SCORE (Alpha-v7 Phase B, Phase 5) —
scripts/walkforward/ruler_v2_rescore.py. STRICTLY REPORT-ONLY.

Pins: the 4 paper-flip classifications (REVIVED/DEMOTED/unchanged) using results where
the significance verdict (path-Sharpe distribution) and the Ruler-v2 verdict
(OOS-series inference) genuinely diverge; the empty-OOS legacy-result note; the
summarize_flips counts; the REPORT-ONLY framing in the rendered table; and the HARD
CONTRACT that re-scoring mutates NO global state (GATE_MODE untouched).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.walkforward import ruler_v2_rescore as rs
from scripts.walkforward.cpcv import CPCVResult


def _oos(sr_ann: float, n=2500, sd=0.01, seed=0):
    """A dated OOS series whose realized annualized Sharpe is EXACTLY `sr_ann`
    (standardized). n=2500 (~10y) so a passing SR (0.6) clears the PAPER HAC
    significance floor (t = sr_ann·√(n/252); 0.6·√(2500/252) ≈ 1.9 → p ≈ 0.03 < 0.05)
    while a 0.10 SR stays insignificant — the verdicts don't ride on sampling noise."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    z = (z - z.mean()) / z.std()                      # mean 0, unit std
    r = sd * (z + sr_ann / np.sqrt(252))              # daily SR == sr_ann/√252 exactly
    idx = pd.bdate_range("2019-01-02", periods=n)
    return [(d.strftime("%Y-%m-%d"), float(x)) for d, x in zip(idx, r)]


def _result(path_sharpes, oos_sr_ann, *, worst_regime=0.10, n_folds=12, seed=0):
    res = CPCVResult(model_type="test", n_folds=n_folds, n_paths=len(path_sharpes))
    res.path_sharpes = list(path_sharpes)
    res.is_true_walkforward = True
    res.worst_regime_sharpe = worst_regime
    res.residual_alpha_t_hac = 1.0
    res.oos_returns_dated = _oos(oos_sr_ann, seed=seed)
    return res


# path-Sharpe profiles: a dispersed-positive set passes the significance core; a flat
# 0.32 set fails it (tstat=0 on zero dispersion AND mean<0.35); a flat 0.10 set fails.
_SIG_PASS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.4, 0.6, 0.5, 0.5, 0.45, 0.55]
_SIG_FAIL_MARGINAL = [0.32] * 12
_SIG_FAIL_WEAK = [0.10] * 12


def test_revived_paper_flip():
    # significance FAILS (path mean 0.32 < 0.35, zero-dispersion tstat=0) but the OOS
    # SERIES has SR 0.6 → Ruler-v2 PAPER (plausibility) PASSES → REVIVED.
    res = _result(_SIG_FAIL_MARGINAL, oos_sr_ann=0.6)
    row = rs.rescore_result(res, label="h-revive", declared_class="known_marginal")
    assert row.sig_paper_pass is False and row.rv2_paper_pass is True
    assert row.paper_flip == rs.FLIP_REVIVED


def test_demoted_paper_flip():
    # significance PASSES (dispersed positive paths) but the concatenated OOS SERIES
    # has SR 0.1 < the 0.30 plausibility floor → Ruler-v2 PAPER FAILS → DEMOTED.
    res = _result(_SIG_PASS, oos_sr_ann=0.10)
    row = rs.rescore_result(res, label="h-demote", declared_class="positive_alpha")
    assert row.sig_paper_pass is True and row.rv2_paper_pass is False
    assert row.paper_flip == rs.FLIP_DEMOTED
    assert "point_sr_floor" in row.rv2_paper_failed


def test_unchanged_pass_and_fail():
    keep = rs.rescore_result(_result(_SIG_PASS, oos_sr_ann=0.6), label="keep")
    assert keep.paper_flip == rs.FLIP_UNCHANGED_PASS
    dead = rs.rescore_result(_result(_SIG_FAIL_WEAK, oos_sr_ann=0.10), label="dead")
    assert dead.paper_flip == rs.FLIP_UNCHANGED_FAIL


def test_capital_column_fails_without_live_paper():
    # A re-scored kill has no live-paper record → CAPITAL fails on live_paper_present
    # by construction, even when PAPER is revived.
    row = rs.rescore_result(_result(_SIG_FAIL_MARGINAL, oos_sr_ann=0.6), label="x")
    assert row.rv2_capital_pass is False
    assert "live_paper_present" in row.rv2_capital_failed


def test_empty_oos_is_flagged_and_fails_closed():
    res = _result(_SIG_PASS, oos_sr_ann=0.6)
    res.oos_returns_dated = []                       # a legacy result, pre-Phase-2
    row = rs.rescore_result(res, label="legacy")
    assert row.rv2_paper_pass is False
    assert "no oos_returns_dated" in row.notes


def test_table_and_summary():
    items = [
        ("h-revive", "known_marginal", _result(_SIG_FAIL_MARGINAL, oos_sr_ann=0.6)),
        ("h-demote", "positive_alpha", _result(_SIG_PASS, oos_sr_ann=0.10)),
        ("keep", "positive_alpha", _result(_SIG_PASS, oos_sr_ann=0.6)),
        ("dead", "null", _result(_SIG_FAIL_WEAK, oos_sr_ann=0.10)),
    ]
    rows = rs.rescore_table(items)
    counts = rs.summarize_flips(rows)
    assert counts[rs.FLIP_REVIVED] == 1 and counts[rs.FLIP_DEMOTED] == 1
    assert counts[rs.FLIP_UNCHANGED_PASS] == 1 and counts[rs.FLIP_UNCHANGED_FAIL] == 1
    txt = rs.format_rescore_table(rows)
    assert "REPORT-ONLY" in txt and "owner adjudicates" in txt
    assert "REVIVED(paper): 1" in txt


def test_hard_contract_no_global_state_mutation():
    import app.ml.retrain_config as rc
    before = rc.GATE_MODE
    rs.rescore_result(_result(_SIG_PASS, oos_sr_ann=0.6), label="x")
    assert rc.GATE_MODE == before                    # re-score flips NOTHING


def test_rescore_does_not_mutate_input_result():
    # Both gates set requires_human_review_flag on the regime-waiver path — the
    # re-score must operate on a copy and leave the caller's ledger object untouched.
    res = _result(_SIG_PASS, oos_sr_ann=0.6, worst_regime=None)
    res.regime_insufficient_obs = True
    assert res.requires_human_review_flag is False
    rs.rescore_result(res, label="x", regime_waiver_approved=True)
    assert res.requires_human_review_flag is False   # NOT mutated by the re-score


def test_significance_exception_becomes_error_sig_not_revived():
    # A result whose significance scoring raises must NOT be silently FAIL (which could
    # read as a spurious REVIVED) — it becomes ERROR_SIG with the error in notes.
    res = _result(_SIG_PASS, oos_sr_ann=0.6)
    res.path_sharpes = None                          # breaks significance_gate_passed
    row = rs.rescore_result(res, label="boom")
    assert row.paper_flip == rs.FLIP_ERROR_SIG
    assert row.sig_paper_pass is None
    assert "ERROR_SIG" in row.notes
