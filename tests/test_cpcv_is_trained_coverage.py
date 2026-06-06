"""
Alpha-v4 Phase 0.1 — CPCV full-coverage for rules-based scorers.

The BUG-23 overlap guard skips a fold evaluation when a contiguous
(rolling/expanding) TRAINING window would span a prior test fold in the same
combo. That is necessary to stop a *trained* model from learning data tested
elsewhere in the combo. But a rules-based scorer (EventEdgeStrategy:
model.trained_through == date.min — nothing is fit) uses the train window ONLY
for PIT universe construction in run_fold, never for training, so the overlap
cannot leak. The guard was therefore discarding ~half of all fold evaluations
for rules-based strategies and biasing the surviving CPCV path distribution
toward later (bull) regimes.

These tests verify run_cpcv now:
  * BYPASSES the overlap guard for rules-based strategies (is_trained=False, or
    derived from model.trained_through == date.min) → full coverage,
  * still SKIPS for trained strategies (guard ON — unchanged behavior),
  * resolves the explicit is_trained flag over the model-cutoff derivation,
  * conserves coverage exactly: bypassed == (trained skips − rules skips).

Statistical-soundness note: recovering these (correlated) paths does NOT
fabricate significance — path_sharpe_tstat divides by sqrt(N_eff=n_folds),
which is invariant to the number of paths run.
"""
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.walkforward.gates import FoldResult

# Config with KNOWN overlaps: k=4, paths=2, expanding window. Combos where the
# higher test fold b sits more than one fold above the lower test fold a
# (b > a+1) trigger the overlap guard: (0,2)·ti2, (0,3)·ti3, (1,3)·ti3 = 3.
_K = 4
_PATHS = 2
_TOTAL_YEARS = 5
_PURGE = 10
_EMBARGO = 10


def _fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
    # Signature mirrors run_cpcv's call: (fold_idx, n_folds, tr_start, tr_end, te_start, te_end).
    return FoldResult(
        fold=fold_idx, train_start=tr_start, train_end=tr_end,
        test_start=te_start, test_end=te_end,
        trades=10, win_rate=0.5, sharpe=1.0, max_drawdown=0.05,
        total_return=0.10, stop_exit_rate=0.2,
    )


def _mock_strategy(*, is_trained=None, trained_through=date(2018, 1, 1)):
    """MagicMock WF strategy. When is_trained is None the attribute is left as
    MagicMock's auto-attr (truthy, not None / not False) → treated as TRAINED
    (the conservative default that keeps every legacy MagicMock test unchanged).
    """
    s = MagicMock()
    s.model_type = "swing"
    s.version = 1
    s.allow_in_sample = False
    s.per_fold_retrain = False
    s.model = MagicMock()
    s.model.trained_through = trained_through
    s.all_days_sorted = []
    if is_trained is not None:
        s.is_trained = is_trained
    s.run_fold.side_effect = _fold
    return s


def _run(strategy):
    with patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        from scripts.walkforward.cpcv import run_cpcv
        return run_cpcv(
            strategy=strategy, purge_days=_PURGE, embargo_days=_EMBARGO,
            n_folds=_K, n_paths=_PATHS, total_years=_TOTAL_YEARS,
            allow_sacred_holdout=True,
        )


# ── result field ─────────────────────────────────────────────────────────────

def test_n_overlap_bypassed_field_defaults_zero():
    from scripts.walkforward.cpcv import CPCVResult
    r = CPCVResult(model_type="swing", n_folds=_K, n_paths=_PATHS)
    assert hasattr(r, "n_overlap_bypassed")
    assert r.n_overlap_bypassed == 0


# ── EventEdgeStrategy declares itself rules-based ────────────────────────────

def test_event_edge_strategy_declares_rules_based():
    from scripts.walkforward.event_edge import EventEdgeStrategy
    assert EventEdgeStrategy.is_trained is False


def test_pead_strategy_inherits_rules_based():
    from scripts.run_pead_cpcv import PEADStrategy
    assert PEADStrategy.is_trained is False


# ── trained strategy: guard ON (unchanged behavior) ──────────────────────────

def test_trained_strategy_still_skips_overlaps():
    s = _mock_strategy(is_trained=True)
    r = _run(s)
    assert r.n_overlap_bypassed == 0, "trained strategy must NOT bypass the guard"
    # fold-0 skips (3) + overlap skips (3) = 6 for k=4,paths=2 expanding.
    assert r.n_skipped >= 1


def test_default_magicmock_treated_as_trained():
    """A bare MagicMock auto-creates is_trained (truthy, not False) → treated as
    trained → guard ON. This is WHY every legacy MagicMock-based CPCV test keeps
    its current behavior under this change."""
    s = _mock_strategy(is_trained=None)
    r = _run(s)
    assert r.n_overlap_bypassed == 0


# ── rules-based strategy: guard BYPASSED (full coverage) ──────────────────────

def test_rules_based_bypasses_overlap_guard_via_flag():
    s = _mock_strategy(is_trained=False)
    r = _run(s)
    assert r.n_overlap_bypassed > 0, "rules-based strategy must bypass the overlap guard"


def test_rules_based_bypasses_via_model_cutoff_derivation():
    """No explicit is_trained attribute + model.trained_through == date.min must
    be derived as rules-based (the safety-net path)."""
    class _RulesStub:
        model_type = "event"
        version = 0
        allow_in_sample = False
        per_fold_retrain = False
        all_days_sorted = []
        symbols_data = {}

        def __init__(self):
            self.model = SimpleNamespace(trained_through=date.min)
            self.calls = 0

        def run_fold(self, fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
            self.calls += 1
            return _fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end)

    stub = _RulesStub()
    assert not hasattr(stub, "is_trained")
    r = _run(stub)
    assert r.n_overlap_bypassed > 0


def test_explicit_flag_overrides_cutoff():
    """is_trained=True must win even when model.trained_through == date.min."""
    s = _mock_strategy(is_trained=True, trained_through=date.min)
    r = _run(s)
    assert r.n_overlap_bypassed == 0, "explicit is_trained=True must keep the guard ON"


# ── conservation: bypass exactly accounts for the recovered skips ─────────────

def test_coverage_conservation_bypass_equals_skip_difference():
    """The folds a rules-based run RECOVERS are exactly the overlap-skips a
    trained run would have dropped. Robust to fold-boundary specifics:
        trained.n_skipped == rules.n_skipped + rules.n_overlap_bypassed
        rules.run_fold_calls == trained.run_fold_calls + rules.n_overlap_bypassed
    """
    trained = _mock_strategy(is_trained=True)
    rules = _mock_strategy(is_trained=False)
    rt = _run(trained)
    rr = _run(rules)

    assert rr.n_overlap_bypassed > 0
    # The only difference between the two runs is whether overlaps skip or run.
    assert rt.n_skipped == rr.n_skipped + rr.n_overlap_bypassed
    # Fold-0 skips are identical (guard-independent); only overlaps differ.
    assert rr.n_skipped == rt.n_skipped - rr.n_overlap_bypassed
    # Rules-based runs strictly more folds, by exactly the bypassed count.
    assert rules.run_fold.call_count == trained.run_fold.call_count + rr.n_overlap_bypassed
    assert rules.run_fold.call_count > trained.run_fold.call_count


def test_rules_based_strictly_reduces_skip_rate_vs_trained():
    """The fix strictly lowers the skip rate for a rules-based run vs the trained
    run, by exactly the bypassed-overlap fraction. (The remaining skips are the
    no-causal-training-history combos, which are guard-independent; their share
    falls as n_folds grows — at the production k=8 it drops under the 20% bias
    warning bar, but that absolute level is a property of k, not asserted here.)"""
    import math
    total_evals = math.comb(_K, _PATHS) * _PATHS
    rt = _run(_mock_strategy(is_trained=True))
    rr = _run(_mock_strategy(is_trained=False))
    assert rr.n_skipped < rt.n_skipped
    recovered_frac = (rt.n_skipped - rr.n_skipped) / total_evals
    bypass_frac = rr.n_overlap_bypassed / total_evals
    assert recovered_frac == pytest.approx(bypass_frac)
