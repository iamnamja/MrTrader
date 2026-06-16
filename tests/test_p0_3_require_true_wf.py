"""Alpha-v9 P0-3 — REQUIRE_TRUE_WF_FOR_PROMOTION flipped True.

A TRAINED model can no longer reach live on a frozen-mode generalization test; but
RULES-BASED sleeves (no fitted model → out-of-sample by construction) are unaffected.
"""
import numpy as np
import pandas as pd

from app.research import ruler_v2
from app.ml import retrain_config
from scripts.walkforward.cpcv import CPCVResult, run_cpcv


def _dated(returns):
    idx = pd.bdate_range("2018-01-02", periods=len(returns))
    return [(d.strftime("%Y-%m-%d"), float(r)) for d, r in zip(idx, returns)]


def _passing_result(true_wf: bool) -> CPCVResult:
    """A result that clears the PAPER gate on metrics, parameterized by is_true_walkforward
    so the ONLY thing under test is the true-WF pre-check."""
    rng = np.random.default_rng(0)
    r = 0.0009 + rng.normal(0, 0.010, 560)        # plausible + significant -> PAPER pass
    res = CPCVResult(model_type="test", n_folds=12, n_paths=20)
    res.path_sharpes = [1.0] * 20
    res.is_true_walkforward = true_wf
    res.worst_regime_sharpe = 0.10
    res.regime_insufficient_obs = False
    res.residual_alpha_t_hac = 3.0
    res.oos_returns_dated = _dated(r)
    return res


def test_flag_is_on():
    assert retrain_config.REQUIRE_TRUE_WF_FOR_PROMOTION is True


def test_gate_blocks_a_frozen_trained_run_even_when_metrics_pass():
    # is_true_walkforward=False (a frozen-model generalization test) -> NOT promotable
    blocked = _passing_result(true_wf=False)
    assert ruler_v2.gate_passed(blocked, tier="paper") is False


def test_gate_allows_a_true_walkforward_run():
    # same metrics, is_true_walkforward=True -> the true-WF pre-check does not block it
    ok = _passing_result(true_wf=True)
    assert ruler_v2.gate_passed(ok, tier="paper") is True


def test_frozen_wf_report_is_inconclusive_not_retire(monkeypatch):
    """C1 regression: a FROZEN (non-true-WF) WalkForwardReport blocked SOLELY by the
    true-WF requirement must return INCONCLUSIVE (report-only) from gate_outcome() —
    NOT RETIRE. RETIRE would make retrain_cron._restore_previous roll back the freshly
    trained champion. Test under live GATE_MODE='mean_sharpe' so the significance
    branch isn't what produces INCONCLUSIVE."""
    from datetime import date
    from scripts.walkforward.gates import WalkForwardReport, FoldResult, GateOutcome

    monkeypatch.setattr(retrain_config, "GATE_MODE", "mean_sharpe", raising=False)

    rep = WalkForwardReport(model_type="swing", is_true_walkforward=False)
    for _ in range(3):
        rep.folds.append(FoldResult(
            fold=1, train_start=date(2020, 1, 1), train_end=date(2022, 1, 1),
            test_start=date(2022, 1, 1), test_end=date(2023, 1, 1),
            trades=100, win_rate=0.6, sharpe=2.5, max_drawdown=0.02,
            total_return=0.5, stop_exit_rate=0.1, n_obs=250,
            profit_factor=1.8, calmar_ratio=1.5,
            regime_sharpes={"BULL": 1.0, "BEAR": 0.5, "NEUTRAL": 0.8},
        ))
    # metrics would PROMOTE; only the true-WF pre-check blocks it → must be report-only
    assert rep.gate_outcome() is GateOutcome.INCONCLUSIVE
    assert rep.gate_outcome() is not GateOutcome.RETIRE


def test_rules_based_series_strategy_is_marked_true_walkforward():
    """The load-bearing P0-3 fix: a RULES-BASED sleeve (no fitted model) is OOS by
    construction, so run_cpcv must mark it is_true_walkforward=True — otherwise flipping
    the flag would wrongly block every sleeve (carry/tsmom/calendar) promotion."""
    from scripts.walkforward.gate_calibration import SeriesReturnStrategy
    idx = pd.bdate_range("2019-01-01", periods=900)
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    strat = SeriesReturnStrategy("test_rules_based", rets)
    assert strat.per_fold_retrain is False and strat.is_trained is False   # genuinely not trained
    res = run_cpcv(strat, purge_days=5, embargo_days=5, n_folds=4, n_paths=2,
                   total_days=len(idx))
    assert res.is_true_walkforward is True
