"""CRITICAL-2: deployment-adjusted Sharpe / low-deployment warning tests."""
from datetime import date, timedelta

from app.backtesting.strategy_simulator import (
    SimResult, StrategySimulator, compute_deployment_metrics,
)
from scripts.walkforward.gates import FoldResult, WalkForwardReport


def _curve(n: int = 50):
    base = date(2023, 1, 1)
    return [(base + timedelta(days=d - 1), 100000 * (1 + 0.001 * d))
            for d in range(1, n + 1)]


def test_full_deployment_matches_raw_sharpe():
    curve = _curve()
    deps = {d: 1.0 for d, _ in curve}
    avg_dep, dep_adj, warn = compute_deployment_metrics(curve, deps)
    eq = [v for _, v in curve]
    rets = [(eq[i] - eq[i - 1]) / eq[i - 1] for i in range(1, len(eq))]
    raw = StrategySimulator._sharpe(rets, 252)
    assert abs(avg_dep - 1.0) < 1e-9
    assert abs(dep_adj - raw) < 1e-6
    assert warn is False


def test_low_deployment_warning_triggers():
    curve = _curve()
    deps = {d: 0.05 for d, _ in curve}
    avg_dep, dep_adj, warn = compute_deployment_metrics(curve, deps)
    assert abs(avg_dep - 0.05) < 1e-9
    assert warn is True


def test_deployment_boundary_strict_less_than():
    # Strict-< semantics: deployment AT (or above) the threshold must NOT warn,
    # deployment just below it MUST warn. Use a value safely at/above 0.10 to avoid
    # float-mean rounding landing fractionally under the threshold at the exact boundary.
    from app.ml.retrain_config import MIN_DEPLOYMENT_PCT_WARN
    curve = _curve()
    at_threshold = MIN_DEPLOYMENT_PCT_WARN + 1e-9
    deps_at = {d: at_threshold for d, _ in curve}
    avg_at, _, warn_at = compute_deployment_metrics(curve, deps_at)
    assert avg_at >= MIN_DEPLOYMENT_PCT_WARN
    assert warn_at is False

    below = MIN_DEPLOYMENT_PCT_WARN - 0.001
    deps_below = {d: below for d, _ in curve}
    _, _, warn_below = compute_deployment_metrics(curve, deps_below)
    assert warn_below is True


def test_all_zero_deployment_no_div_by_zero():
    curve = _curve()
    deps = {d: 0.0 for d, _ in curve}
    avg_dep, dep_adj, warn = compute_deployment_metrics(curve, deps)
    assert avg_dep == 0.0
    assert dep_adj == 0.0
    assert warn is True


def test_very_low_deployment_scale_capped():
    import math
    curve = _curve()
    deps = {d: 0.001 for d, _ in curve}
    avg_dep, dep_adj, warn = compute_deployment_metrics(curve, deps)
    # scale capped at DEP_ADJ_MAX_SCALE — no inf/nan
    assert math.isfinite(dep_adj)
    assert warn is True


def test_report_gate_detail_low_deployment_does_not_block():
    folds = [
        FoldResult(
            fold=i + 1,
            train_start=date(2020, 1, 1), train_end=date(2022, 1, 1),
            test_start=date(2022, 1, 1), test_end=date(2023, 1, 1),
            trades=50, win_rate=0.6, sharpe=1.5,
            max_drawdown=0.05, total_return=0.3, stop_exit_rate=0.1,
            n_obs=250, profit_factor=2.0, calmar_ratio=1.0,
            avg_capital_deployed_pct=0.05, low_deployment_warning=True,
            # Phase 2: real regime data so the now-enforced regime gate isn't the
            # blocker (this test is about low-deployment, not regime).
            regime_sharpes={"BULL": 1.0, "BEAR": 0.5, "NEUTRAL": 0.8},
        )
        for i in range(2)
    ]
    r = WalkForwardReport(model_type="swing", folds=folds)
    detail = r.gate_detail()
    assert "low_deployment_warning" in detail
    assert detail["low_deployment_warning"][1] is False  # triggered
    assert r.low_deployment is True
    # gate_passed must be unaffected by low deployment.
    assert r.gate_passed() is True


def test_simresult_defaults_backward_compat():
    res = SimResult(
        model_type="swing", starting_capital=1.0, ending_capital=1.0,
        total_return_pct=0.0, annualized_return_pct=0.0, sharpe_ratio=0.0,
        sortino_ratio=0.0, max_drawdown_pct=0.0, calmar_ratio=0.0,
    )
    assert res.avg_capital_deployed_pct == 0.0
    assert res.deployment_adjusted_sharpe == 0.0
    assert res.low_deployment_warning is False
