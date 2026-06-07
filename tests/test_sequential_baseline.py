"""Alpha-v4 P0 — purged sequential-WF baseline (scripts/walkforward/sequential_baseline.py).

The runner is a thin wrapper over the already-tested FoldEngine.run, so these tests
verify the wiring (FoldEngine constructed + run with the passed params; strategy
regime map reused) and the comparison printer, using mocks — no real fold execution.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.walkforward.sequential_baseline import (
    run_sequential_baseline, print_baseline_vs_cpcv,
)


def test_runner_constructs_foldengine_and_runs():
    strategy = MagicMock()
    strategy._global_regime_map = {"d": "BULL"}
    fake_report = MagicMock()

    with patch("scripts.walkforward.engine.FoldEngine") as MockEngine:
        MockEngine.return_value.run.return_value = fake_report
        out = run_sequential_baseline(
            strategy, n_folds=6, purge_days=85, total_years=10,
            embargo_days=85, train_years=4,
        )

    # FoldEngine constructed with the strategy + purge/embargo + the reused regime map.
    _, ckwargs = MockEngine.call_args
    assert MockEngine.call_args[0][0] is strategy
    assert ckwargs["purge_days"] == 85
    assert ckwargs["embargo_days"] == 85
    assert ckwargs["regime_map"] == {"d": "BULL"}
    # run() called through with the WF params.
    _, rkwargs = MockEngine.return_value.run.call_args
    assert rkwargs["n_folds"] == 6
    assert rkwargs["total_years"] == 10
    assert rkwargs["train_years"] == 4
    assert out is fake_report


def test_runner_reuses_strategy_regime_map_when_none_passed():
    strategy = MagicMock()
    strategy._global_regime_map = {"x": "BEAR"}
    with patch("scripts.walkforward.engine.FoldEngine") as MockEngine:
        MockEngine.return_value.run.return_value = MagicMock()
        run_sequential_baseline(strategy, n_folds=4, purge_days=2, total_days=500)
    assert MockEngine.call_args[1]["regime_map"] == {"x": "BEAR"}
    # intraday-style: total_days threaded, total_years stays None.
    rkwargs = MockEngine.return_value.run.call_args[1]
    assert rkwargs["total_days"] == 500
    assert rkwargs["total_years"] is None


def test_comparison_printer_shows_both_columns(capsys):
    wf = MagicMock(avg_sharpe=0.41, min_sharpe=-0.10, worst_regime_sharpe=-0.05,
                   folds=[1, 2, 3, 4, 5])
    cpcv = MagicMock(mean_sharpe=0.55, p5_sharpe=0.01, worst_regime_sharpe=-0.02, n_folds=6)
    print_baseline_vs_cpcv(wf, cpcv)
    out = capsys.readouterr().out
    assert "CPCV" in out and "Sequential-WF" in out
    assert "+0.550" in out  # CPCV mean
    assert "+0.410" in out  # baseline avg


def test_comparison_printer_flags_optimistic_gap(capsys):
    # CPCV mean far above the sequential baseline → low-coverage warning.
    wf = MagicMock(avg_sharpe=0.10, min_sharpe=-0.5, worst_regime_sharpe=-0.5, folds=[1, 2])
    cpcv = MagicMock(mean_sharpe=0.80, p5_sharpe=-0.2, worst_regime_sharpe=-0.4, n_folds=6)
    print_baseline_vs_cpcv(wf, cpcv)
    out = capsys.readouterr().out
    assert "exceeds the sequential baseline" in out


def test_comparison_printer_tolerates_missing_fields(capsys):
    # Defensive: partially-populated results must not raise.
    print_baseline_vs_cpcv(MagicMock(spec=[]), MagicMock(spec=[]))
    out = capsys.readouterr().out
    assert "Sequential-WF baseline vs CPCV" in out
    assert "n/a" in out
