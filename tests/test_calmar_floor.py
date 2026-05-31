"""MEDIUM-1: Calmar vol-floor tests."""
from datetime import date
from unittest.mock import patch

from scripts.walkforward.gates import (
    compute_calmar, FoldResult, WalkForwardReport, CAL_TOTAL_LOSS_SENTINEL,
)
from scripts.walkforward.cpcv import CPCVResult


def test_vol_floor_finite_not_sentinel():
    val = compute_calmar(0.10, 0.0, 1.0, daily_returns=[0.001] * 50)
    assert val != 5.0
    assert val != 0.0
    import math
    assert math.isfinite(val)


def test_vol_floor_falls_back_to_min_dd(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        val = compute_calmar(0.10, 0.0, 1.0, daily_returns=None)
    # MIN_CALMAR_FLOOR_DD = 0.01 → 0.10 / 0.01 = 10.0
    assert abs(val - 10.0) < 1e-6
    assert any("MIN_CALMAR_FLOOR_DD" in r.message for r in caplog.records)


def test_normal_calmar_unchanged():
    # dd > 0 → floor not used; CAGR(0.10,1yr)/0.05 = 2.0
    val = compute_calmar(0.10, 0.05, 1.0)
    assert abs(val - 2.0) < 1e-9


def test_total_loss_sentinel():
    assert compute_calmar(-1.5, 0.0, 1.0) == CAL_TOTAL_LOSS_SENTINEL


def test_legacy_no_dd_returns_zero():
    with patch("app.ml.retrain_config.USE_CALMAR_VOL_FLOOR", False):
        assert compute_calmar(0.10, 0.0, 1.0) == 0.0


def _no_dd_fold(idx: int, calmar: float) -> FoldResult:
    return FoldResult(
        fold=idx,
        train_start=date(2020, 1, 1), train_end=date(2022, 1, 1),
        test_start=date(2022, 1, 1), test_end=date(2023, 1, 1),
        trades=50, win_rate=0.6, sharpe=1.5,
        max_drawdown=0.0, total_return=0.3, stop_exit_rate=0.1,
        n_obs=250, profit_factor=2.0, calmar_ratio=calmar,
    )


def test_report_avg_calmar_floored_not_sentinel():
    # 3 no-DD profitable folds with vol-floored calmar (not 5.0 sentinel).
    floored = compute_calmar(0.30, 0.0, 1.0, daily_returns=[0.001] * 250)
    folds = [_no_dd_fold(i + 1, floored) for i in range(3)]
    r = WalkForwardReport(model_type="swing", folds=folds)
    with patch("app.ml.retrain_config.USE_CALMAR_VOL_FLOOR", True):
        assert abs(r.avg_calmar - floored) < 1e-6
        assert r.avg_calmar != 5.0


def test_cpcv_avg_calmar_parity():
    floored = compute_calmar(0.30, 0.0, 1.0, daily_returns=[0.001] * 250)
    c = CPCVResult(
        model_type="swing", n_folds=6, n_paths=2,
        path_sharpes=[1.5, 1.5, 1.5],
        path_profit_factors=[2.0, 2.0, 2.0],
        path_calmars=[floored, floored, floored],
        path_n_obs=[250, 250, 250],
    )
    with patch("app.ml.retrain_config.USE_CALMAR_VOL_FLOOR", True):
        assert abs(c.avg_calmar - floored) < 1e-6
        assert c.avg_calmar != 5.0
