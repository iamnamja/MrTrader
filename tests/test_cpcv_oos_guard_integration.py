"""Integration tests: CPCV raises OOSViolation when model is in-sample.

Reproduces the v63 bug: model trained through ~2026-05-28, test folds
covering Nov 2024–Apr 2026 are all in-sample. CPCV must raise OOSViolation
unless --allow-in-sample is passed.
"""
from __future__ import annotations

import pytest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from scripts.walkforward.oos_guard import OOSViolation


def _make_strategy(trained_through, allow_in_sample=False, n_days=300):
    """Build a minimal strategy namespace that run_cpcv can consume."""
    model = SimpleNamespace(trained_through=trained_through)
    # all_days_sorted: n_days of trading days before trained_through
    base = trained_through or date(2026, 5, 28)
    all_days = [base - timedelta(days=i) for i in range(n_days, 0, -1)]
    strategy = SimpleNamespace(
        model=model,
        model_type="intraday",
        version=63,
        all_days_sorted=all_days,
        allow_in_sample=allow_in_sample,
    )
    return strategy


def test_run_cpcv_raises_when_model_trained_past_test_windows():
    """Reproduces v63 bug: trained May 2026, test folds all in 2024-2026."""
    from scripts.walkforward.cpcv import run_cpcv
    from app.ml.retrain_config import assert_no_sacred_holdout

    strategy = _make_strategy(trained_through=date(2026, 5, 28))

    boundaries = [
        (date(2022, 1, 1), date(2024, 11, 1), date(2024, 11, 1), date(2025, 2, 1)),
        (date(2022, 1, 1), date(2025, 2, 1), date(2025, 2, 1), date(2025, 5, 1)),
        (date(2022, 1, 1), date(2025, 5, 1), date(2025, 5, 1), date(2025, 8, 1)),
        (date(2022, 1, 1), date(2025, 8, 1), date(2025, 8, 1), date(2025, 11, 1)),
        (date(2022, 1, 1), date(2025, 11, 1), date(2025, 11, 1), date(2026, 2, 1)),
        (date(2022, 1, 1), date(2026, 2, 1), date(2026, 2, 1), date(2026, 5, 1)),
    ]

    with patch("scripts.walkforward.engine.FoldEngine") as mock_engine_cls, \
         patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        engine = MagicMock()
        mock_engine_cls.return_value = engine
        engine._build_trading_day_folds.return_value = boundaries

        with pytest.raises(OOSViolation, match="intraday v63"):
            run_cpcv(
                strategy,
                purge_days=5,
                embargo_days=5,
                n_folds=6,
                n_paths=2,
                allow_sacred_holdout=True,
            )


def test_run_cpcv_allow_in_sample_suppresses_violation():
    """allow_in_sample=True must not raise; result.in_sample_override must be True."""
    from scripts.walkforward.cpcv import run_cpcv

    strategy = _make_strategy(trained_through=date(2026, 5, 28), allow_in_sample=True)

    with patch("scripts.walkforward.engine.FoldEngine") as mock_engine_cls, \
         patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        engine = MagicMock()
        mock_engine_cls.return_value = engine
        boundaries = [
            (date(2022, 1, 1), date(2024, 11, 1), date(2024, 11, 1), date(2025, 2, 1)),
            (date(2022, 1, 1), date(2025, 2, 1), date(2025, 2, 1), date(2025, 5, 1)),
            (date(2022, 1, 1), date(2025, 5, 1), date(2025, 5, 1), date(2025, 8, 1)),
            (date(2022, 1, 1), date(2025, 8, 1), date(2025, 8, 1), date(2025, 11, 1)),
            (date(2022, 1, 1), date(2025, 11, 1), date(2025, 11, 1), date(2026, 2, 1)),
            (date(2022, 1, 1), date(2026, 2, 1), date(2026, 2, 1), date(2026, 5, 1)),
        ]
        engine._build_trading_day_folds.return_value = boundaries

        # Folds will be run via strategy.run_fold — mock it to return a trivial result
        fold_result = SimpleNamespace(
            sharpe=2.5, profit_factor=2.0, calmar_ratio=1.0, n_obs=80,
        )
        strategy.run_fold = MagicMock(return_value=fold_result)

        result = run_cpcv(
            strategy,
            purge_days=0,
            embargo_days=0,
            n_folds=6,
            n_paths=2,
            allow_sacred_holdout=True,
        )

    assert result.in_sample_override is True
    assert result.gate_passed() is False


def test_run_cpcv_passes_for_genuine_oos_model():
    """A model trained well before test windows must pass the OOS guard."""
    from scripts.walkforward.cpcv import run_cpcv

    strategy = _make_strategy(trained_through=date(2024, 9, 30))

    with patch("scripts.walkforward.engine.FoldEngine") as mock_engine_cls, \
         patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        engine = MagicMock()
        mock_engine_cls.return_value = engine
        # All test folds start after 2024-09-30
        boundaries = [
            (date(2022, 1, 1), date(2024, 10, 15), date(2024, 10, 15), date(2025, 1, 15)),
            (date(2022, 1, 1), date(2025, 1, 15), date(2025, 1, 15), date(2025, 4, 15)),
            (date(2022, 1, 1), date(2025, 4, 15), date(2025, 4, 15), date(2025, 7, 15)),
            (date(2022, 1, 1), date(2025, 7, 15), date(2025, 7, 15), date(2025, 10, 15)),
            (date(2022, 1, 1), date(2025, 10, 15), date(2025, 10, 15), date(2026, 1, 15)),
            (date(2022, 1, 1), date(2026, 1, 15), date(2026, 1, 15), date(2026, 4, 15)),
        ]
        engine._build_trading_day_folds.return_value = boundaries

        fold_result = SimpleNamespace(
            sharpe=2.5, profit_factor=2.0, calmar_ratio=1.0, n_obs=80,
        )
        strategy.run_fold = MagicMock(return_value=fold_result)

        # Should not raise
        result = run_cpcv(
            strategy,
            purge_days=5,
            embargo_days=5,
            n_folds=6,
            n_paths=2,
            allow_sacred_holdout=True,
        )

    assert result.in_sample_override is False
    assert len(result.path_sharpes) > 0


def test_run_cpcv_raises_when_trained_through_is_none():
    """trained_through=None must raise OOSViolation (can't verify OOS)."""
    from scripts.walkforward.cpcv import run_cpcv

    strategy = _make_strategy(trained_through=None)

    with patch("scripts.walkforward.engine.FoldEngine") as mock_engine_cls, \
         patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        engine = MagicMock()
        mock_engine_cls.return_value = engine
        boundaries = [
            (date(2022, 1, 1), date(2024, 11, 1), date(2024, 11, 1), date(2025, 2, 1)),
            (date(2022, 1, 1), date(2025, 2, 1), date(2025, 2, 1), date(2025, 5, 1)),
            (date(2022, 1, 1), date(2025, 5, 1), date(2025, 5, 1), date(2025, 8, 1)),
            (date(2022, 1, 1), date(2025, 8, 1), date(2025, 8, 1), date(2025, 11, 1)),
            (date(2022, 1, 1), date(2025, 11, 1), date(2025, 11, 1), date(2026, 2, 1)),
            (date(2022, 1, 1), date(2026, 2, 1), date(2026, 2, 1), date(2026, 5, 1)),
        ]
        engine._build_trading_day_folds.return_value = boundaries

        with pytest.raises(OOSViolation, match="trained_through is None"):
            run_cpcv(
                strategy,
                purge_days=5,
                embargo_days=5,
                n_folds=6,
                n_paths=2,
                allow_sacred_holdout=True,
            )
