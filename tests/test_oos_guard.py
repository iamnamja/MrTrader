"""Tests for the OOS guard that prevents in-sample CPCV/WF results."""
from datetime import date
import pytest

from scripts.walkforward.oos_guard import assert_model_oos, OOSViolation


def _fold(te_start, te_end):
    return (date(2020, 1, 1), date(2024, 1, 1), te_start, te_end)


def test_oos_guard_passes_when_all_folds_after_cutoff():
    assert_model_oos(
        trained_through=date(2024, 1, 1),
        fold_boundaries=[_fold(date(2024, 2, 1), date(2024, 5, 1))],
        purge_days=10,
        model_label="test",
    )


def test_oos_guard_raises_when_fold_overlaps_training():
    with pytest.raises(OOSViolation) as exc:
        assert_model_oos(
            trained_through=date(2026, 5, 28),
            fold_boundaries=[_fold(date(2024, 11, 1), date(2025, 2, 1))],
            purge_days=0,
            model_label="intraday v63",
        )
    assert "intraday v63" in str(exc.value)
    assert "2024-11-01" in str(exc.value)


def test_oos_guard_raises_when_trained_through_unknown():
    with pytest.raises(OOSViolation, match="trained_through is None"):
        assert_model_oos(
            trained_through=None,
            fold_boundaries=[_fold(date(2024, 2, 1), date(2024, 5, 1))],
            purge_days=0,
        )


def test_oos_guard_enforces_purge_gap():
    # te_start must be strictly after trained_through + purge_days
    with pytest.raises(OOSViolation):
        assert_model_oos(
            trained_through=date(2024, 1, 1),
            fold_boundaries=[_fold(date(2024, 1, 5), date(2024, 4, 1))],
            purge_days=10,
        )


def test_oos_guard_allow_in_sample_suppresses_raise(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="scripts.walkforward.oos_guard"):
        assert_model_oos(
            trained_through=date(2026, 5, 28),
            fold_boundaries=[_fold(date(2024, 11, 1), date(2025, 2, 1))],
            purge_days=0,
            model_label="intraday v63",
            allow_in_sample=True,
        )
    assert any("OOS guard" in r.message for r in caplog.records)


def test_oos_guard_passes_multiple_folds_all_after_cutoff():
    folds = [
        _fold(date(2025, 1, 1), date(2025, 4, 1)),
        _fold(date(2025, 4, 2), date(2025, 7, 1)),
        _fold(date(2025, 7, 2), date(2025, 10, 1)),
    ]
    assert_model_oos(
        trained_through=date(2024, 12, 1),
        fold_boundaries=folds,
        purge_days=5,
        model_label="test_multi",
    )


def test_oos_guard_raises_on_second_fold_overlap():
    folds = [
        _fold(date(2025, 1, 1), date(2025, 4, 1)),   # OK
        _fold(date(2024, 6, 1), date(2024, 9, 1)),   # OVERLAP
    ]
    with pytest.raises(OOSViolation):
        assert_model_oos(
            trained_through=date(2024, 12, 1),
            fold_boundaries=folds,
            purge_days=0,
        )
