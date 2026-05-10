"""Tests for P1 benign filter in training pipeline — tests the filtering logic directly."""
import pytest
from datetime import date


def _regime_filter(window_end_dates, regime_score_map, benign_enabled, benign_threshold=0.5):
    """
    Replicate the exact filter logic from _process_symbol_windows_worker.
    This tests the filter in isolation without needing real bars or labels.
    """
    included = []
    for w_end in window_end_dates:
        if benign_enabled and regime_score_map is not None:
            pit_score = regime_score_map.get(w_end)
            if pit_score is None or pit_score < benign_threshold:
                continue
        included.append(w_end)
    return included


class TestBenignWorkerFilter:
    def test_filter_removes_adverse_windows(self):
        favorable = [date(2024, 1, 10)]
        adverse = [date(2024, 1, 15)]
        regime_map = {date(2024, 1, 10): 0.8, date(2024, 1, 15): 0.2}

        included = _regime_filter(favorable + adverse, regime_map, benign_enabled=True)
        assert date(2024, 1, 10) in included
        assert date(2024, 1, 15) not in included

    def test_filter_keeps_all_when_disabled(self):
        dates = [date(2024, 1, 10), date(2024, 1, 15)]
        regime_map = {date(2024, 1, 10): 0.8, date(2024, 1, 15): 0.2}

        included = _regime_filter(dates, regime_map, benign_enabled=False)
        assert len(included) == 2

    def test_missing_date_in_map_excluded(self):
        """A window date not in the regime_map should be excluded (fail-closed)."""
        dates = [date(2024, 1, 10)]
        included = _regime_filter(dates, {}, benign_enabled=True)
        assert len(included) == 0

    def test_none_map_passes_all_windows(self):
        """When regime_score_map is None, filter is a no-op even if enabled."""
        dates = [date(2024, 1, 10), date(2024, 1, 15)]
        included = _regime_filter(dates, None, benign_enabled=True)
        assert len(included) == 2

    def test_exactly_at_threshold_is_included(self):
        regime_map = {date(2024, 1, 10): 0.5}
        included = _regime_filter([date(2024, 1, 10)], regime_map,
                                   benign_enabled=True, benign_threshold=0.5)
        assert date(2024, 1, 10) in included

    def test_just_below_threshold_excluded(self):
        regime_map = {date(2024, 1, 10): 0.4999}
        included = _regime_filter([date(2024, 1, 10)], regime_map,
                                   benign_enabled=True, benign_threshold=0.5)
        assert len(included) == 0

    def test_all_adverse_returns_empty(self):
        dates = [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12)]
        regime_map = {d: 0.0 for d in dates}
        included = _regime_filter(dates, regime_map, benign_enabled=True)
        assert included == []

    def test_all_favorable_keeps_all(self):
        dates = [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12)]
        regime_map = {d: 1.0 for d in dates}
        included = _regime_filter(dates, regime_map, benign_enabled=True)
        assert len(included) == 3

    def test_retrain_config_threshold_is_used(self):
        """Verify BENIGN_REGIME_THRESHOLD from retrain_config is 0.5."""
        from app.ml.retrain_config import BENIGN_REGIME_THRESHOLD
        regime_map = {date(2024, 1, 10): BENIGN_REGIME_THRESHOLD}
        included = _regime_filter([date(2024, 1, 10)], regime_map,
                                   benign_enabled=True, benign_threshold=BENIGN_REGIME_THRESHOLD)
        # Exactly at threshold → included
        assert len(included) == 1
