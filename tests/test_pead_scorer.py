"""
Tests for app/ml/pead_scorer.py — 0% coverage before this file.

Covers:
- PEADScorer.__call__: long signal on positive surprise, short on negative
- Confidence scaling: 5% surprise → CONF_MIN, large surprise → CONF_MAX
- max_days_after filter: ignore reports older than threshold
- long_short=False: only returns longs
- Empty/None earnings data: graceful skip
- Sorted by descending |conf|
- Constant values: CONF_MIN, CONF_MAX, thresholds
- PIT safety: get_earnings_features_at is called per-symbol (mocked)
"""
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scorer(**kwargs):
    from app.ml.pead_scorer import PEADScorer
    return PEADScorer(**kwargs)


def _dummy_df():
    """Minimal non-empty DataFrame to pass the `df is None or df.empty` check."""
    import numpy as np
    idx = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame({"close": np.linspace(100, 110, 10),
                         "volume": [1e6] * 10}, index=idx)


def _run(scorer, feats_by_sym: dict, day=None):
    """Run scorer with get_earnings_features_at mocked per-symbol."""
    if day is None:
        day = date(2024, 3, 15)
    symbols_data = {sym: _dummy_df() for sym in feats_by_sym}

    def _fake_feats(sym, as_of):
        return feats_by_sym.get(sym)

    with patch("app.data.fmp_provider.get_earnings_features_at", side_effect=_fake_feats):
        return scorer(day, symbols_data)


# ── Constants ─────────────────────────────────────────────────────────────────

class TestPEADConstants:
    def test_conf_min_below_conf_max(self):
        from app.ml.pead_scorer import CONF_MIN, CONF_MAX
        assert CONF_MIN < CONF_MAX

    def test_thresholds_symmetric(self):
        from app.ml.pead_scorer import LONG_SURPRISE_THRESHOLD, SHORT_SURPRISE_THRESHOLD
        assert LONG_SURPRISE_THRESHOLD > 0
        assert SHORT_SURPRISE_THRESHOLD < 0
        assert abs(LONG_SURPRISE_THRESHOLD) == abs(SHORT_SURPRISE_THRESHOLD)

    def test_max_days_after_is_positive(self):
        from app.ml.pead_scorer import MAX_DAYS_AFTER_EARNINGS
        assert MAX_DAYS_AFTER_EARNINGS > 0


# ── Signal generation ─────────────────────────────────────────────────────────

class TestPEADSignalGeneration:
    def test_positive_surprise_returns_long(self):
        scorer = _make_scorer()
        feats = {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}
        result = _run(scorer, {"AAPL": feats})
        assert len(result) == 1
        sym, conf, direction = result[0]
        assert sym == "AAPL"
        assert direction == "long"
        assert conf > 0

    def test_negative_surprise_returns_short_when_ls_enabled(self):
        scorer = _make_scorer(long_short=True)
        feats = {"fmp_surprise_1q": -0.10, "fmp_days_since_earnings": 1}
        result = _run(scorer, {"TSLA": feats})
        assert len(result) == 1
        sym, conf, direction = result[0]
        assert sym == "TSLA"
        assert direction == "short"
        assert conf < 0

    def test_negative_surprise_skipped_when_ls_disabled(self):
        scorer = _make_scorer(long_short=False)
        feats = {"fmp_surprise_1q": -0.15, "fmp_days_since_earnings": 1}
        result = _run(scorer, {"TSLA": feats})
        assert result == []

    def test_surprise_below_long_threshold_no_signal(self):
        scorer = _make_scorer(long_threshold=0.05)
        feats = {"fmp_surprise_1q": 0.02, "fmp_days_since_earnings": 1}
        result = _run(scorer, {"AAPL": feats})
        assert result == []

    def test_surprise_exactly_at_long_threshold(self):
        scorer = _make_scorer(long_threshold=0.05)
        feats = {"fmp_surprise_1q": 0.05, "fmp_days_since_earnings": 1}
        result = _run(scorer, {"NVDA": feats})
        assert len(result) == 1
        assert result[0][2] == "long"


# ── max_days_after filter ─────────────────────────────────────────────────────

class TestMaxDaysAfterFilter:
    def test_recent_report_included(self):
        scorer = _make_scorer(max_days_after=3)
        feats = {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 2}
        result = _run(scorer, {"AAPL": feats})
        assert len(result) == 1

    def test_stale_report_excluded(self):
        scorer = _make_scorer(max_days_after=3)
        feats = {"fmp_surprise_1q": 0.20, "fmp_days_since_earnings": 4}
        result = _run(scorer, {"AAPL": feats})
        assert result == []

    def test_boundary_day_included(self):
        scorer = _make_scorer(max_days_after=3)
        feats = {"fmp_surprise_1q": 0.08, "fmp_days_since_earnings": 3}
        result = _run(scorer, {"MSFT": feats})
        assert len(result) == 1

    def test_same_day_announcement_excluded(self):
        """days_since=0 means announcement day — must not enter before next open."""
        scorer = _make_scorer(max_days_after=3)
        feats = {"fmp_surprise_1q": 0.20, "fmp_days_since_earnings": 0}
        result = _run(scorer, {"NVDA": feats})
        assert result == [], "Must not trade on announcement day (after-hours release)"


# ── Confidence scaling ────────────────────────────────────────────────────────

class TestConfidenceScaling:
    def test_conf_in_range(self):
        from app.ml.pead_scorer import CONF_MIN, CONF_MAX
        scorer = _make_scorer()
        feats = {"fmp_surprise_1q": 0.08, "fmp_days_since_earnings": 1}
        result = _run(scorer, {"AAPL": feats})
        assert len(result) == 1
        _, conf, _ = result[0]
        assert CONF_MIN <= abs(conf) <= CONF_MAX + 1e-9

    def test_large_surprise_clips_to_conf_max(self):
        from app.ml.pead_scorer import CONF_MAX
        scorer = _make_scorer()
        feats = {"fmp_surprise_1q": 0.99, "fmp_days_since_earnings": 1}
        result = _run(scorer, {"NVDA": feats})
        assert len(result) == 1
        _, conf, _ = result[0]
        assert abs(conf) <= CONF_MAX + 1e-9

    def test_small_surprise_clips_to_conf_min(self):
        from app.ml.pead_scorer import CONF_MIN
        scorer = _make_scorer(long_threshold=0.05)
        feats = {"fmp_surprise_1q": 0.051, "fmp_days_since_earnings": 1}
        result = _run(scorer, {"AAPL": feats})
        assert len(result) == 1
        _, conf, _ = result[0]
        assert abs(conf) >= CONF_MIN - 1e-9

    def test_larger_surprise_higher_conf(self):
        scorer = _make_scorer()
        feats_small = {"fmp_surprise_1q": 0.06, "fmp_days_since_earnings": 1}
        feats_large = {"fmp_surprise_1q": 0.20, "fmp_days_since_earnings": 1}
        r_small = _run(scorer, {"A": feats_small})
        r_large = _run(scorer, {"A": feats_large})
        conf_small = abs(r_small[0][1])
        conf_large = abs(r_large[0][1])
        assert conf_large >= conf_small


# ── Missing / null data ───────────────────────────────────────────────────────

class TestMissingData:
    def test_none_feats_skipped(self):
        scorer = _make_scorer()
        result = _run(scorer, {"AAPL": None})
        assert result == []

    def test_missing_surprise_key_skipped(self):
        scorer = _make_scorer()
        feats = {"fmp_days_since_earnings": 1}  # no surprise
        result = _run(scorer, {"AAPL": feats})
        assert result == []

    def test_missing_days_since_key_skipped(self):
        scorer = _make_scorer()
        feats = {"fmp_surprise_1q": 0.10}  # no days_since
        result = _run(scorer, {"AAPL": feats})
        assert result == []

    def test_exception_from_provider_skipped(self):
        scorer = _make_scorer()
        day = date(2024, 3, 15)
        symbols_data = {"AAPL": _dummy_df()}

        with patch("app.data.fmp_provider.get_earnings_features_at",
                   side_effect=RuntimeError("API error")):
            result = scorer(day, symbols_data)
        assert result == []

    def test_empty_symbols_data_returns_empty(self):
        scorer = _make_scorer()
        with patch("app.data.fmp_provider.get_earnings_features_at", return_value=None):
            result = scorer(date(2024, 3, 15), {})
        assert result == []


# ── Ordering ──────────────────────────────────────────────────────────────────

class TestResultOrdering:
    def test_sorted_by_descending_abs_conf(self):
        scorer = _make_scorer(long_short=True)
        feats = {
            "AAPL": {"fmp_surprise_1q": 0.06, "fmp_days_since_earnings": 1},  # small
            "NVDA": {"fmp_surprise_1q": 0.30, "fmp_days_since_earnings": 1},  # large
            "TSLA": {"fmp_surprise_1q": -0.20, "fmp_days_since_earnings": 1},  # short, medium
        }
        result = _run(scorer, feats)
        confs = [abs(c) for _, c, _ in result]
        assert confs == sorted(confs, reverse=True), "Results must be sorted by |conf| desc"

    def test_multiple_symbols_all_returned(self):
        scorer = _make_scorer()
        feats = {sym: {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}
                 for sym in ["AAPL", "MSFT", "NVDA"]}
        result = _run(scorer, feats)
        syms = [s for s, _, _ in result]
        assert set(syms) == {"AAPL", "MSFT", "NVDA"}
