"""
Tests for regime.event_regime_sharpes() — the EVENT-LEVEL regime instrument
(Alpha-v6 P0 / blueprint X6) that will RETIRE the paper-only event-sparsity
waiver for event strategies in Phase 3 (H1). In PR1 it is REPORT-ONLY.

Coverage (synthetic events, NO network):
  - correct per-regime cross-event Sharpe (mean/std ddof=1), UN-annualized
    (explicitly no sqrt(252) factor);
  - obs_counts populated with RAW per-regime EVENT counts BEFORE the
    REGIME_MIN_OBS filter, and cleared on entry;
  - buckets below REGIME_MIN_OBS dropped from the result but present in
    obs_counts (sparsity-vs-data-bug disambiguation preserved);
  - {} on empty events / empty regime_map; zero-variance bucket -> 0.0;
  - datetime entry dates map to the date-keyed regime map;
  - unmapped dates are skipped.

Plus the run_cpcv WIRING (mock-strategy pattern from
test_cpcv_is_trained_coverage.py) — REPORT-ONLY, NOT a gate input:
  - daily regime buckets empty + event-level buckets present -> the event-level
    min is surfaced on CPCVResult.event_worst_regime_sharpe, but
    worst_regime_sharpe stays None, source stays "daily", and the FIX-2
    event-sparsity waiver STILL fires (retiring it is H1's Phase-3 job);
  - daily buckets present -> daily wins, byte-for-byte unchanged
    (source stays "daily"; event field not set);
  - neither present -> the existing FIX-2 waiver disambiguation is preserved.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ml.retrain_config import REGIME_MIN_OBS
from scripts.walkforward.gates import FoldResult
from scripts.walkforward.regime import event_regime_sharpes

D0 = date(2024, 1, 2)


def _days(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _events(days, rets):
    return list(zip(days, rets))


def _expected_sharpe(rets) -> float:
    arr = np.array(rets, dtype=float)
    return float(np.mean(arr)) / float(np.std(arr, ddof=1))


@pytest.fixture()
def three_regime_setup():
    """BULL: MIN_OBS+10 events; BEAR: exactly MIN_OBS; NEUTRAL: MIN_OBS-1
    (below the floor -> dropped from the result, kept in obs_counts)."""
    n_bull = REGIME_MIN_OBS + 10
    n_bear = REGIME_MIN_OBS
    n_neut = REGIME_MIN_OBS - 1
    bull_days = _days(D0, n_bull)
    bear_days = _days(D0 + timedelta(days=400), n_bear)
    neut_days = _days(D0 + timedelta(days=800), n_neut)
    regime_map = {d: "BULL" for d in bull_days}
    regime_map.update({d: "BEAR" for d in bear_days})
    regime_map.update({d: "NEUTRAL" for d in neut_days})
    rng = np.random.default_rng(1303)
    bull_rets = list(rng.normal(0.01, 0.02, n_bull))     # healthy positive edge
    bear_rets = list(rng.normal(-0.005, 0.03, n_bear))   # losing regime
    neut_rets = list(rng.normal(0.0, 0.02, n_neut))      # sub-floor bucket
    events = (_events(bull_days, bull_rets)
              + _events(bear_days, bear_rets)
              + _events(neut_days, neut_rets))
    return events, regime_map, {
        "BULL": bull_rets, "BEAR": bear_rets, "NEUTRAL": neut_rets,
    }


def test_per_regime_unannualized_sharpe(three_regime_setup):
    events, regime_map, rets = three_regime_setup
    result = event_regime_sharpes(events, regime_map)
    assert set(result) == {"BULL", "BEAR"}  # NEUTRAL below the floor
    assert result["BULL"] == pytest.approx(_expected_sharpe(rets["BULL"]))
    assert result["BEAR"] == pytest.approx(_expected_sharpe(rets["BEAR"]))
    assert result["BEAR"] < 0 < result["BULL"]


def test_no_annualization_factor(three_regime_setup):
    """Per-event Sharpe must NOT carry sqrt(252): a +0.5-per-event edge would
    read ~+7.9 if annualized — assert the magnitude is the raw mean/std."""
    events, regime_map, rets = three_regime_setup
    result = event_regime_sharpes(events, regime_map)
    raw = _expected_sharpe(rets["BULL"])
    assert result["BULL"] == pytest.approx(raw)
    assert abs(result["BULL"] - raw * np.sqrt(252)) > 1.0  # NOT annualized
    assert abs(result["BULL"]) < 3.0  # per-event units stay small


def test_obs_counts_raw_pre_filter_and_cleared(three_regime_setup):
    events, regime_map, _ = three_regime_setup
    obs = {"STALE": 999}  # must be cleared on entry
    result = event_regime_sharpes(events, regime_map, obs_counts=obs)
    assert "STALE" not in obs
    # RAW counts BEFORE the floor: NEUTRAL is present here even though it was
    # dropped from the result — the caller can tell sparsity from a data-bug.
    assert obs == {
        "BULL": REGIME_MIN_OBS + 10,
        "BEAR": REGIME_MIN_OBS,
        "NEUTRAL": REGIME_MIN_OBS - 1,
    }
    assert "NEUTRAL" not in result


def test_all_buckets_below_floor_gives_empty_result_with_counts():
    n = max(REGIME_MIN_OBS - 2, 1)
    days = _days(D0, n)
    regime_map = {d: "BULL" for d in days}
    obs: dict = {}
    result = event_regime_sharpes(
        _events(days, [0.01] * n), regime_map, obs_counts=obs)
    assert result == {}            # event-sparsity ...
    assert obs == {"BULL": n}      # ... but NOT a data-bug


def test_empty_inputs_return_empty():
    obs = {"STALE": 1}
    assert event_regime_sharpes([], {D0: "BULL"}, obs_counts=obs) == {}
    assert obs == {}
    assert event_regime_sharpes(_events(_days(D0, 5), [0.01] * 5), {}) == {}
    assert event_regime_sharpes(_events(_days(D0, 5), [0.01] * 5), None) == {}


def test_zero_variance_bucket_reports_zero():
    n = REGIME_MIN_OBS + 5
    days = _days(D0, n)
    regime_map = {d: "BULL" for d in days}
    result = event_regime_sharpes(_events(days, [0.01] * n), regime_map)
    assert result == {"BULL": 0.0}  # matches the daily-path convention


def test_datetime_entry_dates_map_to_date_keys():
    n = REGIME_MIN_OBS
    days = _days(D0, n)
    regime_map = {d: "BEAR" for d in days}
    dt_events = [(datetime(d.year, d.month, d.day, 16, 0), 0.01 * ((i % 3) - 1))
                 for i, d in enumerate(days)]
    result = event_regime_sharpes(dt_events, regime_map)
    assert set(result) == {"BEAR"}


def test_unmapped_dates_are_skipped():
    n = REGIME_MIN_OBS + 4
    days = _days(D0, n)
    regime_map = {d: "BULL" for d in days}
    rng = np.random.default_rng(7)
    rets = list(rng.normal(0.005, 0.01, n))
    # Events on dates outside the regime map must not contribute anywhere.
    stray = _events(_days(D0 + timedelta(days=4000), 10), [9.9] * 10)
    obs: dict = {}
    result = event_regime_sharpes(
        _events(days, rets) + stray, regime_map, obs_counts=obs)
    assert obs == {"BULL": n}
    assert result["BULL"] == pytest.approx(_expected_sharpe(rets))


# ─────────────────────────── run_cpcv wiring: the event-level fallback (X6)

def _make_fold_factory(*, regime_sharpes=None, regime_obs_counts=None,
                       event_regime_sharpes=None, event_regime_obs_counts=None):
    def _fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        return FoldResult(
            fold=fold_idx, train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=10, win_rate=0.5, sharpe=1.0, max_drawdown=0.05,
            total_return=0.10, stop_exit_rate=0.2,
            regime_sharpes=dict(regime_sharpes or {}),
            regime_obs_counts=dict(regime_obs_counts or {}),
            event_regime_sharpes=dict(event_regime_sharpes or {}),
            event_regime_obs_counts=dict(event_regime_obs_counts or {}),
        )
    return _fold


def _run_cpcv_with(fold_factory):
    s = MagicMock()
    s.model_type = "event"
    s.version = 1
    s.allow_in_sample = False
    s.per_fold_retrain = False
    s.model = MagicMock()
    s.model.trained_through = date(2018, 1, 1)
    s.all_days_sorted = []
    s.is_trained = False  # rules-based: full fold coverage
    s.run_fold.side_effect = fold_factory
    with patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        from scripts.walkforward.cpcv import run_cpcv
        return run_cpcv(strategy=s, purge_days=10, embargo_days=10,
                        n_folds=4, n_paths=2, total_years=5,
                        allow_sacred_holdout=True)


def test_cpcv_event_level_is_report_only_does_not_gate():
    """Daily buckets all below REGIME_MIN_OBS (the event-sparsity case) with
    event-level Sharpes present: the event-level min is surfaced REPORT-ONLY on
    event_worst_regime_sharpe, but it does NOT feed the gate. worst_regime_sharpe
    stays None, the FIX-2 paper waiver STILL fires (retiring it is H1's
    pre-registered Phase-3 consequence, not PR1's), and the per-event number is
    never compared against the annualized-daily MIN_WORST_REGIME_SHARPE floor."""
    r = _run_cpcv_with(_make_fold_factory(
        regime_sharpes={},                       # daily: starved out
        regime_obs_counts={"BULL": 3},           # ... but obs WERE seen
        event_regime_sharpes={"BULL": 0.40, "BEAR": -0.10},
        event_regime_obs_counts={"BULL": 50, "BEAR": 25},
    ))
    # Report-only: event-level min surfaced, gate input untouched.
    assert r.event_worst_regime_sharpe == pytest.approx(-0.10)
    assert r.worst_regime_sharpe is None
    assert r.worst_regime_sharpe_source == "daily"
    # The waiver path is byte-for-byte unchanged — event data does not retire it.
    assert r.regime_insufficient_obs is True
    _, _, regime_ok, regime_waived = r._significance_backstops_ok(tier="paper")
    assert regime_ok is True and regime_waived is True        # paper-only waiver
    assert "requires_human_review" in r.significance_gate_detail(tier="paper")


def test_cpcv_daily_path_wins_when_present():
    """When the daily buckets populate, they win and the event-level numbers
    are ignored — the existing daily path is byte-for-byte unchanged."""
    r = _run_cpcv_with(_make_fold_factory(
        regime_sharpes={"BULL": 1.0, "BEAR": -0.2},
        regime_obs_counts={"BULL": 60, "BEAR": 40},
        event_regime_sharpes={"BULL": 9.9},      # must NOT leak into the gate
        event_regime_obs_counts={"BULL": 50},
    ))
    assert r.worst_regime_sharpe == pytest.approx(-0.2)
    assert r.worst_regime_sharpe_source == "daily"


def test_cpcv_waiver_disambiguation_preserved_without_event_data():
    """No event-level data (e.g. a non-event strategy): the FIX-2
    event-sparsity waiver behavior is exactly as before."""
    r = _run_cpcv_with(_make_fold_factory(
        regime_sharpes={},
        regime_obs_counts={"BULL": 3},           # sparsity, not a data-bug
    ))
    assert r.worst_regime_sharpe is None
    assert r.regime_insufficient_obs is True
    _, _, regime_ok, regime_waived = r._significance_backstops_ok(tier="paper")
    assert regime_ok is True and regime_waived is True  # paper-only waiver
