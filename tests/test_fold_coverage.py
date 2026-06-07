"""Alpha-v4 P0 — fold-coverage report (scripts/walkforward/coverage.py).

Pure unit tests on build_fold_coverage: a bull-only fold set is flagged; a diverse
fold set passes; missing regime map degrades gracefully; the report serializes and
reconstructs for rendering.
"""
from __future__ import annotations

from datetime import date

from scripts.walkforward.coverage import (
    build_fold_coverage, CoverageReport, _is_stress_label, _majority_regime,
)


def _window(y, m=6):
    """A ~2-week test window in year y."""
    return (date(y, m, 1), date(y, m, 14))


def _coarse3_map(windows, label):
    """Label every calendar day in the given windows with `label`."""
    m = {}
    for (s, e) in windows:
        d = s
        while d <= e:
            m[d] = label
            d = date.fromordinal(d.toordinal() + 1)
    return m


def test_stress_label_detection():
    assert _is_stress_label("BEAR")
    assert _is_stress_label("4DN")      # legacy16 top VIX quartile
    assert _is_stress_label("2DP")      # downtrend
    assert not _is_stress_label("BULL")
    assert not _is_stress_label("1UP")
    assert not _is_stress_label("UNK")


def test_majority_regime_picks_dominant_label():
    win = (date(2022, 3, 1), date(2022, 3, 10))
    rm = {}
    d = win[0]
    while d <= win[1]:
        rm[d] = "BEAR" if d.day <= 7 else "NEUTRAL"
        d = date.fromordinal(d.toordinal() + 1)
    assert _majority_regime(win[0], win[1], rm) == "BEAR"
    assert _majority_regime(win[0], win[1], None) == "UNK"


def test_bull_only_is_flagged_low_coverage():
    # 2 folds, same year, all BULL → fails year-span, regime-diversity, and stress.
    windows = [_window(2024, 3), _window(2024, 9)]
    rm = _coarse3_map(windows, "BULL")
    cov = build_fold_coverage(windows, rm)
    assert cov.coverage_ok is False
    assert cov.has_stress_fold is False
    assert any("year" in w for w in cov.warnings)
    assert any("stress" in w for w in cov.warnings)


def test_diverse_fold_set_passes():
    # 3 distinct years, BULL/NEUTRAL/BEAR present → coverage_ok.
    w_bull = [_window(2021)]
    w_neutral = [_window(2022)]
    w_bear = [_window(2020)]
    rm = {}
    rm.update(_coarse3_map(w_bull, "BULL"))
    rm.update(_coarse3_map(w_neutral, "NEUTRAL"))
    rm.update(_coarse3_map(w_bear, "BEAR"))
    cov = build_fold_coverage(w_bull + w_neutral + w_bear, rm)
    assert cov.coverage_ok is True
    assert cov.has_stress_fold is True
    assert cov.n_distinct_years == 3
    assert cov.n_distinct_regimes == 3
    assert cov.warnings == []


def test_no_regime_map_degrades_gracefully():
    windows = [_window(2019), _window(2021), _window(2023)]
    cov = build_fold_coverage(windows, None)
    # Years are fine (3 distinct) but regime coverage is unknown → not ok.
    assert cov.coverage_ok is False
    assert any("regime map" in w for w in cov.warnings)
    assert cov.by_regime.get("UNK") == 3


def test_report_roundtrips_for_render():
    windows = [_window(2020), _window(2021), _window(2022)]
    rm = {}
    rm.update(_coarse3_map([windows[0]], "BEAR"))
    rm.update(_coarse3_map([windows[1]], "BULL"))
    rm.update(_coarse3_map([windows[2]], "NEUTRAL"))
    cov = build_fold_coverage(windows, rm)
    d = cov.to_dict()
    # to_dict() keys must reconstruct CoverageReport (used by CPCVResult.print()).
    rebuilt = CoverageReport(**d)
    assert "FOLD COVERAGE" in rebuilt.render()
    assert d["coverage_ok"] is True
