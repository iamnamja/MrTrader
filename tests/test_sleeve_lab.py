"""Tests for the Sleeve Lab (Alpha-v7 F0) — the uniform sleeve -> Ruler-v2 pipeline.

Offline by construction: every `evaluate_sleeve` call passes an explicit `regime_map`,
so `load_regime_map` (network) is never hit; CPCV on a precomputed return series fetches
no data (SeriesReturnStrategy.fetch_data is a no-op).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward import sleeve_lab as sl
from scripts.walkforward.sleeve_lab import (
    Sleeve, SleeveReport, evaluate_sleeve, assemble_book, format_sleeve_report,
    register_sleeve, build_sleeve, list_sleeves, SLEEVE_REGISTRY,
)


# ── fixtures ────────────────────────────────────────────────────────────────────
def _series(mu: float, sd: float, *, n: int = 1300, seed: int = 0,
            start: str = "2015-01-02") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(rng.normal(mu, sd, size=n), index=idx)


def _regime_map(index: pd.DatetimeIndex, label: str = "BULL") -> dict:
    """A single-regime map covering every date in `index` (avoids the network)."""
    return {ts.date(): label for ts in index}


# ── Sleeve validation ─────────────────────────────────────────────────────────────
def test_sleeve_accepts_valid_returns():
    s = _series(0.0005, 0.008)
    sleeve = Sleeve("good", "risk_premium", s)
    assert sleeve.component_type == "risk_premium"
    assert isinstance(sleeve.returns.index, pd.DatetimeIndex)
    assert sleeve.regime_waived_by_type is True


def test_sleeve_rejects_bad_component_type():
    with pytest.raises(ValueError):
        Sleeve("bad", "not_a_type", _series(0.0, 0.01))


def test_sleeve_rejects_overlay_until_overlay_path_exists():
    """An overlay modifies the book (not an additive stream); the additive Track-A/B
    model would emit a confidently-wrong verdict, so overlay is fail-loud for now."""
    with pytest.raises(ValueError):
        Sleeve("governor", "overlay", _series(0.0, 0.01))


def test_sleeve_rejects_empty_label():
    with pytest.raises(ValueError):
        Sleeve("", "alpha", _series(0.0, 0.01))


def test_sleeve_rejects_non_series_returns():
    with pytest.raises(TypeError):
        Sleeve("x", "alpha", [0.1, 0.2, 0.3])


def test_sleeve_dedups_and_sorts_index():
    idx = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-02", "2020-01-04"])
    raw = pd.Series([0.1, 0.2, 0.3, 0.4], index=idx)
    sleeve = Sleeve("dup", "alpha", raw)
    assert sleeve.returns.index.is_monotonic_increasing
    assert not sleeve.returns.index.has_duplicates
    assert len(sleeve.returns) == 3  # one dup dropped, keep="first"


def test_sleeve_drops_nan_and_rejects_all_nan():
    s = _series(0.0005, 0.008).copy()
    s.iloc[5] = np.nan
    sleeve = Sleeve("withnan", "alpha", s)
    assert sleeve.returns.isna().sum() == 0
    with pytest.raises(ValueError):
        Sleeve("allnan", "alpha", pd.Series([np.nan, np.nan],
               index=pd.bdate_range("2020-01-01", periods=2)))


def test_alpha_not_regime_waived_by_type():
    assert Sleeve("a", "alpha", _series(0.0, 0.01)).regime_waived_by_type is False


# ── Registry ──────────────────────────────────────────────────────────────────────
def test_registry_register_build_list():
    name = "_unit_test_demo_sleeve"
    SLEEVE_REGISTRY.pop(name, None)

    @register_sleeve(name)
    def _build(mu=0.0004):
        return Sleeve(name, "diversifier", _series(mu, 0.008))

    try:
        assert name in list_sleeves()
        sleeve = build_sleeve(name, mu=0.0005)
        assert isinstance(sleeve, Sleeve)
        assert sleeve.label == name
    finally:
        SLEEVE_REGISTRY.pop(name, None)


def test_registry_rejects_duplicate():
    name = "_unit_test_dup_sleeve"
    SLEEVE_REGISTRY.pop(name, None)

    @register_sleeve(name)
    def _b1():
        return Sleeve(name, "alpha", _series(0.0, 0.01))

    try:
        with pytest.raises(ValueError):
            @register_sleeve(name)
            def _b2():
                return Sleeve(name, "alpha", _series(0.0, 0.01))
    finally:
        SLEEVE_REGISTRY.pop(name, None)


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        build_sleeve("__no_such_sleeve__")


# ── End-to-end evaluation ──────────────────────────────────────────────────────────
def test_evaluate_strong_sleeve_passes_paper():
    """A clean ~SR 1.2 risk_premium clears the PAPER plausibility + light HAC floor."""
    s = _series(0.0008, 0.008, seed=1)
    sleeve = Sleeve("strong_rp", "risk_premium", s)
    rep = evaluate_sleeve(sleeve, regime_map=_regime_map(s.index))
    assert isinstance(rep, SleeveReport)
    assert rep.paper_passed, f"expected PAPER pass; failed={rep.paper_failed}"
    assert 0.30 <= rep.point_sr <= 3.0
    assert rep.hac_p_one_sided < 0.05


def test_evaluate_null_sleeve_fails_paper():
    """A zero-edge null fails the PAPER significance floor (point_SR and/or HAC-p)."""
    s = _series(0.0, 0.008, seed=2)
    sleeve = Sleeve("null", "risk_premium", s)
    rep = evaluate_sleeve(sleeve, regime_map=_regime_map(s.index))
    assert not rep.paper_passed
    assert ("point_sr_floor" in rep.paper_failed) or ("hac_significance" in rep.paper_failed)


def test_capital_fails_closed_without_live_paper():
    """CAPITAL is structurally unreachable on a backtest alone (no live_paper)."""
    s = _series(0.0008, 0.008, seed=1)
    sleeve = Sleeve("strong_rp", "risk_premium", s)
    rep = evaluate_sleeve(sleeve, regime_map=_regime_map(s.index))
    assert not rep.capital_passed
    assert "live_paper_present" in rep.capital_failed


def test_track_b_runs_when_base_supplied():
    cand = _series(0.0005, 0.008, seed=3)
    base = _series(0.0004, 0.010, seed=99)  # the "live book"
    sleeve = Sleeve("cand", "diversifier", cand)
    rep = evaluate_sleeve(sleeve, base_book_returns=base,
                          regime_map=_regime_map(cand.index), n_boot=200)
    assert rep.track_b is not None
    assert hasattr(rep.track_b, "passed")
    assert "TRACK-B" in rep.verdict


def test_no_track_b_when_base_absent():
    s = _series(0.0005, 0.008, seed=4)
    rep = evaluate_sleeve(Sleeve("x", "risk_premium", s),
                          regime_map=_regime_map(s.index))
    assert rep.track_b is None
    assert "TRACK-B" not in rep.verdict


# ── Book assembly + reporting ──────────────────────────────────────────────────────
def test_assemble_book_combines_streams():
    a = Sleeve("a", "risk_premium", _series(0.0005, 0.008, seed=5))
    b = Sleeve("b", "diversifier", _series(0.0004, 0.009, seed=6))
    book = assemble_book([a, b])
    summary = book.summary()
    assert summary["n_days"] > 0
    assert np.isfinite(summary["sharpe"])


def test_assemble_book_requires_common_dates():
    a = Sleeve("a", "alpha", _series(0.0005, 0.008, n=300, start="2015-01-02"))
    b = Sleeve("b", "alpha", _series(0.0005, 0.008, n=300, start="2022-01-03"))
    with pytest.raises(ValueError):
        assemble_book([a, b])


def test_report_is_ascii_and_serializable():
    s = _series(0.0008, 0.008, seed=1)
    rep = evaluate_sleeve(Sleeve("strong_rp", "risk_premium", s),
                          regime_map=_regime_map(s.index))
    txt = format_sleeve_report(rep)
    txt.encode("ascii")  # raises if any non-ASCII glyph crept in (cp1252 console safe)
    d = rep.to_dict()
    assert d["label"] == "strong_rp"
    assert "verdict" in d
    assert set(["paper_passed", "capital_passed", "point_sr"]).issubset(d)


def test_failed_criteria_excludes_informational():
    detail = {
        "point_sr_floor": (0.1, False),
        "hac_significance": (0.2, False),
        "hac_t_report": (1.0, True),
        "pf_report": (1.0, True),
        "requires_human_review": ("x", False),  # informational -> excluded
    }
    failed = sl.failed_criteria(detail)
    assert "point_sr_floor" in failed
    assert "hac_significance" in failed
    assert "requires_human_review" not in failed
    assert "pf_report" not in failed
