"""Tests for app/ml/regime_score_pit.py — PIT regime score computation."""
import pandas as pd
import numpy as np
import pytest
from datetime import date


def _make_macro_df(n=300, vix_level=15.0, vix3m_level=16.0,
                   spy_trend="up", rsp_trend="neutral", hyg_trend="up"):
    """Build a synthetic macro DataFrame with controllable regime state."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    spy_base = 400.0
    if spy_trend == "up":
        spy = pd.Series(spy_base + np.arange(n) * 0.5, index=idx, name="spy")
    else:
        spy = pd.Series(spy_base - np.arange(n) * 0.5, index=idx, name="spy")

    rsp_base = 150.0
    if rsp_trend == "neutral":
        rsp = pd.Series(rsp_base + np.sin(np.arange(n) * 0.1), index=idx, name="rsp")
    elif rsp_trend == "outperform":
        rsp = pd.Series(rsp_base + np.arange(n) * 0.6, index=idx, name="rsp")
    else:
        rsp = pd.Series(rsp_base - np.arange(n) * 0.3, index=idx, name="rsp")

    hyg_base = 80.0
    if hyg_trend == "up":
        hyg = pd.Series(hyg_base + np.arange(n) * 0.02, index=idx, name="hyg")
    else:
        hyg = pd.Series(hyg_base - np.arange(n) * 0.02, index=idx, name="hyg")

    ief = pd.Series(100.0 + np.sin(np.arange(n) * 0.05), index=idx, name="ief")
    vix = pd.Series(vix_level, index=idx, name="vix")
    vix3m = pd.Series(vix3m_level, index=idx, name="vix3m")

    return pd.DataFrame({"spy": spy, "rsp": rsp, "hyg": hyg, "ief": ief,
                         "vix": vix, "vix3m": vix3m})


def test_compute_pit_regime_series_columns():
    from app.ml.regime_score_pit import compute_pit_regime_series
    macro_df = _make_macro_df()
    result = compute_pit_regime_series(macro_df)
    expected_cols = {
        "spy_above_ma50", "spy_above_ma200", "vix_term_ratio",
        "breadth_20d_change", "credit_20d_change", "composite_score",
    }
    assert expected_cols.issubset(set(result.columns))


def test_composite_score_range():
    from app.ml.regime_score_pit import compute_pit_regime_series
    macro_df = _make_macro_df()
    result = compute_pit_regime_series(macro_df)
    valid = result["composite_score"].dropna()
    assert (valid >= 0.0).all(), "composite_score must be >= 0"
    assert (valid <= 1.0).all(), "composite_score must be <= 1"


def test_favorable_regime_all_bullish():
    """All 5 components bullish → composite = 1.0 for mature rows."""
    from app.ml.regime_score_pit import compute_pit_regime_series
    macro_df = _make_macro_df(n=300, vix_level=14.0, vix3m_level=16.0,
                               spy_trend="up", rsp_trend="outperform", hyg_trend="up")
    result = compute_pit_regime_series(macro_df)
    # Last row should be fully favorable (has enough history for all MAs)
    last = result["composite_score"].dropna().iloc[-1]
    assert last == pytest.approx(1.0, abs=0.01), f"Expected ~1.0, got {last}"


def test_adverse_regime_all_bearish():
    """All 5 components bearish → composite = 0.0 for mature rows."""
    from app.ml.regime_score_pit import compute_pit_regime_series
    macro_df = _make_macro_df(n=300, vix_level=35.0, vix3m_level=30.0,
                               spy_trend="down", rsp_trend="underperform", hyg_trend="down")
    result = compute_pit_regime_series(macro_df)
    last = result["composite_score"].dropna().iloc[-1]
    assert last == pytest.approx(0.0, abs=0.01), f"Expected ~0.0, got {last}"


def test_vix_term_ratio_contango():
    """VIX3M > VIX → contango → vix_term_ratio column stores the raw ratio > 1."""
    from app.ml.regime_score_pit import compute_pit_regime_series
    macro_df = _make_macro_df(vix_level=15.0, vix3m_level=18.0)
    result = compute_pit_regime_series(macro_df)
    # Column stores raw ratio; composite uses (ratio >= 1) as binary
    assert result["vix_term_ratio"].dropna().iloc[-1] > 1.0


def test_vix_term_ratio_backwardation():
    """VIX > VIX3M → backwardation → vix_term_ratio column stores raw ratio < 1."""
    from app.ml.regime_score_pit import compute_pit_regime_series
    macro_df = _make_macro_df(vix_level=30.0, vix3m_level=25.0)
    result = compute_pit_regime_series(macro_df)
    assert result["vix_term_ratio"].dropna().iloc[-1] < 1.0


def test_build_regime_score_map_returns_dict(tmp_path):
    """build_regime_score_map returns {date: float} without error when parquet exists."""
    from pathlib import Path
    from app.ml.regime_score_pit import build_regime_score_map
    macro_df = _make_macro_df(n=300)
    pq_path = tmp_path / "macro_history.parquet"
    macro_df.to_parquet(pq_path)

    score_map = build_regime_score_map(macro_parquet=Path(pq_path))
    assert isinstance(score_map, dict)
    assert len(score_map) > 0
    for k, v in score_map.items():
        assert isinstance(k, date)
        assert 0.0 <= v <= 1.0


def test_build_regime_score_map_missing_parquet(tmp_path):
    """build_regime_score_map returns {} when parquet is missing."""
    from pathlib import Path
    from app.ml.regime_score_pit import build_regime_score_map
    score_map = build_regime_score_map(macro_parquet=Path(tmp_path / "nonexistent.parquet"))
    assert score_map == {}


def test_get_current_regime_score_fails_closed(tmp_path):
    """get_current_regime_score returns (0.0, ...) when parquet not found."""
    from pathlib import Path
    from app.ml.regime_score_pit import get_current_regime_score
    score, components = get_current_regime_score(
        macro_parquet=Path(tmp_path / "nonexistent.parquet")
    )
    assert score == 0.0
    assert isinstance(components, dict)
