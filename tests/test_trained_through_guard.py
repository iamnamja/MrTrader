"""Regression locks for the trained_through save-guard (KL-10).

A model whose ``trained_through`` is None cannot be OOS-gate-evaluated; shipping
one (as happened with intraday_v63 / swing_v224) produces silently in-sample
walk-forward/CPCV results. These tests verify:

  1. save() hard-fails when trained_through is None (REQUIRE_TRAINED_THROUGH=True).
  2. The flag can be disabled for explicit diagnostic-only saves.
  3. save() succeeds once trained_through is set.
  4. LambdaRankModel now carries a trained_through attribute (was missing — v224 bug).
  5. trained_through survives pickle round-trips for both LambdaRank and PortfolioSelector.
"""

from datetime import date

import pytest

xgboost = pytest.importorskip("xgboost")


def _make_portfolio_selector():
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel(model_type="xgboost")
    m.is_trained = True  # pretend trained so save() is exercised
    return m


def test_save_rejects_none_trained_through(tmp_path):
    """save() must raise ValueError when trained_through is None (flag default True)."""
    from app.ml import retrain_config

    m = _make_portfolio_selector()
    assert m.trained_through is None
    assert retrain_config.REQUIRE_TRAINED_THROUGH is True

    with pytest.raises(ValueError, match="trained_through is None"):
        m.save(str(tmp_path), version=1, model_name="test")


def test_save_allows_none_when_flag_off(tmp_path, monkeypatch):
    """With REQUIRE_TRAINED_THROUGH=False, a None-cutoff diagnostic save succeeds."""
    monkeypatch.setattr("app.ml.retrain_config.REQUIRE_TRAINED_THROUGH", False)

    m = _make_portfolio_selector()
    assert m.trained_through is None

    path = m.save(str(tmp_path), version=1, model_name="test")
    from pathlib import Path
    assert Path(path).exists()


def test_save_succeeds_with_trained_through_set(tmp_path):
    """A model with a real cutoff saves without raising and writes the artifact."""
    from pathlib import Path

    m = _make_portfolio_selector()
    m.trained_through = date(2025, 1, 1)

    path = m.save(str(tmp_path), version=1, model_name="test")
    assert Path(path).exists()


def test_lambdarank_has_trained_through_attr():
    """LambdaRankModel must declare trained_through (defaulting None) — the v224 fix."""
    pytest.importorskip("lightgbm")
    from app.ml.model import LambdaRankModel

    m = LambdaRankModel()
    assert hasattr(m, "trained_through")
    assert m.trained_through is None


def test_lambdarank_trained_through_survives_pickle(tmp_path):
    """Setting the cutoff then save→load on a fresh LambdaRankModel preserves it."""
    pytest.importorskip("lightgbm")
    from app.ml.model import LambdaRankModel

    m = LambdaRankModel()
    m.is_trained = True
    m.trained_through = date(2024, 6, 30)
    m.save(str(tmp_path), version=7, model_name="swing")

    loaded = LambdaRankModel()
    loaded.load(str(tmp_path), version=7, model_name="swing")
    assert loaded.trained_through == date(2024, 6, 30)


def test_portfolio_selector_meta_persists_trained_through(tmp_path):
    """Regression lock for the v63 bug: trained_through round-trips via .load()."""
    m = _make_portfolio_selector()
    m.trained_through = date(2025, 3, 15)
    m.save(str(tmp_path), version=2, model_name="intraday")

    from app.ml.model import PortfolioSelectorModel
    loaded = PortfolioSelectorModel(model_type="xgboost")
    loaded.load(str(tmp_path), version=2, model_name="intraday")
    assert loaded.trained_through == date(2025, 3, 15)
