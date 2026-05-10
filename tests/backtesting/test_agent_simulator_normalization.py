"""
WF-A1 — AgentSimulator normalization + VIX routing tests.

Verifies:
  1. TS norm state is used when present (v185+ model)
  2. cs_normalize fallback for legacy models (no _ts_norm_state)
  3. predict_with_vix routes to high-VIX sibling when VIX above threshold
  4. predict_with_vix routes to primary when VIX below threshold
  5. window_id passed to TS transform is day.toordinal() (per-day, not today)
  6. PIT regime_score from regime_score_history replaces hardcoded 0.5
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch, call
import numpy as np
import pandas as pd
import pytest

from app.backtesting.agent_simulator import AgentSimulator


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_bars(n: int = 250) -> pd.DataFrame:
    """Synthetic OHLCV bars — enough for EMA-200 and all feature computations."""
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.full(n, 1_000_000),
    }, index=idx)


def _make_model(*, has_ts_norm: bool, has_sibling: bool = False,
                sibling_threshold: float = 20.0) -> MagicMock:
    """Return a mock PortfolioSelectorModel with controllable attributes."""
    model = MagicMock()
    model.is_trained = True
    model.feature_names = ["close", "volume"]  # minimal — matches _make_bars keys
    model.predict.return_value = (np.array([1]), np.array([0.6]))
    model.predict_with_vix.return_value = (np.array([1]), np.array([0.6]))

    if has_ts_norm:
        ts_state = MagicMock()
        model._ts_norm_state = ts_state
    else:
        model._ts_norm_state = None

    if has_sibling:
        sibling = MagicMock()
        sibling.is_trained = True
        sibling.predict.return_value = (np.array([1]), np.array([0.55]))
        model._highvix_sibling = sibling
        model._regime_split_threshold = sibling_threshold
    else:
        model._highvix_sibling = None
        model._regime_split_threshold = None

    return model


def _make_vix_series(value: float, day: date) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(day)])
    return pd.Series([value], index=idx)


# ── Test 1: TS norm called when state present ─────────────────────────────────

def test_pm_score_uses_ts_norm_when_state_present():
    model = _make_model(has_ts_norm=True)
    sim = AgentSimulator(model=model)

    bars = _make_bars()
    day = bars.index[-1].date()
    symbols_data = {"AAPL": bars}

    def _fake_transform(X, symbols, window_ids, state):
        return X, np.ones(len(symbols), dtype=bool)

    with patch("app.ml.ts_normalize.transform", side_effect=_fake_transform), \
         patch("app.backtesting.agent_simulator.cs_normalize") as mock_cs:
        sim._pm_score(day, symbols_data)

    # cs_normalize must NOT have been called — TS norm handled it
    mock_cs.assert_not_called()


# ── Test 2: cs_normalize fallback for legacy models ──────────────────────────

def test_pm_score_falls_back_to_cs_normalize_for_legacy_model():
    model = _make_model(has_ts_norm=False)
    sim = AgentSimulator(model=model)

    bars = _make_bars()
    day = bars.index[-1].date()
    symbols_data = {"AAPL": bars}

    with patch("app.backtesting.agent_simulator.cs_normalize",
               return_value=np.zeros((1, 2))) as mock_cs:
        sim._pm_score(day, symbols_data)

    mock_cs.assert_called_once()


# ── Test 3: predict_with_vix routes to high-VIX sibling above threshold ──────

def test_pm_score_routes_to_highvix_sibling_when_vix_high():
    model = _make_model(has_ts_norm=False, has_sibling=True, sibling_threshold=20.0)
    sim = AgentSimulator(model=model)

    bars = _make_bars()
    day = bars.index[-1].date()
    symbols_data = {"AAPL": bars}
    vix = _make_vix_series(25.0, day)  # above threshold

    sim._pm_score(day, symbols_data, vix_history=vix)

    # predict_with_vix must have been called (model routes internally)
    model.predict_with_vix.assert_called_once()
    called_vix = model.predict_with_vix.call_args[1].get(
        "vix_level", model.predict_with_vix.call_args[0][1]
        if len(model.predict_with_vix.call_args[0]) > 1 else None
    )
    assert called_vix == pytest.approx(25.0)


# ── Test 4: predict_with_vix called (primary model) when VIX below threshold ─

def test_pm_score_calls_predict_with_vix_when_vix_low():
    model = _make_model(has_ts_norm=False, has_sibling=True, sibling_threshold=20.0)
    sim = AgentSimulator(model=model)

    bars = _make_bars()
    day = bars.index[-1].date()
    symbols_data = {"AAPL": bars}
    vix = _make_vix_series(15.0, day)  # below threshold

    sim._pm_score(day, symbols_data, vix_history=vix)

    model.predict_with_vix.assert_called_once()


# ── Test 5: window_id is day.toordinal(), not date.today() ───────────────────

def test_pm_score_window_id_is_day_ordinal():
    model = _make_model(has_ts_norm=True)
    sim = AgentSimulator(model=model)

    bars = _make_bars()
    day = bars.index[-1].date()
    symbols_data = {"AAPL": bars}

    captured_window_ids: list = []

    def _fake_transform(X, symbols, window_ids, state):
        captured_window_ids.extend(window_ids)
        return X, np.ones(len(symbols), dtype=bool)

    with patch("app.backtesting.agent_simulator.AgentSimulator._normalize_for_inference",
               wraps=sim._normalize_for_inference):
        with patch("app.ml.ts_normalize.transform", side_effect=_fake_transform):
            sim._pm_score(day, symbols_data)

    assert len(captured_window_ids) >= 1
    assert captured_window_ids[0] == day.toordinal()


# ── Test 6: PIT regime_score replaces hardcoded 0.5 ─────────────────────────

def test_pm_score_uses_pit_regime_score_from_history():
    model = _make_model(has_ts_norm=False)
    day = date(2024, 3, 15)
    sim = AgentSimulator(model=model, regime_score_history={day: 0.85})

    bars = _make_bars()
    bars.index = pd.date_range(
        end=pd.Timestamp(day), periods=len(bars), freq="B"
    )
    symbols_data = {"AAPL": bars}

    captured_scores: list = []

    original_engineer = None

    def _patched_engineer(sym, bars_df, fetch_fundamentals=False,
                          as_of_date=None, regime_score=0.5, **kwargs):
        captured_scores.append(regime_score)
        return {"close": 100.0, "volume": 1e6}

    fe_mock = MagicMock()
    fe_mock.engineer_features.side_effect = _patched_engineer
    sim._feature_engineer = fe_mock

    sim._pm_score(day, symbols_data)

    assert len(captured_scores) >= 1
    assert captured_scores[0] == pytest.approx(0.85)


def test_pm_score_regime_score_defaults_to_neutral_when_date_missing():
    model = _make_model(has_ts_norm=False)
    day = date(2024, 3, 15)
    sim = AgentSimulator(model=model, regime_score_history={})  # empty

    bars = _make_bars()
    bars.index = pd.date_range(end=pd.Timestamp(day), periods=len(bars), freq="B")
    symbols_data = {"AAPL": bars}

    captured_scores: list = []

    def _patched_engineer(sym, bars_df, fetch_fundamentals=False,
                          as_of_date=None, regime_score=0.5, **kwargs):
        captured_scores.append(regime_score)
        return {"close": 100.0, "volume": 1e6}

    fe_mock = MagicMock()
    fe_mock.engineer_features.side_effect = _patched_engineer
    sim._feature_engineer = fe_mock

    sim._pm_score(day, symbols_data)

    assert len(captured_scores) >= 1
    assert captured_scores[0] == pytest.approx(0.5)
