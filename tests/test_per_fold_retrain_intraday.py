"""
Phase 2 — intraday per-fold retraining (true out-of-sample WF/CPCV).

Mirrors tests/test_per_fold_retrain_swing.py for the intraday seam:
  1. IntradayFoldRetrainer._seed_for determinism.
  2. TrainWindowCache dedup with the intraday retrainer.
  3. IntradayStrategy.run_fold per-fold uses _fold_model (not self.model).
  4. Per-fold OOS guard fires on a bad boundary AND uses a TRADING-day purge.
  5. build_train_matrix_for_window returns a NON-EMPTY X on synthetic intraday
     data (the regression lock — the real builder spine, no mock).
  6. No-leak: every training row's day_ordinal <= train_end.toordinal().
  7. is_true_walkforward True in per-fold intraday CPCV mode.

Tests are hermetic where possible; the non-empty / no-leak tests exercise the
REAL matrix builder on tiny synthetic 5-min + daily data (no network, no model
fit unless xgboost is available).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward.retrainers import IntradayFoldRetrainer, TrainWindowCache
from scripts.walkforward.strategies.intraday import IntradayStrategy
from scripts.walkforward.oos_guard import OOSViolation


# ── 1. Seed determinism ──────────────────────────────────────────────────────

def test_seed_deterministic():
    r = IntradayFoldRetrainer(base_config={}, seed_base=42)
    ts, te1 = date(2023, 1, 2), date(2023, 6, 1)
    te2 = date(2023, 9, 1)
    assert r._seed_for(ts, te1) == r._seed_for(ts, te1)   # same window → same seed
    assert r._seed_for(ts, te1) != r._seed_for(ts, te2)   # different tr_end → different


# ── 2. TrainWindowCache dedups for the intraday retrainer ────────────────────

class _CountingRetrainer:
    def __init__(self):
        self.calls = 0

    def train_for_window(self, symbols_data, spy_data, daily_data, spy_daily_data,
                         tr_start, tr_end):
        self.calls += 1
        return SimpleNamespace(trained_through=tr_end, tag=(tr_start, tr_end))


def test_train_window_cache_dedups_intraday():
    stub = _CountingRetrainer()
    cache = TrainWindowCache(stub)
    ts, te = date(2023, 1, 2), date(2023, 6, 1)
    te2 = date(2023, 9, 1)

    # Intraday fit_inputs = (symbols_data, spy_data, daily_data, spy_daily_data)
    m1 = cache.get(ts, te, {}, None, {}, None)
    m1b = cache.get(ts, te, {}, None, {}, None)
    assert m1 is m1b
    assert stub.calls == 1

    m2 = cache.get(ts, te2, {}, None, {}, None)
    assert m2 is not m1
    assert stub.calls == 2


# ── IntradayStrategy run_fold helpers ────────────────────────────────────────

def _make_strategy(per_fold: bool):
    model = SimpleNamespace(feature_names=["f1"], trained_through=date(2022, 1, 1),
                            tag="FROZEN")
    s = IntradayStrategy(model=model, version=1, symbols=["AAPL"])
    s.per_fold_retrain = per_fold
    s.symbols_data = {}
    s.spy_data = None
    s.spy_daily_data = None
    s._daily_data = {}                       # already cached → _ensure_daily_data is a no-op
    s.all_days_sorted = [date(2023, 1, 2) + timedelta(days=i) for i in range(300)]
    s._global_regime_map = {}
    s._purge_days = 2
    return s


class _CaptureSimulator:
    """Stub IntradayAgentSimulator capturing the model it was constructed with."""
    last_model = None

    def __init__(self, model=None, **kwargs):
        _CaptureSimulator.last_model = model
        self.model = model

    def run(self, *a, **k):
        return SimpleNamespace(
            exit_breakdown={}, total_trades=0, win_rate=0.0, sharpe_ratio=0.0,
            max_drawdown_pct=0.0, total_return_pct=0.0, trades=[], equity_curve=[],
            profit_factor=0.0, avg_capital_deployed_pct=0.0,
            deployment_adjusted_sharpe=0.0, low_deployment_warning=False,
        )


def _run_fold(strategy):
    tr_start, tr_end = date(2023, 1, 2), date(2023, 6, 1)
    te_start, te_end = date(2023, 7, 1), date(2023, 9, 1)
    with patch("app.backtesting.intraday_agent_simulator.IntradayAgentSimulator",
               _CaptureSimulator), \
         patch("app.data.universe_history.members_at", return_value=set()), \
         patch("scripts.walkforward.regime.compute_regime_sharpes", return_value={}):
        return strategy.run_fold(1, 1, tr_start, tr_end, te_start, te_end)


# ── 3. Frozen mode uses self.model ───────────────────────────────────────────

def test_strategy_frozen_mode_uses_self_model():
    s = _make_strategy(per_fold=False)
    _CaptureSimulator.last_model = None
    _run_fold(s)
    assert _CaptureSimulator.last_model is s.model
    assert getattr(_CaptureSimulator.last_model, "tag", None) == "FROZEN"


# ── 4. Per-fold mode uses the fold model + trading-day OOS guard runs ─────────

def test_strategy_per_fold_uses_fold_model_and_trading_day_purge():
    s = _make_strategy(per_fold=True)
    tr_end = date(2023, 6, 1)
    sentinel = SimpleNamespace(feature_names=["f1"], trained_through=tr_end, tag="FOLD")

    class _StubCache:
        def __init__(self):
            self.got = None

        def get(self, ts, te, *inputs):
            self.got = (ts, te)
            return sentinel

    s._train_cache = _StubCache()
    _CaptureSimulator.last_model = None
    with patch("scripts.walkforward.oos_guard.assert_model_oos") as m_oos:
        _run_fold(s)
    assert _CaptureSimulator.last_model is sentinel       # fold model, not frozen
    assert _CaptureSimulator.last_model is not s.model
    m_oos.assert_called_once()
    kwargs = m_oos.call_args.kwargs
    assert kwargs["trained_through"] == tr_end
    # Intraday uses a TRADING-day purge — the trading_day_set must be passed.
    assert kwargs["trading_day_set"] is not None
    assert kwargs["trading_day_set"] == set(s.all_days_sorted)


# ── 5. Per-fold OOS guard fires on overlap ────────────────────────────────────

def test_per_fold_oos_guard_fires():
    s = _make_strategy(per_fold=True)
    # trained_through AFTER te_start → must raise OOSViolation inside run_fold.
    bad = SimpleNamespace(feature_names=["f1"], trained_through=date(2023, 8, 1))

    class _StubCache:
        def get(self, ts, te, *inputs):
            return bad

    s._train_cache = _StubCache()
    with pytest.raises(OOSViolation):
        _run_fold(s)


# ── 6. is_true_walkforward flag plumbing (CPCV) ──────────────────────────────

def test_is_true_walkforward_flag_cpcv_intraday():
    from scripts.walkforward.cpcv import CPCVResult
    assert CPCVResult(model_type="intraday", n_folds=4, n_paths=2).is_true_walkforward is False
    assert CPCVResult(model_type="intraday", n_folds=4, n_paths=2,
                      is_true_walkforward=True).is_true_walkforward is True


# ── 7/8. The matrix builder: non-empty + no-leak on synthetic intraday data ──

def _synthetic_intraday_data(start: date, n_days: int, symbols, bars_per_day: int = 40):
    """Build 5-min intraday bars (bars_per_day per business day) for several symbols."""
    bdays = pd.bdate_range(start=start, periods=n_days)
    rng = np.random.default_rng(7)
    data = {}
    for s in symbols:
        idx_list = []
        rows = []
        base = 100.0 + rng.normal(0, 5)
        for d in bdays:
            # session bars at 5-min cadence starting 09:30
            day_open = datetime.combine(d.date(), time(9, 30))
            px = base + rng.normal(0, 1)
            for b in range(bars_per_day):
                ts = day_open + timedelta(minutes=5 * b)
                drift = rng.normal(0, 0.15)
                close = px + drift
                high = max(px, close) + abs(rng.normal(0, 0.1))
                low = min(px, close) - abs(rng.normal(0, 0.1))
                rows.append({"open": px, "high": high, "low": low, "close": close,
                             "volume": float(rng.integers(10_000, 50_000))})
                idx_list.append(ts)
                px = close
            base = px
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx_list, name="timestamp"))
        data[s] = df
    return data


def _synthetic_daily_data(start: date, n_days: int, symbols):
    """Daily OHLCV bars (one extra year of warmup) for vol-percentile / 52w features."""
    idx = pd.bdate_range(start=start - timedelta(days=400), periods=n_days + 280)
    rng = np.random.default_rng(11)
    data = {}
    for s in symbols:
        prices = 100 + np.cumsum(rng.normal(0, 1, len(idx)))
        df = pd.DataFrame({
            "open": prices, "high": prices + 1, "low": prices - 1,
            "close": prices, "volume": rng.integers(1_000_000, 5_000_000, len(idx)),
        }, index=idx)
        data[s] = df
    return data


def test_build_train_matrix_non_empty_and_no_leak():
    """Regression lock + no-leak for the intraday per-fold matrix builder.

    Runs the REAL builder (no mock) on tiny synthetic 5-min + daily data:
      - matrix is NON-EMPTY with consistent y and real feature columns,
      - NO training row uses a day strictly after train_end (same-day label).
    """
    from app.ml.intraday_training import IntradayModelTrainer

    symbols = ["AAA", "BBB", "CCC", "DDD"]
    # 60 business days of 5-min bars — > MIN_DAYS (20) within the train window.
    sdata = _synthetic_intraday_data(date(2023, 1, 2), n_days=60, symbols=symbols)
    ddata = _synthetic_daily_data(date(2023, 1, 2), n_days=60, symbols=symbols)
    spy_daily = _synthetic_daily_data(date(2023, 1, 2), n_days=60, symbols=["SPY"])["SPY"]

    all_5m_days = sorted({d.date() for df in sdata.values()
                          for d in pd.DatetimeIndex(df.index)})
    train_start = all_5m_days[0]
    # Leave ~10 trading days of FUTURE 5-min bars after train_end so a leak
    # would be observable if the builder failed to clamp train_days.
    train_end = all_5m_days[-10]

    trainer = IntradayModelTrainer(provider="alpaca")
    trainer._allow_sacred_holdout = False

    X, y, fnames, raw = trainer.build_train_matrix_for_window(
        sdata, None, ddata, spy_daily, train_start, train_end,
    )

    X = np.asarray(X)
    assert len(X) > 0, "build_train_matrix_for_window returned an EMPTY matrix"
    assert X.ndim == 2 and X.shape[1] > 0, f"matrix has no feature columns: shape {X.shape}"
    assert len(y) == len(X), f"y/X length mismatch: {len(y)} vs {len(X)}"
    assert len(fnames) == X.shape[1], (
        f"feature_names ({len(fnames)}) != matrix columns ({X.shape[1]})"
    )
    assert np.isfinite(X).all(), "matrix contains non-finite values"

    # No-leak: every training row's day_ordinal must be <= train_end.toordinal().
    assert raw is not None and len(raw) == len(X)
    max_day_ord = int(raw[:, 0].max())
    assert max_day_ord <= train_end.toordinal(), (
        f"LEAK: training row uses day_ordinal {max_day_ord} "
        f"(> train_end {train_end} = {train_end.toordinal()})"
    )


def test_build_train_matrix_span_mismatch_raises():
    """C14-1: a fold window that does NOT overlap any loaded 5-min day must raise
    a clear span-mismatch error — NOT silently return an empty matrix.

    This is the regression lock for the 2nd per-fold empty-matrix bug: the fold
    boundary axis (all_days_sorted) was wider than the loaded bars, so early
    folds got a train window before any 5-min bar existed → 'no training samples'.
    """
    from app.ml.intraday_training import IntradayModelTrainer

    symbols = ["AAA", "BBB", "CCC", "DDD"]
    # 5-min bars only from 2025-01-02 onward...
    sdata = _synthetic_intraday_data(date(2025, 1, 2), n_days=40, symbols=symbols)
    trainer = IntradayModelTrainer(provider="alpaca")
    trainer._allow_sacred_holdout = False

    # ...but ask for a 2024 window (mirrors all_days_sorted starting a year early).
    with pytest.raises(RuntimeError, match="does not overlap any loaded 5-min"):
        trainer.build_train_matrix_for_window(
            sdata, None, {}, None, date(2024, 1, 2), date(2024, 6, 25),
        )


def test_fetch_data_clamps_all_days_to_window(monkeypatch):
    """C14-1: fetch_data must clamp all_days_sorted AND symbols_data to the
    requested [start, end], independent of what load_many returns. Otherwise the
    fold-boundary day-axis can precede the matrix-builder's bar coverage."""
    import app.data.intraday_cache as ic
    import yfinance as yf

    # load_many returns bars spanning a FULL year before `start` (simulating a
    # cache/provider that ignored the start filter).
    wide_idx = pd.DatetimeIndex(
        pd.date_range("2024-01-02", "2025-06-01", freq="h", tz="UTC"), name="timestamp"
    )
    wide_df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=wide_idx,
    )
    monkeypatch.setattr(ic, "available_symbols", lambda: ["AAA"])
    monkeypatch.setattr(ic, "load_many", lambda syms, start, end: {"AAA": wide_df})
    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())

    strat = IntradayStrategy(model=SimpleNamespace(trained_through=date(2000, 1, 1)),
                             version=1, symbols=["AAA"])
    strat.fetch_data(date(2025, 1, 1), date(2025, 6, 1))

    assert strat.all_days_sorted, "all_days_sorted unexpectedly empty"
    assert strat.all_days_sorted[0] >= date(2025, 1, 1), (
        f"all_days_sorted not clamped: starts {strat.all_days_sorted[0]}"
    )
    assert strat.all_days_sorted[-1] <= date(2025, 6, 1)
    # symbols_data bars must be clamped too (lock-step with the day-axis).
    bar_dates = pd.DatetimeIndex(strat.symbols_data["AAA"].index).date
    assert bar_dates.min() >= date(2025, 1, 1)


def test_fit_in_memory_trains_on_synthetic_matrix():
    """fit_in_memory produces a usable model (ensemble + feature_names) without
    save/version/DB. Skips if xgboost is unavailable."""
    pytest.importorskip("xgboost")
    from app.ml.intraday_training import IntradayModelTrainer

    symbols = ["AAA", "BBB", "CCC", "DDD"]
    sdata = _synthetic_intraday_data(date(2023, 1, 2), n_days=60, symbols=symbols)
    ddata = _synthetic_daily_data(date(2023, 1, 2), n_days=60, symbols=symbols)
    spy_daily = _synthetic_daily_data(date(2023, 1, 2), n_days=60, symbols=["SPY"])["SPY"]
    all_5m_days = sorted({d.date() for df in sdata.values()
                          for d in pd.DatetimeIndex(df.index)})

    trainer = IntradayModelTrainer(provider="alpaca")
    trainer._allow_sacred_holdout = False
    X, y, fnames, raw = trainer.build_train_matrix_for_window(
        sdata, None, ddata, spy_daily, all_5m_days[0], all_5m_days[-5],
    )
    assert len(X) > 0
    model = trainer.fit_in_memory(X, y, fnames, raw, seed=99)
    assert model.feature_names == fnames
    assert getattr(model, "ensemble_models", None)
    assert len(model.ensemble_models) == 3
    # Deterministic seeds: same seed → identical ensemble random_states.
    model2 = trainer.fit_in_memory(X, y, fnames, raw, seed=99)
    seeds1 = [m.get_params()["random_state"] for m in model.ensemble_models]
    seeds2 = [m.get_params()["random_state"] for m in model2.ensemble_models]
    assert seeds1 == seeds2
