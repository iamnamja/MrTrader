"""
Phase 1 — swing per-fold retraining (true out-of-sample WF/CPCV).

Covers the new seam: TrainWindowCache dedup, SwingFoldRetrainer determinism,
SwingStrategy.run_fold frozen-vs-per-fold model selection + per-fold OOS guard,
the is_true_walkforward flag plumbing, the REQUIRE_TRUE_WF_FOR_PROMOTION gate,
and the no-leak property of build_train_matrix_for_window.

Tests are hermetic — the trainer/simulator are stubbed everywhere except the
no-leak test, which exercises the real matrix-builder spine logic with a tiny
synthetic dataset (no model fit, no network).
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward.retrainers import SwingFoldRetrainer, TrainWindowCache
from scripts.walkforward.strategies.swing import SwingStrategy
from scripts.walkforward.oos_guard import OOSViolation


# ── 1. Seed determinism ──────────────────────────────────────────────────────

def test_seed_deterministic():
    r = SwingFoldRetrainer(base_config={}, seed_base=42)
    ts, te1 = date(2020, 1, 1), date(2021, 1, 1)
    te2 = date(2021, 6, 1)
    assert r._seed_for(ts, te1) == r._seed_for(ts, te1)  # same window → same seed
    assert r._seed_for(ts, te1) != r._seed_for(ts, te2)  # different tr_end → different


# ── 2. TrainWindowCache dedups ───────────────────────────────────────────────

class _CountingRetrainer:
    def __init__(self):
        self.calls = 0

    def train_for_window(self, symbols_data, spy_prices, regime_map, tr_start, tr_end):
        self.calls += 1
        return SimpleNamespace(trained_through=tr_end, tag=(tr_start, tr_end))


def test_train_window_cache_dedups():
    stub = _CountingRetrainer()
    cache = TrainWindowCache(stub)
    ts, te = date(2020, 1, 1), date(2021, 1, 1)
    te2 = date(2021, 6, 1)

    m1 = cache.get(ts, te, {}, None, None)
    m1b = cache.get(ts, te, {}, None, None)
    assert m1 is m1b               # identity — same object returned
    assert stub.calls == 1         # underlying train called once for this window

    m2 = cache.get(ts, te2, {}, None, None)
    assert m2 is not m1            # different window → different object
    assert stub.calls == 2         # one more call for the new window


# ── SwingStrategy run_fold helpers ───────────────────────────────────────────

def _make_strategy(per_fold: bool):
    model = SimpleNamespace(feature_names=["f1"], trained_through=date(2019, 1, 1),
                            tag="FROZEN")
    s = SwingStrategy(model=model, version=1, symbols=["AAPL"],
                      feature_cache_disable=True)
    s.per_fold_retrain = per_fold
    s.symbols_data = {}
    s.spy_prices = None
    s._global_regime_map = {}
    s._purge_days = 10
    return s


class _CaptureSimulator:
    """Stub AgentSimulator capturing the model it was constructed with."""
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
    tr_start, tr_end = date(2020, 1, 1), date(2021, 1, 1)
    te_start, te_end = date(2021, 4, 1), date(2021, 7, 1)
    with patch("app.backtesting.agent_simulator.AgentSimulator", _CaptureSimulator), \
         patch("app.data.universe_history.pit_union", return_value=set()), \
         patch("app.data.universe_history.historical_trade_symbols", return_value=[]), \
         patch("scripts.walkforward.regime.compute_regime_sharpes", return_value={}):
        return strategy.run_fold(1, 1, tr_start, tr_end, te_start, te_end)


# ── 3. Frozen mode uses self.model ───────────────────────────────────────────

def test_strategy_frozen_mode_uses_self_model():
    s = _make_strategy(per_fold=False)
    _CaptureSimulator.last_model = None
    _run_fold(s)
    assert _CaptureSimulator.last_model is s.model
    assert getattr(_CaptureSimulator.last_model, "tag", None) == "FROZEN"


# ── 4. Per-fold mode uses the fold model + OOS guard runs ─────────────────────

def test_strategy_per_fold_uses_fold_model():
    s = _make_strategy(per_fold=True)
    tr_end = date(2021, 1, 1)
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
    assert _CaptureSimulator.last_model is sentinel       # fold model, not self.model
    assert _CaptureSimulator.last_model is not s.model
    m_oos.assert_called_once()                            # per-fold OOS guard invoked
    # Guard called with the fold model's trained_through.
    assert m_oos.call_args.kwargs["trained_through"] == tr_end


# ── 5. Per-fold OOS guard fires on overlap ────────────────────────────────────

def test_per_fold_oos_guard_fires():
    s = _make_strategy(per_fold=True)
    # trained_through AFTER te_start → must raise OOSViolation inside run_fold.
    bad = SimpleNamespace(feature_names=["f1"], trained_through=date(2021, 6, 1))

    class _StubCache:
        def get(self, ts, te, *inputs):
            return bad

    s._train_cache = _StubCache()
    s._purge_days = 10
    with pytest.raises(OOSViolation):
        _run_fold(s)


# ── 6. is_true_walkforward flag plumbing ──────────────────────────────────────

def test_is_true_walkforward_flag_cpcv():
    from scripts.walkforward.cpcv import CPCVResult
    assert CPCVResult(model_type="swing", n_folds=6, n_paths=2).is_true_walkforward is False
    assert CPCVResult(model_type="swing", n_folds=6, n_paths=2,
                      is_true_walkforward=True).is_true_walkforward is True


def test_is_true_walkforward_flag_report():
    from scripts.walkforward.gates import WalkForwardReport
    assert WalkForwardReport(model_type="swing").is_true_walkforward is False
    assert WalkForwardReport(model_type="swing",
                             is_true_walkforward=True).is_true_walkforward is True


# ── 7. Frozen cannot promote when REQUIRE_TRUE_WF_FOR_PROMOTION=True ──────────

def _passing_wf_report(is_true_wf: bool):
    from scripts.walkforward.gates import WalkForwardReport, FoldResult
    r = WalkForwardReport(model_type="swing", is_true_walkforward=is_true_wf)
    for _ in range(3):
        r.folds.append(FoldResult(
            fold=1, train_start=date(2022, 1, 1), train_end=date(2023, 1, 1),
            test_start=date(2023, 1, 1), test_end=date(2023, 6, 1),
            trades=100, win_rate=0.60, sharpe=2.5, max_drawdown=0.05,
            total_return=0.20, stop_exit_rate=0.1, model_version=1,
            profit_factor=1.8, calmar_ratio=1.5, n_obs=120,
            regime_sharpes={"BULL": 1.0, "BEAR": 0.5, "NEUTRAL": 0.8},
        ))
    return r


def _passing_cpcv_result(is_true_wf: bool):
    from scripts.walkforward.cpcv import CPCVResult
    r = CPCVResult(model_type="swing", n_folds=6, n_paths=2, is_true_walkforward=is_true_wf)
    r.path_sharpes = [2.0, 2.2, 2.4, 2.1, 2.3, 2.0]
    r.path_profit_factors = [1.8] * 6
    r.path_calmars = [1.5] * 6
    r.path_n_obs = [120] * 6
    r.worst_regime_sharpe = 0.5
    return r


def test_frozen_cannot_promote_when_required_report():
    from app.ml import retrain_config

    # Default (flag False): frozen report passes its metric gates.
    assert _passing_wf_report(is_true_wf=False).gate_passed() is True

    with patch.object(retrain_config, "REQUIRE_TRUE_WF_FOR_PROMOTION", True):
        assert _passing_wf_report(is_true_wf=False).gate_passed() is False  # frozen blocked
        assert _passing_wf_report(is_true_wf=True).gate_passed() is True    # per-fold allowed


def test_cpcv_swing_gate_ok_reflects_failure():
    """Display-bug lock: a None result, a zero-path result, or a failed-gate
    result must all report NOT-OK so the overall `passed` flag goes False
    (previously _run_cpcv_swing's result was ignored and a failed/empty CPCV
    still printed 'ALL GATES PASSED')."""
    from scripts.walkforward_tier3 import _cpcv_swing_gate_ok
    from scripts.walkforward.cpcv import CPCVResult, N_TRIALS_TESTED

    _args = SimpleNamespace(dsr_n=N_TRIALS_TESTED, paper_gate=False)

    # None (no model / skipped) → not ok.
    assert _cpcv_swing_gate_ok(None, _args) is False

    # Zero surviving paths (every fold skipped — the exact per-fold-empty-matrix
    # failure mode) → not ok even though gate_passed would be evaluated.
    empty = CPCVResult(model_type="swing", n_folds=6, n_paths=2, is_true_walkforward=True)
    assert not empty.path_sharpes
    assert _cpcv_swing_gate_ok(empty, _args) is False

    # A genuinely passing per-fold result → ok.
    good = _passing_cpcv_result(is_true_wf=True)
    assert _cpcv_swing_gate_ok(good, _args) is True


def test_frozen_cannot_promote_when_required_cpcv():
    from app.ml import retrain_config

    assert _passing_cpcv_result(is_true_wf=False).gate_passed() is True

    with patch.object(retrain_config, "REQUIRE_TRUE_WF_FOR_PROMOTION", True):
        assert _passing_cpcv_result(is_true_wf=False).gate_passed() is False
        assert _passing_cpcv_result(is_true_wf=True).gate_passed() is True


# ── 8. THE killer test: build_train_matrix_for_window has no label leak ───────

def _synthetic_symbols_data(start: date, n_days: int, symbols):
    """Build trading-day (business-day) bars for several symbols spanning n_days."""
    idx = pd.bdate_range(start=start, periods=n_days)
    data = {}
    rng = np.random.default_rng(0)
    for s in symbols:
        prices = 100 + np.cumsum(rng.normal(0, 1, len(idx)))
        df = pd.DataFrame({
            "open": prices, "high": prices + 1, "low": prices - 1,
            "close": prices, "volume": rng.integers(1_000_000, 5_000_000, len(idx)),
        }, index=idx)
        data[s] = df
    return data


def test_build_train_matrix_no_leak():
    """Slicing to [train_start, train_end] must clamp the date spine so that NO
    training window's FORWARD_DAYS label looks past train_end. We spy on
    _windows_to_matrix to read the exact (all_dates, window_starts) the builder
    constructs and verify max label date <= train_end."""
    from app.ml.training import ModelTrainer, WINDOW_DAYS

    symbols = ["AAA", "BBB", "CCC"]
    # Span well past train_end so a leak would be possible if not clamped.
    sdata = _synthetic_symbols_data(date(2018, 1, 1), n_days=600, symbols=symbols)
    # spy_prices is a Series of close prices (mirrors SwingStrategy.spy_prices).
    spy = _synthetic_symbols_data(date(2018, 1, 1), n_days=600, symbols=["SPY"])["SPY"]["close"]

    train_start = date(2018, 1, 1)
    train_end = date(2019, 6, 30)  # leave ~7 months of FUTURE bars after train_end

    trainer = ModelTrainer(model_type="lambdarank", label_scheme="lambdarank",
                           use_feature_store=False, n_workers=1)

    captured = {}

    def _spy_windows(symbols_data, all_dates, window_starts, *a, **k):
        captured["all_dates"] = list(all_dates)
        captured["window_starts"] = list(window_starts)
        # Return an empty matrix — we only care about the spine, not the fit.
        trainer._last_feature_names = ["f1"]
        return [], [], []

    with patch.object(trainer, "_windows_to_matrix", side_effect=_spy_windows):
        trainer.build_train_matrix_for_window(
            sdata, train_start, train_end, spy_prices=spy, regime_score_map={},
        )

    all_dates = captured["all_dates"]
    window_starts = captured["window_starts"]
    assert all_dates, "builder produced no date spine"
    assert window_starts, "builder produced no windows"

    # Property 1: the clamped spine never exceeds train_end.
    assert max(all_dates) <= train_end, (
        f"date spine max {max(all_dates)} exceeds train_end {train_end} — clamp failed"
    )

    # The builder labels at LABEL_HORIZON_DAYS (it sets the module FORWARD_DAYS to
    # that horizon only WHILE it runs, then restores it). Use the production
    # horizon for the no-leak arithmetic — reading the module constant after the
    # call would see the restored default and understate the label reach.
    from app.ml.retrain_config import LABEL_HORIZON_DAYS as _LHD
    fwd = int(_LHD)

    # Property 2 (the real no-leak guarantee): for EVERY training window the
    # worker would keep (w_end_idx + fwd < len(all_dates)), the label
    # date all_dates[w_end_idx + fwd] must not exceed train_end.
    max_label_date = None
    for w_start in window_starts:
        w_end_idx = w_start + WINDOW_DAYS
        future_idx = w_end_idx + fwd
        if future_idx >= len(all_dates):
            continue  # worker drops this window — no label used
        label_date = all_dates[future_idx]
        max_label_date = label_date if max_label_date is None else max(max_label_date, label_date)
        assert label_date <= train_end, (
            f"LEAK: window starting idx {w_start} uses label date {label_date} "
            f"> train_end {train_end}"
        )
    assert max_label_date is not None and max_label_date <= train_end


# ── 8b. THE regression lock: matrix must be NON-EMPTY end-to-end ──────────────

def test_build_train_matrix_is_non_empty():
    """Regression lock for the per-fold empty-matrix bug.

    The original PR mocked _windows_to_matrix, so it never exercised the real
    feature/label/regime spine and a vacuously-empty X passed the no-leak test.
    The production failure: SwingStrategy passes its {date: 'BULL'/'BEAR'} regime
    *label* map as regime_score_map; the worker did float('BULL') -> ValueError,
    swallowed by its bare except -> EVERY window dropped -> empty matrix.

    This test runs the REAL builder (no mock) on realistic multi-symbol synthetic
    data, passing exactly that string-label map, and asserts the matrix is
    non-empty with consistent y and real feature columns. It FAILS against the
    pre-fix code (X is empty) and PASSES after the fix (label map is rejected and
    the PIT composite-score map is rebuilt).
    """
    from app.ml.training import ModelTrainer

    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    # ~3 years of business days — plenty of windows even at the 20d horizon.
    sdata = _synthetic_symbols_data(date(2021, 1, 1), n_days=780, symbols=symbols)
    spy = _synthetic_symbols_data(date(2021, 1, 1), n_days=780, symbols=["SPY"])["SPY"]["close"]

    train_start = date(2021, 4, 30)
    train_end = date(2023, 12, 29)

    # The exact shape SwingStrategy._global_regime_map has: {date: str label}.
    label_regime_map = {d.date(): "BULL" for d in spy.index}

    trainer = ModelTrainer(model_type="lambdarank", label_scheme="lambdarank",
                           use_feature_store=False, n_workers=1)
    trainer._allow_sacred_holdout = False

    X, y, fnames, meta = trainer.build_train_matrix_for_window(
        sdata, train_start, train_end,
        spy_prices=spy, regime_score_map=label_regime_map, fetch_fundamentals=False,
    )

    X = np.asarray(X)
    assert len(X) > 0, "build_train_matrix_for_window returned an EMPTY matrix"
    assert X.ndim == 2 and X.shape[1] > 0, f"matrix has no feature columns: shape {X.shape}"
    assert len(y) == len(X), f"y/X length mismatch: {len(y)} vs {len(X)}"
    assert len(meta) == len(X), f"meta/X length mismatch: {len(meta)} vs {len(X)}"
    assert len(fnames) == X.shape[1], (
        f"feature_names ({len(fnames)}) != matrix columns ({X.shape[1]})"
    )
    # Sanity: real engineered feature columns are present (not a degenerate frame).
    # (regime_score itself is in PRUNED_FEATURES so it is intentionally absent from
    # the final vector — its presence as input is proven by the non-empty matrix,
    # since the buggy path crashed inside engineer_features on the string label.)
    assert any(f.startswith("ema") for f in fnames), f"no EMA features in matrix: {fnames[:8]}"
    assert np.isfinite(X).all(), "matrix contains non-finite values"


def test_build_train_matrix_uses_production_horizon():
    """The per-fold builder must label at LABEL_HORIZON_DAYS (production parity),
    and must restore the module-level horizon constants on exit so it never
    contaminates other code in the same process."""
    import app.ml.training as T
    from app.ml.retrain_config import LABEL_HORIZON_DAYS

    symbols = ["AAA", "BBB", "CCC"]
    sdata = _synthetic_symbols_data(date(2021, 1, 1), n_days=600, symbols=symbols)
    spy = _synthetic_symbols_data(date(2021, 1, 1), n_days=600, symbols=["SPY"])["SPY"]["close"]

    trainer = T.ModelTrainer(model_type="lambdarank", label_scheme="lambdarank",
                             use_feature_store=False, n_workers=1)
    trainer._allow_sacred_holdout = False

    captured = {}

    def _spy_windows(symbols_data, all_dates, window_starts, *a, **k):
        # Capture the horizon the builder set WHILE it runs.
        captured["forward_days"] = T.FORWARD_DAYS
        captured["step_days"] = T.STEP_DAYS
        trainer._last_feature_names = ["f1"]
        return [], [], []

    before = (T.FORWARD_DAYS, T.STEP_DAYS, T.EMBARGO_WINDOWS)
    with patch.object(trainer, "_windows_to_matrix", side_effect=_spy_windows):
        trainer.build_train_matrix_for_window(
            sdata, date(2021, 4, 30), date(2022, 12, 30),
            spy_prices=spy, regime_score_map={},
        )
    after = (T.FORWARD_DAYS, T.STEP_DAYS, T.EMBARGO_WINDOWS)

    assert captured["forward_days"] == int(LABEL_HORIZON_DAYS), (
        f"per-fold labeled at horizon {captured['forward_days']}, "
        f"expected production LABEL_HORIZON_DAYS={LABEL_HORIZON_DAYS}"
    )
    assert captured["step_days"] == int(LABEL_HORIZON_DAYS)
    assert before == after, (
        f"module horizon not restored: before={before} after={after}"
    )
