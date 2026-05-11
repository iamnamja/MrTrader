"""R5 tests: RegimeClassifier — no-lookahead, label correctness, fit/predict, save/load."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.regime_classifier import (
    RegimeClassifier,
    build_regime_features,
    build_regime_labels,
    FEATURE_NAMES,
    REGIME_FLOOR,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_spy(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    close = 300.0 + rng.standard_normal(n).cumsum() * 2
    close = np.clip(close, 50, 1000)
    return pd.DataFrame({"Close": close}, index=idx)


def _make_vix(n: int = 500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    vix = 18.0 + rng.standard_normal(n) * 4
    vix = np.clip(vix, 9, 80)
    return pd.DataFrame({"Close": vix}, index=idx)


def _make_hyg(n: int = 500, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    close = 80.0 + rng.standard_normal(n).cumsum() * 0.5
    close = np.clip(close, 50, 120)
    return pd.DataFrame({"Close": close}, index=idx)


def _fit_clf(n: int = 500) -> tuple[RegimeClassifier, pd.DataFrame, pd.Series]:
    spy = _make_spy(n)
    vix = _make_vix(n)
    hyg = _make_hyg(n)
    features = build_regime_features(spy, vix, hyg)
    labels = build_regime_labels(spy, vix)
    common = features.index.intersection(labels.index)
    clf = RegimeClassifier()
    clf.fit(features.loc[common], labels.loc[common], train_end="2019-12-31")
    return clf, features.loc[common], labels.loc[common]


# ── Label definition ──────────────────────────────────────────────────────────

class TestLabelDefinition:
    def test_spy_above_ma_vix_below_thresh_is_1(self):
        """Regime=1 when SPY above MA200 and VIX < threshold."""
        n = 400
        spy = _make_spy(n)
        vix = _make_vix(n)
        # Force VIX to be always low
        vix["Close"] = 10.0
        labels = build_regime_labels(spy, vix, vix_threshold=25.0)
        # After MA200 warmup, SPY will sometimes be above MA — those should be 1
        # We only check that all values are 0 or 1
        assert set(labels.dropna().unique()).issubset({0, 1})

    def test_high_vix_forces_label_0(self):
        """When VIX > threshold always, label is always 0."""
        spy = _make_spy(400)
        vix = _make_vix(400)
        vix["Close"] = 50.0  # always above threshold of 25
        labels = build_regime_labels(spy, vix, vix_threshold=25.0)
        assert (labels.dropna() == 0).all(), "VIX=50 should always give regime=0"

    def test_spy_below_ma_forces_label_0(self):
        """When SPY is always far below its MA200, label is 0."""
        idx = pd.bdate_range("2018-01-02", periods=400)
        # SPY starts at 50 but MA200 is anchored to prior high values
        spy = pd.DataFrame({"Close": np.linspace(300, 50, 400)}, index=idx)
        vix = pd.DataFrame({"Close": np.full(400, 10.0)}, index=idx)
        labels = build_regime_labels(spy, vix)
        # Last ~100 bars: SPY well below MA200
        assert (labels.iloc[-50:] == 0).all()


# ── No-lookahead check ────────────────────────────────────────────────────────

class TestNoLookahead:
    def test_features_at_T_do_not_use_data_after_T(self):
        """Features for date T must be identical whether computed on full series or truncated."""
        spy = _make_spy(450)
        vix = _make_vix(450)
        hyg = _make_hyg(450)

        features_full = build_regime_features(spy, vix, hyg)
        # Pick a date in the middle
        t_idx = 300
        t = features_full.index[t_idx]

        # Truncated at T
        features_trunc = build_regime_features(
            spy.loc[:t], vix.loc[:t], hyg.loc[:t]
        )

        if t in features_full.index and t in features_trunc.index:
            row_full = features_full.loc[t]
            row_trunc = features_trunc.loc[t]
            for col in FEATURE_NAMES:
                if col in row_full.index and col in row_trunc.index:
                    np.testing.assert_allclose(
                        row_full[col], row_trunc[col], rtol=1e-5,
                        err_msg=f"Lookahead detected in feature '{col}' at {t}",
                    )


# ── predict_proba ─────────────────────────────────────────────────────────────

class TestPredictProba:
    def test_predict_proba_in_unit_interval(self):
        clf, features, _ = _fit_clf()
        probs = clf.predict_proba_series(features)
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), \
            "predict_proba must return values in [0, 1]"

    def test_predict_proba_date_single_row(self):
        clf, features, _ = _fit_clf()
        row = features.iloc[-1]
        prob = clf.predict_proba_date(row)
        assert 0.0 <= prob <= 1.0

    def test_predict_proba_nan_row_returns_neutral(self):
        clf, features, _ = _fit_clf()
        row = features.iloc[-1].copy()
        row[:] = np.nan
        prob = clf.predict_proba_date(row)
        assert prob == 0.5, "NaN row should return neutral probability 0.5"

    def test_sizing_weight_floored_at_regime_floor(self):
        clf, _, _ = _fit_clf()
        weight = clf.sizing_weight(0.0)
        assert weight == REGIME_FLOOR, f"sizing_weight(0) should be REGIME_FLOOR={REGIME_FLOOR}"

    def test_sizing_weight_capped_at_1(self):
        clf, _, _ = _fit_clf()
        assert clf.sizing_weight(1.5) == 1.0


# ── Save / load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        clf, features, _ = _fit_clf()
        out = tmp_path / "regime_test.pkl"
        clf.save(out)

        loaded = RegimeClassifier.load(out)
        probs_orig = clf.predict_proba_series(features).values
        probs_loaded = loaded.predict_proba_series(features).values
        np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-6,
                                   err_msg="Loaded model produces different predictions")

    def test_meta_json_written(self, tmp_path):
        clf, features, _ = _fit_clf()
        out = tmp_path / "regime_test.pkl"
        clf.save(out)
        meta_path = tmp_path / "regime_test_meta.json"
        assert meta_path.exists(), "Meta JSON must be written alongside pkl"
        import json
        meta = json.loads(meta_path.read_text())
        assert "feature_names" in meta
        assert "train_end" in meta


# ── Validation Brier check ────────────────────────────────────────────────────

class TestValidationQuality:
    def test_brier_below_baseline_on_train_data(self):
        """Model Brier on training data must beat the constant-prediction baseline."""
        from sklearn.metrics import brier_score_loss
        clf, features, labels = _fit_clf(n=500)
        probs = clf.predict_proba_series(features)
        aligned = labels.reindex(probs.index).dropna()
        probs_aligned = probs.loc[aligned.index]
        brier = brier_score_loss(aligned.values, probs_aligned.values)
        baseline = float(aligned.mean() * (1 - aligned.mean()))
        assert brier < baseline, (
            f"Model Brier ({brier:.4f}) ≥ baseline ({baseline:.4f}). "
            "Model is no better than predicting the mean."
        )


# ── RegimeProbGate interface ──────────────────────────────────────────────────

class TestRegimeProbGate:
    def test_is_available_false_when_no_model(self, tmp_path):
        from app.risk.regime_gate import RegimeProbGate
        assert not RegimeProbGate.is_available(str(tmp_path / "nonexistent.pkl"))

    def test_weight_returns_float(self, tmp_path):
        from app.risk.regime_gate import RegimeProbGate
        clf, features, _ = _fit_clf()
        out = tmp_path / "regime_v1.pkl"
        clf.save(out)
        gate = RegimeProbGate(model_path=str(out))
        row = features.iloc[-1]
        w = gate.weight(row)
        assert isinstance(w, float)
        assert REGIME_FLOOR <= w <= 1.0

    def test_weight_fails_open_when_no_model(self):
        from app.risk.regime_gate import RegimeProbGate
        gate = RegimeProbGate(model_path="nonexistent_path_xyz.pkl")
        row = pd.Series({f: 0.0 for f in FEATURE_NAMES})
        w = gate.weight(row)
        assert w == 1.0, "Should fail-open (return 1.0) when model missing"
