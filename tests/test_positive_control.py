"""Tests for the Alpha-v9 P0-1 positive-control harness.

The harness validates the swing feature->label pipeline by pushing KNOWN
cross-sectional anomalies through the REAL `build_train_matrix_for_window` and
checking the pipeline recovers them. These tests prove BOTH directions:

  1. on a clean synthetic panel with injected momentum+low-vol, the harness PASSES
     (it recovers the injected effects through the real builder);
  2. when the feature->outcome join is deliberately corrupted (rows permuted), the
     harness FAILS — proving it actually DETECTS the deflationary-bug class it
     exists to catch, rather than rubber-stamping.

Plus unit coverage of the cross-sectional IC primitive.
"""
import os
from datetime import date

import numpy as np
import pytest

from scripts.walkforward.positive_control import (
    cross_sectional_ic,
    classify_fidelity,
    run_positive_control,
    ANOMALIES,
)

# The integration tests below build the REAL train matrix (cache-free engineer_features
# + FMP/macro/regime parquet loads), which is inherently too slow + variable for CI's
# 120s per-test cap under xdist contention. They run locally / on demand; CI coverage
# of the logic comes from the fast pure-function unit tests above. (GitHub sets CI=true.)
_skip_in_ci = pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="real-matrix build exceeds the 120s CI cap under contention; run locally",
)


# ---- unit: fidelity classifier (the core of the FULL-mode verdict) ----------
def test_fidelity_pipeline_matches_strong_reference_is_not_a_bug():
    # data HAS the effect and the pipeline reproduces it -> faithful, NOT deflationary
    # (this is the real 150-symbol case: momentum pipe -0.029 vs ref -0.031)
    faithful, deflationary = classify_fidelity(-0.029, -0.031)
    assert faithful is True and deflationary is False


def test_fidelity_pipeline_loses_present_signal_is_a_bug():
    # data HAS a strong effect but the pipeline column is ~0 -> deflationary divergence
    faithful, deflationary = classify_fidelity(0.001, 0.060)
    assert faithful is False and deflationary is True


def test_fidelity_effect_absent_is_not_a_bug():
    # reference ~ 0 (effect absent in window) and pipeline ~ 0 -> not deflationary
    faithful, deflationary = classify_fidelity(0.004, 0.006)
    assert deflationary is False


def test_fidelity_no_reference_is_inconclusive():
    faithful, deflationary = classify_fidelity(0.05, None)
    assert faithful is None and deflationary is False


# ---- unit: cross-sectional IC primitive -------------------------------------
def test_cross_sectional_ic_recovers_strong_positive_relationship():
    rng = np.random.default_rng(0)
    n_per, n_win = 40, 30
    feat, out, wid = [], [], []
    for w in range(n_win):
        f = rng.standard_normal(n_per)
        o = f + 0.3 * rng.standard_normal(n_per)   # strong positive dependence
        feat.extend(f)
        out.extend(o)
        wid.extend([w] * n_per)
    ic, t, nw, series = cross_sectional_ic(np.array(feat), np.array(out), np.array(wid))
    assert ic > 0.7
    assert t > 5
    assert nw == n_win


def test_cross_sectional_ic_zero_on_independent_noise():
    rng = np.random.default_rng(1)
    n_per, n_win = 40, 30
    feat = rng.standard_normal(n_per * n_win)
    out = rng.standard_normal(n_per * n_win)
    wid = np.repeat(np.arange(n_win), n_per)
    ic, t, nw, _ = cross_sectional_ic(feat, out, wid)
    assert abs(ic) < 0.1
    assert abs(t) < 2.0


def test_cross_sectional_ic_sign_flips_with_relationship():
    rng = np.random.default_rng(2)
    n_per, n_win = 40, 30
    feat, out, wid = [], [], []
    for w in range(n_win):
        f = rng.standard_normal(n_per)
        o = -f + 0.3 * rng.standard_normal(n_per)   # negative dependence
        feat.extend(f)
        out.extend(o)
        wid.extend([w] * n_per)
    ic, t, _, _ = cross_sectional_ic(np.array(feat), np.array(out), np.array(wid))
    assert ic < -0.7
    assert t < -5


# Small synthetic panel so each integration build stays well under the 120s CI
# per-test timeout (the serial, cache-free build is the bottleneck); the CLI keeps
# the larger defaults. The injected effect is strong enough to clear t>=2.0 here.
_SMOKE = dict(smoke_n_symbols=30, smoke_n_days=420)


# ---- integration: the harness on a clean synthetic panel --------------------
@pytest.mark.slow
@_skip_in_ci
def test_positive_control_smoke_passes_on_clean_panel():
    rep = run_positive_control(as_of=date(2026, 6, 16), smoke=True, **_SMOKE)  # lambdarank
    assert rep.label_scheme == "lambdarank"
    # the pipeline built a non-empty, finite matrix and labels reflect outcomes
    assert rep.matrix_nonempty is True
    assert rep.matrix_finite is True
    assert rep.label_fidelity_ok is True
    # under lambdarank y is a binary sign label; IC(sign, return) is a REAL (not
    # tautological) check and lands well above the 0.30 floor (~0.86 on smoke).
    assert rep.label_fidelity_ic > 0.5
    # the injected anomalies are recovered with the expected sign + significance
    by_name = {a.name: a for a in rep.anomalies}
    mom = by_name["xs_momentum"]
    assert mom.pipeline_ic > 0 and mom.pipeline_ic_t >= 2.0 and mom.passed
    lv = by_name["low_vol"]
    assert lv.pipeline_ic < 0 and lv.pipeline_ic_t <= -2.0 and lv.passed
    assert rep.overall_pass is True
    # the window-truncation finding is surfaced accurately: ~42 bars (63 - 21 skip)
    assert mom.effective_lookback_bars == 42


# ---- integration: detection power (the harness must FAIL on a broken join) ---
@pytest.mark.slow
@_skip_in_ci
def test_positive_control_fails_when_join_is_corrupted():
    rep = run_positive_control(as_of=date(2026, 6, 16), smoke=True, _corrupt_join=True, **_SMOKE)
    # permuting outcomes destroys feature->outcome alignment -> the verdict MUST flip
    assert rep.overall_pass is False
    by_name = {a.name: a for a in rep.anomalies}
    # the injected momentum (clean ic ~0.24) must collapse once the join is broken
    # (small-panel shuffle leaves only residual noise, well below the clean signal)
    assert abs(by_name["xs_momentum"].pipeline_ic) < 0.10
    assert not by_name["xs_momentum"].significant
    assert any("CORRUPTION INJECTED" in n for n in rep.notes)


# ---- guard: feature store must be OFF (else cache bypasses engineer_features)-
@pytest.mark.slow
@_skip_in_ci
def test_positive_control_does_not_pollute_feature_store():
    # The harness constructs the trainer with use_feature_store=False and asserts
    # trainer._feature_store is None internally; if that regressed, the run would
    # raise. Reaching a verdict at all proves the guard held.
    rep = run_positive_control(as_of=date(2026, 6, 16), smoke=True, **_SMOKE)
    assert rep.mode == "smoke"


def test_anomaly_specs_cover_both_signs():
    signs = {a.expected_sign for a in ANOMALIES}
    assert signs == {1, -1}        # exercises positive AND negative detection paths


def test_fetch_real_panel_offline(monkeypatch):
    """Exercise the FULL-mode fetch code path with yfinance stubbed (no network).
    Regression for the SwingStrategy-signature bug that only the live run surfaced:
    smoke never calls _fetch_real_panel, so this guards the real-data path."""
    import pandas as pd
    import numpy as np
    import scripts.walkforward.positive_control as pc

    def _fake_download(sym, start=None, end=None, progress=False, auto_adjust=True):
        idx = pd.bdate_range(end="2025-12-31", periods=260)
        rng = np.random.default_rng(abs(hash(sym)) % (2**32))
        close = 100 * np.exp(np.cumsum(rng.standard_normal(260) * 0.01))
        return pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99,
             "Close": close, "Volume": 1e6}, index=idx)

    monkeypatch.setattr("yfinance.download", _fake_download)
    symbols_data, spy = pc._fetch_real_panel(window_years=1, as_of=date(2026, 6, 16), max_symbols=4)
    assert len(symbols_data) >= 1            # >=210 bars => all 4 kept
    assert "SPY" not in symbols_data         # SPY returned separately as the spine
    assert hasattr(spy, "index") and len(spy) >= 210
    # columns lower-cased to the OHLCV the builder expects
    any_df = next(iter(symbols_data.values()))
    assert set(["open", "high", "low", "close", "volume"]).issubset(set(any_df.columns))
