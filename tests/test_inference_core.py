"""
Tests for app/research/inference.py — the Ruler-v2 pure inference core.

Known-answer fixtures per RULER_V2_DESIGN.md risk register: HAC-SR recovers the
mean t-test and the HAC variance exceeds the IID variance under positive
autocorrelation (R2); the stationary bootstrap is ~0.5 on noise / ~1.0 on drift and
reproducible (R3); PBO is ~0.5 on noise, low on a dominant config, high on an
overfit (spike) matrix, and exposes the IS=OOS leak signature (R1); the too-few-obs
/ degenerate-vol contract holds; the multifactor_alpha re-export is identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research import inference as inf


# ───────────────────────────── HAC Sharpe (Lo 2002) ─────────────────────────

def test_hac_sharpe_detects_strong_positive_drift():
    rng = np.random.default_rng(1)
    r = 0.0010 + rng.normal(0, 0.005, 600)        # ~3.2 daily SR-ish, strong
    res = inf.hac_sharpe(r)
    assert res.gating is True
    assert res.t_stat > 4 and res.p_one_sided < 0.001
    assert res.sr_ann == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    # the implied SE is back-derived from t: sr_ann / se_implied == t.
    assert res.sr_ann / res.se_sr_ann_implied == pytest.approx(res.t_stat)


def test_hac_sharpe_not_significant_on_zero_mean_noise():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 600)
    r = r - r.mean()                              # exactly zero mean -> t=0, p=0.5
    res = inf.hac_sharpe(r)
    assert res.gating is True
    assert res.t_stat == pytest.approx(0.0, abs=1e-9)
    assert res.p_one_sided == pytest.approx(0.5)


def test_hac_longrun_var_exceeds_iid_under_positive_autocorr():
    # AR(1) with rho=0.5 -> positive autocorrelation -> NW long-run var > gamma_0.
    rng = np.random.default_rng(3)
    n = 2000
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1.0)
    u = x - x.mean()
    g0 = float(u @ u) / n
    lrv = inf._nw_longrun_var_of_mean(u, lag=10)
    assert lrv > g0 * 1.5                          # autocorrelation inflates it
    # An IID series: long-run var ~ gamma_0.
    iid = rng.normal(0, 1.0, n)
    ui = iid - iid.mean()
    assert inf._nw_longrun_var_of_mean(ui, 10) == pytest.approx(float(ui @ ui) / n, rel=0.3)


def test_hac_sharpe_t_shrinks_under_positive_autocorrelation():
    # Same mean/vol, but positive autocorrelation -> larger SE -> smaller |t| than
    # the naive iid t. (The Type-I protection the legacy gate lacked.)
    rng = np.random.default_rng(4)
    n = 1500
    eps = rng.normal(0, 1.0, n)
    ar = np.empty(n)
    ar[0] = eps[0]
    for t in range(1, n):
        ar[t] = 0.6 * ar[t - 1] + eps[t]
    ar = ar - ar.mean() + 0.05                      # inject a small positive mean
    iid = rng.normal(0.05, ar.std(), n)             # matched mean/vol, no autocorr
    t_ar = inf.hac_sharpe(ar).t_stat
    t_iid = inf.hac_sharpe(iid).t_stat
    assert t_ar < t_iid                             # autocorrelation discounts the t


def test_hac_sharpe_guards():
    assert inf.hac_sharpe(np.zeros(40)).gating is False        # too few obs
    r = inf.hac_sharpe(np.full(100, 0.001))                    # degenerate vol
    assert r.gating is False and r.reason == "degenerate vol"


# ───────────────────────── stationary bootstrap ─────────────────────────────

def test_bootstrap_one_on_strong_drift():
    rng = np.random.default_rng(5)
    drift = inf.stationary_bootstrap_sr(0.001 + rng.normal(0, 0.004, 500), n_reps=800)
    assert drift.p_sr_gt_0 > 0.97                  # strong positive -> ~1 (robust)
    assert drift.ci_low_95 < drift.sr_ann_point < drift.ci_high_95


def test_bootstrap_well_centered_on_noise_averaged_over_draws():
    # A SINGLE zero-edge draw's p can land anywhere (it bootstraps the point SR,
    # std~0.30 across seeds) — so assert the MEAN over many independent noise draws
    # is ~0.5, not a single draw inside a narrow band (that would be seed-flaky).
    ps = []
    for s in range(40):
        rng = np.random.default_rng(100 + s)
        ps.append(inf.stationary_bootstrap_sr(rng.normal(0, 0.01, 400),
                                              n_reps=300, seed=s).p_sr_gt_0)
    assert abs(np.mean(ps) - 0.5) < 0.08           # well-centered on the null


def test_bootstrap_reproducible_with_seed():
    rng = np.random.default_rng(6)
    r = 0.0005 + rng.normal(0, 0.008, 400)
    a = inf.stationary_bootstrap_sr(r, n_reps=500, seed=42)
    b = inf.stationary_bootstrap_sr(r, n_reps=500, seed=42)
    assert a.p_sr_gt_0 == b.p_sr_gt_0 and a.ci_low_95 == b.ci_low_95


def test_bootstrap_block_len_longer_for_persistent_series():
    rng = np.random.default_rng(7)
    n = 1000
    iid = rng.normal(0, 1.0, n)
    ar = np.empty(n)
    ar[0] = iid[0]
    for t in range(1, n):
        ar[t] = 0.7 * ar[t - 1] + iid[t]
    assert inf._auto_block_len(ar) > inf._auto_block_len(iid)


def test_bootstrap_guards():
    assert inf.stationary_bootstrap_sr(np.zeros(30)).gating is False
    assert inf.stationary_bootstrap_sr(np.full(200, 0.01)).gating is False


# ───────────────────────────── PBO (CSCV) ───────────────────────────────────

def test_pbo_about_half_on_pure_noise():
    rng = np.random.default_rng(8)
    M, S = 20, 12
    perf = rng.normal(0, 1.0, (M, S))              # no config has real skill
    res = inf.pbo_cscv(perf)
    assert res.gating is True
    assert 0.35 < res.pbo < 0.65                   # ~0.5 = the overfit null


def test_pbo_low_for_a_genuinely_dominant_config():
    M, S = 10, 12
    perf = np.full((M, S), 0.0)
    perf[0, :] = 5.0                               # config 0 dominates every block
    res = inf.pbo_cscv(perf)
    assert res.pbo < 0.05                          # IS-best == OOS-best -> no overfit
    assert res.prob_oos_loss == 0.0


def test_pbo_high_for_an_overfit_spike_matrix():
    # Each config spikes in exactly one unique block: the IS-best (its spike in the
    # IS half) is ~zero OOS -> below the OOS median -> high PBO.
    S = 12
    perf = np.eye(S) * 10.0                         # M=S configs, identity spikes
    res = inf.pbo_cscv(perf)
    assert res.pbo > 0.8                            # textbook overfit


def test_pbo_leak_signature_is_documented_false_negative():
    # If the caller leaks (each config's IS performance == its OOS performance for
    # every split — here: identical per-block performance), the IS-best is trivially
    # OOS-best -> PBO ~ 0 (a FALSE "no overfit"). Pinning the contract: PBO cannot
    # detect a leaking caller.
    rng = np.random.default_rng(9)
    scores = rng.normal(0, 1.0, 15)
    leaked = np.tile(scores[:, None], (1, 12))     # every block identical per config
    assert inf.pbo_cscv(leaked).pbo < 0.05


def test_pbo_fails_closed_on_non_finite_cell():
    rng = np.random.default_rng(13)
    perf = rng.normal(0, 1.0, (10, 12))
    perf[3, 5] = np.nan                            # e.g. a block with no trades
    res = inf.pbo_cscv(perf)
    assert res.gating is False and np.isnan(res.pbo)
    assert "non-finite" in res.reason


def test_pbo_tie_fair_rank_not_optimistically_biased():
    # Discretized (tie-heavy) perf must not be biased DOWN by a `<=` rank. The
    # mid-rank convention keeps a genuinely-overfit spike matrix high even when
    # quantized.
    S = 12
    perf = np.round(np.eye(S) * 10.0)              # integer ties everywhere off-diag
    res = inf.pbo_cscv(perf)
    assert res.pbo > 0.8                           # still textbook overfit


def test_pbo_single_config_undefined():
    res = inf.pbo_cscv(np.ones((1, 8)))
    assert res.gating is False and np.isnan(res.pbo)
    assert "single config" in res.reason


def test_pbo_handles_odd_blocks_by_dropping_last():
    rng = np.random.default_rng(10)
    res = inf.pbo_cscv(rng.normal(0, 1.0, (8, 7)))  # odd S=7 -> uses 6
    assert res.gating is True and 0.0 <= res.pbo <= 1.0


def test_pbo_sampling_path_deterministic_and_unbiased():
    # When n_splits forces the sampling path (instead of full enumeration), it's
    # seed-deterministic and still ~0.5 on noise (unbiased).
    rng = np.random.default_rng(11)
    perf = rng.normal(0, 1.0, (20, 14))
    a = inf.pbo_cscv(perf, n_splits=500, seed=7)
    b = inf.pbo_cscv(perf, n_splits=500, seed=7)
    assert a.pbo == b.pbo and a.n_splits == 500
    assert 0.30 < a.pbo < 0.70


def test_bootstrap_respects_manual_block_len():
    rng = np.random.default_rng(12)
    r = 0.0005 + rng.normal(0, 0.008, 400)
    res = inf.stationary_bootstrap_sr(r, n_reps=300, mean_block_len=25.0)
    assert res.mean_block_len == 25.0


# ───────────────────────── multifactor_alpha re-export ──────────────────────

def test_multifactor_alpha_reexport_identical():
    from app.research.options_xs_ls import multifactor_alpha as reexport
    assert reexport is inf.multifactor_alpha
    idx = pd.bdate_range("2023-01-01", periods=200)
    rng = np.random.default_rng(11)
    f = pd.DataFrame({"SPY": rng.normal(0, 0.01, 200)}, index=idx)
    y = pd.Series(0.0002 + 0.7 * f["SPY"].to_numpy() + rng.normal(0, 0.003, 200),
                  index=idx)
    out = inf.multifactor_alpha(y, f, hac_lag=5)
    assert out["betas"]["SPY"] == pytest.approx(0.7, abs=0.05)
    assert np.isfinite(out["t_alpha_hac"])


def test_multifactor_alpha_does_not_raise_on_inf_factor():
    # pandas .dropna() does NOT drop ±inf and lstsq raises on it — the finite-mask
    # must keep the contract (fail to the zero-fill, never raise).
    idx = pd.bdate_range("2023-01-01", periods=120)
    rng = np.random.default_rng(14)
    f = pd.DataFrame({"SPY": rng.normal(0, 0.01, 120)}, index=idx)
    f.iloc[10, 0] = np.inf
    y = pd.Series(0.0002 + 0.7 * f["SPY"].to_numpy(), index=idx)
    out = inf.multifactor_alpha(y, f)              # must not raise
    assert out["n"] == 119                         # the inf row dropped, rest kept
    assert np.isfinite(out["t_alpha_hac"])
