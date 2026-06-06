"""
Unit tests for the reusable CAPM beta-isolation core of the Phase-1 PEAD
attribution tool (scripts/pead_phase1_attribution.py::_capm).

_capm is pure numpy/pandas (the app/network imports live inside the other
functions), so it tests in isolation. We verify it recovers known beta/alpha,
that the beta-removed Sharpe is the market-hedged stream (mean = alpha, NOT the
mean-zero OLS residual), and that the Newey-West HAC t-stat differs from OLS on
autocorrelated residuals (the reason we report HAC for overlapping multi-week
holds).
"""
import numpy as np
import pandas as pd
import pytest

from scripts.pead_phase1_attribution import _capm, ANN


def _series(arr, start="2022-01-03"):
    idx = pd.date_range(start=start, periods=len(arr), freq="B")
    return pd.Series(arr, index=idx)


def test_recovers_known_beta_zero_alpha():
    rng = np.random.default_rng(0)
    r_spy = rng.normal(0.0004, 0.011, 1500)
    r_book = 0.5 * r_spy + rng.normal(0.0, 0.002, 1500)  # beta 0.5, no alpha
    m = _capm(_series(r_book), _series(r_spy))
    assert abs(m["beta"] - 0.5) < 0.05
    assert abs(m["alpha_ann"]) < 0.05            # economically ~0 annual alpha
    # (No assertion on the zero-alpha t-stat: with n=1500 and a TRUE alpha of 0,
    #  |t| > 2 occurs ~5% of the time by chance — a seed-flaky bound. Beta recovery
    #  + negligible alpha magnitude already capture "known beta, no alpha".)
    assert m["r2"] > 0.85                         # mostly explained by SPY


def test_recovers_positive_alpha_zero_beta():
    rng = np.random.default_rng(1)
    r_spy = rng.normal(0.0004, 0.011, 1500)
    alpha_d = 0.0006                              # +0.06%/day ~ +15%/yr
    r_book = alpha_d + 0.0 * r_spy + rng.normal(0.0, 0.003, 1500)
    m = _capm(_series(r_book), _series(r_spy))
    assert abs(m["beta"]) < 0.05
    assert m["alpha_ann"] > 0.10                  # clearly positive
    assert m["t_alpha_ols"] > 3.0                 # significant
    # Beta-removed Sharpe ~ raw Sharpe here (beta ~0), and clearly positive.
    assert m["resid_sharpe"] > 1.0


def test_beta_removed_sharpe_is_hedged_stream_not_mean_zero_residual():
    """The reported 'resid_sharpe' must be Sharpe(y - beta*x) (mean = alpha),
    NOT the mean-zero OLS residual y-alpha-beta*x (whose Sharpe is ~0). With a
    NEGATIVE alpha + positive beta on a rising market, raw Sharpe is positive but
    the beta-removed Sharpe must be NEGATIVE (the PEAD finding)."""
    rng = np.random.default_rng(2)
    r_spy = np.abs(rng.normal(0.0008, 0.009, 1500))  # persistently rising market
    alpha_d = -0.0002                                 # negative alpha
    r_book = alpha_d + 0.3 * r_spy + rng.normal(0.0, 0.002, 1500)
    m = _capm(_series(r_book), _series(r_spy))
    assert m["raw_sharpe"] > 0.0                       # positive raw (beta rides the market)
    assert m["alpha_ann"] < 0.0                        # negative alpha
    assert m["resid_sharpe"] < 0.0                     # hedge out beta -> loses money
    # And it must NOT be the trivially-zero mean-zero residual Sharpe.
    assert abs(m["resid_sharpe"]) > 0.05


def test_hac_t_differs_from_ols_on_autocorrelated_residuals():
    """Positively autocorrelated residuals (overlapping holds) inflate the OLS
    t-stat; the HAC correction shrinks it. The two must not be identical."""
    rng = np.random.default_rng(3)
    r_spy = rng.normal(0.0004, 0.011, 2000)
    # AR(1) residual with strong positive autocorrelation.
    e = np.zeros(2000)
    for i in range(1, 2000):
        e[i] = 0.6 * e[i - 1] + rng.normal(0.0, 0.002)
    r_book = 0.0004 + 0.2 * r_spy + e
    m = _capm(_series(r_book), _series(r_spy), hac_lag=5)
    assert m["t_alpha_ols"] != pytest.approx(m["t_alpha_hac"], abs=1e-6)
    # HAC se >= OLS se under positive autocorrelation -> |t_hac| <= |t_ols|.
    assert abs(m["t_alpha_hac"]) <= abs(m["t_alpha_ols"]) + 1e-9


def test_annualization_constant():
    assert ANN == 252
