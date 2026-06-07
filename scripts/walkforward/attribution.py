"""
Market-residualized alpha attribution — Alpha-v4 P0 gate recalibration.

A book's Sharpe can be mostly market beta (PEAD's lesson: +0.55 raw Sharpe but
HAC alpha t≈-0.95, β≈0.14 → it loses money once SPY is hedged out). The honest
robustness signal is the **residual-alpha t-stat**: regress book returns on SPY
(CAPM) and test whether the intercept (alpha) is significant after Newey-West HAC
correction for the autocorrelation of overlapping multi-week holds.

This is the canonical implementation, lifted verbatim from
scripts/pead_phase1_attribution._capm (which now re-exports from here) so the gate
pipeline and the standalone PEAD attribution script share one definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def capm_alpha(r_book: pd.Series, r_spy: pd.Series, hac_lag: int = 5) -> dict:
    """OLS r_book = alpha + beta*r_spy with OLS and Newey-West HAC alpha t-stats.

    Returns a dict: n, raw_sharpe, beta, alpha_ann, alpha_bps_d, t_alpha_ols,
    t_alpha_hac, resid_sharpe (beta-hedged), r2. Aligns the two series on their
    common index and drops NaNs. Returns a zero-filled dict if < 30 aligned obs
    (too few to estimate) so callers never divide by zero / over-interpret noise.
    """
    r_spy = r_spy.reindex(r_book.index).dropna()
    r_book = r_book.reindex(r_spy.index).dropna()
    r_spy = r_spy.reindex(r_book.index)
    x = r_spy.to_numpy()
    y = r_book.to_numpy()
    n = len(y)
    if n < 30:
        return {
            "n": n, "raw_sharpe": 0.0, "beta": 0.0, "alpha_ann": 0.0,
            "alpha_bps_d": 0.0, "t_alpha_ols": 0.0, "t_alpha_hac": 0.0,
            "resid_sharpe": 0.0, "r2": 0.0,
        }
    X = np.column_stack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_d, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    dof = n - 2
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se_alpha_ols = float(np.sqrt(s2 * XtX_inv[0, 0]))
    t_alpha_ols = alpha_d / se_alpha_ols if se_alpha_ols > 0 else 0.0

    # Newey-West HAC sandwich on the full [1, x] design (robust to the
    # autocorrelation/heteroskedasticity of overlapping multi-week holds).
    bread = XtX_inv
    S = np.zeros((2, 2))
    u = resid
    Xr = X * u[:, None]  # n x 2 score contributions
    S += Xr.T @ Xr
    for L in range(1, hac_lag + 1):
        w = 1.0 - L / (hac_lag + 1.0)  # Bartlett kernel
        G = Xr[L:].T @ Xr[:-L]
        S += w * (G + G.T)
    cov_hac = bread @ S @ bread
    se_alpha_hac = float(np.sqrt(max(cov_hac[0, 0], 0.0)))
    t_alpha_hac = alpha_d / se_alpha_hac if se_alpha_hac > 0 else 0.0

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    raw_sharpe = float(y.mean() / y.std() * np.sqrt(ANN)) if y.std() > 0 else 0.0
    # Beta-REMOVED (market-hedged) return stream = y - beta*x; its mean is alpha,
    # so its Sharpe is the honest "what survives after hedging out SPY" number.
    hedged = y - beta * x
    resid_sharpe = float(hedged.mean() / hedged.std() * np.sqrt(ANN)) if hedged.std() > 0 else 0.0
    return {
        "n": n, "raw_sharpe": raw_sharpe, "beta": beta,
        "alpha_ann": alpha_d * ANN, "alpha_bps_d": alpha_d * 1e4,
        "t_alpha_ols": t_alpha_ols, "t_alpha_hac": t_alpha_hac,
        "resid_sharpe": resid_sharpe, "r2": r2,
    }
