"""Alpha-v4 P0 — residual-alpha (CAPM/HAC) diagnostic + gate recalibration.

Covers:
- capm_alpha reproduces the beta-driven verdict (low HAC-t when returns are mostly
  SPY beta) and detects genuine alpha (high HAC-t for a market-neutral drift), and
  guards against tiny samples.
- The residual-alpha diagnostic is NON-GATING this PR: a result that passes the
  significance gate still passes regardless of a terrible residual-alpha t-stat.
- The CAPITAL mean-Sharpe floor was recalibrated 0.50 → 0.45.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.walkforward.attribution import capm_alpha
from scripts.walkforward.cpcv import CPCVResult


def _idx(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_capm_flags_beta_driven_returns():
    # book = 0.3*SPY + small noise, ZERO true intercept → HAC alpha-t should be low
    # (mostly market beta — PEAD's lesson) and beta ≈ 0.3.
    rng = np.random.default_rng(1)
    n = 400
    spy = pd.Series(rng.normal(0.0004, 0.01, n), index=_idx(n))
    book = 0.3 * spy + pd.Series(rng.normal(0.0, 0.002, n), index=spy.index)
    m = capm_alpha(book, spy)
    assert m["n"] == n
    assert abs(m["beta"] - 0.3) < 0.1
    assert m["t_alpha_hac"] < 1.0          # beta-driven → weak/insignificant alpha


def test_capm_detects_genuine_alpha():
    # Market-neutral positive drift (no beta) → significant positive HAC alpha-t.
    rng = np.random.default_rng(2)
    n = 400
    spy = pd.Series(rng.normal(0.0004, 0.01, n), index=_idx(n))
    book = pd.Series(rng.normal(0.0010, 0.003, n), index=spy.index)  # ~+0.1%/day drift
    m = capm_alpha(book, spy)
    assert m["t_alpha_hac"] > 2.0
    assert m["alpha_ann"] > 0.0
    assert m["resid_sharpe"] > 0.0


def test_capm_guards_tiny_sample():
    s = pd.Series([0.01, -0.01, 0.02], index=_idx(3))
    m = capm_alpha(s, s)
    assert m["n"] == 3
    assert m["t_alpha_hac"] == 0.0 and m["beta"] == 0.0


def _passing_paper_result(**kw):
    # PEAD-like path vector: %pos=1.0, P5≈0.009, mean≈0.546.
    paths = [0.009] + [0.575] * 19
    return CPCVResult(
        model_type="test", n_folds=8, n_paths=2,
        path_sharpes=paths, path_profit_factors=[1.5] * 20,
        path_calmars=[1.0] * 20, path_n_obs=[250] * 20,
        worst_regime_sharpe=0.5, is_true_walkforward=True, **kw,
    )


def test_residual_alpha_is_non_gating(significance_gate_mode):
    from unittest.mock import patch
    good = _passing_paper_result(residual_alpha_t_hac=None)
    terrible = _passing_paper_result(residual_alpha_t_hac=-10.0, residual_beta=0.9,
                                     residual_sharpe=-0.5, residual_n=300)
    # Pin the path t-stat to the logged PEAD value so the paper gate is reachable.
    with patch.object(CPCVResult, "path_sharpe_tstat", property(lambda self: 2.26)):
        # A catastrophic residual-alpha must NOT change the paper gate verdict —
        # it's a diagnostic this PR, deliberately excluded from gate pass/fail.
        assert good.significance_gate_passed(tier="paper") is True
        assert terrible.significance_gate_passed(tier="paper") is True


def test_residual_alpha_absent_from_gate_detail(significance_gate_mode):
    from unittest.mock import patch
    r = _passing_paper_result(residual_alpha_t_hac=-10.0)
    with patch.object(CPCVResult, "path_sharpe_tstat", property(lambda self: 2.26)):
        detail = r.significance_gate_detail(tier="paper")
    assert "residual_alpha_t_hac" not in detail  # diagnostic, not a gate criterion


def test_capital_mean_floor_recalibrated_to_045():
    from app.ml.retrain_config import CAPITAL_GATE_MIN_MEAN_SHARPE
    assert CAPITAL_GATE_MIN_MEAN_SHARPE == 0.45
