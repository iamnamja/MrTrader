"""Tests for OPT-5 threshold-robustness sweep (scripts/run_pead_implied_threshold_sweep.py).

These guard the two correctness landmines in the sweep harness — both of which would
silently corrupt a decision-feeding robustness verdict rather than fail loudly:

  1. METRIC MAPPING (_m): the beta-hedged Sharpe lives on CPCVResult as `residual_sharpe`
     (NOT `residual_alpha_sharpe`/`hedged_sharpe`). A wrong name makes `_m()` return NaN,
     which silently forces the verdict to FRAGILE no matter the data. This is a regression
     guard for exactly that bug.
  2. VERDICT CLASSIFICATION (classify_robustness): the plateau-vs-spike logic, and its
     honest handling of a non-computable hedged Sharpe (NaN) — which must surface as
     "hedged unavailable", never as a silent FRAGILE.

classify_robustness is a pure function (no I/O), so it is tested directly on synthetic
metric dicts.
"""
from types import SimpleNamespace

import pytest

from scripts.run_pead_implied_threshold_sweep import (
    _fin,
    _m,
    classify_robustness,
)

NAN = float("nan")
THRESHOLDS = [0.75, 1.0, 1.25]


def _metrics(sharpe, ra_sharpe, pf=1.2, t=2.0, ra_t=1.5, pos=0.7):
    """Build an _m()-shaped metric dict for the classifier."""
    return {"sharpe": sharpe, "t": t, "ra_t": ra_t, "ra_sharpe": ra_sharpe,
            "pf": pf, "pos": pos}


# ───────────────────────── 1. _m metric mapping ────────────────────────────────

def test_m_reads_residual_sharpe_for_hedged():
    """REGRESSION: hedged Sharpe must be read from CPCVResult.residual_sharpe.

    The original sweep looked for `residual_alpha_sharpe`/`hedged_sharpe` — neither
    exists on CPCVResult — so ra_sharpe was always NaN, which hardcoded the verdict to
    FRAGILE. _m() must pull the real `residual_sharpe` attribute.
    """
    res = SimpleNamespace(
        mean_sharpe=0.85,
        path_sharpe_tstat=2.1,
        residual_alpha_t_hac=1.7,
        residual_sharpe=0.62,
        avg_profit_factor=1.3,
        pct_positive=0.72,
    )
    m = _m(res)
    assert m["sharpe"] == 0.85
    assert m["t"] == 2.1
    assert m["ra_t"] == 1.7
    assert m["ra_sharpe"] == 0.62  # <- the fix: NOT NaN
    assert m["pf"] == 1.3
    assert m["pos"] == 0.72


def test_m_returns_nan_when_residual_sharpe_absent():
    """When SPY-align < 30 obs, CPCVResult.residual_sharpe is None -> _m() yields NaN,
    which the verdict path must treat as 'hedged unavailable' (covered below)."""
    res = SimpleNamespace(
        mean_sharpe=0.5,
        path_sharpe_tstat=1.0,
        residual_alpha_t_hac=None,
        residual_sharpe=None,
        avg_profit_factor=1.1,
        pct_positive=0.6,
    )
    m = _m(res)
    assert not _fin(m["ra_sharpe"])
    assert not _fin(m["ra_t"])
    assert m["sharpe"] == 0.5  # mean still finite


# ───────────────────────── 2. verdict classification ───────────────────────────

def test_robust_when_plateau_and_hedged_lift():
    """Mean-Sharpe lifts at ALL thresholds with a tight spread AND hedged lifts too."""
    base = _metrics(0.40, 0.30)
    rows = {0.75: _metrics(0.72, 0.55),
            1.0: _metrics(0.75, 0.58),
            1.25: _metrics(0.70, 0.54)}
    out = classify_robustness(base, rows, THRESHOLDS)
    assert out["label"] == "ROBUST"
    assert out["spread"] <= 0.40


def test_not_fragile_regression_with_correct_hedged():
    """REGRESSION twin of test_m_*: with hedged correctly populated and a real plateau,
    the verdict must be ROBUST — pre-fix (NaN hedged) this same data returned FRAGILE."""
    base = _metrics(0.30, 0.20)
    rows = {t: _metrics(0.55, 0.40) for t in THRESHOLDS}  # +0.25 mean, +0.20 hedged, flat
    out = classify_robustness(base, rows, THRESHOLDS)
    assert out["label"] == "ROBUST"
    assert out["label"] != "FRAGILE"


def test_fragile_when_lift_only_at_one_threshold():
    """Spike at 1.0 only (0.75 drops, 1.25 flat) -> overfit-suspect -> FRAGILE."""
    base = _metrics(0.40, 0.30)
    rows = {0.75: _metrics(0.30, 0.20),   # Δ -0.10
            1.0: _metrics(0.95, 0.70),    # Δ +0.55
            1.25: _metrics(0.42, 0.31)}   # Δ +0.02
    out = classify_robustness(base, rows, THRESHOLDS)
    assert out["label"] == "FRAGILE"


def test_directional_when_lifts_but_spread_wide():
    """Lifts at all thresholds (>0.10) but magnitude varies widely (spread > 0.40)."""
    base = _metrics(0.30, 0.20)
    rows = {0.75: _metrics(0.45, 0.35),   # Δ +0.15
            1.0: _metrics(0.95, 0.70),    # Δ +0.65
            1.25: _metrics(0.80, 0.60)}   # Δ +0.50  -> spread 0.50 > 0.40
    out = classify_robustness(base, rows, THRESHOLDS)
    assert out["label"] == "DIRECTIONAL"
    assert out["spread"] > 0.40


def test_robust_mean_only_when_hedged_unavailable():
    """Mean-Sharpe plateaus and lifts, but hedged Sharpe is NaN (SPY-align < 30 obs):
    must surface as ROBUST_MEAN_ONLY with an explicit caveat — never silent FRAGILE."""
    base = _metrics(0.40, NAN)
    rows = {t: _metrics(0.72, NAN) for t in THRESHOLDS}
    out = classify_robustness(base, rows, THRESHOLDS)
    assert out["label"] == "ROBUST_MEAN_ONLY"
    assert not out["hedged_ok"]
    assert "HEDGED UNAVAILABLE" in out["verdict"]


def test_inconclusive_when_mean_sharpe_nonfinite():
    """A non-finite mean-Sharpe (a CPCV metric failed to populate) must be INCONCLUSIVE,
    not silently bucketed as FRAGILE."""
    base = _metrics(NAN, 0.30)
    rows = {t: _metrics(0.70, 0.50) for t in THRESHOLDS}
    out = classify_robustness(base, rows, THRESHOLDS)
    assert out["label"] == "INCONCLUSIVE"
    assert not out["mean_ok"]


def test_fragile_hedged_unavailable_does_not_claim_hedged():
    """When the mean-Sharpe lift itself is not robust AND hedged is unavailable, the
    FRAGILE message must say it judged on mean-Sharpe only (not pretend it checked hedged)."""
    base = _metrics(0.40, NAN)
    rows = {0.75: _metrics(0.30, NAN),    # Δ -0.10
            1.0: _metrics(0.95, NAN),     # Δ +0.55
            1.25: _metrics(0.42, NAN)}    # Δ +0.02
    out = classify_robustness(base, rows, THRESHOLDS)
    assert out["label"] == "FRAGILE"
    assert "hedged Sharpe unavailable" in out["verdict"]


# ───────────────────────── 3. _fin helper ──────────────────────────────────────

@pytest.mark.parametrize("val,expected", [
    (0.0, True), (1.5, True), (-3.2, True),
    (NAN, False), (float("inf"), False), (float("-inf"), False), (None, False), ("x", False),
])
def test_fin(val, expected):
    assert _fin(val) is expected
