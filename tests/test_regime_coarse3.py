"""Phase 2 — Regime gate overhaul tests.

Covers:
  - coarse3 BULL/BEAR/NEUTRAL labeling
  - Look-ahead regression (MEDIUM-4): expanding-quantile VIX threshold means the
    label for date t is identical whether the series ends at t or extends further
  - Warmup window → NEUTRAL
  - compute_regime_sharpes drops buckets < REGIME_MIN_OBS
  - worst_regime_sharpe=None + ALLOW_NO_REGIME_GATE flag behaviour (WF + CPCV)
  - MIN_WORST_REGIME_SHARPE boundary
  - legacy16 backward-compat
"""
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward.regime import (
    _load_coarse3, _load_legacy16, compute_regime_sharpes,
)
from scripts.walkforward.gates import WalkForwardReport, FoldResult
from scripts.walkforward.cpcv import CPCVResult
from app.ml.retrain_config import REGIME_VIX_WARMUP_DAYS, REGIME_MIN_OBS


def _series(values, start="2020-01-01"):
    idx = pd.date_range(start=start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def _coarse3(spy_vals, vix_vals, start="2020-01-01"):
    spy = _series(spy_vals, start)
    vix = _series(vix_vals, start)
    common = spy.index
    return _load_coarse3(spy, vix, common, common[0].date(), common[-1].date())


# ── 1. BULL/BEAR/NEUTRAL labeling ──────────────────────────────────────────────
def test_coarse3_labels_bull_bear_neutral():
    n = 300
    # Build a regime map and assert a clearly-bull day, a clearly-bear day, and a
    # neutral day get the expected labels.
    # BULL phase: SPY strongly rising (well above MAs), VIX very low.
    spy_bull = [100.0 + i * 0.5 for i in range(n)]
    # Constant low VIX: expanding p25 == p75 == 10, last value 10 satisfies
    # vix_val <= vix_lo (BULL) without tripping vix_val >= vix_hi first only if
    # the SPY-below-MA-bear BEAR branch is false. Use a slightly declining VIX so
    # the final value sits strictly at/below the lower quantile.
    vix_bull = list(np.linspace(15.0, 10.0, n))  # VIX trending down → last is min
    rmap = _coarse3(spy_bull, vix_bull)
    last_day = sorted(rmap.keys())[-1]
    assert rmap[last_day] == "BULL"

    # BEAR phase: SPY falling below 200d MA.
    spy_bear = [300.0 - i * 0.5 for i in range(n)]
    vix_bear = [18.0] * n
    rmap_bear = _coarse3(spy_bear, vix_bear)
    last_bear = sorted(rmap_bear.keys())[-1]
    assert rmap_bear[last_bear] == "BEAR"

    # BEAR via high VIX even with flat SPY.
    spy_flat = [100.0] * n
    vix_spike = [15.0] * (n - 1) + [80.0]
    rmap_spike = _coarse3(spy_flat, vix_spike)
    last_spike = sorted(rmap_spike.keys())[-1]
    assert rmap_spike[last_spike] == "BEAR"


def test_coarse3_neutral_label():
    n = 300
    # SPY flat (right at MA), VIX mid-range → NEUTRAL (not extreme either way).
    spy = [100.0 + np.sin(i / 10.0) * 0.2 for i in range(n)]
    vix = [20.0] * n  # constant → percentiles all equal; not <= p25 strictly below
    rmap = _coarse3(spy, vix)
    # With constant VIX, vix_lo == vix_hi == vix_val so neither BULL (needs <= lo
    # AND above MA-bull) nor high-VIX BEAR fires cleanly; many days NEUTRAL.
    labels = set(rmap.values())
    assert "NEUTRAL" in labels


# ── 2. Look-ahead regression (MEDIUM-4) ────────────────────────────────────────
def test_coarse3_no_lookahead():
    """Label for date t must not change when future data is appended."""
    n = 250
    rng = np.random.default_rng(42)
    spy_vals = list(100.0 + np.cumsum(rng.normal(0.05, 1.0, n + 200)))
    vix_vals = list(20.0 + rng.normal(0, 5.0, n + 200).clip(-10, 40))

    short = _coarse3(spy_vals[:n], vix_vals[:n])
    long = _coarse3(spy_vals, vix_vals)

    # Every date present in the short map must have an identical label in the long map.
    shared = set(short.keys()) & set(long.keys())
    assert len(shared) > 50  # sanity: meaningful overlap
    mismatches = {d: (short[d], long[d]) for d in shared if short[d] != long[d]}
    assert not mismatches, f"look-ahead leak: {list(mismatches.items())[:5]}"


# ── 3. Warmup ───────────────────────────────────────────────────────────────────
def test_coarse3_warmup_all_neutral():
    n = REGIME_VIX_WARMUP_DAYS + 50
    spy = [100.0 + i * 0.5 for i in range(n)]  # would be BULL but warmup forces NEUTRAL
    vix = [10.0] * n
    rmap = _coarse3(spy, vix)
    days_sorted = sorted(rmap.keys())
    warmup_days = days_sorted[:REGIME_VIX_WARMUP_DAYS]
    assert all(rmap[d] == "NEUTRAL" for d in warmup_days), \
        "first REGIME_VIX_WARMUP_DAYS days must be NEUTRAL"


# ── 4. compute_regime_sharpes drops buckets < REGIME_MIN_OBS ───────────────────
def test_compute_regime_sharpes_min_obs_boundary():
    assert REGIME_MIN_OBS == 20  # boundary assumption for this test
    base = date(2021, 1, 4)
    # Build an equity curve with two regimes: 'BEAR' gets 19 obs (dropped),
    # 'BULL' gets 20 obs (kept). compute_regime_sharpes needs i-1 prior point,
    # so prepend a seed day per regime that is NOT labeled.
    equity_curve = []
    regime_map = {}
    eq = 100.0
    d = base
    # 21 BULL equity points → 20 returns labeled BULL
    for i in range(21):
        equity_curve.append((d, eq))
        regime_map[d] = "BULL"
        eq *= 1.001
        d += timedelta(days=1)
    # 20 BEAR equity points → 19 returns labeled BEAR (first BEAR point's return
    # is computed against the last BULL point and labeled BEAR).
    # To get exactly 19 BEAR returns we add 19 more points (the transition return
    # is labeled by the new day = BEAR).
    for i in range(19):
        equity_curve.append((d, eq))
        regime_map[d] = "BEAR"
        eq *= 0.999
        d += timedelta(days=1)

    res = compute_regime_sharpes(equity_curve, base, d, regime_map=regime_map)
    assert "BULL" in res, "20-obs bucket should be kept"
    assert "BEAR" not in res, "19-obs bucket should be dropped (< REGIME_MIN_OBS)"


# ── 5/6. worst_regime_sharpe=None + ALLOW_NO_REGIME_GATE (WalkForwardReport) ────
def _passing_wf_report(regime_sharpes):
    """Build a WF report that passes everything except possibly regime."""
    folds = []
    for i in range(3):
        folds.append(FoldResult(
            fold=i + 1,
            train_start=date(2020, 1, 1), train_end=date(2021, 1, 1),
            test_start=date(2021, 1, 2), test_end=date(2022, 1, 1),
            trades=50, win_rate=0.6, sharpe=1.5, max_drawdown=0.05,
            total_return=0.20, stop_exit_rate=0.1,
            profit_factor=1.5, calmar_ratio=1.0, n_obs=250,
            regime_sharpes=dict(regime_sharpes),
        ))
    return WalkForwardReport(model_type="test", folds=folds,
                             is_true_walkforward=True)  # P0-3: isolate the regime gate


def test_wf_none_regime_fails_by_default():
    rep = _passing_wf_report({})  # empty → worst_regime_sharpe None
    assert rep.worst_regime_sharpe is None
    with patch("app.ml.retrain_config.ALLOW_NO_REGIME_GATE", False):
        assert rep.gate_passed() is False
        detail = rep.gate_detail()
        assert detail["worst_regime_sharpe"][1] is False


def test_wf_none_regime_bypass_when_flag_true():
    rep = _passing_wf_report({})
    assert rep.worst_regime_sharpe is None
    with patch("app.ml.retrain_config.ALLOW_NO_REGIME_GATE", True):
        # Not blocked by regime — gate should pass (all other metrics OK).
        assert rep.gate_passed() is True
        detail = rep.gate_detail()
        assert detail["worst_regime_sharpe"][1] is True


# ── 7. MIN_WORST_REGIME_SHARPE boundary (WalkForwardReport) ────────────────────
def test_wf_regime_boundary():
    # -0.4 (>= -0.5) passes
    rep_ok = _passing_wf_report({"BULL": -0.4, "BEAR": -0.4})
    assert rep_ok.worst_regime_sharpe == pytest.approx(-0.4)
    assert rep_ok.gate_passed() is True
    # -0.6 fails
    rep_fail = _passing_wf_report({"BULL": 1.0, "BEAR": -0.6})
    assert rep_fail.worst_regime_sharpe == pytest.approx(-0.6)
    assert rep_fail.gate_passed() is False


# ── 8. CPCV parity ──────────────────────────────────────────────────────────────
def _passing_cpcv(worst_regime):
    return CPCVResult(
        model_type="test", n_folds=6, n_paths=2,
        path_sharpes=[1.5, 1.4, 1.6, 1.5, 1.5],
        path_profit_factors=[1.5] * 5,
        path_calmars=[1.0] * 5,
        path_n_obs=[250] * 5,
        worst_regime_sharpe=worst_regime,
        is_true_walkforward=True,  # P0-3: promotable run is true-WF; isolate the regime gate
    )


def test_cpcv_none_regime_fails_by_default():
    res = _passing_cpcv(None)
    with patch("app.ml.retrain_config.ALLOW_NO_REGIME_GATE", False):
        assert res.gate_passed() is False
        assert res.gate_detail()["worst_regime_sharpe"][1] is False


def test_cpcv_none_regime_bypass_when_flag_true():
    res = _passing_cpcv(None)
    with patch("app.ml.retrain_config.ALLOW_NO_REGIME_GATE", True):
        # regime not the blocker; gate should pass on the other metrics.
        assert res.gate_passed() is True
        assert res.gate_detail()["worst_regime_sharpe"][1] is True


# ── 9. legacy16 backward compat ─────────────────────────────────────────────────
def test_legacy16_label_format():
    n = 120
    spy = [100.0 + i * 0.3 for i in range(n)]  # rising → trend U, momentum P
    vix = list(np.linspace(10, 40, n))
    spy_s = _series(spy)
    vix_s = _series(vix)
    common = spy_s.index
    with patch("app.ml.retrain_config.REGIME_SCHEME", "legacy16"):
        rmap = _load_legacy16(spy_s, vix_s, common, common[0].date(), common[-1].date())
    assert rmap, "legacy16 should produce labels"
    # Labels match format <quartile 1-4><U|D><P|N>, e.g. "1UP", "4DN".
    for lbl in rmap.values():
        assert len(lbl) == 3
        assert lbl[0] in "1234"
        assert lbl[1] in "UD"
        assert lbl[2] in "PN"
    # Late rising days should be trend-up momentum-positive.
    last = sorted(rmap.keys())[-1]
    assert rmap[last][1] == "U"
    assert rmap[last][2] == "P"
