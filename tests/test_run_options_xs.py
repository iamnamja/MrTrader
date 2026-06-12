"""
Tests for scripts/run_options_xs_cpcv.py — the P4 confirmatory runner logic
(the non-I/O pieces: rebalance grid, forward weekly returns, the three primary
tests on synthetic known-edge / null panels, the frozen verdict rule, and the
CPCV backstop). The data loaders (pit_union / _get_daily_bars) are integration-
tested by the smoke build, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_options_xs_cpcv as R


# ───────────────────────── rebalance grid ────────────────────────────────────

def test_weekly_rebalance_dates_are_mondays_in_range():
    from datetime import date
    days = R.weekly_rebalance_dates(date(2024, 3, 1), date(2024, 3, 31))
    assert all(d.weekday() == 0 for d in days)           # all Mondays
    assert days[0] == pd.Timestamp("2024-03-04")          # first Monday
    assert days[-1] == pd.Timestamp("2024-03-25")


# ───────────────────────── forward weekly returns ────────────────────────────

def test_forward_weekly_returns_pit_and_value():
    idx = pd.bdate_range("2024-03-01", periods=30)
    closes = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    rebal = R.weekly_rebalance_dates(idx[0].date(), idx[-1].date())
    fwd = R._forward_weekly_returns(closes, rebal)
    # Each value is close[next Monday]/close[this Monday] - 1, all positive (rising).
    assert all(v > 0 for v in fwd.values())
    d0, d1 = rebal[0], rebal[1]
    want = float(closes.loc[:d1].iloc[-1]) / float(closes.loc[:d0].iloc[-1]) - 1.0
    assert fwd[d0] == pytest.approx(want)
    # The LAST rebalance has no "next" -> absent.
    assert rebal[-1] not in fwd


def test_forward_weekly_returns_empty_on_no_history():
    assert R._forward_weekly_returns(pd.Series(dtype=float), []) == {}


# ───────────────────────── the three primary tests ───────────────────────────

def _known_edge_panel(direction=+1, seed=0, n_weeks=60, n_names=120, strength=0.012):
    """Synthetic panel with a REAL monotone signal->forward-return edge in
    `direction`, low-noise positive weekly spread, and benign factors."""
    rng = np.random.default_rng(seed)
    weeks = pd.bdate_range("2023-01-02", periods=n_weeks, freq="W-MON")
    weekly_rows, pooled_rows = [], []
    for w in weeks:
        sig = rng.uniform(0, 1, n_names)
        # forward return increases (direction +1) or decreases (-1) in the signal.
        y = direction * strength * (sig - 0.5) + rng.normal(0, 0.002, n_names)
        for i in range(n_names):
            pooled_rows.append({"week": w, "symbol": f"S{i:03d}",
                                "signal": float(sig[i]), "fwd_ret": float(y[i])})
        # Decile L/S spread for this week ~ top-bottom mean (positive by construction).
        order = np.argsort(sig)
        bottom = y[order[:n_names // 10]].mean()
        top = y[order[-n_names // 10:]].mean()
        spread = (top - bottom) if direction > 0 else (bottom - top)
        weekly_rows.append({"week": w, "ls_spread_gross": spread,
                            "ls_spread_net": spread - 0.0002, "turnover": 1.0,
                            "n_long": n_names // 10, "n_short": n_names // 10,
                            "n_qualified": n_names})
    weekly = pd.DataFrame(weekly_rows)
    pooled = pd.DataFrame(pooled_rows)
    factors = pd.DataFrame({
        "SPY": rng.normal(0, 0.01, n_weeks),
        "IWM_SPY": rng.normal(0, 0.005, n_weeks),
        "VIXY": rng.normal(0, 0.02, n_weeks),
    }, index=weeks)
    return weekly, pooled, factors


def test_primary_tests_detect_a_real_edge_and_verdict_pass():
    weekly, pooled, factors = _known_edge_panel(direction=+1, seed=1)
    t = R.run_primary_tests(weekly, pooled, factors, direction=+1)
    assert t["week_clustered_t"] > 2.0 and t["p_one_sided"] < 0.025
    assert t["decile_dir_ok"] is True
    assert t["alpha_positive"] is True
    assert R.verdict(t) == "PASS"


def test_primary_tests_respect_negative_direction():
    # direction -1: high signal -> LOW return; the book longs low-signal names.
    weekly, pooled, factors = _known_edge_panel(direction=-1, seed=2)
    t = R.run_primary_tests(weekly, pooled, factors, direction=-1)
    assert t["week_clustered_t"] > 2.0
    assert t["decile_dir_ok"] is True                 # monotone in the -1 direction
    assert np.sign(t["decile_spearman_rho"]) == -1    # signal anti-correlated w/ y
    assert R.verdict(t) == "PASS"


def test_primary_tests_null_panel_is_killed():
    rng = np.random.default_rng(9)
    weeks = pd.bdate_range("2023-01-02", periods=60, freq="W-MON")
    pooled = pd.DataFrame([
        {"week": w, "symbol": f"S{i:03d}", "signal": rng.uniform(),
         "fwd_ret": rng.normal(0, 0.01)}                # signal independent of y
        for w in weeks for i in range(120)])
    weekly = pd.DataFrame([
        {"week": w, "ls_spread_gross": rng.normal(0, 0.01),
         "ls_spread_net": rng.normal(0, 0.01), "turnover": 1.0,
         "n_long": 12, "n_short": 12, "n_qualified": 120} for w in weeks])
    factors = pd.DataFrame({"SPY": rng.normal(0, 0.01, 60)}, index=weeks)
    t = R.run_primary_tests(weekly, pooled, factors, direction=+1)
    assert R.verdict(t) == "KILL"                      # no significant edge


# ───────────────────────── verdict rule ──────────────────────────────────────

def test_verdict_requires_all_three():
    base = {"week_clustered_t": 3.0, "p_one_sided": 0.001,
            "decile_dir_ok": True, "alpha_positive": True}
    assert R.verdict(base) == "PASS"
    assert R.verdict({**base, "week_clustered_t": 1.5}) == "KILL"   # t fails
    assert R.verdict({**base, "decile_dir_ok": False}) == "KILL"    # mono fails
    assert R.verdict({**base, "alpha_positive": False}) == "KILL"   # alpha fails
    assert R.verdict({"error": "no weeks built"}) == "INCONCLUSIVE"


# ───────────────────────── CPCV backstop ─────────────────────────────────────

def test_cpcv_backstop_reports_fold_sharpes():
    weekly = pd.DataFrame({
        "week": pd.bdate_range("2023-01-02", periods=40, freq="W-MON"),
        "ls_spread_net": np.linspace(0.001, 0.004, 40),
    })
    bs = R.cpcv_backstop(weekly, n_folds=4)
    assert bs["available"] is True
    assert len(bs["fold_sharpes"]) == 4
    assert "worst_fold_sharpe" in bs and "mean_fold_sharpe" in bs


def test_cpcv_backstop_unavailable_when_too_short():
    weekly = pd.DataFrame({"week": pd.bdate_range("2023-01-02", periods=5,
                                                  freq="W-MON"),
                           "ls_spread_net": np.zeros(5)})
    assert R.cpcv_backstop(weekly, n_folds=4)["available"] is False


# ───────────────────────── decision mapping (R4 safety) ──────────────────────

def test_verdict_to_decision_are_valid_registry_members():
    from app.research.registry import DECISIONS
    for v in ("PASS", "KILL", "INCONCLUSIVE"):
        assert R.verdict_to_decision(v) in DECISIONS
    assert R.verdict_to_decision("PASS") == "promote_paper"
    assert R.verdict_to_decision("KILL") == "kill"


# ───────────────────────── monotonicity is TREND, not strict ─────────────────

def test_decile_dir_ok_is_trend_not_strict_monotone():
    # Decile-mean forward returns trend UP in the signal but with ONE reversal
    # (D6 > D7): strict step-monotonicity is False, yet the gradient is clearly
    # in-direction (sign(rho)=+1, top-minus-bottom>0) -> decile_dir_ok True.
    rng = np.random.default_rng(3)
    means = np.array([0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.0055,
                      0.007, 0.008])
    weeks = pd.bdate_range("2023-01-02", periods=8, freq="W-MON")
    pooled_rows, weekly_rows = [], []
    for w in weeks:
        for dec in range(10):
            for j in range(15):
                pooled_rows.append({
                    "week": w, "symbol": f"D{dec}_{j}",
                    "signal": dec / 10.0 + rng.uniform(0, 0.09),
                    "fwd_ret": float(means[dec] + rng.normal(0, 0.0004))})
        weekly_rows.append({"week": w, "ls_spread_net": means[-1] - means[0],
                            "ls_spread_gross": means[-1] - means[0],
                            "turnover": 1.0})
    t = R.run_primary_tests(pd.DataFrame(weekly_rows), pd.DataFrame(pooled_rows),
                            pd.DataFrame({"SPY": rng.normal(0, 0.01, 8)},
                                         index=weeks), direction=+1)
    assert t["decile_is_monotone_strict"] is False   # the D6>D7 reversal
    assert t["decile_dir_ok"] is True                # trend is in-direction


def test_primary_tests_survive_sparse_pooled_panel():
    weeks = pd.bdate_range("2023-01-02", periods=30, freq="W-MON")
    pooled = pd.DataFrame([{"week": weeks[0], "symbol": f"S{i}",
                            "signal": float(i), "fwd_ret": 0.01 * i}
                           for i in range(5)])     # < 10 -> decile_report would raise
    weekly = pd.DataFrame([{"week": w, "ls_spread_net": 0.001,
                            "ls_spread_gross": 0.001, "turnover": 1.0}
                           for w in weeks])
    t = R.run_primary_tests(weekly, pooled,
                            pd.DataFrame({"SPY": np.zeros(30)}, index=weeks),
                            direction=+1)           # must NOT raise
    assert t["decile_dir_ok"] is False


# ───────────────────────── PIT exclusion (safety-critical) ───────────────────

def test_latest_known_features_excludes_not_yet_knowable():
    feats = pd.DataFrame([
        {"underlying": "A", "date": pd.Timestamp("2024-03-01"),
         "knowable_date": pd.Timestamp("2024-03-04"), "cpiv_matched_delta": 0.1},
        {"underlying": "A", "date": pd.Timestamp("2024-03-11"),
         "knowable_date": pd.Timestamp("2024-03-12"), "cpiv_matched_delta": 0.9},
    ])
    # as_of Monday 03-11: the 03-11 chain is knowable 03-12 (Tue) -> EXCLUDED;
    # the freshest usable signal is the 03-01 row (knowable 03-04).
    out = R._latest_known_features(feats, ["A"], pd.Timestamp("2024-03-11"))
    assert out.loc["A", "cpiv_matched_delta"] == 0.1
    # as_of 03-12: the fresh row is now knowable.
    out2 = R._latest_known_features(feats, ["A"], pd.Timestamp("2024-03-12"))
    assert out2.loc["A", "cpiv_matched_delta"] == 0.9
