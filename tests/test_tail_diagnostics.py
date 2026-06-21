"""GL-1 tail-diagnostics tests (app/research/tail_diagnostics.py).

Guards the detectors: an exceedance-correlation that doesn't condition correctly, a down/up beta
that mixes the regimes, or a crisis-replay that mis-compounds would all corrupt the VRP / defensive-
sleeve verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import tail_diagnostics as td


def _idx(n):
    return pd.bdate_range("2010-01-01", periods=n)


def test_exceedance_correlation_picks_up_stress_clustering():
    rng = np.random.default_rng(0)
    n = 1000
    idx = _idx(n)
    cond = pd.Series(rng.normal(0, 1, n), index=idx)               # conditioner (e.g. SPY)
    worst = cond <= cond.quantile(0.05)
    nw = int(worst.sum())
    # two sleeves uncorrelated normally, but on the worst days they CO-MOVE via a shared shock
    # (tail clustering -> high stress-conditional correlation)
    a = pd.Series(rng.normal(0, 1, n), index=idx)
    b = pd.Series(rng.normal(0, 1, n), index=idx)
    shock = rng.normal(0, 2.0, nw)
    a.loc[worst] = shock + rng.normal(0, 0.3, nw)
    b.loc[worst] = shock + rng.normal(0, 0.3, nw)
    sleeves = pd.DataFrame({"a": a, "b": b})
    uncond = float(sleeves.corr().iloc[0, 1])
    exc = td.exceedance_correlation(sleeves, cond, 0.05)
    assert exc["n_days"] == int(np.floor(0.05 * n)) or abs(exc["n_days"] - 0.05 * n) <= 1
    assert exc["avg_offdiag"] > uncond + 0.3            # tail correlation >> unconditional


def test_down_up_beta_recovers_asymmetry():
    rng = np.random.default_rng(1)
    n = 2000
    idx = _idx(n)
    mkt = pd.Series(rng.normal(0, 0.01, n), index=idx)
    # sleeve = 0.6*mkt on down days, 0.1*mkt on up days -> down_beta 0.6, up_beta 0.1
    s = np.where(mkt < 0, 0.6 * mkt, 0.1 * mkt) + rng.normal(0, 1e-5, n)
    res = td.down_up_beta(pd.Series(s, index=idx), mkt)
    assert abs(res["down_beta"] - 0.6) < 0.05
    assert abs(res["up_beta"] - 0.1) < 0.05
    assert res["asymmetry"] > 0.4                       # down >> up = negative convexity


def test_crisis_replay_compounds_per_window():
    idx = _idx(60)
    s = pd.Series(0.0, index=idx)
    s.loc["2010-01-04":"2010-01-08"] = 0.10             # +10%/day for 5 days in the window
    sleeves = pd.DataFrame({"x": s})
    out = td.crisis_replay(sleeves, {"W": ("2010-01-04", "2010-01-08")})
    assert abs(out.loc["W", "x"] - (1.10 ** 5 - 1.0)) < 1e-9


def test_inverse_vol_book_weights_inverse_to_vol():
    idx = _idx(500)
    rng = np.random.default_rng(2)
    lo = pd.Series(rng.normal(0, 0.005, 500), index=idx)    # low vol
    hi = pd.Series(rng.normal(0, 0.02, 500), index=idx)     # 4x vol
    book = td.inverse_vol_book(pd.DataFrame({"lo": lo, "hi": hi}))
    # recover the weights by regressing the book on the two sleeves; inverse-vol => w_lo/w_hi ~
    # vol_hi/vol_lo ~ 4 (the low-vol sleeve gets ~4x the weight)
    X = np.column_stack([lo.to_numpy(), hi.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, book.to_numpy(), rcond=None)
    assert 3.0 < coef[0] / coef[1] < 5.0


def test_run_tail_diagnostics_end_to_end_and_flags():
    rng = np.random.default_rng(3)
    n = 1500
    idx = _idx(n)
    spy = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    trend = pd.Series(-0.2 * spy.to_numpy() + rng.normal(0, 0.005, n), index=idx)  # crisis-convex
    carry = pd.Series(0.3 * spy.to_numpy() + rng.normal(0, 0.006, n), index=idx)   # risk-on
    xsmom = pd.Series(0.1 * spy.to_numpy() + rng.normal(0, 0.006, n), index=idx)
    vrp = pd.Series(np.where(spy < 0, 1.0 * spy, 0.2 * spy) + rng.normal(0, 0.004, n),
                    index=idx)                                                     # short-crisis
    sleeves = pd.DataFrame({"trend": trend, "carry": carry, "xsmom": xsmom, "vrp": vrp})
    res = td.run_tail_diagnostics(sleeves, spy, vrp_col="vrp")
    assert set(res.sleeves) == {"trend", "carry", "xsmom", "vrp"}
    assert np.isfinite(res.spy_exceedance[0.05])
    assert isinstance(res.one_bet, bool) and isinstance(res.vrp_worsens_tail, bool)
    # VRP here is explicitly short-crisis -> must be flagged as worsening the tail
    assert res.vrp_worsens_tail is True
    # determinism
    res2 = td.run_tail_diagnostics(sleeves, spy, vrp_col="vrp")
    assert res.one_bet == res2.one_bet and res.vrp_worsens_tail == res2.vrp_worsens_tail
