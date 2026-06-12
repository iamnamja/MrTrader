"""
Tests for app/research/options_xs_ls.py — the P4 options-as-signal L/S core.

Decile dollar-neutral construction (neutrality, gross, direction, equal-weight,
min-names skip), the H4c conditioned composite, spread-return renormalization on
missing names, the 5-factor frame, and multifactor_alpha (known-alpha recovery +
exact agreement with attribution.capm_alpha on the single-factor case).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research import options_xs_ls as xs
from scripts.preregister_options_xs_features import HYPOTHESES
from scripts.walkforward.attribution import capm_alpha


# ───────────────────────── direction map matches the prereg ──────────────────

def test_feature_direction_covers_all_five_and_matches_prereg_signs():
    feats = {h["params"]["feature"] for h in HYPOTHESES}
    assert set(xs.FEATURE_DIRECTION) == feats
    # The frozen prereg "direction" strings carry POSITIVE/NEGATIVE; the sign map
    # must agree (POSITIVE coeff -> +1 long-high; NEGATIVE -> -1 long-low).
    for h in HYPOTHESES:
        f = h["params"]["feature"]
        d = h["direction"]
        want = +1 if "POSITIVE" in d.split("expected coefficient")[-1] else -1
        assert xs.FEATURE_DIRECTION[f] == want, f"{f}: {d!r}"


# ───────────────────────── decile L/S construction ───────────────────────────

def _signal(n):
    return pd.Series(np.arange(n, dtype=float), index=[f"S{i:03d}" for i in range(n)])


def test_decile_weights_dollar_neutral_high_minus_low_and_equal_weight():
    w = xs.decile_ls_weights(_signal(100), direction=+1, n_deciles=10)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)        # net 0 (dollar-neutral)
    assert w.abs().sum() == pytest.approx(2.0)             # gross = 2 (long +1/short -1)
    longs, shorts = w[w > 0], w[w < 0]
    assert len(longs) == 10 and len(shorts) == 10          # decile each leg
    assert longs.nunique() == 1 and shorts.nunique() == 1  # equal-weight within leg
    assert longs.sum() == pytest.approx(1.0)               # long leg sums +1
    assert longs.iloc[0] == pytest.approx(0.10)            # 1.0 / 10


def test_direction_plus_longs_top_minus_longs_bottom():
    s = _signal(100)
    wp = xs.decile_ls_weights(s, direction=+1)
    wm = xs.decile_ls_weights(s, direction=-1)
    # +1: top decile (highest signal S090..S099) is long.
    assert (wp[wp > 0].index == [f"S{i:03d}" for i in range(90, 100)]).all()
    # -1: bottom decile (S000..S009) is long.
    assert (wm[wm > 0].index == [f"S{i:03d}" for i in range(0, 10)]).all()
    # The two books are exact opposites.
    assert wp.add(wm, fill_value=0.0).abs().sum() == pytest.approx(0.0, abs=1e-12)


def test_decile_skips_when_too_few_names():
    assert xs.decile_ls_weights(_signal(xs.MIN_NAMES_FOR_DECILES - 1),
                                direction=+1).empty


def test_decile_handles_uneven_universe_size():
    # 47 names: deciles aren't equal-sized, but the book stays neutral + gross 2.
    w = xs.decile_ls_weights(_signal(47), direction=+1)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert w.abs().sum() == pytest.approx(2.0)
    assert (w > 0).any() and (w < 0).any()


# ───────────────────────── H4c conditioned composite ─────────────────────────

def test_build_signal_passthrough_for_plain_feature():
    feats = pd.DataFrame({"cpiv_matched_delta": [0.1, np.nan, 0.3]},
                         index=["A", "B", "C"])
    sig = xs.build_signal(feats, "cpiv_matched_delta")
    assert list(sig.index) == ["A", "C"]      # NaN row dropped
    assert sig.loc["A"] == 0.1


def test_build_signal_h4c_composite_high_when_both_high():
    # D has the highest O/S AND the highest put/call -> highest composite.
    feats = pd.DataFrame({
        "opt_share_volume_ratio": [1.0, 2.0, 3.0, 9.0],
        "put_call_volume_ratio":  [0.5, 1.0, 1.5, 9.0],
    }, index=["A", "B", "C", "D"])
    sig = xs.build_signal(feats, xs.CONDITIONED_FEATURE)
    assert sig.idxmax() == "D"
    assert sig.idxmin() == "A"
    assert (sig <= 1.0).all() and (sig >= 0.0).all()  # avg of two pct-ranks


# ───────────────────────── spread return + renormalization ───────────────────

def test_ls_spread_return_weighted_sum():
    w = pd.Series({"A": 0.25, "B": 0.25, "C": -0.25, "D": -0.25})
    r = pd.Series({"A": 0.02, "B": 0.04, "C": -0.01, "D": 0.03})
    # long mean 0.03, short mean 0.01 -> spread 0.03 - 0.01 = 0.02
    assert xs.ls_spread_return(w, r) == pytest.approx(0.5 * (0.06) - 0.5 * (0.02))


def test_ls_spread_return_renormalizes_when_a_name_drops():
    w = pd.Series({"A": 0.25, "B": 0.25, "C": -0.25, "D": -0.25})
    r = pd.Series({"A": 0.02, "B": 0.04, "D": 0.05})  # C missing (delisted)
    # high-minus-low: long leg A,B renorm to +0.5 each; short leg = D alone -> -1.0.
    want = 0.5 * 0.02 + 0.5 * 0.04 - 1.0 * 0.05  # = -0.02 (non-degenerate)
    assert want == pytest.approx(-0.02)
    assert xs.ls_spread_return(w, r) == pytest.approx(want)
    # And neutrality is preserved (short leg correctly stayed negative).
    assert xs.ls_spread_return(w, pd.Series({"A": 1.0, "B": 1.0, "D": 1.0})) \
        == pytest.approx(0.0, abs=1e-12)


def test_ls_spread_zero_when_a_whole_leg_vanishes():
    w = pd.Series({"A": 0.5, "C": -0.5})
    r = pd.Series({"A": 0.02})  # entire short leg gone
    assert xs.ls_spread_return(w, r) == 0.0


def test_turnover_union_of_names():
    prev = pd.Series({"A": 0.5, "C": -0.5})
    new = pd.Series({"A": 0.25, "B": 0.25, "C": -0.5})
    # |0.25-0.5| + |0.25-0| + |-0.5+0.5| = 0.25 + 0.25 + 0 = 0.5
    assert xs.turnover(prev, new) == pytest.approx(0.5)


# ───────────────────────── factor frame ──────────────────────────────────────

def _closes(level, n=40, seed=0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(level * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_factor_frame_builds_style_spreads_vs_spy():
    closes = {s: _closes(100, seed=i) for i, s in
              enumerate(["SPY", "IWM", "MTUM", "VLUE", "VIXY"])}
    F = xs.build_factor_frame(closes)
    assert list(F.columns) == ["SPY", "IWM_SPY", "MTUM_SPY", "VLUE_SPY", "VIXY"]
    # IWM_SPY is exactly the return spread IWM - SPY.
    iwm_r = closes["IWM"].pct_change()
    spy_r = closes["SPY"].pct_change()
    exp = (iwm_r - spy_r).reindex(F.index)
    assert np.allclose(F["IWM_SPY"].to_numpy(), exp.to_numpy())


def test_factor_frame_requires_spy_and_omits_missing():
    assert xs.build_factor_frame({"IWM": _closes(100)}).empty  # no SPY
    F = xs.build_factor_frame({"SPY": _closes(100, seed=1),
                               "VIXY": _closes(20, seed=2)})
    assert "SPY" in F.columns and "VIXY" in F.columns
    assert "IWM_SPY" not in F.columns                          # IWM absent
    assert F.attrs.get("missing_factors") == ["IWM", "MTUM", "VLUE"]


# ───────────────────────── multifactor alpha ─────────────────────────────────

def test_multifactor_alpha_recovers_known_alpha():
    idx = pd.bdate_range("2023-01-01", periods=300)
    rng = np.random.default_rng(11)
    F = pd.DataFrame({
        "SPY": rng.normal(0, 0.01, len(idx)),
        "IWM_SPY": rng.normal(0, 0.005, len(idx)),
        "VIXY": rng.normal(0, 0.02, len(idx)),
    }, index=idx)
    true_alpha = 0.0003  # 3 bps/day
    betas = np.array([0.8, 0.3, -0.1])
    y = true_alpha + F.to_numpy() @ betas + rng.normal(0, 0.001, len(idx))
    out = xs.multifactor_alpha(pd.Series(y, index=idx), F, hac_lag=5)
    assert out["alpha_bps_d"] == pytest.approx(true_alpha * 1e4, abs=0.5)
    assert out["betas"]["SPY"] == pytest.approx(0.8, abs=0.05)
    assert out["t_alpha_hac"] > 3.0          # 3bps/day with tiny noise is strong
    assert 0.0 <= out["r2"] <= 1.0


def test_multifactor_alpha_matches_capm_on_single_factor():
    idx = pd.bdate_range("2023-01-01", periods=200)
    rng = np.random.default_rng(7)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    y = pd.Series(0.0002 + 0.7 * spy.to_numpy() + rng.normal(0, 0.003, len(idx)),
                  index=idx)
    mf = xs.multifactor_alpha(y, spy.to_frame("SPY"), hac_lag=5)
    cm = capm_alpha(y, spy, hac_lag=5)
    assert mf["alpha_bps_d"] == pytest.approx(cm["alpha_bps_d"], rel=1e-6)
    assert mf["betas"]["SPY"] == pytest.approx(cm["beta"], rel=1e-6)
    assert mf["t_alpha_hac"] == pytest.approx(cm["t_alpha_hac"], rel=1e-6)
    assert mf["t_alpha_ols"] == pytest.approx(cm["t_alpha_ols"], rel=1e-6)


def test_multifactor_alpha_zero_filled_when_too_few_obs():
    idx = pd.bdate_range("2024-01-01", periods=10)
    F = pd.DataFrame({"SPY": np.zeros(10)}, index=idx)
    out = xs.multifactor_alpha(pd.Series(np.zeros(10), index=idx), F)
    assert out["n"] == 10 and out["t_alpha_hac"] == 0.0 and out["betas"] == {}
