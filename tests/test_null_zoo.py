"""GL-0 null-strategy-zoo tests (app/research/null_zoo.py).

Guards the detector itself: a permutation that leaks the real signal, a Track-B replication that
diverges from the canonical gate, or a Deflated-Sharpe that isn't monotone in the trial count would
all make the verdict untrustworthy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import null_zoo as nz
from app.research import futures_carry as fc


def _panels(n_days=260, n_mkts=5, seed=1):
    """Small synthetic panels with a REAL cross-sectional carry edge (lagged carry sign predicts
    next-day return), an independent base book, and a momentum signal off synthetic prices."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    mkts = [f"M{i}" for i in range(n_mkts)]
    # persistent carry signal (AR-ish)
    carry = pd.DataFrame(np.cumsum(rng.normal(0, 0.1, (n_days, n_mkts)), axis=0),
                         index=idx, columns=mkts)
    carry = (carry - carry.mean()) / carry.std()
    # returns: lagged carry edge + noise -> carry_backtest should earn alpha
    edge = 0.0008 * np.sign(carry.shift(1).fillna(0.0).to_numpy())
    rets = pd.DataFrame(edge + rng.normal(0, 0.01, (n_days, n_mkts)), index=idx, columns=mkts)
    prices = 100.0 * (1.0 + rets).cumprod()
    roll_days = pd.DataFrame(False, index=idx, columns=mkts)
    base = pd.Series(rng.normal(0.0003, 0.008, n_days), index=idx, name="base")  # independent
    return rets, carry, prices, roll_days, base


# ---- cross_sectional_permute ----
def test_cross_sectional_permute_shape_and_determinism():
    _, carry, _, _, _ = _panels()
    a = nz.cross_sectional_permute(carry, np.random.default_rng(0))
    b = nz.cross_sectional_permute(carry, np.random.default_rng(0))
    assert a.shape == carry.shape
    pd.testing.assert_frame_equal(a, b)               # seeded -> deterministic


def test_cross_sectional_permute_preserves_per_date_distribution():
    _, carry, _, _, _ = _panels()
    out = nz.cross_sectional_permute(carry, np.random.default_rng(3))
    for i in range(0, len(carry), 50):                # each row is a permutation of its own values
        assert sorted(out.iloc[i].values) == sorted(carry.iloc[i].values)


def test_cross_sectional_permute_preserves_nan_present_set():
    """The BLOCKER fix: NaN cells must stay NaN (the per-date present-set is preserved exactly), so
    the null book is built on the SAME effective universe as the observed book."""
    _, carry, _, _, _ = _panels()
    c = carry.copy()
    c.iloc[5, 0] = np.nan                              # market M0 absent on day 5
    c.iloc[6, [1, 2]] = np.nan                         # M1,M2 absent on day 6
    out = nz.cross_sectional_permute(c, np.random.default_rng(7))
    # exact same missingness mask
    assert (out.isna().to_numpy() == c.isna().to_numpy()).all()
    # present values on a partially-present row are a permutation of the present originals
    present_in = sorted(c.iloc[6].dropna().values)
    present_out = sorted(out.iloc[6].dropna().values)
    assert present_in == present_out


def test_circular_time_shift_is_a_roll():
    _, carry, _, _, _ = _panels()
    out = nz.circular_time_shift(carry, np.random.default_rng(5), min_shift=30)
    assert out.shape == carry.shape
    # values are conserved (a roll permutes rows)
    assert np.isclose(np.sort(out.to_numpy(), axis=None),
                      np.sort(carry.to_numpy(), axis=None)).all()


# ---- track_b_stat ----
def test_track_b_stat_degenerate_returns_nan():
    # a constant (zero-vol) candidate -> (nan, nan) so degenerate draws are EXCLUDED from the null
    # distribution rather than piling mass at t=0 (which would bias the empirical p downward)
    _, _, _, _, base = _panels()
    t, sr = nz.track_b_stat(pd.Series(0.0, index=base.index), base)
    assert np.isnan(t) and np.isnan(sr)


def test_track_b_stat_matches_multifactor_alpha_path():
    from app.research.inference import multifactor_alpha
    from scripts.walkforward.book_gate import _vol_target_candidate, ANN
    _, _, _, _, base = _panels()
    rng = np.random.default_rng(7)
    cand = pd.Series(0.3 * base.to_numpy() + rng.normal(0, 0.006, len(base)),
                     index=base.index)
    t_mine, sr_mine = nz.track_b_stat(cand, base)
    # replicate the canonical path directly
    aligned = pd.concat([base.rename("base"), cand.rename("cand")], axis=1, join="inner").dropna()
    cand_vt, _ = _vol_target_candidate(aligned["cand"], float(aligned["base"].std() * np.sqrt(ANN)))
    ev = pd.DataFrame({"base": aligned["base"], "cand": cand_vt}).dropna()
    mfa = multifactor_alpha(ev["cand"], ev[["base"]], hac_lag=5)
    assert abs(t_mine - mfa["t_alpha_hac"]) < 1e-9
    assert abs(sr_mine - mfa["resid_sharpe"]) < 1e-9


# ---- deflated_sharpe ----
def test_deflated_sharpe_in_unit_interval_and_monotone_in_trials():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0006, 0.01, 1500))     # SR ~0.95 ann
    d1 = nz.deflated_sharpe(r, n_trials=1, var_sr_trials=1e-4)
    d20 = nz.deflated_sharpe(r, n_trials=20, var_sr_trials=1e-4)
    d100 = nz.deflated_sharpe(r, n_trials=100, var_sr_trials=1e-4)
    for d in (d1, d20, d100):
        assert 0.0 <= d <= 1.0
    assert d1 >= d20 >= d100                            # more trials -> harder bar -> lower DSR


def test_deflated_sharpe_higher_for_stronger_sharpe():
    rng = np.random.default_rng(13)
    weak = pd.Series(rng.normal(0.0001, 0.01, 1500))
    strong = pd.Series(rng.normal(0.0010, 0.01, 1500))
    assert nz.deflated_sharpe(strong, 20, 1e-4) > nz.deflated_sharpe(weak, 20, 1e-4)


# ---- _past_max_diff ----
def test_past_max_diff_detects_only_past_changes():
    idx = pd.bdate_range("2020-01-01", periods=100)
    a = pd.Series(np.arange(100.0), index=idx)
    cut = idx[80]
    same, d = nz._past_max_diff(a, a.copy(), cut)
    assert same and d == 0.0
    b = a.copy()
    b.iloc[10] += 1.0                                   # a PAST change
    same, d = nz._past_max_diff(a, b, cut)
    assert (not same) and d == 1.0
    c = a.copy()
    c.iloc[90] += 5.0                                   # a FUTURE change (>= cut) -> ignored
    same, d = nz._past_max_diff(a, c, cut)
    assert same and d == 0.0


# ---- run_null_zoo end-to-end ----
def test_run_null_zoo_end_to_end_structure_and_determinism():
    rets, carry, prices, roll_days, base = _panels()
    from app.research import futures_factors as ff
    mom = ff.xs_momentum_signal(prices, lookback=60, skip=5)
    cfg = fc.CarryConfig(roll_cost_bps=3.0, vol_lookback=20, min_xs_width=2)
    r1 = nz.run_null_zoo(rets, carry, mom, roll_days, base, prices=prices, cfg=cfg, n_nulls=12, seed=0)
    r2 = nz.run_null_zoo(rets, carry, mom, roll_days, base, prices=prices, cfg=cfg, n_nulls=12, seed=0)
    assert r1.verdict in {"BASKET_REAL", "CARRY_ONLY", "RESIDUE"}
    assert r1.verdict == r2.verdict and abs(r1.null_book_p - r2.null_book_p) < 1e-12  # seeded
    assert np.isfinite(r1.t_obs_book) and np.isfinite(r1.t_obs_carry)
    assert 0.0 <= r1.null_book_p <= 1.0 and 0.0 <= r1.dsr_n20 <= 1.0


def test_look_ahead_audit_pit_clean_on_well_formed_panels():
    rets, carry, prices, roll_days, _ = _panels()
    from app.research import futures_factors as ff
    mom = ff.xs_momentum_signal(prices, lookback=60, skip=5)
    cfg = fc.CarryConfig(roll_cost_bps=3.0, vol_lookback=20, min_xs_width=2)
    out = nz.look_ahead_audit(rets, carry, mom, roll_days, prices=prices, cfg=cfg,
                              corrupt_tail_days=30)
    assert out["pit_clean"] is True
    assert out["carry_engine_future_blind"] and out["xsmom_signal_future_blind"]
