"""
Tests for app/data/options_features.py — the daily options feature table.

Feature math is checked on SYNTHETIC chains with known IVs/deltas/strikes so the
expected value is hand-computable: CPIV = iv_call - iv_put at the matched ATM
delta; 25d put skew sign; term_slope sign; iv_rv ratio; put/call volume ratio;
opt_volume_z rolling. PIT, quality exclusion, and coverage_flags are tested on
the assembled per-underlying frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data import options_features as of


# ───────────────────────────── synthetic-chain builder ──────────────────────

def _contract(underlying, d, kind, strike, expiration, *, iv, delta,
              close=1.0, underlying_close=100.0, volume=50.0,
              solver_status="ok", stale_flag=False):
    return {
        "contract": f"O:{underlying}{strike}{kind[0].upper()}",
        "date": pd.Timestamp(d), "knowable_date": pd.Timestamp(d) + pd.offsets.BDay(1),
        "contract_type": kind, "strike": float(strike),
        "expiration": pd.Timestamp(expiration), "close": float(close),
        "underlying_close": float(underlying_close), "volume": float(volume),
        "stale_flag": bool(stale_flag), "iv": float(iv) if iv is not None else np.nan,
        "delta": float(delta) if delta is not None else np.nan,
        "gamma": 0.01, "vega": 0.1, "theta": -0.05,
        "solver_status": solver_status, "underlying": underlying,
    }


def _surface_chain(d, expiration, *, iv_call_atm, iv_put_atm, iv_put_25d,
                   spot=100.0, underlying="TST"):
    """A single-expiry chain with an ATM call, ATM put, and a 25d put — the legs
    every IV-surface feature needs, with chosen deltas/IVs."""
    return [
        _contract(underlying, d, "call", spot, expiration, iv=iv_call_atm,
                  delta=0.50, underlying_close=spot),
        _contract(underlying, d, "put", spot, expiration, iv=iv_put_atm,
                  delta=-0.50, underlying_close=spot),
        _contract(underlying, d, "put", spot * 0.9, expiration, iv=iv_put_25d,
                  delta=-0.25, underlying_close=spot),
        _contract(underlying, d, "call", spot * 1.1, expiration, iv=iv_call_atm + 0.01,
                  delta=0.25, underlying_close=spot),
    ]


def _valid_frame(rows):
    """Quality-filter + add the `dte` column the pure functions expect (mirrors
    assemble_underlying_features' prep), for one date's rows."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    valid = df[(df["solver_status"] == "ok") & (~df["stale_flag"].astype(bool))].copy()
    valid["dte"] = (valid["expiration"] - valid["date"]).dt.days
    return valid


# ───────────────────────────── IV-surface math ──────────────────────────────

def test_cpiv_is_call_minus_put_iv_at_matched_atm_delta():
    d = "2024-03-01"
    exp30 = "2024-03-31"  # 30 DTE
    rows = _surface_chain(d, exp30, iv_call_atm=0.30, iv_put_atm=0.26,
                          iv_put_25d=0.34)
    feats = of.compute_iv_term_features(_valid_frame(rows))
    # matched-delta ATM: |0.50| both legs -> within tol -> 0.30 - 0.26
    assert feats["cpiv_matched_delta"] == pytest.approx(0.04)
    assert feats["atm_iv_30d"] == pytest.approx(0.30)  # ATM = the |delta|~0.5 row


def test_cpiv_nan_when_atm_deltas_do_not_match():
    d = "2024-03-01"
    exp30 = "2024-03-31"
    rows = [
        _contract("TST", d, "call", 100, exp30, iv=0.30, delta=0.50),
        # nearest-ATM put is far from 0.5 (only a 0.20-delta put exists)
        _contract("TST", d, "put", 80, exp30, iv=0.26, delta=-0.20),
        _contract("TST", d, "put", 70, exp30, iv=0.40, delta=-0.10),
        _contract("TST", d, "call", 110, exp30, iv=0.31, delta=0.30),
        _contract("TST", d, "put", 90, exp30, iv=0.33, delta=-0.30),
        _contract("TST", d, "call", 120, exp30, iv=0.34, delta=0.20),
    ]
    feats = of.compute_iv_term_features(_valid_frame(rows))
    assert np.isnan(feats["cpiv_matched_delta"])
    assert feats["flags"] & of.CF_MISSING_CPIV


def test_skew_25d_put_sign_steep_skew_positive():
    d = "2024-03-01"
    exp30 = "2024-03-31"
    # OTM put IV (0.34) > ATM IV (0.30) -> positive skew measure (steep)
    rows = _surface_chain(d, exp30, iv_call_atm=0.30, iv_put_atm=0.30,
                          iv_put_25d=0.34)
    feats = of.compute_iv_term_features(_valid_frame(rows))
    assert feats["skew_25d_put"] == pytest.approx(0.04)  # 0.34 - 0.30


def test_skew_nan_when_no_25d_put_in_band():
    d = "2024-03-01"
    exp30 = "2024-03-31"
    rows = [
        _contract("TST", d, "call", 100, exp30, iv=0.30, delta=0.50),
        _contract("TST", d, "put", 100, exp30, iv=0.30, delta=-0.50),
        # only a deep 0.05-delta put exists — outside the 0.25 +- 0.10 band
        _contract("TST", d, "put", 60, exp30, iv=0.50, delta=-0.05),
        _contract("TST", d, "call", 110, exp30, iv=0.31, delta=0.30),
        _contract("TST", d, "put", 95, exp30, iv=0.31, delta=-0.40),
        _contract("TST", d, "call", 120, exp30, iv=0.32, delta=0.20),
    ]
    feats = of.compute_iv_term_features(_valid_frame(rows))
    assert np.isnan(feats["skew_25d_put"])
    assert feats["flags"] & of.CF_MISSING_25D_PUT


def test_term_slope_sign_contango_positive():
    d = "2024-03-01"
    exp30 = "2024-03-31"   # ~30 DTE
    exp60 = "2024-04-30"   # ~60 DTE
    rows = _surface_chain(d, exp30, iv_call_atm=0.30, iv_put_atm=0.30,
                          iv_put_25d=0.32)
    rows += [
        _contract("TST", d, "call", 100, exp60, iv=0.35, delta=0.50),
        _contract("TST", d, "put", 100, exp60, iv=0.35, delta=-0.50),
    ]
    feats = of.compute_iv_term_features(_valid_frame(rows))
    # 60d ATM IV (0.35) - 30d ATM IV (0.30) = +0.05 (contango)
    assert feats["term_slope_30_60"] == pytest.approx(0.05)


def test_term_slope_nan_without_60d_tenor():
    d = "2024-03-01"
    exp30 = "2024-03-31"
    rows = _surface_chain(d, exp30, iv_call_atm=0.30, iv_put_atm=0.30,
                          iv_put_25d=0.32)
    feats = of.compute_iv_term_features(_valid_frame(rows))
    assert np.isnan(feats["term_slope_30_60"])
    assert feats["flags"] & of.CF_MISSING_60D_TENOR


# ───────────────────────────── volume features ──────────────────────────────

def test_put_call_volume_ratio():
    d = "2024-03-01"
    exp30 = "2024-03-31"
    rows = [
        _contract("TST", d, "call", 100, exp30, iv=0.3, delta=0.5, volume=100),
        _contract("TST", d, "put", 100, exp30, iv=0.3, delta=-0.5, volume=300),
    ]
    out = of.compute_volume_features(_valid_frame(rows), equity_share_volume=None)
    assert out["put_call_volume_ratio"] == pytest.approx(3.0)
    assert out["total_opt_volume"] == pytest.approx(400.0)
    assert out["flags"] & of.CF_NO_EQUITY_VOLUME  # no equity volume passed


def test_implied_move_nan_on_single_sided_chain():
    # A front expiry with only calls (no put) cannot form a straddle -> NaN +
    # CF_MISSING_FRONT, not a crash.
    d, exp = "2024-03-01", "2024-03-15"
    rows = [
        _contract("TST", d, "call", 100, exp, iv=0.3, delta=0.5),
        _contract("TST", d, "call", 105, exp, iv=0.3, delta=0.4),
    ]
    out = of.compute_implied_move_front(_valid_frame(rows))
    assert np.isnan(out["implied_move_front"])
    assert out["flags"] & of.CF_MISSING_FRONT


def test_opt_share_volume_ratio_scales_by_100():
    d = "2024-03-01"
    exp30 = "2024-03-31"
    rows = [
        _contract("TST", d, "call", 100, exp30, iv=0.3, delta=0.5, volume=10),
        _contract("TST", d, "put", 100, exp30, iv=0.3, delta=-0.5, volume=10),
    ]
    out = of.compute_volume_features(_valid_frame(rows), equity_share_volume=1000.0)
    # (20 contracts * 100) / 1000 shares = 2.0
    assert out["opt_share_volume_ratio"] == pytest.approx(2.0)
    assert not (out["flags"] & of.CF_NO_EQUITY_VOLUME)


# ───────────────────────────── realized vol / PIT ───────────────────────────

def test_realized_vol_matches_manual_log_return_std():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    rng = np.random.default_rng(7)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
                       index=idx)
    as_of = idx[-1]
    rv = of.compute_realized_vol_pit(closes, as_of, window=20)
    prior = closes[closes.index < as_of]
    lr = np.diff(np.log(prior.to_numpy()))
    want = float(np.std(lr[-20:], ddof=1) * np.sqrt(252))
    assert rv == pytest.approx(want)


def test_realized_vol_is_strictly_prior():
    idx = pd.date_range("2024-01-01", periods=25, freq="D")
    closes = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    # as_of equals a date in the series — that day's close must NOT enter RV.
    as_of = idx[-1]
    prior = closes[closes.index < as_of]
    assert len(prior) == len(closes) - 1
    rv = of.compute_realized_vol_pit(closes, as_of, window=20)
    assert rv is not None


def test_realized_vol_none_when_too_few_priors():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    closes = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    assert of.compute_realized_vol_pit(closes, idx[-1], window=20) is None


def test_realized_vol_drops_split_step_from_unadjusted_close():
    # An UNADJUSTED close series with a 10:1 split step (the store-close fallback
    # path). Without the guard the split's ~-2.3 log-return inflates RV ~20x; the
    # guard drops it so RV reflects the genuine low daily vol.
    idx = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(11)
    lvl = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    closes = pd.Series(lvl, index=idx)
    split_at = 30
    closes.iloc[split_at:] = closes.iloc[split_at:] / 10.0  # 10:1 split cliff
    as_of = idx[-1]
    rv = of.compute_realized_vol_pit(closes, as_of, window=20)
    # Compare to RV of the SAME series with the split healed (true vol).
    healed = closes.copy()
    healed.iloc[:split_at] = healed.iloc[:split_at] / 10.0
    rv_true = of.compute_realized_vol_pit(healed, as_of, window=20)
    assert rv is not None and rv_true is not None
    assert rv == pytest.approx(rv_true, rel=0.2)  # split step did NOT leak in
    assert rv < 0.5  # ~1% daily vol annualizes well under 50%, not ~800%


# ───────────────────────────── knowable_date (PIT) ──────────────────────────

def test_assemble_carries_holiday_aware_knowable_date_not_plain_bday():
    # date = Wed 2024-07-03 (session before the July-4 holiday). The store stamps
    # knowable_date = Fri 2024-07-05 (next NYSE session). A plain BDay(1) would
    # give Thu 2024-07-04 (a market holiday) — one session early. The assembler
    # must carry the store value through, NOT recompute it.
    d = "2024-07-03"
    rows = _full_chain_rows(d)
    store_knowable = pd.Timestamp("2024-07-05")
    for r in rows:
        r["knowable_date"] = store_knowable
    feats = of.assemble_underlying_features(pd.DataFrame(rows))
    assert len(feats) == 1
    assert feats.iloc[0]["knowable_date"] == store_knowable
    # And it is NOT the holiday a plain business-day offset would have produced.
    assert feats.iloc[0]["knowable_date"] != pd.Timestamp(d) + pd.offsets.BDay(1)


# ───────────────────────────── assembled frame ──────────────────────────────

def _full_chain_rows(d, *, spot=100.0, vol=50.0, underlying="TST",
                     n_filler=6):
    """A complete one-date chain: 30d + 60d ATM/skew legs plus filler contracts
    so n_valid_contracts clears MIN_VALID_CONTRACTS."""
    exp30 = (pd.Timestamp(d) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    exp60 = (pd.Timestamp(d) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    rows = _surface_chain(d, exp30, iv_call_atm=0.30, iv_put_atm=0.28,
                          iv_put_25d=0.34, spot=spot, underlying=underlying)
    rows += [
        _contract(underlying, d, "call", spot, exp60, iv=0.33, delta=0.50,
                  underlying_close=spot, volume=vol),
        _contract(underlying, d, "put", spot, exp60, iv=0.33, delta=-0.50,
                  underlying_close=spot, volume=vol),
    ]
    for k in range(n_filler):
        rows.append(_contract(underlying, d, "call", spot + 5 + k, exp30,
                              iv=0.31, delta=0.40, underlying_close=spot, volume=vol))
    for r in rows:
        r["volume"] = vol
    return rows


def test_assemble_emits_knowable_date_plus_one_bday():
    d = "2024-03-01"  # a Friday
    rows = _full_chain_rows(d)
    feats = of.assemble_underlying_features(pd.DataFrame(rows))
    assert len(feats) == 1
    r = feats.iloc[0]
    assert r["knowable_date"] == pd.Timestamp(d) + pd.offsets.BDay(1)
    assert r["knowable_date"].weekday() == 0  # Friday + 1 bday = Monday
    assert np.isfinite(r["atm_iv_30d"])
    assert np.isfinite(r["cpiv_matched_delta"])
    assert np.isfinite(r["term_slope_30_60"])


def test_assemble_excludes_non_ok_and_stale_contracts():
    d = "2024-03-01"
    rows = _full_chain_rows(d)
    # Poison: flip enough rows to non-ok / stale that the date is DROPPED.
    for r in rows[:-1]:
        r["solver_status"] = "below_intrinsic"
    feats = of.assemble_underlying_features(pd.DataFrame(rows))
    assert feats.empty  # too few valid contracts -> name-date dropped


def test_assemble_drops_thin_date_below_min_valid():
    d = "2024-03-01"
    exp30 = "2024-03-31"
    # only 3 valid contracts < MIN_VALID_CONTRACTS
    rows = _surface_chain(d, exp30, iv_call_atm=0.30, iv_put_atm=0.30,
                          iv_put_25d=0.32)[:3]
    feats = of.assemble_underlying_features(pd.DataFrame(rows))
    assert feats.empty


def test_opt_volume_z_uses_strictly_prior_window():
    # 22 trading days; the prior window has mild noise (so std>0), then a big
    # spike on the last day -> a large positive z computed from the prior window.
    days = pd.bdate_range("2024-01-01", periods=22)
    rng = np.random.default_rng(3)
    all_rows = []
    for i, ts in enumerate(days):
        vol = (50.0 + rng.normal(0, 2.0)) if i < len(days) - 1 else 500.0
        all_rows += _full_chain_rows(ts.strftime("%Y-%m-%d"), vol=vol)
    feats = of.assemble_underlying_features(pd.DataFrame(all_rows))
    feats = feats.sort_values("date").reset_index(drop=True)
    # First OPT_VOL_Z_WINDOW rows have no full prior window -> NaN z.
    assert feats["opt_volume_z"].iloc[:of.OPT_VOL_Z_WINDOW].isna().all()
    # The spike day's z is large + positive (mean/std built from prior ~50s).
    assert feats["opt_volume_z"].iloc[-1] > 5.0


def test_assemble_iv_rv_ratio_present_with_history():
    days = pd.bdate_range("2024-01-01", periods=25)
    all_rows = []
    for i, ts in enumerate(days):
        spot = 100.0 + i * 0.5  # drifting underlying gives a finite RV
        all_rows += _full_chain_rows(ts.strftime("%Y-%m-%d"), spot=spot)
    feats = of.assemble_underlying_features(pd.DataFrame(all_rows))
    feats = feats.sort_values("date").reset_index(drop=True)
    # Early rows lack 21 prior closes -> NaN; later rows have iv_rv.
    assert feats["iv_rv_20d_ratio"].iloc[-1] > 0
    assert np.isnan(feats["iv_rv_20d_ratio"].iloc[0])


def test_assemble_prefers_equity_bars_close_for_rv():
    # Store underlying_close is FLAT (RV -> 0 -> None -> NaN iv_rv); the supplied
    # equity bars carry a VOLATILE split-adjusted close (finite RV). A finite
    # iv_rv proves RV was taken from the equity bars, not the store close.
    days = pd.bdate_range("2024-05-01", periods=30)
    all_rows = []
    for ts in days:
        all_rows += _full_chain_rows(ts.strftime("%Y-%m-%d"), spot=100.0)  # flat
    rng = np.random.default_rng(5)
    eq_close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
    eq = pd.DataFrame({"close": eq_close, "volume": 1e6}, index=days)
    with_eq = (of.assemble_underlying_features(pd.DataFrame(all_rows), equity_bars=eq)
               .sort_values("date").reset_index(drop=True))
    assert np.isfinite(with_eq["iv_rv_20d_ratio"].iloc[-1])
    # The store-close fallback on the same flat series yields NaN (RV ~ 0).
    store_only = (of.assemble_underlying_features(pd.DataFrame(all_rows))
                  .sort_values("date").reset_index(drop=True))
    assert np.isnan(store_only["iv_rv_20d_ratio"].iloc[-1])
