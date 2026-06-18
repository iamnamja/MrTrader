"""P4-2 — futures return-panel + universe tests (real readers via tmp parquet mirror).

The load-bearing correctness checks: difference-adjusted returns (ΔCCB/unadj_prev) stay
sane even when back-adjusted close goes NEGATIVE; near-zero-denominator blowups are
winsorized; the synthetic price recovers the returns; the universe filter excludes micros /
illiquid / short-history / ultra-low-vol STIRs.
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd
import pytest

from app.data import norgate_provider as ng

fd = importlib.import_module("app.research.futures_data")


def _write_market(dirpath, market, rets, *, unadj=100.0, volume=10000.0, n_extra=0):
    """Write a continuous-mirror parquet whose difference-adjusted returns == `rets`.
    ccb = unadj + cumsum(rets*unadj) so ΔCCB/unadj_prev == rets exactly (unadj constant)."""
    rets = np.asarray(rets, dtype=float)
    n = len(rets) + 1
    dates = pd.bdate_range("2005-01-03", periods=n)
    ccb = unadj + np.concatenate([[0.0], np.cumsum(rets * unadj)])
    ua = np.full(n, unadj)
    frames = []
    for ptype, c in (("backadjusted", ccb), ("unadjusted", ua)):
        frames.append(pd.DataFrame({
            "date": dates, "open": c, "high": c, "low": c, "close": c,
            "volume": volume, "open_interest": volume, "delivery_month": 200503,
            "price_type": ptype}))
    pd.concat(frames, ignore_index=True).to_parquet(
        os.path.join(dirpath, f"{market}.parquet"), index=False)


@pytest.fixture()
def mirror(monkeypatch, tmp_path):
    cont = tmp_path / "continuous"
    cont.mkdir()
    monkeypatch.setattr(ng, "CONTINUOUS_DIR", str(cont))
    importlib.reload(fd)
    monkeypatch.setattr(fd.ng, "CONTINUOUS_DIR", str(cont))
    return str(cont)


def test_true_returns_sane_when_ccb_goes_negative(mirror):
    rng = np.random.default_rng(0)
    rets = rng.normal(-0.001, 0.02, 3000)            # negative drift -> CCB crosses zero
    _write_market(mirror, "CLX", rets)
    # confirm CCB actually goes negative (the difference-adjustment landmine)
    cb = ng.load_continuous("CLX", price_type="backadjusted")["close"]
    assert (cb < 0).any()
    r = fd.true_returns("CLX")
    assert np.isfinite(r).all()
    # recovered returns match the injected ones (unadj constant) to tolerance
    assert np.allclose(r.to_numpy(), rets[-len(r):], atol=1e-9)


def _write_market_raw(dirpath, market, ccb, ua, *, start="2005-01-03"):
    """Write CCB + unadjusted arrays directly (to exercise the varying-denominator path)."""
    n = len(ccb)
    dates = pd.bdate_range(start, periods=n)
    frames = []
    for ptype, c in (("backadjusted", np.asarray(ccb, float)), ("unadjusted", np.asarray(ua, float))):
        frames.append(pd.DataFrame({
            "date": dates, "open": c, "high": c, "low": c, "close": c,
            "volume": 10000.0, "open_interest": 10000.0, "delivery_month": 200503,
            "price_type": ptype}))
    pd.concat(frames, ignore_index=True).to_parquet(
        os.path.join(dirpath, f"{market}.parquet"), index=False)


def test_true_returns_uses_prior_unadjusted_denominator(mirror):
    # VARYING unadjusted level: r_t must be ΔCCB_t / Unadj_{t-1} exactly (not Unadj_t, not
    # a constant). A bug using the same-day denominator would NOT match.
    ccb = [100.0, 101.0, 103.5, 102.0]
    ua = [50.0, 60.0, 75.0, 40.0]
    _write_market_raw(mirror, "VAR", ccb, ua)
    r = fd.true_returns("VAR", cap=1.0)
    expected = [(101.0 - 100.0) / 50.0, (103.5 - 101.0) / 60.0, (102.0 - 103.5) / 75.0]
    assert np.allclose(r.to_numpy(), expected, atol=1e-12)


def test_true_returns_drops_negative_denominator_day(mirror):
    # CL-2020 pattern: prior UNADJUSTED price goes negative -> that return would be sign-
    # flipped; the guard must DROP it (NaN), not book a fake gain inside the winsor band.
    ccb = [100.0, 90.0, 81.0, 85.0]
    ua = [20.0, -5.0, -8.0, 10.0]          # prior-day negative on the 3rd and 4th rows
    _write_market_raw(mirror, "NEGD", ccb, ua)
    r = fd.true_returns("NEGD", cap=1.0)
    # row1 ok (prior ua=20>0); rows where prior ua<=0 are dropped -> no sign-flipped gain
    assert r.iloc[0] == pytest.approx((90.0 - 100.0) / 20.0)
    assert len(r) == 1                      # the two negative-prior-denominator days dropped
    assert (r > 0).sum() == 0               # no spurious positive return


def test_winsorization_caps_blowups(mirror):
    rets = np.r_[np.full(2999, 0.001), [5.0]]         # one absurd +500% day
    _write_market(mirror, "BLW", rets)
    r = fd.true_returns("BLW", cap=0.5)
    assert r.max() <= 0.5 + 1e-12 and r.min() >= -0.5 - 1e-12


def test_synthetic_price_positive_and_recovers_returns(mirror):
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0003, 0.015, 3000)
    _write_market(mirror, "ESX", rets)
    panel = fd.synthetic_price_panel(["ESX"])
    assert (panel["ESX"] > 0).all()                   # never negative
    rec = panel["ESX"].pct_change().dropna()
    assert np.allclose(rec.to_numpy(), fd.true_returns("ESX").to_numpy()[-len(rec):], atol=1e-9)


def test_liquid_universe_excludes_micros(mirror):
    rng = np.random.default_rng(2)
    for m in ("ES", "MES", "NQ"):
        _write_market(mirror, m, rng.normal(0.0003, 0.015, 3000))
    uni = fd.liquid_universe()
    assert "ES" in uni and "NQ" in uni and "MES" not in uni   # micro dropped


def test_liquid_universe_filters_short_history_illiquid_and_stir(mirror):
    rng = np.random.default_rng(3)
    _write_market(mirror, "ES", rng.normal(0.0003, 0.015, 3000))               # good
    _write_market(mirror, "SHORT", rng.normal(0.0003, 0.015, 500))             # too short
    _write_market(mirror, "ILLQ", rng.normal(0.0003, 0.015, 3000), volume=10)  # illiquid
    _write_market(mirror, "SR3", rng.normal(0.0, 0.0003, 3000))                # ultra-low vol STIR
    uni = fd.liquid_universe()
    assert uni == ["ES"]
