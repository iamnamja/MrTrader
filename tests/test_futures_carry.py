"""P4-2 — futures carry tests (real reader via tmp parquet mirror).

Checks the carry SIGN convention (backwardation -> positive, contango -> negative), the
annualization formula, and that the cross-sectional carry backtest is PIT (signal shifted)
and long-high / short-low carry.
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd
import pytest

from app.data import norgate_provider as ng

fd = importlib.import_module("app.research.futures_data")
fc = importlib.import_module("app.research.futures_carry")


def _write_two_contract_market(dirpath, market, front_px, next_px, *, n=120):
    """Two contracts delivering H=Mar / M=Jun 2011 (3-month gap), with data spanning mid-2010
    — every date is well before expiry (>5d), so both stay live throughout and expiry comes
    from the SCHEDULED contract code (not the realized last date). carry sign = sign(front-next)."""
    dates = pd.bdate_range("2010-06-01", periods=n)
    rows = []
    for code, px in (("H", front_px), ("M", next_px)):     # H=Mar, M=Jun 2011
        rows.append(pd.DataFrame({"date": dates, "contract": f"{market}-2011{code}",
                                  "close": float(px), "open": px, "high": px, "low": px,
                                  "volume": 1000.0, "open_interest": 1000.0}))
    pd.concat(rows, ignore_index=True).to_parquet(
        os.path.join(dirpath, f"{market}.parquet"), index=False)


@pytest.fixture()
def cmirror(monkeypatch, tmp_path):
    contr = tmp_path / "contracts"
    contr.mkdir()
    monkeypatch.setattr(ng, "CONTRACTS_DIR", str(contr))
    importlib.reload(fc)
    monkeypatch.setattr(fc.ng, "CONTRACTS_DIR", str(contr))
    return str(contr)


def test_carry_sign_backwardation_positive(cmirror):
    # front (102) > next (100) -> backwardation -> positive carry
    _write_two_contract_market(cmirror, "BWD", 102.0, 100.0)
    s = fc.carry_series("BWD")
    assert not s.empty and (s > 0).all()


def test_carry_sign_contango_negative(cmirror):
    # front (98) < next (100) -> contango -> negative carry
    _write_two_contract_market(cmirror, "CON", 98.0, 100.0)
    s = fc.carry_series("CON")
    assert not s.empty and (s < 0).all()


def test_carry_annualization_formula(cmirror):
    # front 105, next 100; scheduled gap H(Mar)->M(Jun) 2011 = ~92 calendar days = 0.252y.
    _write_two_contract_market(cmirror, "ANN", 105.0, 100.0)
    s = fc.carry_series("ANN")
    # carry = (105-100)/100 / 0.252 ~= 0.198 annualized — a sane positive number.
    assert 0.05 < s.mean() < 0.5


def test_carry_backtest_is_pit_and_runs(cmirror):
    rng = np.random.default_rng(0)
    # two markets so the cross-section is non-degenerate
    _write_two_contract_market(cmirror, "AAA", 102.0, 100.0)
    _write_two_contract_market(cmirror, "BBB", 98.0, 100.0)
    carry = fc.carry_panel(["AAA", "BBB"])
    idx = carry.index
    rets = pd.DataFrame({"AAA": rng.normal(0, 0.01, len(idx)),
                         "BBB": rng.normal(0, 0.01, len(idx))}, index=idx)
    r = fc.carry_backtest(rets, carry, fc.CarryConfig(min_xs_width=2))
    assert len(r) > 0 and np.isfinite(r).all()


def test_carry_backtest_longs_high_carry(cmirror):
    # AAA persistently high carry, BBB low -> with deterministic equal returns, the book
    # should profit when the high-carry market outperforms.
    _write_two_contract_market(cmirror, "AAA", 110.0, 100.0)     # high carry
    _write_two_contract_market(cmirror, "BBB", 90.0, 100.0)      # low (negative) carry
    carry = fc.carry_panel(["AAA", "BBB"])
    idx = carry.index
    rets = pd.DataFrame({"AAA": 0.001, "BBB": -0.001}, index=idx)
    r = fc.carry_backtest(rets, carry, fc.CarryConfig(min_xs_width=2))
    assert r.sum() > 0


# ── regression guards added during the 2026-06-18 hardening review ──────────────
def _flat_carry_panel(markets, idx):
    return pd.DataFrame({m: 0.0 for m in markets}, index=idx)


def test_carry_backtest_lag_prevents_lookahead(cmirror):
    """The signal MUST be lagged: a carry tilt on day T may only earn day T+1's return.
    Constructed so that REMOVING the `.shift(1)` would zero the captured PnL → this test
    fails on a look-ahead regression (the old test used constant inputs and could not)."""
    mkts = [f"M{i}" for i in range(6)]
    idx = pd.bdate_range("2010-01-04", periods=20)
    carry = _flat_carry_panel(mkts, idx)
    T = 10
    carry.iloc[T, 0] = 1.0                       # M0 high carry ONLY on day T
    for j in range(1, 6):
        carry.iloc[T, j] = -0.2
    rets = pd.DataFrame(0.0, index=idx, columns=mkts)
    rets.iloc[T + 1, 0] = 0.10                   # the tradable move is the NEXT day, on M0
    cfg = fc.CarryConfig(min_xs_width=2, vol_lookback=3, rebalance_days=1,
                         book_vol_max_leverage=4.0)
    r = fc.carry_backtest(rets, carry, cfg)
    # PnL must land on T+1 (lagged signal caught the move), and NOT on T (no look-ahead)
    assert r.loc[idx[T + 1]] > 0
    assert abs(r.loc[idx[T]]) < 1e-9


def test_carry_backtest_flat_on_thin_cross_section(cmirror):
    """A cross-section thinner than min_xs_width trades NOTHING (no noise-betting)."""
    mkts = ["A", "B"]
    idx = pd.bdate_range("2010-01-04", periods=80)
    carry = pd.DataFrame({"A": 0.3, "B": -0.3}, index=idx)      # only 2 markets
    rng = np.random.default_rng(1)
    rets = pd.DataFrame({m: rng.normal(0, 0.01, len(idx)) for m in mkts}, index=idx)
    r = fc.carry_backtest(rets, carry, fc.CarryConfig(min_xs_width=5))
    assert (r.abs() < 1e-12).all()              # below width threshold -> flat, finite, no crash
