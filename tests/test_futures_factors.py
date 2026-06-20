"""P1.2 — futures factor-zoo signal tests (PIT + correctness)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import futures_factors as ff


def _prices(n=400, k=3, seed=0):
    idx = pd.bdate_range("2015-01-02", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"M{i}": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
                         for i in range(k)}, index=idx)


def test_xs_momentum_signal_formula_and_pit():
    P = _prices()
    sig = ff.xs_momentum_signal(P, lookback=252, skip=21)
    # value at t = P[t-21]/P[t-252]-1 exactly
    t = 300
    expected = P["M0"].iloc[t - 21] / P["M0"].iloc[t - 252] - 1.0
    assert sig["M0"].iloc[t] == np.float64(expected) or abs(sig["M0"].iloc[t] - expected) < 1e-12
    # PIT: first lookback rows are NaN (no history)
    assert sig.iloc[:252].isna().all().all()
    # prefix-invariant (uses only past) -> recomputing on a prefix matches
    pref = ff.xs_momentum_signal(P.iloc[:320])
    assert np.allclose(pref["M0"].iloc[:320].to_numpy(), sig["M0"].iloc[:320].to_numpy(),
                       equal_nan=True)


def test_value_signal_is_negative_long_horizon_return():
    P = _prices(n=1400)
    sig = ff.value_signal(P, lookback=1260)
    t = 1300
    expected = -(P["M0"].iloc[t] / P["M0"].iloc[t - 1260] - 1.0)
    assert abs(sig["M0"].iloc[t] - expected) < 1e-12


def test_curve_momentum_is_carry_change():
    idx = pd.bdate_range("2015-01-02", periods=200)
    carry = pd.DataFrame({"A": np.linspace(0, 1, 200), "B": np.linspace(1, 0, 200)}, index=idx)
    sig = ff.curve_momentum_signal(carry, lookback=63)
    assert abs(sig["A"].iloc[100] - (carry["A"].iloc[100] - carry["A"].iloc[37])) < 1e-12


def test_skew_signal_sign():
    # a right-skewed series -> negative signal (we prefer LOW skew)
    idx = pd.bdate_range("2015-01-02", periods=300)
    rng = np.random.default_rng(1)
    r = pd.DataFrame({"A": rng.exponential(0.01, 300) - 0.01}, index=idx)  # right-skewed
    sig = ff.skew_signal(r, window=252, min_periods=120)
    assert sig["A"].dropna().iloc[-1] < 0


def test_futures_book_is_equal_weight_of_subsleeves(monkeypatch):
    # P1.3: futures_book = 0.5*(carry + xsmom) of the two sub-sleeve return series (inner-joined).
    import scripts.walkforward.sleeves as sl
    from scripts.walkforward.sleeve_lab import Sleeve
    idx = pd.bdate_range("2015-01-02", periods=100)
    c = pd.Series(0.001, index=idx)
    x = pd.Series(0.003, index=idx)
    monkeypatch.setattr(sl, "build_futures_carry",
                        lambda **k: Sleeve(label="c", component_type="risk_premium", returns=c))
    monkeypatch.setattr(sl, "build_futures_xsmom",
                        lambda **k: Sleeve(label="x", component_type="diversifier", returns=x))
    book = sl.build_futures_book()
    assert book.registration_id == "P1-3-FUT-BOOK"
    assert np.allclose(book.returns.to_numpy(), 0.002)        # 0.5*(0.001+0.003)


def test_xs_factor_backtest_aliases_carry_engine():
    from app.research import futures_carry as fc
    idx = pd.bdate_range("2015-01-02", periods=300)
    rng = np.random.default_rng(2)
    rets = pd.DataFrame({m: rng.normal(0, 0.01, 300) for m in ["A", "B", "C", "D", "E"]}, index=idx)
    signal = rets.rolling(20).mean()           # any signal panel
    cfg = fc.CarryConfig(min_xs_width=2)
    a = ff.xs_factor_backtest(rets, signal, cfg)
    b = fc.carry_backtest(rets, signal.reindex_like(rets), cfg)
    pd.testing.assert_series_equal(a, b)
