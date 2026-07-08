"""CH2 sizing gate-harness + CH2b correlation-multiplier tests (app/research/ch2_sizing.py).

Pins the pure pieces — the DUAL-gate decision, the held-book weighted correlation, and the
correlation→multiplier mapping (incl. the PIT shift). The heavy CPCV `gate_multiplier` /
`run_ch2b` are exercised by running the module, not the unit suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research import ch2_sizing as ch2


BAR = {"mean_sharpe": 0.7009, "bear_sharpe": -0.77}


# ── gate_decision (the DUAL gate) ─────────────────────────────────────────────
def test_gate_pass_needs_both_beat_and_bear():
    # beats + BEAR not regressed -> PASS
    assert ch2.gate_decision(0.75, -0.70, BAR) == (True, True, True)


def test_gate_fail_when_not_beating_even_if_bear_better():
    beats, bear_ok, passed = ch2.gate_decision(0.69, 0.10, BAR)
    assert beats is False and bear_ok is True and passed is False


def test_gate_fail_when_bear_regresses_even_if_beating():
    # beats mean_sharpe but BEAR drops below baseline -> FAIL (the whole point of the dual gate)
    beats, bear_ok, passed = ch2.gate_decision(0.80, -0.90, BAR)
    assert beats is True and bear_ok is False and passed is False


def test_gate_none_bear_treated_as_non_regressing():
    assert ch2.gate_decision(0.75, None, BAR) == (True, True, True)
    assert ch2.gate_decision(0.69, None, BAR)[2] is False   # mean prong still binds


def test_gate_equal_mean_sharpe_does_not_beat():
    # strictly-greater: matching the baseline is NOT beating it (adding a knob for nothing)
    assert ch2.gate_decision(0.7009, 0.0, BAR)[0] is False


# ── held_book_corr (weighted book correlation) ────────────────────────────────
def _prices_from_returns(rets: pd.DataFrame) -> pd.DataFrame:
    return 100.0 * (1.0 + rets).cumprod()


def test_held_book_corr_weights_by_the_held_names():
    # SPY,QQQ perfectly co-move (corr 1); TLT independent. A book holding ONLY SPY+QQQ must read
    # ~1.0; the idle TLT (weight 0) must not dilute it (that was the full-universe bug).
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 120)
    rets = pd.DataFrame({"SPY": base, "QQQ": base, "TLT": rng.normal(0, 0.01, 120)}, index=idx)
    prices = _prices_from_returns(rets)
    w = pd.DataFrame(0.0, index=idx, columns=["SPY", "QQQ", "TLT"])
    w["SPY"] = 0.5
    w["QQQ"] = 0.5   # TLT weight 0 throughout
    c = ch2.held_book_corr(prices, w, window=30)
    assert c.dropna().iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_held_book_corr_hedge_lowers_it():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, 120)
    rets = pd.DataFrame({"SPY": base, "TLT": -base}, index=idx)   # perfectly anti-correlated
    prices = _prices_from_returns(rets)
    w = pd.DataFrame({"SPY": [0.5] * 120, "TLT": [0.5] * 120}, index=idx)
    c = ch2.held_book_corr(prices, w, window=30)
    assert c.dropna().iloc[-1] == pytest.approx(-1.0, abs=1e-6)


def test_held_book_corr_single_name_book_is_max_concentration():
    # a 1-name book is MAXIMAL concentration -> 1.0 (so the multiplier cuts), not NaN/no-cut.
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rets = pd.DataFrame({"SPY": np.random.default_rng(2).normal(0, 0.01, 60),
                         "TLT": np.random.default_rng(3).normal(0, 0.01, 60)}, index=idx)
    prices = _prices_from_returns(rets)
    w = pd.DataFrame({"SPY": [1.0] * 60, "TLT": [0.0] * 60}, index=idx)  # only ONE name held
    assert ch2.held_book_corr(prices, w, window=30).dropna().iloc[-1] == pytest.approx(1.0)


def test_held_book_corr_empty_book_is_nan():
    # no positions held -> no signal (NaN -> the harness fills to m=1.0, no cut)
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rets = pd.DataFrame({"SPY": np.random.default_rng(2).normal(0, 0.01, 60),
                         "TLT": np.random.default_rng(3).normal(0, 0.01, 60)}, index=idx)
    w = pd.DataFrame({"SPY": [0.0] * 60, "TLT": [0.0] * 60}, index=idx)
    assert ch2.held_book_corr(_prices_from_returns(rets), w, window=30).dropna().empty


# ── correlation_gross_multiplier (mapping; UN-shifted — the harness owns the lag) ──
def test_multiplier_maps_corr_band_to_gross_unshifted(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    fake_corr = pd.Series([0.0, 0.60, 0.75, 0.90, 0.95, 0.60, 0.30, 0.0], index=idx)
    monkeypatch.setattr(ch2, "held_book_corr", lambda *a, **k: fake_corr)
    m = ch2.correlation_gross_multiplier(pd.DataFrame(index=idx), pd.DataFrame(index=idx),
                                         window=3, corr_lo=0.60, corr_hi=0.90, floor=0.50)
    # NO shift here (info-through-t): corr[t] maps directly at t. corr=0.75 -> 1-0.5*0.5 = 0.75
    assert m.iloc[2] == pytest.approx(0.75)
    assert m.iloc[3] == pytest.approx(0.50)     # corr 0.90 -> floor
    assert m.iloc[0] == pytest.approx(1.0)      # corr 0.0 below lo -> 1.0


def test_multiplier_clamps_between_floor_and_one(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    fake = pd.Series([-0.5, 0.5, 1.2, 0.95, 0.0, 0.7], index=idx)   # incl. out-of-band values
    monkeypatch.setattr(ch2, "held_book_corr", lambda *a, **k: fake)
    m = ch2.correlation_gross_multiplier(pd.DataFrame(index=idx), pd.DataFrame(index=idx),
                                         window=3, corr_lo=0.60, corr_hi=0.90, floor=0.50)
    assert m.dropna().min() >= 0.50 - 1e-9 and m.dropna().max() <= 1.0 + 1e-9


# ── governed_returns (harness-owned PIT lag + re-sizing turnover) ─────────────
def test_governed_returns_applies_pit_lag():
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    base = pd.Series([0.01, -0.02, 0.03, 0.00, 0.05, -0.01], index=idx)
    raw = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=idx)   # signal through t
    _, m = ch2.governed_returns(raw, base, apply_lag=True, cost_bps=0.0)
    # m[t] = raw[t-1] (harness shift); m[0] = warmup fill 1.0
    assert m.iloc[0] == pytest.approx(1.0)
    assert m.iloc[3] == pytest.approx(0.7) and m.iloc[5] == pytest.approx(0.9)


def test_governed_returns_identity_multiplier_reproduces_base():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    base = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50), index=idx)
    gov, _ = ch2.governed_returns(pd.Series(1.0, index=idx), base, apply_lag=True, cost_bps=0.0)
    assert np.allclose(gov.to_numpy(), base.to_numpy())      # m≡1 → exact reproduction


def test_governed_returns_charges_resize_turnover():
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    base = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)         # isolate the cost term
    raw = pd.Series([1.0, 1.0, 0.5, 0.5], index=idx)          # a 0.5 step in m
    gov, m = ch2.governed_returns(raw, base, apply_lag=True, cost_bps=2.0)
    # after shift m = [1,1,1,0.5]; the |Δm|=0.5 step at idx[3] costs 0.5 * 2bps
    assert gov.iloc[-1] == pytest.approx(-0.5 * 2.0 / 1e4)


# ── paired_delta_sharpe_pvalue (the significance guard) ───────────────────────
def test_paired_pvalue_is_one_when_governed_equals_base():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    base = pd.Series(np.random.default_rng(1).normal(0.0004, 0.01, 300), index=idx)
    assert ch2.paired_delta_sharpe_pvalue(base, base, n_boot=200) == pytest.approx(1.0)


def test_paired_pvalue_low_when_governed_dominates():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    base = pd.Series(np.random.default_rng(2).normal(0.0002, 0.01, 300), index=idx)
    governed = base + 0.001              # strictly higher mean, same vol → higher Sharpe every draw
    assert ch2.paired_delta_sharpe_pvalue(governed, base, n_boot=200) < 0.05


# ── held_book_corr PIT (no future leak) ──────────────────────────────────────
def test_held_book_corr_is_pit_no_future_leak():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    rng = np.random.default_rng(5)
    rets = pd.DataFrame({"SPY": rng.normal(0, 0.01, 120), "TLT": rng.normal(0, 0.01, 120)},
                        index=idx)
    w = pd.DataFrame({"SPY": [0.5] * 120, "TLT": [0.5] * 120}, index=idx)
    prices = _prices_from_returns(rets)
    c0 = ch2.held_book_corr(prices, w, window=30)
    # perturb ONLY the last 10 rows (the future relative to t=100) and re-check t<=100 unchanged
    rets2 = rets.copy()
    rets2.iloc[110:] += 0.05
    c1 = ch2.held_book_corr(_prices_from_returns(rets2), w, window=30)
    assert c0.iloc[:101].equals(c1.iloc[:101])


def test_load_baseline_bar_reads_frozen_artifact():
    bar = ch2.load_baseline_bar()
    assert bar["mean_sharpe"] == pytest.approx(0.7009, abs=1e-4)
    assert bar["bear_sharpe"] is not None and bar["bear_sharpe"] < 0   # BEAR is negative
