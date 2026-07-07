"""CH1 per-name risk-gate tests (app/live_trading/per_name_gate.py).

The gate closes the live trend book's per-name safety gap (correlation / heat / concentration).
These pin the book-level metrics, the signed weighted-average correlation, and — critically — the
FAIL-SAFE shadow contract (a gate error must never raise / never block a live rebalance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.live_trading import per_name_gate as png
from app.live_trading.risk_policy import RISK_POLICY_V1 as P


def _corr(pairs: dict, names: list) -> pd.DataFrame:
    """Build a symmetric correlation matrix from {(a,b): rho} off-diagonals (diag = 1)."""
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names, dtype=float)
    for (a, b), r in pairs.items():
        m.at[a, b] = r
        m.at[b, a] = r
    return m


# ---- weighted_avg_book_corr (the primary metric) ----
def test_weighted_avg_corr_two_names_equals_pair_corr():
    w = {"SPY": 0.5, "QQQ": 0.5}
    c = _corr({("SPY", "QQQ"): 0.9}, ["SPY", "QQQ"])
    assert png.weighted_avg_book_corr(w, c) == pytest.approx(0.9)


def test_weighted_avg_corr_is_signed_hedge_lowers_it():
    # a negatively-correlated hedge (TLT) makes the book LESS of a single bet (wavg < 0 here)
    w = {"SPY": 0.5, "TLT": 0.5}
    c = _corr({("SPY", "TLT"): -0.4}, ["SPY", "TLT"])
    assert png.weighted_avg_book_corr(w, c) == pytest.approx(-0.4)


def test_weighted_avg_corr_single_name_is_none():
    assert png.weighted_avg_book_corr({"SPY": 1.0}, _corr({}, ["SPY"])) is None


def test_weighted_avg_corr_skips_missing_and_nan_pairs():
    # QQQ absent from the matrix + a NaN pair are skipped, not crash
    w = {"SPY": 0.4, "QQQ": 0.3, "IWM": 0.3}
    c = _corr({("SPY", "IWM"): np.nan}, ["SPY", "IWM"])   # QQQ not in matrix; SPY-IWM NaN
    assert png.weighted_avg_book_corr(w, c) is None       # nothing usable -> None


# ---- evaluate (book-level checks) ----
def test_diversified_book_allows():
    w = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "UUP": 0.25}
    c = _corr({("SPY", "TLT"): -0.3, ("SPY", "GLD"): 0.0, ("SPY", "UUP"): -0.2,
               ("TLT", "GLD"): 0.1, ("TLT", "UUP"): 0.1, ("GLD", "UUP"): -0.1}, list(w))
    v = png.evaluate(w, c)
    assert v.allow and not v.breaches
    assert v.details["weighted_avg_book_corr"] < png.BOOK_CORR_GATE_AT


def test_normal_equity_led_book_passes_below_gate():
    # THE calibration test: a normal risk-on trend book is mostly equity ETFs, structurally
    # ~0.85 correlated. That must NOT breach the coarse extreme-concentration gate (0.90) — else
    # enforce would HOLD every equity-led week. The metric is still recorded (for shadow calibration).
    names = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
    w = {n: 0.1 for n in names}
    pairs = {(a, b): 0.85 for i, a in enumerate(names) for b in names[i + 1:]}
    v = png.evaluate(w, _corr(pairs, names))
    assert v.allow and not any("book_correlation" in b for b in v.breaches)
    assert v.details["weighted_avg_book_corr"] == pytest.approx(0.85)   # observed, not blocked


def test_one_bet_book_breaches_correlation():
    # book has collapsed to genuinely one bet (~0.97 everywhere) -> STRONG breach (> 0.95)
    names = ["SPY", "QQQ", "IWM", "EFA"]
    w = {n: 0.2 for n in names}
    pairs = {(a, b): 0.97 for i, a in enumerate(names) for b in names[i + 1:]}
    v = png.evaluate(w, _corr(pairs, names))
    assert not v.allow
    assert any("book_correlation" in b for b in v.breaches)
    assert any("STRONG" in b for b in v.breaches)   # 0.97 > strong (0.95)


def test_empty_book_allows():
    v = png.evaluate({}, None)
    assert v.allow and not v.breaches and v.details["max_name_weight"] == 0.0


def test_concentration_breach_on_oversized_name():
    w = {"SPY": 0.40, "TLT": 0.10}   # SPY 40% > 25% cap
    v = png.evaluate(w, _corr({("SPY", "TLT"): 0.0}, ["SPY", "TLT"]))
    assert not v.allow and any("per_name_concentration SPY" in b for b in v.breaches)


def test_concentration_at_cap_does_not_breach():
    w = {"SPY": P.max_single_instrument_notional_frac, "TLT": 0.10}   # exactly 25% -> OK
    v = png.evaluate(w, _corr({("SPY", "TLT"): 0.0}, ["SPY", "TLT"]))
    assert not any("per_name_concentration" in b for b in v.breaches)


def test_concentration_cap_respects_configured_per_name_cap():
    # if the sleeve is configured to allow 40% per name, a 40% weight must NOT breach (else raising
    # trend_max_position_pct > 0.25 would permanently HOLD). Effective cap = max(policy, configured).
    w = {"SPY": 0.40, "TLT": 0.10}
    assert not any("per_name_concentration" in b
                   for b in png.evaluate(w, None, per_name_cap=0.40).breaches)
    # but a weight ABOVE even the configured cap (clip-integrity failure) still breaches
    assert any("per_name_concentration SPY" in b
               for b in png.evaluate({"SPY": 0.45}, None, per_name_cap=0.40).breaches)


def test_heat_breach_when_gross_is_extreme():
    # heat = gross * FALLBACK_RISK_PCT(2%); to exceed 6% need gross > 3.0. 13 names @ 0.25 = 3.25.
    w = {f"N{i}": P.max_single_instrument_notional_frac for i in range(13)}
    v = png.evaluate(w, None)   # corr irrelevant here
    assert any("portfolio_heat" in b for b in v.breaches)
    assert v.details["portfolio_heat_frac"] == pytest.approx(3.25 * 0.02)


def test_normal_trend_gross_has_no_heat_breach():
    w = {"SPY": 0.2, "TLT": 0.2, "GLD": 0.1}   # gross 0.5 -> heat 1% << 6%
    v = png.evaluate(w, None)
    assert not any("portfolio_heat" in b for b in v.breaches)


# ---- _corr_from_prices ----
def test_corr_from_prices_builds_matrix_from_panel():
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 60)
    df = pd.DataFrame({
        "SPY": 100 + np.cumsum(base),
        "QQQ": 100 + np.cumsum(base + rng.normal(0, 0.05, 60)),   # near-identical to SPY
        "TLT": 100 + np.cumsum(-base + rng.normal(0, 0.05, 60)),  # opposite
    }, index=idx)
    c = png._corr_from_prices(df, ["SPY", "QQQ", "TLT", "MISSING"])
    assert c.at["SPY", "QQQ"] > 0.8      # co-moving
    assert c.at["SPY", "TLT"] < -0.5     # opposite
    assert "MISSING" not in c.columns


def test_corr_from_prices_too_few_names_is_empty():
    df = pd.DataFrame({"SPY": [1.0, 2.0, 3.0]})
    assert png._corr_from_prices(df, ["SPY"]).empty


# ---- shadow / enforce / fail-safe ----
def test_shadow_mode_sees_breach_without_blocking_semantics():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    base = np.cumsum(np.random.default_rng(1).normal(0, 1, 40))
    df = pd.DataFrame({"SPY": 100 + base, "QQQ": 100 + base * 1.001}, index=idx)  # ~1.0 corr
    v = png.shadow_per_name_gate({"SPY": 0.5, "QQQ": 0.5}, df, mode=png.SHADOW, label="trend")
    assert v.mode == png.SHADOW
    assert any("book_correlation" in b for b in v.breaches)   # it SEES the one-bet book


def test_enforce_mode_verdict_reflects_breach():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    base = np.cumsum(np.random.default_rng(2).normal(0, 1, 40))
    df = pd.DataFrame({"SPY": 100 + base, "QQQ": 100 + base * 1.001}, index=idx)
    v = png.shadow_per_name_gate({"SPY": 0.5, "QQQ": 0.5}, df, mode=png.ENFORCE, label="t")
    assert v.mode == png.ENFORCE and not v.allow


def test_gate_is_fail_safe_never_raises():
    # malformed weights (a non-numeric weight) must NOT raise and must fail-safe to allow=True
    v = png.shadow_per_name_gate({"SPY": "notanumber"}, None, mode=png.SHADOW, label="trend")
    assert v.allow is True and v.error is not None
