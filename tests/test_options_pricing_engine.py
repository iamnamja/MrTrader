"""OPT-1a: options pricing/greeks engine correctness.

Textbook BS values, put-call parity, American>=European (BjS), BjS<->CRR cross-check,
IV round-trip, and greeks sign/sanity. These are what make computed historical IV/greeks
trustworthy enough to backtest on (the OPT-1 confidence gate).
"""
from __future__ import annotations

import math

import pytest

from app.options.contracts import OptionsPricingEngine
from app.options.pricing_engine import (
    bs_price, bs_greeks, american_price, crr_price, PricingEngine, ENGINE,
)


def test_engine_satisfies_frozen_contract():
    # The engine must duck-type to the OPT-0 frozen OptionsPricingEngine Protocol
    # (price/implied_vol/greeks) so every downstream layer can depend on the seam.
    assert isinstance(ENGINE, OptionsPricingEngine)
    assert isinstance(PricingEngine(), OptionsPricingEngine)


# ── Black-Scholes textbook values ─────────────────────────────────────────────

def test_bs_textbook_values():
    # S=K=100, T=1, r=5%, q=0, vol=20% -> well-known BS values.
    c = bs_price(100, 100, 1.0, 0.05, 0.0, 0.20, "call")
    p = bs_price(100, 100, 1.0, 0.05, 0.0, 0.20, "put")
    assert c == pytest.approx(10.4506, abs=1e-3)
    assert p == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity_european():
    S, K, T, r, q, sig = 105, 100, 0.75, 0.04, 0.015, 0.25
    c = bs_price(S, K, T, r, q, sig, "call")
    p = bs_price(S, K, T, r, q, sig, "put")
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


# ── American (Bjerksund-Stensland) ────────────────────────────────────────────

def test_american_call_no_dividend_equals_european():
    # With q<=0 it's never optimal to early-exercise a call -> American == European.
    for K in (90, 100, 110):
        am = american_price(100, K, 1.0, 0.05, 0.0, 0.25, "call")
        eu = bs_price(100, K, 1.0, 0.05, 0.0, 0.25, "call")
        assert am == pytest.approx(eu, abs=1e-6)


def test_american_geq_european():
    # American >= European always (early-exercise right has non-negative value).
    cases = [("call", 0.04), ("put", 0.0), ("put", 0.02), ("call", 0.03)]
    for kind, q in cases:
        am = american_price(100, 100, 1.0, 0.05, q, 0.30, kind)
        eu = bs_price(100, 100, 1.0, 0.05, q, 0.30, kind)
        assert am >= eu - 1e-9


def test_bjs_matches_crr_reference():
    # BjS-1993 should track a 400-step CRR binomial within a few cents.
    for kind, q in [("put", 0.0), ("put", 0.03), ("call", 0.04), ("call", 0.06)]:
        for K in (90, 100, 110):
            bjs = american_price(100, K, 0.75, 0.05, q, 0.28, kind)
            crr = crr_price(100, K, 0.75, 0.05, q, 0.28, kind, "american", steps=500)
            assert bjs == pytest.approx(crr, abs=0.15), f"{kind} K={K} q={q}: bjs={bjs} crr={crr}"


def test_american_never_below_intrinsic():
    # Deep ITM American put must be >= intrinsic.
    am = american_price(70, 100, 0.5, 0.05, 0.0, 0.30, "put")
    assert am >= (100 - 70) - 1e-9


def test_high_dividend_call_matches_crr():
    # REGRESSION (CRITICAL): when dividend yield > rate, BjS-1993 h(T) goes positive and
    # the trigger formula breaks down, underpricing the call ~95%. The degenerate-regime
    # CRR fallback must recover the true (early-exercise) value.
    for q, T, sig in [(0.20, 1.0, 0.05), (0.20, 2.0, 0.05), (0.15, 1.5, 0.10)]:
        bjs = american_price(100, 100, T, 0.05, q, sig, "call")
        crr = crr_price(100, 100, T, 0.05, q, sig, "call", "american", steps=2000)
        assert bjs == pytest.approx(crr, abs=0.05), f"q={q} T={T}: {bjs} vs {crr}"


def test_dividend_call_early_exercise_premium_appears():
    # With q>0 (but still q<r so non-degenerate) an American call carries a STRICTLY
    # positive early-exercise premium over European.
    am = american_price(100, 90, 1.0, 0.06, 0.04, 0.25, "call")
    eu = bs_price(100, 90, 1.0, 0.06, 0.04, 0.25, "call")
    assert am > eu + 1e-4


def test_short_expiry_sane():
    # A few days to expiry: price ~ intrinsic + small time value, finite, >= intrinsic.
    for kind in ("call", "put"):
        am = american_price(101, 100, 3 / 365.0, 0.05, 0.0, 0.30, kind)
        eu = bs_price(101, 100, 3 / 365.0, 0.05, 0.0, 0.30, kind)
        assert am >= eu - 1e-9
        assert math.isfinite(am)


# ── Implied vol ───────────────────────────────────────────────────────────────

def test_iv_roundtrip_european():
    S, K, T, r, q, sig = 100, 95, 0.5, 0.04, 0.01, 0.33
    price = bs_price(S, K, T, r, q, sig, "call")
    iv = ENGINE.implied_vol(price, S, K, T, r, q, "call", style="european")
    assert iv == pytest.approx(sig, abs=1e-3)


def test_iv_roundtrip_american():
    S, K, T, r, q, sig = 100, 100, 0.4, 0.05, 0.03, 0.27
    price = american_price(S, K, T, r, q, sig, "put")
    iv = ENGINE.implied_vol(price, S, K, T, r, q, "put", style="american")
    assert iv == pytest.approx(sig, abs=2e-3)


def test_iv_at_intrinsic_floor_returns_none():
    # REGRESSION (HIGH): a deep-ITM American price pinned to the intrinsic floor carries
    # no vol information -> the solver must return None, not march to the bracket top (3.0).
    price = american_price(70, 100, 0.5, 0.10, 0.0, 0.10, "put")
    assert price == pytest.approx(30.0, abs=1e-6)  # sits exactly on intrinsic
    iv = ENGINE.implied_vol(price, 70, 100, 0.5, 0.10, 0.0, "put", style="american")
    assert iv is None


def test_iv_below_intrinsic_returns_none():
    # A price below intrinsic is unattainable -> None (not a bogus IV).
    # ITM put K=130 > S=100 -> intrinsic ~30; a quote of 1.0 is below intrinsic.
    assert ENGINE.implied_vol(1.0, 100, 130, 0.5, 0.05, 0.0, "put", style="european") is None
    assert ENGINE.implied_vol(0.0, 100, 100, 0.5, 0.05, 0.0, "call") is None


# ── Greeks ────────────────────────────────────────────────────────────────────

def test_european_greeks_signs():
    g = bs_greeks(100, 100, 0.5, 0.04, 0.0, 0.25, "call")
    assert 0.0 < g["delta"] < 1.0
    assert g["gamma"] > 0 and g["vega"] > 0
    gp = bs_greeks(100, 100, 0.5, 0.04, 0.0, 0.25, "put")
    assert -1.0 < gp["delta"] < 0.0
    # call delta - put delta == e^{-qT} (here q=0 -> ~1)
    assert g["delta"] - gp["delta"] == pytest.approx(1.0, abs=1e-6)


def test_american_greeks_finite_diff_sane():
    eng = PricingEngine()
    g = eng.greeks(100, 100, 0.5, 0.05, 0.03, 0.30, "call", style="american")
    assert 0.0 < g["delta"] < 1.0
    assert g["gamma"] > 0 and g["vega"] > 0


def test_american_gamma_no_kink_spike_at_boundary():
    # REGRESSION (MEDIUM): central-difference gamma must not spike where the American price
    # peels off the intrinsic floor. Near the early-exercise boundary gamma stays bounded
    # and close to the smooth interior value, not ~10x it.
    eng = PricingEngine()
    g = eng.greeks(92, 100, 0.3, 0.12, 0.0, 0.18, "put", style="american")
    assert 0.0 <= g["gamma"] < 0.12   # smooth ATM gamma here is ~0.03; pre-fix spiked to ~0.30


def test_engine_dispatches_style():
    eng = PricingEngine()
    eu = eng.price(100, 100, 1.0, 0.05, 0.05, 0.25, "put", style="european")
    am = eng.price(100, 100, 1.0, 0.05, 0.05, 0.25, "put", style="american")
    assert am >= eu - 1e-9  # American put >= European put with dividends
