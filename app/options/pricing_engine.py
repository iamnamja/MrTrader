"""
Options pricing + greeks engine — OPT-1a (the confidence keystone).

Polygon Developer serves IV/greeks only in the CURRENT snapshot, so historical IV/greeks
must be COMPUTED. This module computes them from EOD price + underlying + rate + dividend:

  * Black-Scholes-European (closed form) — fast path, exact for European.
  * Bjerksund-Stensland 1993 (American approximation) — equity options are American;
    early exercise matters for ITM puts and dividend-paying calls.
  * CRR binomial — independent reference for cross-checking (tests), not the hot path.
  * Implied-vol solver (bisection — robust, no derivative blowups near expiry).

The OPT-0 spike showed BS-European IV already matches Polygon's served IV to ~0.86
vol-points near-ATM (unbiased); American + real dividends mainly tighten the ITM/OTM tails
and remove the all-contract bias. Pure / no I/O → unit-tested against textbook values,
put-call parity, and BS↔BjS↔CRR cross-checks.

Implements the `OptionsPricingEngine` contract (app/options/contracts.py).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _intrinsic(S: float, K: float, kind: str) -> float:
    return max(0.0, (S - K) if kind == "call" else (K - S))


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes-European (closed form)
# ─────────────────────────────────────────────────────────────────────────────

def _d1_d2(S, K, T, r, q, sigma):
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> float:
    """European Black-Scholes price (continuous dividend yield q)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return _intrinsic(S, K, kind)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if kind == "call":
        return S * math.exp(-q * T) * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
    return K * math.exp(-r * T) * _cdf(-d2) - S * math.exp(-q * T) * _cdf(-d1)


def bs_greeks(S, K, T, r, q, sigma, kind) -> Dict[str, float]:
    """European greeks. delta (unitless), gamma (per $1), vega (per 1.00 vol = per 100%),
    theta (per year), rho (per 1.00 rate). Callers scale to per-1% as needed."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Degenerate: delta is the exercise indicator; others ~0.
        itm = (S > K) if kind == "call" else (S < K)
        return {"delta": (1.0 if kind == "call" else -1.0) if itm else 0.0,
                "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrtT = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    gamma = disc_q * _pdf(d1) / (S * sigma * sqrtT)
    vega = S * disc_q * _pdf(d1) * sqrtT
    if kind == "call":
        delta = disc_q * _cdf(d1)
        theta = (-S * disc_q * _pdf(d1) * sigma / (2 * sqrtT)
                 - r * K * disc_r * _cdf(d2) + q * S * disc_q * _cdf(d1))
        rho = K * T * disc_r * _cdf(d2)
    else:
        delta = -disc_q * _cdf(-d1)
        theta = (-S * disc_q * _pdf(d1) * sigma / (2 * sqrtT)
                 + r * K * disc_r * _cdf(-d2) - q * S * disc_q * _cdf(-d1))
        rho = -K * T * disc_r * _cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ─────────────────────────────────────────────────────────────────────────────
# Bjerksund-Stensland 1993 American approximation
# ─────────────────────────────────────────────────────────────────────────────

def _phi(S, T, gamma, H, i_bnd, r, b, sigma):
    s2 = sigma * sigma
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * s2) * T
    kappa = 2.0 * b / s2 + (2.0 * gamma - 1.0)
    sqrtT = sigma * math.sqrt(T)
    d = -(math.log(S / H) + (b + (gamma - 0.5) * s2) * T) / sqrtT
    d2 = d - 2.0 * math.log(i_bnd / S) / sqrtT
    # Clamp the exponential to avoid overflow at extreme vols (the IV solver probes
    # high sigma to bracket the root); finite is all the bisection needs.
    return math.exp(min(lam, 50.0)) * (S ** gamma) * (_cdf(d) - ((i_bnd / S) ** kappa) * _cdf(d2))


def _bjs_call(S, K, T, r, b, sigma):
    """Bjerksund-Stensland 1993 American CALL with cost-of-carry b (= r - q)."""
    if b >= r:
        # No early-exercise premium (q <= 0) -> American call == European call.
        return bs_price(S, K, T, r, r - b, sigma, "call")
    s2 = sigma * sigma
    beta = (0.5 - b / s2) + math.sqrt((b / s2 - 0.5) ** 2 + 2.0 * r / s2)
    b_inf = beta / (beta - 1.0) * K
    b0 = max(K, r / (r - b) * K)
    ht = -(b * T + 2.0 * sigma * math.sqrt(T)) * b0 / (b_inf - b0)
    # Clamp: at very low vol ht explodes (the IV solver probes sigma->0 to bracket the
    # root). exp is then floored to a finite value; american_price re-floors the result
    # at max(intrinsic, european), so the degenerate boundary stays correct.
    trigger = b0 + (b_inf - b0) * (1.0 - math.exp(min(ht, 50.0)))
    if S >= trigger:
        return S - K
    alpha = (trigger - K) * trigger ** (-beta)
    return (alpha * S ** beta
            - alpha * _phi(S, T, beta, trigger, trigger, r, b, sigma)
            + _phi(S, T, 1.0, trigger, trigger, r, b, sigma)
            - _phi(S, T, 1.0, K, trigger, r, b, sigma)
            - K * _phi(S, T, 0.0, trigger, trigger, r, b, sigma)
            + K * _phi(S, T, 0.0, K, trigger, r, b, sigma))


def _bjs_degenerate(T, r_arg, b_arg, sigma) -> bool:
    """True when the BjS-1993 h(T) term goes positive (b*T + 2*sigma*sqrt(T) < 0) — the
    strongly-negative-carry regime where the trigger boundary formula breaks down and
    underprices badly (e.g. dividend-yield > rate calls). We route those to CRR instead.
    `b_arg`/`r_arg` are the carry/rate as PASSED to _bjs_call (transformed for puts)."""
    return (b_arg * T + 2.0 * sigma * math.sqrt(T)) < 0.0


def american_price(S, K, T, r, q, sigma, kind) -> float:
    """American option price via Bjerksund-Stensland 1993. Put uses the BjS put-call
    transformation P(S,K,r,b) = C(K,S, r-b, -b). In the degenerate strongly-negative-carry
    regime (BjS h(T) breaks down) we fall back to an exact CRR binomial. Floors at intrinsic."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return _intrinsic(S, K, kind)
    b = r - q
    # carry/rate actually handed to _bjs_call (puts go through the put-call transform)
    r_arg, b_arg = (r, b) if kind == "call" else (r - b, -b)
    if _bjs_degenerate(T, r_arg, b_arg, sigma):
        return crr_price(S, K, T, r, q, sigma, kind, "american", steps=500)
    try:
        if kind == "call":
            val = _bjs_call(S, K, T, r, b, sigma)
        else:
            val = _bjs_call(K, S, T, r - b, -b, sigma)
    except OverflowError:
        # Only at extreme vol (the IV solver's upper bracket probe). The American price
        # is bounded by the no-arbitrage cap (call <= S, put <= K); use it so the solver
        # still gets a finite, correctly-signed (price-too-high) value.
        val = S if kind == "call" else K
    # American value can never be below intrinsic or below the European value.
    return max(val, _intrinsic(S, K, kind), bs_price(S, K, T, r, q, sigma, kind))


# ─────────────────────────────────────────────────────────────────────────────
# CRR binomial (reference / cross-check)
# ─────────────────────────────────────────────────────────────────────────────

def crr_price(S, K, T, r, q, sigma, kind, style="american", steps: int = 400) -> float:
    """Cox-Ross-Rubinstein binomial — independent reference for tests."""
    if T <= 0 or sigma <= 0:
        return _intrinsic(S, K, kind)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    # terminal payoffs
    vals = []
    for i in range(steps + 1):
        ST = S * (u ** (steps - i)) * (d ** i)
        vals.append(_intrinsic(ST, K, kind))
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            cont = disc * (p * vals[i] + (1.0 - p) * vals[i + 1])
            if style == "american":
                ST = S * (u ** (step - i)) * (d ** i)
                vals[i] = max(cont, _intrinsic(ST, K, kind))
            else:
                vals[i] = cont
    return vals[0]


# ─────────────────────────────────────────────────────────────────────────────
# Engine (implements app/options/contracts.OptionsPricingEngine)
# ─────────────────────────────────────────────────────────────────────────────

class PricingEngine:
    """BS-European + Bjerksund-Stensland American + bisection IV + greeks
    (American greeks via central finite-difference on the BjS price)."""

    def price(self, S, K, T, r, q, sigma, kind, style="american") -> float:
        if style == "european":
            return bs_price(S, K, T, r, q, sigma, kind)
        return american_price(S, K, T, r, q, sigma, kind)

    def implied_vol(self, price, S, K, T, r, q, kind, style="american") -> Optional[float]:
        """Bisection IV solve in [1e-4, 3.0] (300% vol bracket — ample for equity
        options). None if price is below intrinsic or outside the bracket
        (degenerate/arbitrageable quote)."""
        if price is None or price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return None
        if price < _intrinsic(S, K, kind) - 1e-8:
            return None

        def f(sig):
            return self.price(S, K, T, r, q, sig, kind, style) - price

        lo, hi = 1e-4, 3.0
        flo, fhi = f(lo), f(hi)
        # Price pinned to the intrinsic floor (deep-ITM American on the early-exercise
        # boundary): f is flat ~0 at low sigma, so vol is not identifiable -> None
        # (rather than letting bisection march spuriously to the bracket top).
        if abs(flo) < 1e-7:
            return None
        if flo * fhi > 0:
            return None  # price not attainable within sane vol bracket
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if abs(fm) < 1e-7:
                return mid
            if flo * fm < 0:
                hi = mid
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi)

    def greeks(self, S, K, T, r, q, sigma, kind, style="american") -> Dict[str, float]:
        if style == "european":
            return bs_greeks(S, K, T, r, q, sigma, kind)
        # American greeks via central finite difference on the BjS price (delta/gamma/
        # vega/theta/rho), which captures the early-exercise curvature BS greeks miss.
        hS = max(S * 1e-3, 1e-4)
        hv = 1e-4
        hT = min(T * 1e-3, 1.0 / 365.0) if T > 1.0 / 365.0 else T * 0.5
        hr = 1e-4
        p = lambda **kw: american_price(  # noqa: E731
            kw.get("S", S), K, kw.get("T", T), kw.get("r", r), q,
            kw.get("sigma", sigma), kind)
        base = american_price(S, K, T, r, q, sigma, kind)
        up_s, dn_s = p(S=S + hS), p(S=S - hS)
        delta = (up_s - dn_s) / (2 * hS)
        # Gamma via 2nd difference, but the American price has a kink where it peels off
        # the early-exercise (intrinsic) floor. A central difference straddling that kink
        # spikes spuriously, so detect a floored bump and switch to a one-sided 2nd
        # difference on the smooth side (a fully-floored point is locally linear -> 0).

        def _floored(price, Sx):
            return abs(price - _intrinsic(Sx, K, kind)) < 1e-9
        b_floor = _floored(base, S)
        u_floor = _floored(up_s, S + hS)
        d_floor = _floored(dn_s, S - hS)
        if b_floor:
            gamma = 0.0
        elif d_floor and not u_floor:          # floor below -> use the upper smooth side
            gamma = (p(S=S + 2 * hS) - 2 * up_s + base) / (hS * hS)
        elif u_floor and not d_floor:          # floor above -> use the lower smooth side
            gamma = (base - 2 * dn_s + p(S=S - 2 * hS)) / (hS * hS)
        else:
            gamma = (up_s - 2 * base + dn_s) / (hS * hS)
        vega = (p(sigma=sigma + hv) - p(sigma=sigma - hv)) / (2 * hv)
        theta = -(p(T=T) - p(T=max(T - hT, 1e-9))) / hT if T > hT else 0.0
        rho = (p(r=r + hr) - p(r=r - hr)) / (2 * hr)
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# Module-level singleton for convenience.
ENGINE = PricingEngine()
