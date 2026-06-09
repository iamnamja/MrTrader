"""
OPT-0 feasibility spike — does our computed IV match Polygon's served IV?

The whole options program rests on COMPUTING historical IV/greeks ourselves (Polygon serves
them only in the current snapshot). This spike pulls the CURRENT snapshot (which DOES serve
real IV) for a basket of liquid underlyings, recomputes IV from the option's EOD close +
underlying + r + q via Black-Scholes, and reports the error distribution vs Polygon's IV.

It sets the OPT-1 accuracy tolerance empirically and reveals whether BS-European (q=0) is
close enough or whether American-exercise + dividends (Bjerksund-Stensland) are required.
Throwaway/diagnostic — seeds scripts/validate_options_engine.py in OPT-1.

Run: PYTHONIOENCODING=utf-8 python -m scripts.spike_options_iv_check
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import requests

from app.config import settings

UNDERLYINGS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
               "TSLA", "JPM", "XLE", "GLD", "TLT", "AMD"]
R = 0.043           # flat risk-free proxy for the spike (OPT-1 uses a real rate series)
Q = 0.0             # dividend yield ignored in the spike (note: biases dividend-payers)
EXP_LO, EXP_HI = 20, 45   # days-to-expiry window
N1 = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731


def _bs_price(S, K, T, r, q, sigma, kind):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if kind == "call" else (K - S))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * N1(d1) - K * math.exp(-r * T) * N1(d2)
    return K * math.exp(-r * T) * N1(-d2) - S * math.exp(-q * T) * N1(-d1)


def _bs_iv(price, S, K, T, r, q, kind):
    """Bisection IV solve. Returns None if outside a sane bracket."""
    if price <= 0 or T <= 0:
        return None
    lo, hi = 1e-4, 5.0
    plo = _bs_price(S, K, T, r, q, lo, kind) - price
    phi = _bs_price(S, K, T, r, q, hi, kind) - price
    if plo * phi > 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pm = _bs_price(S, K, T, r, q, mid, kind) - price
        if abs(pm) < 1e-6:
            return mid
        if plo * pm < 0:
            hi = mid
        else:
            lo, plo = mid, pm
    return 0.5 * (lo + hi)


def _fetch(underlying, key):
    today = dt.date.today()
    rows = []
    for kind, ct in (("call", "call"), ("put", "put")):
        r = requests.get(f"https://api.polygon.io/v3/snapshot/options/{underlying}", params={
            "apiKey": key, "limit": 250, "contract_type": ct,
            "expiration_date.gte": (today + dt.timedelta(days=EXP_LO)).isoformat(),
            "expiration_date.lte": (today + dt.timedelta(days=EXP_HI)).isoformat(),
        }, timeout=25)
        for c in (r.json().get("results") or []):
            g = c.get("greeks") or {}
            day = c.get("day") or {}
            d = c.get("details") or {}
            und = (c.get("underlying_asset") or {}).get("price")
            iv = c.get("implied_volatility")
            close = day.get("close")
            delta = g.get("delta")
            if not (iv and close and und and delta is not None):
                continue
            rows.append({
                "S": float(und), "K": float(d["strike_price"]),
                "exp": d["expiration_date"], "kind": kind,
                "price": float(close), "poly_iv": float(iv), "delta": float(delta),
            })
    return rows


def main() -> int:
    key = settings.polygon_api_key
    today = dt.date.today()
    errs, near_atm_errs, n = [], [], 0
    for u in UNDERLYINGS:
        try:
            rows = _fetch(u, key)
        except Exception as exc:
            print(f"  {u}: fetch failed: {exc}")
            continue
        for row in rows:
            T = (dt.date.fromisoformat(row["exp"]) - today).days / 365.0
            my_iv = _bs_iv(row["price"], row["S"], row["K"], T, R, Q, row["kind"])
            if my_iv is None:
                continue
            err = my_iv - row["poly_iv"]
            errs.append(err)
            n += 1
            if abs(abs(row["delta"]) - 0.5) < 0.15:   # near-ATM
                near_atm_errs.append(err)
        print(f"  {u}: {len(rows)} contracts with IV/close/delta")

    def _stats(label, e):
        if not e:
            print(f"  {label}: no data")
            return
        a = np.array(e)
        print(f"  {label}: n={len(a)}  median|err|={np.median(np.abs(a)):.4f}  "
              f"mean_err={a.mean():+.4f}  p90|err|={np.percentile(np.abs(a),90):.4f}  "
              f"(IV vol-points)")

    print("\n" + "=" * 78)
    print("  OPT-0 SPIKE: computed BS-European IV vs Polygon served IV (q=0, r=4.3%)")
    print("=" * 78)
    _stats("ALL contracts", errs)
    _stats("NEAR-ATM (|delta-0.5|<0.15)", near_atm_errs)
    print("=" * 78)
    print("  Interpretation: small near-ATM |err| (a few vol-pts) => BS is a viable fast path;")
    print("  systematic mean_err on dividend names => OPT-1 needs Bjerksund-Stensland + dividends.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
