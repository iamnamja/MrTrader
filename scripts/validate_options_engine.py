"""
OPT-1a confidence keystone — validate the computed-IV/greeks engine vs Polygon's served values.

The options program rests on COMPUTING historical IV/greeks ourselves (Polygon Developer serves
them only in the CURRENT snapshot). This script recomputes IV from each contract's EOD close +
underlying + risk-free rate + dividend yield using app.options.pricing_engine, then compares to
Polygon's own served IV (and delta) for the SAME current snapshot — the one window where we have
ground truth. It reports the error distribution and PASS/FAILs against the OPT-1 tolerance.

Upgrades over the OPT-0 spike (scripts/spike_options_iv_check.py):
  * American exercise via Bjerksund-Stensland (equity options are American), not European-only.
  * Real per-underlying dividend yield (from Polygon /v3/reference/dividends), not q=0 — this is
    what drives the all-contract mean bias toward zero on dividend payers.
  * Engine delta vs served delta cross-check (greeks sanity, not just IV).
  * PASS/FAIL exit code so it can run as a nightly health gate.

OPT-1 acceptance (the confidence gate):
  * near-ATM median |IV err|  < 0.010 (1 vol-point)
  * all-contract mean IV bias  in [-0.010, +0.010]

Run: PYTHONIOENCODING=utf-8 python -m scripts.validate_options_engine
     PYTHONIOENCODING=utf-8 python -m scripts.validate_options_engine --european   # ablation
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import requests

from app.config import settings
from app.options.pricing_engine import ENGINE

UNDERLYINGS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
               "TSLA", "JPM", "XLE", "GLD", "TLT", "AMD"]
R = 0.043                 # flat risk-free proxy (OPT-2 will wire a real rate series)
EXP_LO, EXP_HI = 20, 45   # days-to-expiry window
NEAR_ATM_BAND = 0.15      # |delta - 0.5| < band  => "near-ATM"
MIN_VOL = 10              # contracts with day-volume below this have stale closes (the
#                          live snapshot pairs an option's last trade with the LIVE spot;
#                          untraded tails are stale -> a data-timing artifact, not engine
#                          error. We'd never trade those marks anyway. EOD-bar backtests
#                          (OPT-1b) pair same-day option+underlying closes, so no mismatch.

# OPT-1 acceptance thresholds
TOL_NEAR_ATM_MED = 0.010
TOL_ALL_BIAS = 0.010

_BASE = "https://api.polygon.io"


def _dividend_yield(underlying: str, key: str, spot: float) -> float:
    """Annualized cash-dividend yield from the most recent declared dividend × frequency / spot.
    Returns 0.0 on any failure or non-payer (the safe, unbiased default)."""
    if not spot or spot <= 0:
        return 0.0
    try:
        r = requests.get(f"{_BASE}/v3/reference/dividends", params={
            "apiKey": key, "ticker": underlying, "limit": 1,
            "order": "desc", "sort": "ex_dividend_date",
        }, timeout=20)
        res = (r.json().get("results") or [])
        if not res:
            return 0.0
        d = res[0]
        cash = float(d.get("cash_amount") or 0.0)
        freq = int(d.get("frequency") or 0)   # 0,1,2,4,12 payments/yr
        if cash <= 0 or freq <= 0:
            return 0.0
        return (cash * freq) / spot
    except Exception:
        return 0.0


def _fetch_snapshot(underlying: str, key: str):
    """CURRENT options snapshot rows carrying served IV + delta + EOD close + spot."""
    today = dt.date.today()
    rows = []
    for ct in ("call", "put"):
        r = requests.get(f"{_BASE}/v3/snapshot/options/{underlying}", params={
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
                "exp": d["expiration_date"], "kind": ct,
                "price": float(close), "poly_iv": float(iv), "poly_delta": float(delta),
                "volume": float(day.get("volume") or 0.0),
            })
    return rows


def _stats(label, e):
    if not e:
        print(f"  {label:<34}: no data")
        return None
    a = np.array(e)
    med = float(np.median(np.abs(a)))
    bias = float(a.mean())
    print(f"  {label:<34}: n={len(a):>4}  median|err|={med:.4f}  "
          f"mean_bias={bias:+.4f}  p90|err|={np.percentile(np.abs(a), 90):.4f}")
    return {"n": len(a), "median_abs": med, "bias": bias}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--european", action="store_true",
                    help="ablation: price European (q=0 BS) instead of American + dividends")
    ap.add_argument("--no-dividends", action="store_true", help="force q=0")
    ap.add_argument("--min-vol", type=float, default=MIN_VOL,
                    help="drop contracts with day-volume below this (stale-close filter)")
    args = ap.parse_args()
    style = "european" if args.european else "american"

    key = settings.polygon_api_key
    today = dt.date.today()
    errs, near_atm_errs, delta_errs = [], [], []
    n_contracts = 0

    for u in UNDERLYINGS:
        try:
            rows = _fetch_snapshot(u, key)
        except Exception as exc:
            print(f"  {u}: snapshot fetch failed: {exc}")
            continue
        spot = rows[0]["S"] if rows else 0.0
        q = 0.0 if (args.no_dividends or args.european) else _dividend_yield(u, key, spot)
        kept = 0
        for row in rows:
            if row["volume"] < args.min_vol:
                continue
            T = (dt.date.fromisoformat(row["exp"]) - today).days / 365.0
            if T <= 0:
                continue
            kept += 1
            my_iv = ENGINE.implied_vol(row["price"], row["S"], row["K"], T, R, q,
                                       row["kind"], style=style)
            if my_iv is None:
                continue
            errs.append(my_iv - row["poly_iv"])
            n_contracts += 1
            if abs(abs(row["poly_delta"]) - 0.5) < NEAR_ATM_BAND:
                near_atm_errs.append(my_iv - row["poly_iv"])
            # greek cross-check: engine delta at the served IV vs served delta
            g = ENGINE.greeks(row["S"], row["K"], T, R, q, row["poly_iv"],
                              row["kind"], style=style)
            delta_errs.append(g["delta"] - row["poly_delta"])
        print(f"  {u}: {len(rows)} contracts, {kept} kept (vol>={args.min_vol:g}) (q={q:.4f})")

    print("\n" + "=" * 80)
    print(f"  OPT-1a ENGINE VALIDATION — computed {style} IV vs Polygon served IV "
          f"(r={R:.3f}, vol>={args.min_vol:g})")
    print("=" * 80)
    all_stat = _stats("ALL contracts (IV)", errs)
    atm_stat = _stats("NEAR-ATM (IV)", near_atm_errs)
    _stats("ALL contracts (delta)", delta_errs)
    print("=" * 80)

    if all_stat is None or atm_stat is None:
        print("  RESULT: NO DATA (market closed / no snapshot) — cannot evaluate.")
        return 2
    atm_ok = atm_stat["median_abs"] < TOL_NEAR_ATM_MED
    bias_ok = abs(all_stat["bias"]) < TOL_ALL_BIAS
    print(f"  near-ATM median|err| {atm_stat['median_abs']:.4f} < {TOL_NEAR_ATM_MED}: "
          f"{'PASS' if atm_ok else 'FAIL'}")
    print(f"  all-contract |bias|  {abs(all_stat['bias']):.4f} < {TOL_ALL_BIAS}: "
          f"{'PASS' if bias_ok else 'FAIL'}")
    ok = atm_ok and bias_ok
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 80)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
