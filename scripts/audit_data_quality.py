"""
audit_data_quality.py -- Alpha-v10: a read-only data-quality sweep over everything we've
saved/downloaded, to catch outliers / representation artifacts that could have silently
moved a research verdict (the way the CL 2020-04-21 negative-denominator sign-flip did).

Read-only. Prints a structured report; writes nothing. Run:
    venv/Scripts/python scripts/audit_data_quality.py
    venv/Scripts/python scripts/audit_data_quality.py --section futures
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.data import norgate_provider as ng
from app.research import futures_data as fd

DATA = "data"


# ----------------------------------------------------------------------------- helpers
def _flag(cond: bool) -> str:
    return "  !!" if cond else ""


def _stale_max_run(s: pd.Series) -> int:
    """Longest run of identical consecutive values (frozen-feed detector)."""
    if len(s) < 2:
        return len(s)                       # 0 for empty, 1 for a singleton
    same = (s.values[1:] == s.values[:-1])
    best = run = 0
    for x in same:
        run = run + 1 if x else 0
        best = max(best, run)
    return best + 1 if best else 1


def _ohlc_violations(df: pd.DataFrame) -> int:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bad = (h < l) | (h < o) | (h < c) | (l > o) | (l > c)
    return int(bad.sum())


def _ohlc_violation_magnitude(df: pd.DataFrame) -> float:
    """Worst OHLC breach as a fraction of price -- distinguishes a real broken bar
    from sub-penny adjusted-close vs raw-OHLC rounding (yfinance adjusts close but
    not O/H/L, so close can sit a hair outside [low, high])."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    over = (pd.concat([o, c], axis=1).max(axis=1) - h).clip(lower=0)
    under = (l - pd.concat([o, c], axis=1).min(axis=1)).clip(lower=0)
    breach = pd.concat([over, under, (l - h).clip(lower=0)], axis=1).max(axis=1)
    rel = (breach / c.abs().where(c.abs() > 0)).replace([np.inf, -np.inf], np.nan)
    return float(rel.max()) if rel.notna().any() else 0.0


def _gaps(idx: pd.DatetimeIndex, max_bday_gap: int) -> int:
    if len(idx) < 2:
        return 0
    d = pd.Series(idx).diff().dt.days.dropna()
    # count gaps larger than max_bday_gap calendar days (rough; weekends/holidays expected)
    return int((d > max_bday_gap).sum())


# ----------------------------------------------------------------------------- futures
def audit_futures() -> None:
    print("\n" + "=" * 90)
    print("NORGATE FUTURES  (data/norgate_futures/continuous/)")
    print("=" * 90)
    markets = sorted(f[:-8] for f in os.listdir(ng.CONTINUOUS_DIR) if f.endswith(".parquet"))
    print(f"{len(markets)} markets present\n")

    hdr = f"{'mkt':<6}{'rows':>6}{'dup':>5}{'nonmono':>8}{'ohlc':>6}{'neg_ua':>7}" \
          f"{'wins%':>7}{'stale':>6}{'rawmax%':>9}"
    print(hdr)
    print("-" * len(hdr))
    neg_ua_markets: List[Tuple[str, int]] = []
    high_wins: List[Tuple[str, float]] = []
    big_stale: List[Tuple[str, int]] = []
    for m in markets:
        try:
            ua = ng.load_continuous(m, price_type="unadjusted")
        except Exception as e:
            print(f"{m:<6} LOAD ERROR: {e}")
            continue
        idx = ua.index
        dup = int(pd.Series(idx).duplicated().sum())
        nonmono = int((pd.Series(idx).diff().dt.days.dropna() < 0).sum())
        ohlc = _ohlc_violations(ua) if {"open", "high", "low", "close"}.issubset(ua.columns) else -1
        neg_ua = int((ua["close"] <= 0).sum())
        # raw (un-winsorized) return distribution via the true-return formula, cap=None
        r_raw = fd.true_returns(m, cap=None)
        rawmax = float(r_raw.abs().max() * 100) if len(r_raw) else 0.0
        # winsor hit rate = fraction of days the |raw return| exceeds the cap
        wins = float((r_raw.abs() > fd.RETURN_CAP).mean() * 100) if len(r_raw) else 0.0
        stale = _stale_max_run(ua["close"])
        if neg_ua:
            neg_ua_markets.append((m, neg_ua))
        if wins > 0.5:
            high_wins.append((m, wins))
        if stale >= 10:
            big_stale.append((m, stale))
        print(f"{m:<6}{len(ua):>6}{dup:>5}{nonmono:>8}{ohlc:>6}{neg_ua:>7}"
              f"{wins:>7.2f}{stale:>6}{rawmax:>9.1f}"
              + _flag(dup or nonmono or ohlc > 0 or wins > 1.0 or stale >= 20))

    print("\n--- futures summary ---")
    print(f"markets with negative unadjusted close (sign-flip risk; guard NaNs these): "
          f"{[m for m, _ in neg_ua_markets]}")
    print(f"high winsor-hit (>0.5% of days clipped): "
          f"{[(m, round(w, 2)) for m, w in sorted(high_wins, key=lambda x: -x[1])]}")
    print(f"long stale runs (>=10 identical closes): "
          f"{[(m, n) for m, n in sorted(big_stale, key=lambda x: -x[1])]}")


# ----------------------------------------------------------------------------- macro
def audit_macro() -> None:
    print("\n" + "=" * 90)
    print("MACRO HISTORY  (data/macro/macro_history.parquet)")
    print("=" * 90)
    p = os.path.join(DATA, "macro", "macro_history.parquet")
    if not os.path.exists(p):
        print("  (missing)")
        return
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
    print(f"rows {len(df)}  range {df.index.min().date()} -> {df.index.max().date()}")
    print(f"duplicate dates: {int(pd.Series(df.index).duplicated().sum())}")
    print(f"calendar gaps >5d: {_gaps(df.index, 5)}\n")
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        nans = int(s.isna().sum())
        nonpos = int((s <= 0).sum())
        ret = s.pct_change(fill_method=None).abs()
        mx = float(ret.max() * 100) if ret.notna().any() else 0.0
        print(f"  {col:<8} nan={nans:<4} nonpos={nonpos:<4} max_1d_move={mx:>6.1f}%"
              + _flag(nans > 0 or nonpos > 0 or mx > 60))
    if {"vix", "vix3m"}.issubset(df.columns):
        inv = int((pd.to_numeric(df["vix"]) > pd.to_numeric(df["vix3m"])).sum())
        print(f"\n  vix>vix3m (backwardation) days: {inv} "
              f"({100*inv/len(df):.1f}% -- real spikes; flag only if absurd)")


# ----------------------------------------------------------------------------- finra
def audit_finra() -> None:
    print("\n" + "=" * 90)
    print("FINRA SHORT VOLUME  (data/finra_short_volume.parquet)")
    print("=" * 90)
    p = os.path.join(DATA, "finra_short_volume.parquet")
    if not os.path.exists(p):
        print("  (missing)")
        return
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
    print(f"rows {len(df)}  range {df.index.min().date()} -> {df.index.max().date()}")
    print(f"duplicate dates: {int(pd.Series(df.index).duplicated().sum())}  "
          f"calendar gaps >5d: {_gaps(df.index, 5)}\n")
    ratio_cols = [c for c in df.columns if "ratio" in c.lower()]
    for c in ratio_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        oob = int(((s < 0) | (s > 1)).sum())
        nans = int(s.isna().sum())
        print(f"  {c:<22} nan={nans:<4} out-of-[0,1]={oob}"
              + _flag(oob > 0))
    # aggregate volumes should be positive
    for c in [c for c in df.columns if "vol" in c.lower() and "ratio" not in c.lower()]:
        s = pd.to_numeric(df[c], errors="coerce")
        print(f"  {c:<22} nonpos={int((s <= 0).sum())}")


# ----------------------------------------------------------------------------- fundamentals
def audit_fundamentals() -> None:
    print("\n" + "=" * 90)
    print("FMP FUNDAMENTALS  (data/fundamentals/fmp_fundamentals_history.parquet)")
    print("=" * 90)
    p = os.path.join(DATA, "fundamentals", "fmp_fundamentals_history.parquet")
    if not os.path.exists(p):
        print("  (missing)")
        return
    df = pd.read_parquet(p)
    print(f"rows {len(df)}  symbols {df['symbol'].nunique() if 'symbol' in df else '?'}")
    if {"symbol", "as_of_date"}.issubset(df.columns):
        dup = int(df.duplicated(["symbol", "as_of_date"]).sum())
        print(f"duplicate (symbol, as_of_date): {dup}" + _flag(dup > 0))
    # filing date must be on/after the period it reports
    if {"as_of_date", "period_end"}.issubset(df.columns):
        a = pd.to_datetime(df["as_of_date"], errors="coerce")
        pe = pd.to_datetime(df["period_end"], errors="coerce")
        impossible = int((a < pe).sum())
        print(f"as_of_date < period_end (look-ahead/impossible filing): {impossible}"
              + _flag(impossible > 0))
    for col in [c for c in ("shares_outstanding", "revenue") if c in df.columns]:
        s = pd.to_numeric(df[col], errors="coerce")
        print(f"  {col:<20} negative={int((s < 0).sum())}  nan={int(s.isna().sum())}"
              + _flag((s < 0).sum() > 0))


# ----------------------------------------------------------------------------- equities
def audit_equities(max_symbols: int = 10_000) -> None:
    print("\n" + "=" * 90)
    print("EQUITY DAILY CACHE  (data/cache/daily/*.parquet)")
    print("=" * 90)
    d = os.path.join(DATA, "cache", "daily")
    if not os.path.isdir(d):
        print("  (missing)")
        return
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))[:max_symbols]
    print(f"{len(files)} symbols cached\n")
    n_ohlc = n_nonpos = n_dup = n_stale = n_bigret = 0
    worst_ret: List[Tuple[str, float, str]] = []
    worst_stale: List[Tuple[str, int]] = []
    for f in files:
        sym = f[:-8]
        try:
            df = pd.read_parquet(os.path.join(d, f))
        except Exception:
            continue
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            continue
        df = df.sort_index()
        ohlc = _ohlc_violations(df)
        ohlc_mag = _ohlc_violation_magnitude(df)
        nonpos = int((df["close"] <= 0).sum())
        dup = int(pd.Series(df.index).duplicated().sum())
        stale = _stale_max_run(df["close"])
        ret = df["close"].pct_change(fill_method=None).abs()
        mx = float(ret.max()) if ret.notna().any() else 0.0
        mxdate = str(ret.idxmax().date()) if ret.notna().any() and mx > 0 else "-"
        # only count a MATERIAL OHLC breach (>0.5%) -- below that is adj-close vs
        # raw-OHLC rounding, not a broken bar
        n_ohlc += ohlc > 0 and ohlc_mag > 0.005
        n_nonpos += nonpos > 0
        n_dup += dup > 0
        if stale >= 20:
            n_stale += 1
            worst_stale.append((sym, stale))
        if mx > 2.0:                       # >200% one-day move = data-error candidate (splits)
            n_bigret += 1
            worst_ret.append((sym, mx * 100, mxdate))
    print(f"symbols with MATERIAL OHLC breach (>0.5%): {n_ohlc}")
    print(f"symbols with non-positive close: {n_nonpos}")
    print(f"symbols with duplicate dates : {n_dup}")
    print(f"symbols with stale run >=20  : {n_stale}")
    print(f"symbols with >200% 1-day move : {n_bigret} (split/data-error candidates)")
    print("\n  worst 1-day moves:")
    for sym, pct, dt in sorted(worst_ret, key=lambda x: -x[1])[:20]:
        print(f"    {sym:<8} {pct:>9.0f}%  {dt}")
    print("\n  longest stale runs:")
    for sym, n in sorted(worst_stale, key=lambda x: -x[1])[:15]:
        print(f"    {sym:<8} {n} identical closes")


SECTIONS = {
    "futures": audit_futures,
    "macro": audit_macro,
    "finra": audit_finra,
    "fundamentals": audit_fundamentals,
    "equities": audit_equities,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=list(SECTIONS) + ["all"], default="all")
    args = ap.parse_args()
    if args.section == "all":
        for fn in SECTIONS.values():
            fn()
    else:
        SECTIONS[args.section]()
    print("\n" + "=" * 90)
    print("DONE -- read-only sweep complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
