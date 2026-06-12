"""
Backfill computed IV + greeks for every (contract, date) bar in data/options_bars.parquet
(FUSE B — Alpha-v6 P2 slow fuse).

Polygon serves NO historical IV/greeks, so we compute them from the EOD close +
underlying raw close + risk-free rate + dividend yield via app/options/pricing_engine
(the OPT-1a validated engine). Output is per-underlying hive-partitioned parquet:
``data/options_greeks/underlying={U}/part-0.parquet`` — one file per underlying gives
free resume (skip existing) and bounded memory.

THE solver hot path (mandatory — measured, do not "simplify" back to the naive call):
a naive ``ENGINE.implied_vol(..., style="american")`` costs ~31 ms/solve for puts
because the bisection's sigma->1e-4 probe routes q<r puts into the CRR-500 fallback
(pricing_engine.american_price -> _bjs_degenerate). Instead we solve the EUROPEAN IV
first (pure BS bisection, ~0.016 ms) and refine the American root by bisection
BRACKETED in ~[0.4*ivE, 2.5*ivE] (the American IV is always <= the European IV, so the
root lives in (0, ivE]); the tight bracket never probes sigma->0, so CRR never fires.
Measured: ~0.108 ms/solve with exact IV recovery -> ~0.4h wall @8 workers vs ~122h naive.

Inputs (rate series, per-underlying dividend-yield schedule, raw closes) are prepared
ONCE in the parent — workers are pure CPU (ProcessPoolExecutor; this is compute-bound,
not I/O-bound like backfill_options.py's thread pool):
  * Risk-free: FRED DGS3MO daily series (decimal, calendar forward-filled). OPT-1a used
    a flat 0.043; this wires the real curve front-end. Flat fallback if FRED is down.
  * Dividend yield: Polygon /v3/reference/dividends full history -> at each ex-date,
    q = TTM cash dividends / raw close as-of ex-date (step function; q=0 before the
    first ex-date and for non-payers). Generalizes scripts/validate_options_engine.py.
  * Underlying closes: Polygon /v2/aggs adjusted=false — TRULY as-traded, same
    provenance/scale as the store's unadjusted OCC strikes across splits. yfinance is
    NOT usable here: auto_adjust=False only skips DIVIDEND adjustment, closes stay
    SPLIT-adjusted (NVDA 2023-06-01 prints 39.77 vs the store's ~394 strikes).

Usage
-----
    python scripts/backfill_computed_greeks.py --spike --underlyings SPY   # timing probe
    python scripts/backfill_computed_greeks.py --workers 6                 # full (resumes)
    python scripts/backfill_computed_greeks.py --underlyings SPY QQQ --force
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from app.notifications import notifier  # noqa: E402
from app.options.pricing_engine import ENGINE, american_price, bs_price  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_computed_greeks")

_ROOT = Path(__file__).resolve().parent.parent
BARS_PATH = _ROOT / "data" / "options_bars.parquet"
CONTRACTS_PATH = _ROOT / "data" / "options_contracts.parquet"
OUT_DIR = _ROOT / "data" / "options_greeks"

DEFAULT_RATE = 0.043          # flat fallback if FRED is unreachable (OPT-1a's proxy)
RATE_SERIES = "DGS3MO"
RATE_START = date(2022, 1, 1)  # store starts ~2022-06; margin for ffill

IV_LO, IV_HI = 1e-4, 3.0       # engine's sane-vol bracket (pricing_engine.implied_vol)
WARM_LO_MULT, WARM_HI_MULT = 0.4, 2.5

# In-file columns. `underlying` is NOT stored in-file — the hive partition directory
# (underlying={U}) supplies it when the dataset root is read, and duplicating it would
# clash with pyarrow hive partitioning.
GREEKS_COLS = ["contract", "date", "knowable_date", "contract_type", "strike",
               "expiration", "close", "underlying_close", "volume", "stale_flag",
               "iv", "delta", "gamma", "vega", "theta", "solver_status"]

SOLVER_STATUSES = ("ok", "european_fallback", "below_intrinsic", "pinned",
                   "out_of_bracket", "no_underlying", "expired")


def _intrinsic(S: float, K: float, kind: str) -> float:
    return max(0.0, (S - K) if kind == "call" else (K - S))


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start American IV solver (the R2-measured hot path)
# ─────────────────────────────────────────────────────────────────────────────

def solve_iv_american(price, S, K, T, r, q, kind) -> Tuple[Optional[float], str]:
    """American IV via European warm start -> (iv, solver_status).

    Status mapping mirrors pricing_engine.implied_vol's None-paths EXPLICITLY (the
    engine collapses them all to None; a 112M-row backfill must never silently
    default): below_intrinsic (incl. price<=0 degenerate quotes), pinned (price on a
    zero-vol floor — intrinsic for American, or the BS sigma->0 limit — so vol is not
    identifiable), out_of_bracket (not attainable within [1e-4, 3.0]). `expired` /
    `no_underlying` are assigned by the caller before pricing.

    The American bisection bracket's low edge is clamped strictly above the
    CRR-degenerate threshold (b_arg*T + 2*sigma*sqrt(T) < 0 — pricing_engine
    ._bjs_degenerate) so american_price never falls back to the 500-step binomial.
    If the warm bracket does not straddle the root (large early-exercise premium),
    fall back to the European IV — the best identifiable answer at ~0.02ms — and
    label it ``european_fallback`` (NOT ``ok``): the true American IV is <= ivE, so
    the fallback is biased, concentrated in deep-ITM/high-early-exercise rows;
    downstream must be able to filter it.
    """
    if T <= 0:
        return None, "expired"
    if price is None or price <= 0 or S <= 0 or K <= 0:
        return None, "below_intrinsic"
    intr = _intrinsic(S, K, kind)
    if price < intr - 1e-8:
        return None, "below_intrinsic"
    if (intr > 0 and price <= intr + 1e-7) or \
            abs(price - bs_price(S, K, T, r, q, IV_LO, kind)) < 1e-7:
        return None, "pinned"

    iv_e = ENGINE.implied_vol(price, S, K, T, r, q, kind, style="european")
    if iv_e is None:
        return None, "out_of_bracket"

    b_arg = (r - q) if kind == "call" else (q - r)
    crr_floor = max(0.0, -b_arg) * math.sqrt(T) / 2.0
    lo = max(WARM_LO_MULT * iv_e, crr_floor * 1.05 + 1e-6, IV_LO)
    hi = min(WARM_HI_MULT * iv_e, IV_HI)
    if hi <= lo:
        return iv_e, "european_fallback"

    def f(sig):
        return american_price(S, K, T, r, q, sig, kind) - price

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return iv_e, "european_fallback"
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-7:
            return mid, "ok"
        if flo * fm < 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi), "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Parent-side input preparation (network I/O — done once, passed to workers)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_exc(exc: Exception) -> str:
    """Key-safe exception summary. requests' exception text (incl. what
    raise_for_status() raises) embeds the FULL request URL — which carries
    ``apiKey=``/``api_key=`` query params for Polygon/FRED — so never log
    ``str(exc)``; log only the type name + HTTP status code."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"{type(exc).__name__}(HTTP {status})" if status else type(exc).__name__


def build_rate_series(start: date = RATE_START, end: Optional[date] = None,
                      series_id: str = RATE_SERIES) -> Dict[date, float]:
    """Daily risk-free series from FRED (decimal), calendar forward-filled (weekends/
    holidays carry the last print; the head is back-filled so every store date maps).
    Official API when fred_api_key is set (app/macro/fred_client._fetch caps at 24 obs,
    so we fetch directly with the full window); fredgraph CSV (keyless) otherwise.
    Empty dict on total failure -> workers fall back to DEFAULT_RATE flat."""
    end = end or date.today()
    obs: List[Tuple[date, float]] = []
    try:
        from app.config import settings
        api_key = getattr(settings, "fred_api_key", None)
        if api_key:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params={"series_id": series_id, "api_key": api_key,
                                     "file_type": "json", "sort_order": "asc",
                                     "observation_start": start.isoformat()},
                             timeout=30)
            r.raise_for_status()
            for o in r.json().get("observations", []):
                if o.get("value") not in (None, "", "."):
                    obs.append((date.fromisoformat(o["date"]), float(o["value"]) / 100.0))
        if not obs:
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                             params={"id": series_id, "cosd": start.isoformat()},
                             timeout=30)
            r.raise_for_status()
            for line in r.text.splitlines()[1:]:
                d, _, v = line.partition(",")
                if v.strip() not in ("", "."):
                    obs.append((date.fromisoformat(d.strip()), float(v) / 100.0))
    except Exception as exc:
        logger.warning("FRED %s fetch failed (%s) — flat %.3f fallback",
                       series_id, _safe_exc(exc), DEFAULT_RATE)
        return {}
    if not obs:
        return {}
    s = pd.Series(dict(obs)).sort_index()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(pd.date_range(start, end, freq="D")).ffill().bfill()
    logger.info("rate series %s: %d days  %s→%s  last=%.4f",
                series_id, len(s), start, end, float(s.iloc[-1]))
    return {ts.date(): float(v) for ts, v in s.items()}


def fetch_raw_closes(underlying: str, start: str = "2021-06-01") -> Dict[date, float]:
    """AS-TRADED (truly unadjusted) daily closes from Polygon aggregates
    (``adjusted=false``) — same provenance as the bars store and the same scale as
    its unadjusted OCC strikes across splits.

    yfinance must NEVER be used here: ``auto_adjust=False`` only skips DIVIDEND
    adjustment — its closes stay SPLIT-adjusted, so S is wrong by the split ratio
    for every pre-split bar of a split name (NVDA 2023-06-01: yfinance 39.77 vs the
    store's ~394 strikes; as-traded is 397.70), and it inflates the dividend yield
    (nominal cash / split-adjusted close). Date convention matches the daily-aggs
    pattern in app/data/polygon_provider._fetch_daily_rest (t = epoch ms UTC ->
    normalized UTC date == the ET trading date). Empty dict on failure or empty
    response — the underlying's rows are then marked ``no_underlying`` LOUDLY;
    never silently fall back to a split-adjusted source.
    """
    from app.config import settings
    key = settings.polygon_api_key
    if not key:
        logger.warning("  %s: POLYGON_API_KEY missing — no closes "
                       "(all rows -> no_underlying)", underlying)
        return {}
    closes: Dict[date, float] = {}
    url = (f"https://api.polygon.io/v2/aggs/ticker/{underlying.upper()}"
           f"/range/1/day/{start}/{date.today().isoformat()}")
    params: Dict[str, object] = {"adjusted": "false", "limit": 50000, "apiKey": key}
    try:
        for _ in range(20):  # next_url pagination cap (50k/page >> the store window)
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            body = r.json()
            for res in body.get("results") or []:
                c = res.get("c")
                if c is not None and float(c) > 0:
                    d = pd.to_datetime(res["t"], unit="ms", utc=True).normalize()
                    closes[d.date()] = float(c)
            nxt = body.get("next_url")
            if not nxt:
                break
            url, params = nxt, {"apiKey": key}
    except Exception as exc:
        logger.warning("  %s: Polygon unadjusted closes FAILED (%s) — all rows -> "
                       "no_underlying (NO yfinance fallback: it is split-adjusted)",
                       underlying, _safe_exc(exc))
        return {}
    if not closes:
        logger.warning("  %s: Polygon unadjusted closes EMPTY — all rows -> "
                       "no_underlying", underlying)
    return closes


def build_div_schedule(underlying: str,
                       closes: Dict[date, float]) -> List[Tuple[date, float]]:
    """Step-function dividend-yield schedule [(ex_date, q), ...] ascending.

    q at each ex-date = TTM cash dividends / raw close as-of that ex-date
    (generalizes validate_options_engine._dividend_yield from latest-only to a full
    PIT-stepped history). Empty list (=> q=0) for non-payers or on any failure —
    the safe, unbiased default the validator also uses.

    Known limitation (accepted): the schedule is a step function that holds the LAST
    ex-date's TTM q forever — a name that suspends/cuts dividends keeps a stale
    positive q after its final ex-date instead of decaying to 0. Affects only
    post-suspension rows; revisit if a panel name actually suspends.
    """
    from app.config import settings
    key = settings.polygon_api_key
    if not key or not closes:
        return []
    try:
        divs: List[Tuple[date, float]] = []
        url = "https://api.polygon.io/v3/reference/dividends"
        params = {"apiKey": key, "ticker": underlying, "limit": 1000,
                  "order": "asc", "sort": "ex_dividend_date"}
        for _ in range(10):  # next_url pagination cap (1000/page is years of history)
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            body = r.json()
            for d in body.get("results") or []:
                cash = float(d.get("cash_amount") or 0.0)
                exd = d.get("ex_dividend_date")
                if cash > 0 and exd:
                    divs.append((date.fromisoformat(exd), cash))
            nxt = body.get("next_url")
            if not nxt:
                break
            url, params = nxt, {"apiKey": key}
        if not divs:
            return []
        divs.sort()
        close_dates = sorted(closes)
        sched: List[Tuple[date, float]] = []
        for i, (exd, _cash) in enumerate(divs):
            ttm = sum(c for d, c in divs if exd - timedelta(days=365) < d <= exd)
            j = bisect_right(close_dates, exd)
            if j == 0:
                continue  # no close on/before this ex-date -> can't form a yield
            spot = closes[close_dates[j - 1]]
            if spot > 0:
                sched.append((exd, ttm / spot))
        return sched
    except Exception as exc:
        logger.warning("  %s: dividend schedule failed (q=0 fallback): %s",
                       underlying, _safe_exc(exc))
        return []


def _q_asof(div_schedule: List[Tuple[date, float]], ex_dates: List[date], d: date) -> float:
    """Yield as of d = the most recent ex-date's q; 0.0 before the first ex-date.
    NOTE: holds the last TTM q indefinitely past the final ex-date (stale for
    dividend suspensions — see build_div_schedule's known limitation)."""
    i = bisect_right(ex_dates, d)
    return div_schedule[i - 1][1] if i > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Worker (pure CPU — one underlying per task)
# ─────────────────────────────────────────────────────────────────────────────

def process_underlying(underlying: str, bars_path: str, contracts_path: str,
                       out_dir: str, rates: Dict[date, float],
                       div_schedule: List[Tuple[date, float]],
                       closes: Dict[date, float],
                       log_every: int = 250_000) -> dict:
    """Compute iv/greeks for every (contract, date) bar of one underlying and write
    ``{out_dir}/underlying={U}/part-0.parquet`` atomically (tmp + os.replace).
    Returns a summary dict with the solver_status histogram."""
    if not logging.getLogger().handlers:  # spawned process: no inherited config
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
    t0 = time.time()
    bars = pd.read_parquet(bars_path, filters=[("underlying", "==", underlying)])
    if bars.empty:
        return {"underlying": underlying, "rows": 0, "statuses": {}, "elapsed_s": 0.0}
    meta = pd.read_parquet(
        contracts_path, filters=[("underlying", "==", underlying)],
        columns=["contract", "contract_type", "strike", "expiration"])
    bars = bars.merge(meta, on="contract", how="left")
    missing = bars["strike"].isna()
    if missing.any():  # contract absent from the universe file -> OCC-parse fallback
        from app.data.options_provider import parse_occ
        for idx in bars.index[missing]:
            m = parse_occ(bars.at[idx, "contract"])
            if m:
                bars.at[idx, "contract_type"] = m["contract_type"]
                bars.at[idx, "strike"] = m["strike"]
                bars.at[idx, "expiration"] = pd.Timestamp(m["expiration"])
    bars = bars.dropna(subset=["strike", "expiration"])

    ex_dates = [d for d, _ in div_schedule]
    statuses: Dict[str, int] = {s: 0 for s in SOLVER_STATUSES}
    out_rows: List[dict] = []
    n = len(bars)
    for i, row in enumerate(bars.itertuples(index=False), 1):
        d = row.date.date()
        exp = row.expiration.date() if hasattr(row.expiration, "date") else row.expiration
        kind = "call" if str(row.contract_type).lower().startswith("c") else "put"
        K = float(row.strike)
        price = float(row.close) if pd.notna(row.close) else None
        vol = float(row.volume) if pd.notna(row.volume) else float("nan")
        stale = (not math.isfinite(vol)) or vol == 0.0
        T = (exp - d).days / 365.0
        S = closes.get(d)
        r = rates.get(d, DEFAULT_RATE)
        q = _q_asof(div_schedule, ex_dates, d)

        if T <= 0:
            iv, status = None, "expired"
        elif S is None or S <= 0:
            iv, status = None, "no_underlying"
        else:
            iv, status = solve_iv_american(price, S, K, T, r, q, kind)
        if iv is not None:
            g = ENGINE.greeks(S, K, T, r, q, iv, kind, style="american")
            delta, gamma, vega, theta = g["delta"], g["gamma"], g["vega"], g["theta"]
        else:
            delta = gamma = vega = theta = float("nan")
        statuses[status] += 1
        out_rows.append({
            "contract": row.contract, "date": row.date,
            "knowable_date": row.knowable_date, "contract_type": kind,
            "strike": K, "expiration": pd.Timestamp(exp),
            "close": price if price is not None else float("nan"),
            "underlying_close": S if S is not None else float("nan"),
            "volume": vol, "stale_flag": bool(stale),
            "iv": iv if iv is not None else float("nan"),
            "delta": delta, "gamma": gamma, "vega": vega, "theta": theta,
            "solver_status": status,
        })
        if i % log_every == 0:
            rate_s = i / max(time.time() - t0, 1e-9)
            logging.info("  %s: %d/%d rows (%.0f rows/s, ETA %.0f min)",
                         underlying, i, n, rate_s, (n - i) / rate_s / 60.0)

    out = pd.DataFrame(out_rows, columns=GREEKS_COLS)
    part_dir = Path(out_dir) / f"underlying={underlying}"
    part_dir.mkdir(parents=True, exist_ok=True)
    final = part_dir / "part-0.parquet"
    # Dot-prefixed tmp name: pyarrow dataset discovery only ignores '.'/'_'-prefixed
    # files, so a crashed mid-write leftover must not be named part-0.parquet.tmp —
    # that would break every pd.read_parquet("data/options_greeks/") until cleaned.
    tmp = str(part_dir / ".part-0.parquet.tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, final)
    return {"underlying": underlying, "rows": len(out),
            "statuses": {k: v for k, v in statuses.items() if v},
            "elapsed_s": round(time.time() - t0, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Spike (timing probe) + main
# ─────────────────────────────────────────────────────────────────────────────

def run_spike(underlying: str, n: int, workers: int) -> int:
    """Time the warm-start solve on a sample of one underlying's real bars and project
    the full-store wall time. Run this BEFORE launching the full backfill — it is the
    guard against silently regressing onto the 31 ms/solve CRR path (R2 spike)."""
    rates = build_rate_series()
    closes = fetch_raw_closes(underlying)
    split_check = date(2023, 6, 1)  # pre-NVDA-split date: as-traded must be ~397, NOT 39.77
    if split_check in closes:
        logger.info("spike: %s as-traded close %s = %.2f (split sanity — must match the "
                    "store's unadjusted OCC strike scale, NOT a split-adjusted close)",
                    underlying, split_check, closes[split_check])
    sched = build_div_schedule(underlying, closes)
    ex_dates = [d for d, _ in sched]
    bars = pd.read_parquet(BARS_PATH, filters=[("underlying", "==", underlying)])
    meta = pd.read_parquet(CONTRACTS_PATH, filters=[("underlying", "==", underlying)],
                           columns=["contract", "contract_type", "strike", "expiration"])
    bars = bars.merge(meta, on="contract", how="inner").dropna(subset=["strike", "expiration"])
    sample = bars.sample(n=min(n, len(bars)), random_state=7)
    logger.info("spike: %s — %d sampled of %d bars", underlying, len(sample), len(bars))

    n_done = n_ok = 0
    statuses: Dict[str, int] = {s: 0 for s in SOLVER_STATUSES}
    t0 = time.perf_counter()
    for row in sample.itertuples(index=False):
        d = row.date.date()
        exp = row.expiration.date() if hasattr(row.expiration, "date") else row.expiration
        kind = "call" if str(row.contract_type).lower().startswith("c") else "put"
        T = (exp - d).days / 365.0
        S = closes.get(d)
        if T <= 0 or S is None:
            continue
        price = float(row.close) if pd.notna(row.close) else None
        iv, status = solve_iv_american(price, S, float(row.strike), T,
                                       rates.get(d, DEFAULT_RATE),
                                       _q_asof(sched, ex_dates, d), kind)
        n_done += 1
        n_ok += status == "ok"
        statuses[status] += 1
    elapsed = time.perf_counter() - t0
    ms = elapsed / max(n_done, 1) * 1000.0
    try:
        import pyarrow.parquet as pq
        total_rows = pq.ParquetFile(BARS_PATH).metadata.num_rows
    except Exception:
        total_rows = 0
    proj_h = total_rows * ms / 1000.0 / 3600.0 / max(workers, 1)
    logger.info("spike RESULT: %d solves in %.2fs -> %.3f ms/solve, solved(ok)=%.1f%%; "
                "store=%d rows -> projected ~%.1f h @%d workers (IV only; greeks add ~3x)",
                n_done, elapsed, ms, 100.0 * n_ok / max(n_done, 1),
                total_rows, proj_h, workers)
    logger.info("spike status histogram: %s", {k: v for k, v in statuses.items() if v})
    return 0 if ms < 1.0 else 1  # >1ms/solve means the CRR guard regressed — investigate


def _all_underlyings() -> List[str]:
    u = pd.read_parquet(CONTRACTS_PATH, columns=["underlying"])["underlying"].unique()
    return sorted(u)


def pending_underlyings(underlyings: List[str], out_dir: Path, force: bool) -> List[str]:
    """Resume filter: drop underlyings whose part file already exists (unless --force).
    Per-underlying output files make resume free — a crashed run just reruns this."""
    if force:
        return list(underlyings)
    return [u for u in underlyings
            if not (out_dir / f"underlying={u}" / "part-0.parquet").exists()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Backfill computed IV/greeks for data/options_bars.parquet")
    p.add_argument("--underlyings", nargs="+", default=None,
                   help="Subset (default: every underlying in options_contracts.parquet)")
    p.add_argument("--workers", type=int, default=6,
                   help="Process pool size (CPU-bound)")
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--max-underlyings", type=int, default=None, help="Cap (debug)")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if the underlying's output already exists "
                        "(default is --resume: skip existing part files)")
    p.add_argument("--spike", action="store_true",
                   help="Time N solves on one underlying and exit (no files written)")
    p.add_argument("--spike-n", type=int, default=2000, help="Spike sample size")
    args = p.parse_args()

    if args.spike:
        u = (args.underlyings or ["SPY"])[0].upper()
        return run_spike(u, args.spike_n, args.workers)

    out_dir = Path(args.out_dir)
    underlyings = ([u.upper() for u in args.underlyings] if args.underlyings
                   else _all_underlyings())
    if args.max_underlyings:
        underlyings = underlyings[: args.max_underlyings]
    todo = pending_underlyings(underlyings, out_dir, args.force)
    if len(todo) < len(underlyings):
        logger.info("resume: skipping %d underlyings with existing output",
                    len(underlyings) - len(todo))
    underlyings = todo
    if not underlyings:
        logger.info("nothing to do (all outputs exist — use --force to recompute)")
        return 0

    t0 = time.time()
    rates = build_rate_series()

    # Parent-side I/O prep (threaded — network-bound), then pure-CPU process pool.
    logger.info("preparing closes + dividend schedules for %d underlyings…",
                len(underlyings))
    prep: Dict[str, Tuple[Dict[date, float], List[Tuple[date, float]]]] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {}
        for u in underlyings:
            futs[ex.submit(fetch_raw_closes, u)] = u
        closes_by_u = {futs[f]: f.result() for f in as_completed(futs)}
        futs = {ex.submit(build_div_schedule, u, closes_by_u[u]): u for u in underlyings}
        for f in as_completed(futs):
            u = futs[f]
            prep[u] = (closes_by_u[u], f.result())
    no_close = [u for u in underlyings if not prep[u][0]]
    if no_close:
        logger.warning("%d underlyings have NO closes (all rows -> no_underlying): %s…",
                       len(no_close), no_close[:8])

    summaries: List[dict] = []
    total_statuses: Dict[str, int] = {s: 0 for s in SOLVER_STATUSES}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_underlying, u, str(BARS_PATH), str(CONTRACTS_PATH),
                          str(out_dir), rates, prep[u][1], prep[u][0]): u
                for u in underlyings}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                s = fut.result()
            except Exception as exc:
                logger.error("  %s FAILED: %s", u, exc)
                s = {"underlying": u, "rows": 0, "statuses": {}, "error": str(exc)}
            summaries.append(s)
            for k, v in s.get("statuses", {}).items():
                total_statuses[k] = total_statuses.get(k, 0) + v
            done += 1
            elapsed = time.time() - t0
            eta_min = elapsed / done * (len(underlyings) - done) / 60.0
            logger.info("[%d/%d] %s: %d rows in %.0fs  (ETA ~%.0f min)",
                        done, len(underlyings), u, s.get("rows", 0),
                        s.get("elapsed_s", 0), eta_min)

    total_rows = sum(s.get("rows", 0) for s in summaries)
    n_failed = sum(1 for s in summaries if "error" in s)
    hist = {k: v for k, v in total_statuses.items() if v}
    logger.info("=" * 78)
    logger.info("DONE: %d underlyings, %d rows in %.1f h  status histogram: %s  "
                "(%d underlyings errored)",
                len(summaries), total_rows, (time.time() - t0) / 3600.0, hist, n_failed)
    logger.info("=" * 78)
    notifier.enqueue("phase_complete", {
        "phase": "P2 greeks backfill (FUSE B)",
        "tasks_done": f"{len(summaries)} underlyings / {total_rows} (contract,date) rows",
        "outcome": f"solver histogram: {hist}; {n_failed} underlyings errored",
        "next_phase": "P2 IV-surface/term-structure features on computed greeks",
        "notes": f"out={out_dir}; rate=FRED {RATE_SERIES} "
                 f"({'live' if rates else f'flat {DEFAULT_RATE} fallback'}); "
                 f"warm-start American IV (R2 path)",
    })
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
