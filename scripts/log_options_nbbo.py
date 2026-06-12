"""
Nightly options NBBO/spread logger (FUSE A — Alpha-v6 P1c slow fuse).

Why this exists: we have NO historical options quote data (EOD OHLCV only), so the
simulators guess spreads from heuristics. This job snapshots the live chain's bid/ask
once per day at 15:55 ET for a FROZEN panel of liquid underlyings and appends the
flattened spread observations to data/options_spread_obs.parquet. After ~4-6 weeks the
accumulated panel calibrates a spread-structure model — the fitter/CalibratedSpreadModel
is DELIBERATELY deferred until that data exists (do not build it yet).

Source is Alpaca's ``indicative`` feed (free tier; possibly delayed/synthesized), NOT
Polygon — the $79 Polygon plan serves no options NBBO (see app/data/alpaca_options.py).
Every row records the feed name so the quality caveat stays auditable. Idempotent:
re-runs the same day dedup on (contract, obs_date) keep-last.

Usage
-----
    python scripts/log_options_nbbo.py                 # full frozen panel
    python scripts/log_options_nbbo.py --panel SPY     # subset (smoke test / catch-up)
    python scripts/log_options_nbbo.py --force         # run despite a non-trading day

Scheduled nightly at 15:55 ET via app/orchestrator.py (job_id=options_nbbo_logger).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import pandas as pd  # noqa: E402

from app.data.alpaca_options import (  # noqa: E402
    fetch_latest_underlying_prices, fetch_option_snapshots,
)
from app.data.options_provider import parse_occ  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("log_options_nbbo")

# FROZEN observation panel — broad-index ETFs + liquid single names, seeded from
# scripts/backfill_options.DEFAULT_UNDERLYINGS (the names we backtest/trade) plus a
# few more high-OI singles for spread-structure coverage. Do NOT edit casually: the
# spread fitter needs a STABLE panel so per-name spread series stay comparable across
# the accumulation window. If it must change, append (never remove) and note the date.
PANEL = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX",
    "JPM", "BAC", "GS", "XOM", "INTC", "BA", "DIS", "MU", "AVGO", "CRM",
    "XLE", "GLD", "TLT",
]

MAX_DTE = 70          # expiration cap — bounds chain size/pagination per underlying
FEED = "indicative"   # Alpaca free-tier options feed (recorded per row)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = _DATA_DIR / "options_spread_obs.parquet"

OBS_COLS = [
    "obs_date", "obs_ts", "feed", "underlying", "contract", "contract_type",
    "strike", "expiration", "dte", "bid", "ask", "mid", "spread_pct",
    "bid_size", "ask_size", "iv", "oi", "day_close", "day_volume",
    "underlying_price", "moneyness",
]


def _et_today():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return datetime.now().date()


def _is_trading_day(d) -> bool:
    """True iff *d* is an NYSE session day (weekday AND not a market holiday).
    Reuses the NYSE calendar behind options_provider.knowable_date — the 15:55 job
    is weekday-scheduled, so without this it fires on holidays and logs phantom
    obs_date rows carrying the PRIOR session's quotes."""
    from app.data.options_provider import _NYSE_BDAY
    return bool(_NYSE_BDAY.is_on_offset(pd.Timestamp(d)))


def flatten_snapshots(
    underlying: str,
    snapshots: Dict[str, dict],
    obs_date,
    *,
    feed: str = FEED,
    underlying_price: Optional[float] = None,
    max_dte: int = MAX_DTE,
) -> Tuple[List[dict], int]:
    """Flatten one underlying's snapshot dict into observation rows. Pure (no I/O).

    Returns (rows, n_dropped) where n_dropped counts contracts skipped for having no
    bid or no ask, a CROSSED quote (bid > ask — a bad print on the indicative feed
    whose negative spread_pct would poison the spread fitter; locked bid == ask is
    kept), or an unparseable/expired OCC ticker — logged so a one-sided, crossed, or
    empty book shows up in the run summary instead of vanishing silently.

    Alpaca snapshot keys are OCC tickers WITHOUT the ``O:`` prefix; we store the
    prefixed form so `contract` joins directly against options_bars.parquet.
    """
    rows: List[dict] = []
    dropped = 0
    obs_date = pd.Timestamp(obs_date).normalize()
    for occ, snap in (snapshots or {}).items():
        contract = occ if occ.startswith("O:") else f"O:{occ}"
        meta = parse_occ(contract)
        if not meta:
            dropped += 1
            continue
        dte = (pd.Timestamp(meta["expiration"]) - obs_date).days
        if dte < 0 or dte > max_dte:
            dropped += 1
            continue
        quote = snap.get("latestQuote") or {}
        bid, ask = quote.get("bp"), quote.get("ap")
        if not bid or not ask or bid <= 0 or ask <= 0:
            dropped += 1
            continue
        if bid > ask:  # crossed book — drop (counted), never a negative spread_pct
            dropped += 1
            continue
        mid = 0.5 * (bid + ask)
        day = snap.get("dailyBar") or {}
        spot = underlying_price if underlying_price and underlying_price > 0 else None
        rows.append({
            "obs_date": obs_date,
            "obs_ts": pd.to_datetime(quote.get("t"), utc=True, errors="coerce"),
            "feed": feed,
            "underlying": underlying,
            "contract": contract,
            "contract_type": meta["contract_type"],
            "strike": float(meta["strike"]),
            "expiration": pd.Timestamp(meta["expiration"]),
            "dte": int(dte),
            "bid": float(bid),
            "ask": float(ask),
            "mid": float(mid),
            "spread_pct": float((ask - bid) / mid),
            "bid_size": float(quote.get("bs") or 0.0),
            "ask_size": float(quote.get("as") or 0.0),
            "iv": _opt_float(snap.get("impliedVolatility")),
            "oi": _opt_float(snap.get("openInterest")),   # not served on indicative today
            "day_close": _opt_float(day.get("c")),
            "day_volume": _opt_float(day.get("v")),
            "underlying_price": spot if spot is not None else float("nan"),
            "moneyness": float(meta["strike"]) / spot if spot else float("nan"),
        })
    return rows, dropped


def _opt_float(v) -> float:
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def append_observations(new_rows: pd.DataFrame, out_path: Path = OUT_PATH) -> int:
    """Append+dedup to the spread-obs store on (contract, obs_date), keep last —
    re-running the same day overwrites that day's rows (idempotent). Atomic write
    (tmp + os.replace, pattern scripts/merge_options_parquet.py)."""
    if new_rows is None or new_rows.empty:
        return 0
    frames = [new_rows[OBS_COLS]]
    if out_path.exists():
        frames.insert(0, pd.read_parquet(out_path))
    merged = (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["contract", "obs_date"], keep="last")
              .sort_values(["obs_date", "underlying", "contract"])
              .reset_index(drop=True))[OBS_COLS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return len(merged)


def run_nbbo_logging(
    panel: Optional[List[str]] = None,
    *,
    max_dte: int = MAX_DTE,
    out_path: Path = OUT_PATH,
    obs_date=None,
    force: bool = False,
) -> dict:
    """Snapshot the chain for every panel underlying and append the spread rows.

    Sync entry point (the orchestrator runs it via run_in_executor). Per-underlying
    failures are logged and skipped; a TOTAL failure (zero rows) is logged as an error
    and reported in the summary (status="no_data") — no notifier event type fits a
    data-logger failure, so the orchestrator's error log is the alert channel.
    ``obs_date`` defaults to today (ET); override only for tests (quotes are live-only,
    so a backdated obs_date would mislabel the observation day).

    Non-trading days (the 15:55 schedule is weekday-only but still fires on market
    HOLIDAYS) are skipped (status="skipped") — the chain's quotes would be the prior
    session's, logged under a phantom obs_date. ``force=True`` (CLI ``--force``)
    overrides for manual catch-up runs. FAIL-SAFE: if the calendar check itself
    errors, proceed with a warning — a missed calibration day is worse than a
    flagged one.
    """
    t0 = time.time()
    panel = [u.upper() for u in (panel or PANEL)]
    obs_date = obs_date or _et_today()
    if not force:
        try:
            trading = _is_trading_day(obs_date)
        except Exception as exc:
            logger.warning("trading-day check failed (%s) — proceeding fail-safe", exc)
            trading = True
        if not trading:
            summary = {"status": "skipped", "reason": "not_a_trading_day",
                       "obs_date": str(obs_date)}
            logger.info("NBBO logging SKIPPED: %s is not a trading day (holiday/"
                        "weekend) — quotes would be the prior session's. "
                        "Use --force to override.", obs_date)
            return summary
    logger.info("NBBO logging: %d underlyings, feed=%s, dte<=%d, obs_date=%s",
                len(panel), FEED, max_dte, obs_date)

    try:
        spots = fetch_latest_underlying_prices(panel)
    except Exception as exc:
        logger.warning("underlying price fetch failed (moneyness will be NaN): %s", exc)
        spots = {}

    all_rows: List[dict] = []
    dropped = 0
    failed: List[str] = []
    for u in panel:
        try:
            snaps = fetch_option_snapshots(
                u, feed=FEED, exp_lo=obs_date,
                exp_hi=obs_date + timedelta(days=max_dte))
        except Exception as exc:
            logger.error("  %s: snapshot fetch failed: %s", u, exc)
            failed.append(u)
            continue
        rows, n_drop = flatten_snapshots(u, snaps, obs_date,
                                         underlying_price=spots.get(u), max_dte=max_dte)
        all_rows.extend(rows)
        dropped += n_drop
        logger.info("  %s: %d contracts -> %d quoted rows (%d dropped no-bid/ask/dte)",
                    u, len(snaps), len(rows), n_drop)

    new_df = pd.DataFrame(all_rows, columns=OBS_COLS)
    total = append_observations(new_df, out_path) if not new_df.empty else None
    summary = {
        "status": "ok" if not new_df.empty else "no_data",
        "obs_date": str(obs_date),
        "feed": FEED,
        "underlyings": len(panel),
        "failed_underlyings": failed,
        "rows_written": len(new_df),
        "rows_dropped_no_quote": dropped,
        "store_rows_total": total,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if new_df.empty:
        logger.error("NBBO logging produced ZERO rows — feed down or after-hours blackout? %s",
                     summary)
    else:
        logger.info("NBBO logging done: %s", summary)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description="Nightly options NBBO/spread logger (Alpaca indicative feed)")
    p.add_argument("--panel", nargs="+", default=None,
                   help=f"Underlying subset (default: frozen {len(PANEL)}-name panel)")
    p.add_argument("--max-dte", type=int, default=MAX_DTE,
                   help="Drop contracts expiring beyond this many days")
    p.add_argument("--out", type=str, default=None,
                   help=f"Output parquet (default: {OUT_PATH})")
    p.add_argument("--force", action="store_true",
                   help="Run even on a non-trading day (manual catch-up/backfill; "
                        "the scheduled job skips holidays/weekends)")
    args = p.parse_args()
    out = Path(args.out) if args.out else OUT_PATH
    summary = run_nbbo_logging(args.panel, max_dte=args.max_dte, out_path=out,
                               force=args.force)
    return 0 if summary["status"] in ("ok", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
