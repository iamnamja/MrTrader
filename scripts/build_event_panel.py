"""
build_event_panel.py — build data/event_panel.parquet (Alpha-v6 Phase 3, PR3).

The earnings-event research table (app/research/event_panel.py — read its
module docstring for the FROZEN methodological decisions and every column
definition). One row per (symbol, announce_date), R1K PIT, 2019->2026,
equity columns only; option columns reserved as all-NaN for Phase 2.

What a build run does, in order:
  1. resolve the universe (PIT R1K union over the window via quarterly
     members_at checkpoints, unless --symbols/--smoke overrides);
  2. load bars (DataCache read-through + yfinance fill), ^VIX, SPY, sector
     ETFs, sector map, FMP /stable/earnings histories (limit=120, on-disk
     cached);
  3. assemble the panel (sacred-holdout guarded, PIT by construction);
  4. run the FMP announce-date spot check (>=20 well-known prints, +-1 day
     tolerance) — logged; failures WARN loudly but do not abort (FMP is the
     population source either way; the check is the calibration evidence);
  5. run validate_panel_pit on a sample — a single mismatch ABORTS the build
     (a leaking panel must never be written);
  6. atomic parquet write (tmp + os.replace);
  7. print the coverage report (events/yr, qualified/yr, %full-20d-window).

Usage:
  python scripts/build_event_panel.py --start 2019-01-01 --end 2026-06-10
  python scripts/build_event_panel.py --smoke            # 3 symbols, tiny, fast
  python scripts/build_event_panel.py --symbols AAPL,MSFT --output data/p.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

logger = logging.getLogger("build_event_panel")

DEFAULT_START = date(2019, 1, 1)
DEFAULT_OUTPUT = ROOT / "data" / "event_panel.parquet"
SMOKE_SYMBOLS = ["AAPL", "MSFT", "NVDA"]
SMOKE_OUTPUT = ROOT / "data" / "event_panel_smoke.parquet"

# FMP announce-date spot check: well-known, widely-reported earnings dates.
# Tolerance +-1 calendar day (vendor date conventions differ on BMO/AMC
# boundaries — announce_ts_flag is UNK in PR3 by frozen decision 4).
SPOT_CHECK_PRINTS = [
    ("AAPL", "2023-05-04"), ("AAPL", "2024-05-02"), ("AAPL", "2024-08-01"),
    ("AAPL", "2024-11-01"), ("AAPL", "2025-01-30"),
    ("MSFT", "2024-04-25"), ("MSFT", "2024-10-30"),
    ("NVDA", "2024-05-22"), ("NVDA", "2024-08-28"), ("NVDA", "2024-11-20"),
    ("NVDA", "2025-02-26"),
    ("META", "2024-04-24"), ("META", "2024-07-31"), ("META", "2024-10-30"),
    ("GOOGL", "2024-04-25"), ("GOOGL", "2024-10-29"),
    ("AMZN", "2024-04-30"), ("AMZN", "2024-10-31"),
    ("TSLA", "2024-04-23"), ("TSLA", "2024-10-23"), ("TSLA", "2025-01-29"),
    ("JPM", "2024-04-12"), ("JPM", "2024-10-11"),
]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [event_panel] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def resolve_universe(start: date, end: date, index: str = "russell1000") -> list:
    """PIT R1K union over [start, end] sampled at QUARTERLY checkpoints —
    unlike the two-endpoint pit_union, this catches names that both joined
    and left mid-window (a 7-year window has many)."""
    from app.data.universe_history import members_at

    syms: set = set()
    d = start
    while d <= end:
        syms.update(members_at(index, d))
        d += timedelta(days=91)
    syms.update(members_at(index, end))
    # Index/synthetic inputs are never panel events.
    syms -= {"SPY", "^VIX", "VIX"}
    return sorted(syms)


def fmp_spot_check(prints=SPOT_CHECK_PRINTS, tolerance_days: int = 1) -> dict:
    """The 5-minute FMP announce-date audit: for each well-known print, does
    the fetched earnings history contain a record within +-tolerance days?
    Logged per-print; returns {checked, hits, misses: [...]}.
    """
    from app.research.event_panel import fetch_earnings_history

    hits = 0
    misses = []
    for sym, expected in prints:
        exp = date.fromisoformat(expected)
        recs = fetch_earnings_history(sym)
        dates = []
        for r in recs:
            try:
                dates.append(date.fromisoformat(str(r["date"])[:10]))
            except (ValueError, TypeError):
                continue
        hit = any(abs((d - exp).days) <= tolerance_days for d in dates)
        if hit:
            hits += 1
        else:
            nearest = min(dates, key=lambda d: abs((d - exp).days)) if dates else None
            misses.append({"symbol": sym, "expected": expected,
                           "nearest": nearest.isoformat() if nearest else None})
            logger.warning("SPOT-CHECK MISS %s %s (nearest FMP record: %s)",
                           sym, expected, nearest)
    logger.info("FMP announce-date spot check: %d/%d well-known prints matched "
                "(+-%dd tolerance)", hits, len(prints), tolerance_days)
    return {"checked": len(prints), "hits": hits, "misses": misses}


def atomic_write_parquet(panel, path: Path) -> None:
    """tmp + os.replace, mirroring DataCache._write_df (C3 atomicity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    panel.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def coverage_report(panel) -> str:
    """ASCII coverage table: events/yr, qualified/yr, % full 20d window."""
    import pandas as pd
    from app.research.event_panel import QF_INCOMPLETE_FWD20

    lines = []
    bar = "-" * 72
    lines.append(bar)
    lines.append(f"  {'year':>6} | {'events':>7} | {'qualified':>9} | "
                 f"{'full 20d window':>16}")
    lines.append(bar)
    years = pd.to_datetime(panel["announce_date"]).dt.year
    for yr in sorted(years.unique()):
        sub = panel[years == yr]
        full = (sub["quality_flags"] & QF_INCOMPLETE_FWD20) == 0
        lines.append(f"  {yr:>6} | {len(sub):>7d} | "
                     f"{int(sub['pead_qualified'].sum()):>9d} | "
                     f"{100.0 * full.mean():>15.1f}%")
    full_all = (panel["quality_flags"] & QF_INCOMPLETE_FWD20) == 0
    lines.append(bar)
    lines.append(f"  {'TOTAL':>6} | {len(panel):>7d} | "
                 f"{int(panel['pead_qualified'].sum()):>9d} | "
                 f"{100.0 * full_all.mean():>15.1f}%")
    lines.append(bar)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the earnings-event panel (data/event_panel.parquet)")
    ap.add_argument("--start", default=DEFAULT_START.isoformat(),
                    metavar="YYYY-MM-DD", help="first announce date (2019-01-01)")
    ap.add_argument("--end", default=None, metavar="YYYY-MM-DD",
                    help="last announce date (default: today - 1 day)")
    ap.add_argument("--output", default=None,
                    help=f"output parquet ({DEFAULT_OUTPUT.relative_to(ROOT)}; "
                         f"--smoke defaults to {SMOKE_OUTPUT.relative_to(ROOT)})")
    ap.add_argument("--symbols", default=None,
                    help="comma-separated symbol override (skips universe resolve)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"tiny smoke build on {SMOKE_SYMBOLS} -> "
                         f"{SMOKE_OUTPUT.name} (path-proving only)")
    ap.add_argument("--pit-sample", type=int, default=25,
                    help="events sampled for the PIT validation (25)")
    ap.add_argument("--skip-spot-check", action="store_true",
                    help="skip the FMP announce-date spot check (tests only)")
    args = ap.parse_args(argv)

    _setup_logging()
    try:
        if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from app.research.event_panel import build_event_panel, validate_panel_pit

    start = date.fromisoformat(args.start)
    end = (date.fromisoformat(args.end) if args.end
           else datetime.now().date() - timedelta(days=1))

    if args.smoke:
        symbols = list(SMOKE_SYMBOLS)
        output = Path(args.output) if args.output else SMOKE_OUTPUT
        logger.info("SMOKE build: %s (path-proving only, NOT the research panel)",
                    symbols)
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        output = Path(args.output) if args.output else DEFAULT_OUTPUT
    else:
        symbols = resolve_universe(start, end)
        output = Path(args.output) if args.output else DEFAULT_OUTPUT
        logger.info("PIT R1K universe over %s -> %s: %d symbols",
                    start, end, len(symbols))

    panel, inputs = build_event_panel(symbols, start, end)
    if panel.empty:
        logger.error("Panel is EMPTY — nothing written.")
        return 1

    # FMP announce-date spot check (calibration evidence; WARNs, never aborts).
    spot = None
    if not args.skip_spot_check:
        spot = fmp_spot_check()

    # PIT validation — a leaking panel is never written.
    pit = validate_panel_pit(
        panel, bars=inputs["bars"], spy_bars=inputs["spy_bars"],
        vix_close=inputs["vix_close"], earnings=inputs["earnings"],
        sample_n=args.pit_sample,
    )
    if not pit["ok"]:
        logger.error("PIT VALIDATION FAILED (%d mismatches) — panel NOT written: %s",
                     len(pit["mismatches"]), pit["mismatches"][:5])
        return 1
    logger.info("PIT validation OK: %d sampled events reproduce from "
                "announce-truncated inputs", pit["checked"])

    atomic_write_parquet(panel, output)
    logger.info("Panel written: %s (%d events, %d columns)",
                output, len(panel), panel.shape[1])

    print()
    print("  EVENT PANEL COVERAGE" + ("   [SMOKE]" if args.smoke else ""))
    print(coverage_report(panel))
    if spot is not None:
        print(f"  FMP spot check: {spot['hits']}/{spot['checked']} well-known "
              f"prints matched (misses: {len(spot['misses'])})")
    print(f"  PIT validation: OK on {pit['checked']} sampled events")
    print(f"  Output: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
