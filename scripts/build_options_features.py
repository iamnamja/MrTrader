"""
build_options_features.py — build the daily options FEATURE TABLE
(data/options_features.parquet) from the computed-greeks store
(Alpha-v6 Phase 4, PR4a).

One row per (underlying, date), via app/data/options_features.py, from each
underlying's greeks-store slice (data/options_greeks/underlying={U}/part-0.parquet,
built by scripts/backfill_computed_greeks.py). Equity share volume (for
opt_share_volume_ratio) comes from the PIT daily-bars loader reused from
app/research/event_panel._get_daily_bars.

OUTPUT FORMAT — SINGLE FILE: data/options_features.parquet. The table is small
(one row per name-day, ~hundreds of rows/name vs the greeks store's ~1M), and the
two consumers — app/data/options_quality.filter_options_universe and the P4 L/S
scorers — read the WHOLE table and filter by knowable_date (PIT). A single file
is the cleanest read for them. RESUME is via per-underlying intermediate parts
under data/options_features_parts/ (one .parquet per underlying, written
atomically); the final single file is a concat of the parts. A crashed run
re-runs only the missing parts. --force recomputes.

Structure mirrors scripts/backfill_computed_greeks.py: argparse, per-underlying
resumable parts + atomic (tmp + os.replace) writes, progress logging, and a final
coverage summary (per-feature non-NaN coverage + coverage_flags histogram).

Usage
-----
    python -m scripts.build_options_features --smoke               # SPY AAPL NVDA MSFT AMZN
    python -m scripts.build_options_features --underlyings SPY QQQ
    python -m scripts.build_options_features --workers 8           # full (resumes)
    python -m scripts.build_options_features --start 2023-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import pandas as pd  # noqa: E402

from app.data.options_features import (  # noqa: E402
    CF_MISSING_25D_PUT, CF_MISSING_60D_TENOR, CF_MISSING_ATM, CF_MISSING_CPIV,
    CF_MISSING_FRONT, CF_NO_EQUITY_VOLUME, CF_THIN_CHAIN, FEATURE_COLS,
    assemble_underlying_features, filter_dates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_options_features")

_ROOT = Path(__file__).resolve().parent.parent
GREEKS_DIR = _ROOT / "data" / "options_greeks"
OUT_FILE = _ROOT / "data" / "options_features.parquet"
PARTS_DIR = _ROOT / "data" / "options_features_parts"

SMOKE_UNDERLYINGS = ["SPY", "AAPL", "NVDA", "MSFT", "AMZN"]

# Feature columns whose non-NaN coverage we report in the final summary.
_COVERAGE_FEATURES = [
    "atm_iv_30d", "implied_move_front", "cpiv_matched_delta", "skew_25d_put",
    "term_slope_30_60", "iv_rv_20d_ratio", "opt_share_volume_ratio",
    "put_call_volume_ratio", "opt_volume_z",
]
_FLAG_NAMES = {
    CF_MISSING_ATM: "missing_atm", CF_MISSING_25D_PUT: "missing_25d_put",
    CF_MISSING_60D_TENOR: "missing_60d_tenor", CF_NO_EQUITY_VOLUME: "no_equity_volume",
    CF_THIN_CHAIN: "thin_chain", CF_MISSING_CPIV: "missing_cpiv",
    CF_MISSING_FRONT: "missing_front",
}


def _all_underlyings() -> List[str]:
    """Every underlying with a greeks-store part file."""
    if not GREEKS_DIR.exists():
        return []
    return sorted(p.name.split("=", 1)[1] for p in GREEKS_DIR.glob("underlying=*")
                  if (p / "part-0.parquet").exists())


def _equity_bars(symbol: str, dates: pd.Series) -> Optional[pd.DataFrame]:
    """Daily equity bars (split-adjusted close + share volume) covering the
    feature date span, reusing event_panel._get_daily_bars (PIT, split-healed).
    Returns a date-indexed OHLCV frame or None on failure. The start is padded
    ~45 calendar days BEFORE the first feature date so the 20d realized-vol
    (which needs ~21 prior closes) is warm from the first emitted row."""
    if dates is None or dates.empty:
        return None
    try:
        from app.data.cache import get_cache
        from app.research.event_panel import _get_daily_bars
        start = pd.Timestamp(dates.min()).date() - timedelta(days=45)
        end = pd.Timestamp(dates.max()).date() + timedelta(days=5)
        return _get_daily_bars(get_cache(), symbol, start, end)
    except Exception as exc:  # never fail the build on a volume-source hiccup
        logger.warning("  %s: equity bars unavailable (%s) — opt_share_volume_ratio "
                       "will be NaN", symbol, type(exc).__name__)
        return None


def process_underlying(underlying: str, greeks_dir: str, parts_dir: str,
                       start: Optional[str], end: Optional[str]) -> dict:
    """Build one underlying's feature rows and write its part atomically. Returns
    a summary dict (rows + per-flag counts) for the coverage roll-up."""
    if not logging.getLogger().handlers:  # spawned process: no inherited config
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
    t0 = time.time()
    part = Path(greeks_dir) / f"underlying={underlying}" / "part-0.parquet"
    if not part.exists():
        return {"underlying": underlying, "rows": 0, "elapsed_s": 0.0}
    greeks = pd.read_parquet(part)
    if "underlying" not in greeks.columns:
        greeks["underlying"] = underlying  # in-file schema omits it (hive partition)

    # Equity bars for the O/S ratio (span = the chain's own dates).
    dates = pd.to_datetime(greeks["date"]).drop_duplicates()
    eq_bars = _equity_bars(underlying, dates)

    feats = assemble_underlying_features(greeks, equity_bars=eq_bars)
    # PIT history (iv_rv, opt_volume_z) is built across the full slice, THEN the
    # requested [start, end] window is applied (so a windowed build still has
    # correct trailing stats).
    s = date.fromisoformat(start) if start else None
    e = date.fromisoformat(end) if end else None
    feats = filter_dates(feats, s, e)

    Path(parts_dir).mkdir(parents=True, exist_ok=True)
    final = Path(parts_dir) / f"{underlying}.parquet"
    tmp = Path(parts_dir) / f".{underlying}.parquet.tmp"
    feats.to_parquet(tmp, index=False)
    os.replace(tmp, final)

    flag_counts = {name: int((feats["coverage_flags"] & bit).astype(bool).sum())
                   for bit, name in _FLAG_NAMES.items()} if not feats.empty else {}
    return {"underlying": underlying, "rows": int(len(feats)),
            "flags": flag_counts, "elapsed_s": round(time.time() - t0, 1)}


def pending_underlyings(underlyings: List[str], parts_dir: Path,
                        force: bool) -> List[str]:
    """Resume filter: drop underlyings whose part already exists (unless --force)."""
    if force:
        return list(underlyings)
    return [u for u in underlyings if not (parts_dir / f"{u}.parquet").exists()]


def assemble_final(parts_dir: Path, out_file: Path) -> int:
    """Concat EVERY underlying part present under parts_dir into the single
    feature table (atomic write). Deliberately assembles ALL parts on disk — not
    just this run's requested subset — so a single-name refresh
    (`--underlyings AAPL`) re-runs only that part yet still rebuilds the FULL
    table and never silently TRUNCATES data/options_features.parquet down to the
    subset. (For an isolated subset output, point --parts-dir/--output elsewhere.)
    Returns the total row count."""
    frames = []
    parts = sorted(p for p in parts_dir.glob("*.parquet")
                   if not p.name.startswith("."))  # skip .tmp atomic-write staging
    for p in parts:
        df = pd.read_parquet(p)
        if not df.empty:
            frames.append(df)
    if not frames:
        logger.warning("no non-empty parts under %s — %s not written",
                       parts_dir, out_file)
        return 0
    full = pd.concat(frames, ignore_index=True)
    full = full[FEATURE_COLS].sort_values(["date", "underlying"], kind="mergesort")
    full = full.reset_index(drop=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.parent / f".{out_file.name}.tmp"
    full.to_parquet(tmp, index=False)
    os.replace(tmp, out_file)
    logger.info("assembled %d parts -> %d rows", len(frames), len(full))
    return len(full)


def _coverage_summary(out_file: Path) -> None:
    """Print per-feature non-NaN coverage + a coverage_flags histogram + a sample
    row (the smoke-build evidence)."""
    if not out_file.exists():
        return
    df = pd.read_parquet(out_file)
    n = len(df)
    logger.info("=" * 78)
    logger.info("FEATURE TABLE: %d rows, %d underlyings, %s -> %s",
                n, df["underlying"].nunique(),
                df["date"].min().date() if n else "-",
                df["date"].max().date() if n else "-")
    logger.info("per-feature non-NaN coverage:")
    for col in _COVERAGE_FEATURES:
        cov = float(df[col].notna().mean()) * 100.0 if n else 0.0
        logger.info("  %-24s %6.1f%%", col, cov)
    flag_hist = {name: int((df["coverage_flags"] & bit).astype(bool).sum())
                 for bit, name in _FLAG_NAMES.items()}
    logger.info("coverage_flags histogram (rows with each bit set): %s", flag_hist)
    if n:
        # A populated sample row: prefer one with the full IV surface present.
        full_mask = df[["atm_iv_30d", "cpiv_matched_delta", "term_slope_30_60"]] \
            .notna().all(axis=1)
        sample = df[full_mask].iloc[0] if full_mask.any() else df.iloc[n // 2]
        logger.info("sample row:\n%s", sample.to_string())
    logger.info("=" * 78)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build the daily options feature table from the greeks store")
    p.add_argument("--underlyings", nargs="+", default=None,
                   help="Subset (default: every underlying with a greeks part)")
    p.add_argument("--smoke", action="store_true",
                   help=f"Build only {SMOKE_UNDERLYINGS} (path proof; overrides "
                        f"--underlyings)")
    p.add_argument("--start", type=str, default=None, help="Min date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="Max date (YYYY-MM-DD)")
    p.add_argument("--output", type=str, default=str(OUT_FILE))
    p.add_argument("--parts-dir", type=str, default=str(PARTS_DIR))
    p.add_argument("--workers", type=int, default=4, help="Process pool size")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if an underlying's part already exists")
    args = p.parse_args()

    out_file = Path(args.output)
    parts_dir = Path(args.parts_dir)

    if args.smoke:
        underlyings = list(SMOKE_UNDERLYINGS)
    elif args.underlyings:
        underlyings = [u.upper() for u in args.underlyings]
    else:
        underlyings = _all_underlyings()
    if not underlyings:
        logger.error("no underlyings to build (greeks store empty at %s?)", GREEKS_DIR)
        return 1

    todo = pending_underlyings(underlyings, parts_dir, args.force)
    if len(todo) < len(underlyings):
        logger.info("resume: skipping %d underlyings with existing parts",
                    len(underlyings) - len(todo))

    t0 = time.time()
    summaries: List[dict] = []
    if todo:
        # workers=1 -> run inline (smoke / debugging is easier to read).
        if args.workers <= 1:
            for i, u in enumerate(todo, 1):
                s = process_underlying(u, str(GREEKS_DIR), str(parts_dir),
                                       args.start, args.end)
                summaries.append(s)
                logger.info("[%d/%d] %s: %d rows in %.1fs", i, len(todo), u,
                            s.get("rows", 0), s.get("elapsed_s", 0.0))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(process_underlying, u, str(GREEKS_DIR),
                                  str(parts_dir), args.start, args.end): u
                        for u in todo}
                done = 0
                for fut in as_completed(futs):
                    u = futs[fut]
                    try:
                        s = fut.result()
                    except Exception as exc:
                        logger.error("  %s FAILED: %s", u, exc)
                        s = {"underlying": u, "rows": 0, "error": str(exc)}
                    summaries.append(s)
                    done += 1
                    logger.info("[%d/%d] %s: %d rows in %.1fs", done, len(todo), u,
                                s.get("rows", 0), s.get("elapsed_s", 0.0))

    total = assemble_final(parts_dir, out_file)
    logger.info("assembled %s: %d rows in %.1fs", out_file, total, time.time() - t0)
    _coverage_summary(out_file)
    n_failed = sum(1 for s in summaries if "error" in s)
    if n_failed:
        logger.warning("%d underlyings errored", n_failed)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
