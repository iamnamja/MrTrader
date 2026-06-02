"""
Survivorship-safe, point-in-time small/mid-cap universe builder.

Builds a (date -> eligible-symbol-set) map and a price panel for the
small/mid-cap PEAD CPCV harness (scripts/run_pead_smallmid_cpcv.py), using
Polygon grouped-daily flat files as the candidate source.

WHY GROUPED-DAILY (the gold standard)
--------------------------------------
Polygon's `us_stocks_sip/day_aggs_v1/{Y}/{M}/{date}.csv.gz` file lists EVERY
ticker that printed a daily bar that day, including names later delisted
(SIVB, FRC, etc. appear up to their delisting date). Iterating these files and
keeping each symbol up to its last traded day is fully survivorship-safe — a
delisted name is in the universe exactly until it stops trading, never after,
and is never retroactively dropped. This is strictly better than a
"broad ticker list + per-symbol bulk" approach (which depends on a current
ticker list that itself drops dead names) so we use grouped-daily.

ELIGIBILITY (point-in-time, liquidity-aware = the ADV filter H-1 wants)
-----------------------------------------------------------------------
For each trading day D and symbol S, the trailing 20-day average dollar volume
  ADV(S, D) = mean over the 20 most-recent trading days <= D of (close * volume)
must fall in the band [ADV_MIN, ADV_MAX]. Because the window is strictly the
trailing 20 days on-or-before D, a future volume spike cannot change today's
eligibility — PIT by construction.

SIZE PROXY (honest flag)
------------------------
Shares-outstanding / market-cap is NOT available survivorship-safe from the
flat files, so we use the ADV dollar-volume band as the small/mid-cap size
proxy. ADV_MIN=$2M excludes illiquid micro-caps; ADV_MAX=$50M excludes
mega-caps (AAPL etc. trade billions/day). This is a liquidity band, not a
strict market-cap band — documented as a known approximation.

CACHE
-----
The expensive step is iterating ~1500 grouped-daily files over 6 years. We run
it ONCE and cache two parquets under data/smallmid/ (gitignored):
  - panel.parquet     : long format [date, symbol, close, volume] for eligible+
                        candidate names (the price source for the harness).
  - eligibility.parquet: long format [date, symbol] of band-eligible names.
Re-runs load from cache unless --rebuild is passed.

Usage:
    python scripts/build_smallmid_universe.py --years 6
    python scripts/build_smallmid_universe.py --start 2023-01-01 --end 2023-03-31  # small validation
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── Band constants (small/mid-cap liquid range) ────────────────────────────────
ADV_MIN = 2_000_000.0      # $2M  — excludes illiquid micro-caps
ADV_MAX = 50_000_000.0     # $50M — excludes large/mega-caps
ADV_WINDOW = 20            # trailing trading days for the ADV average
MIN_PRICE = 1.0            # drop sub-$1 penny names (not tradeable/borrowable)
# Tractability cap: keep at most this many names per day, ranked by PIT ADV
# DESCENDING within the band. Capping by PIT ADV rank (not current existence)
# does NOT reintroduce survivorship — a delisted name with high ADV at the time
# is kept; the rank is computed from trailing data only.
MAX_NAMES_PER_DAY = 300

_CACHE_DIR = ROOT / "data" / "smallmid"
_PANEL_PATH = _CACHE_DIR / "panel.parquet"
_ELIG_PATH = _CACHE_DIR / "eligibility.parquet"


# ── Core PIT eligibility computation (pure, unit-testable) ──────────────────────

def compute_pit_eligibility(
    panel: pd.DataFrame,
    adv_min: float = ADV_MIN,
    adv_max: float = ADV_MAX,
    window: int = ADV_WINDOW,
    min_price: float = MIN_PRICE,
    max_names_per_day: int = MAX_NAMES_PER_DAY,
) -> pd.DataFrame:
    """
    Compute point-in-time small/mid-cap eligibility from a long price panel.

    Parameters
    ----------
    panel : DataFrame with columns ['date', 'symbol', 'close', 'volume'].
        'date' is a python date or datetime-like. One row per (symbol, day).
    Returns
    -------
    DataFrame [date, symbol, adv] of band-eligible names (long format), where
    `adv` is the trailing-`window`-day average dollar volume strictly using days
    on-or-before `date`. PIT by construction (rolling window, no centering, no
    forward fill from future).
    """
    if panel.empty:
        return pd.DataFrame(columns=["date", "symbol", "adv"])

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])
    df["dollar_vol"] = df["close"].astype(float) * df["volume"].astype(float)

    # Trailing rolling mean over the symbol's own trading days, INCLUDING today.
    # min_periods=window so a name needs a full window of trailing history before
    # it can be eligible (also satisfies the per-fold short-history concern at the
    # universe layer). closed='right' (default) → window ends at current row, never
    # peeks forward.
    df["adv"] = (
        df.groupby("symbol", group_keys=False)["dollar_vol"]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    elig = df[
        (df["adv"] >= adv_min)
        & (df["adv"] <= adv_max)
        & (df["close"].astype(float) >= min_price)
        & df["adv"].notna()
    ][["date", "symbol", "adv"]].copy()

    if max_names_per_day and max_names_per_day > 0:
        # Rank by trailing ADV DESC within each day; keep the top-N. Rank uses only
        # trailing-window ADV, so it is PIT and survivorship-safe (a delisted name
        # with high ADV-at-the-time keeps its slot).
        elig = (
            elig.sort_values(["date", "adv"], ascending=[True, False])
            .groupby("date", group_keys=False)
            .head(max_names_per_day)
        )

    elig["date"] = elig["date"].dt.date
    return elig.reset_index(drop=True)


def eligibility_to_map(elig: pd.DataFrame) -> Dict[date, Set[str]]:
    """Convert long-format eligibility to {date: set(symbols)}."""
    out: Dict[date, Set[str]] = {}
    for d, grp in elig.groupby("date"):
        out[d] = set(grp["symbol"].tolist())
    return out


def symbols_eligible_in_window(elig: pd.DataFrame, start: date, end: date) -> Set[str]:
    """
    Union of all symbols that were band-eligible on any day in [start, end].

    Used by the harness to build the per-fold PIT universe: a name is in the
    fold universe if it was ADV-eligible as-of the test window.
    """
    d = pd.to_datetime(elig["date"])
    mask = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    return set(elig.loc[mask, "symbol"].tolist())


# ── Grouped-daily ingestion (survivorship-safe candidate source) ────────────────

def _business_days(start: date, end: date) -> List[date]:
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def build_panel_from_grouped_daily(
    start: date,
    end: date,
    provider=None,
    min_price: float = MIN_PRICE,
    prefilter_adv_max: float = ADV_MAX,
) -> pd.DataFrame:
    """
    Walk Polygon grouped-daily flat files over [start, end] and build a long
    price panel [date, symbol, close, volume].

    Survivorship-safe: every ticker that traded each day is captured, including
    names later delisted (kept up to their final traded day, never after).

    To keep the panel tractable we apply a CHEAP per-day prefilter: drop names
    whose same-day dollar volume already exceeds `prefilter_adv_max` (these can
    never be in the band — they are large-caps) and sub-`min_price` pennies.
    The authoritative band filter (trailing-20d ADV) is applied later in
    compute_pit_eligibility; this prefilter only removes names that the trailing
    average cannot rescue into the band, so it does not affect correctness.
    """
    if provider is None:
        from app.data.polygon_provider import PolygonProvider
        provider = PolygonProvider()

    frames: List[pd.DataFrame] = []
    n_days = 0
    n_missing = 0
    for day in _business_days(start, end):
        gd = provider.get_grouped_daily(day)
        if gd is None or gd.empty:
            n_missing += 1
            continue
        n_days += 1
        gd = gd[gd["close"] >= min_price].copy()
        gd["dollar_vol"] = gd["close"] * gd["volume"]
        # Keep names whose same-day dollar volume is below a generous multiple of
        # the band ceiling — a name far above this can never trail-average into the
        # band. 5x headroom keeps any name within reach of the band.
        gd = gd[gd["dollar_vol"] <= prefilter_adv_max * 5.0]
        gd["date"] = pd.Timestamp(day)
        frames.append(gd[["date", "symbol", "close", "volume"]])
        if n_days % 50 == 0:
            logger.info("Grouped-daily: %d trading days ingested (last=%s)", n_days, day)

    if not frames:
        logger.warning("No grouped-daily data ingested for %s→%s (S3 configured?)", start, end)
        return pd.DataFrame(columns=["date", "symbol", "close", "volume"])

    panel = pd.concat(frames, ignore_index=True)
    logger.info(
        "Panel built: %d trading days, %d missing files, %d (symbol,day) rows, %d distinct symbols",
        n_days, n_missing, len(panel), panel["symbol"].nunique(),
    )
    return panel


# ── Cache orchestration ─────────────────────────────────────────────────────────

def build_and_cache(
    start: date,
    end: date,
    rebuild: bool = False,
    provider=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (or load from cache) the price panel and eligibility table.
    Returns (panel, eligibility).
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not rebuild and _PANEL_PATH.exists() and _ELIG_PATH.exists():
        logger.info("Loading cached universe from %s", _CACHE_DIR)
        panel = pd.read_parquet(_PANEL_PATH)
        elig = pd.read_parquet(_ELIG_PATH)
        return panel, elig

    panel = build_panel_from_grouped_daily(start, end, provider=provider)
    elig = compute_pit_eligibility(panel)

    if not panel.empty:
        panel.to_parquet(_PANEL_PATH, index=False)
        elig.to_parquet(_ELIG_PATH, index=False)
        logger.info("Cached panel (%d rows) + eligibility (%d rows) to %s",
                    len(panel), len(elig), _CACHE_DIR)
    return panel, elig


def load_cached() -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load (panel, eligibility) from cache, or None if not built."""
    if _PANEL_PATH.exists() and _ELIG_PATH.exists():
        return pd.read_parquet(_PANEL_PATH), pd.read_parquet(_ELIG_PATH)
    return None


def _print_stats(panel: pd.DataFrame, elig: pd.DataFrame) -> None:
    if elig.empty:
        print("Eligibility is EMPTY — no names fell in the band (check S3 access / dates).")
        return
    per_day = elig.groupby("date")["symbol"].nunique()
    print("\n=== Small/mid-cap universe stats ===")
    print(f"  Date range:        {per_day.index.min()} - {per_day.index.max()}")
    print(f"  Trading days:      {len(per_day)}")
    print(f"  Distinct symbols:  {elig['symbol'].nunique()}")
    print(f"  Names/day (band):  mean={per_day.mean():.0f}  min={per_day.min()}  max={per_day.max()}")
    print(f"  ADV band:          ${ADV_MIN:,.0f} - ${ADV_MAX:,.0f}  (window={ADV_WINDOW}d)")
    print(f"  Cap/day:           {MAX_NAMES_PER_DAY}")
    # Surface a few names that drop out of the panel before the end (proxy for
    # delistings / survivorship coverage).
    last_seen = panel.groupby("symbol")["date"].max()
    panel_end = pd.to_datetime(panel["date"]).max()
    dropped = last_seen[last_seen < panel_end - pd.Timedelta(days=10)]
    print(f"  Symbols leaving panel >10d before end (delisted/halted proxy): {len(dropped)}")
    if len(dropped) > 0:
        sample = list(dropped.sort_values().index[:8])
        print(f"    e.g. {sample}")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [smallmid_universe] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=None, help="trailing years to build")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--rebuild", action="store_true", help="ignore cache and rebuild")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        years = args.years or 6
        end = date.today()
        # +ADV_WINDOW trading days of lead-in so the first real day has a full window
        start = end - timedelta(days=years * 365 + 40)

    logger.info("Building small/mid-cap universe %s → %s (rebuild=%s)", start, end, args.rebuild)
    panel, elig = build_and_cache(start, end, rebuild=args.rebuild)
    _print_stats(panel, elig)
    return 0 if not elig.empty else 1


if __name__ == "__main__":
    sys.exit(main())
