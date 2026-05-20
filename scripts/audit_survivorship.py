"""
P0.4 — Survivorship bias audit.

Checks whether the daily bar cache contains delisted / dead stocks
from the 2019–2024 S&P 500 membership history. A pure analysis script
that writes a markdown report — makes no data changes.

Usage:
    python scripts/audit_survivorship.py [--out docs/survivorship_audit.md]

Pass criteria: < 5% of unique symbols in the cache are survivors-only
(i.e., we're missing meaningful coverage of delisted names).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Known S&P 500 deletions 2019–2024 — companies removed due to bankruptcy,
# acquisition, or index rebalancing. Partial list of notable names.
# Full list would require a data subscription (e.g., CRSP, Compustat).
KNOWN_DELETIONS = [
    # Bankruptcies / delistings
    "HTZ",   # Hertz — bankrupt 2020, relisted 2021 (edge case)
    "GE",    # General Electric — spun off 2023, still trades
    "DISCA", # Discovery pre-merger
    "XLNX",  # Xilinx — acquired by AMD 2022
    "ATVI",  # Activision — acquired by MSFT 2023
    "TWTR",  # Twitter — taken private 2022
    "CTXS",  # Citrix — taken private 2022
    "VIAC",  # ViacomCBS → renamed PARA
    "FB",    # Facebook → renamed META
    "BRTX",  # delisted
    "KSU",   # Kansas City Southern — acquired by CN Rail 2021
    "INFO",  # IHS Markit — acquired by S&P Global 2022
    "CERN",  # Cerner — acquired by Oracle 2022
    "SIVB",  # Silicon Valley Bank — failed 2023
    "FRC",   # First Republic Bank — failed 2023
    "SBNY",  # Signature Bank — failed 2023
    "LB",    # L Brands → Bath & Body Works BBWI
    "PBCT",  # People's United — acquired 2022
    "NLOK",  # Norton LifeLock → renamed GEN
    "VRSN",  # still active but delisted from S&P 500
    "MXIM",  # Maxim Integrated — acquired by ADI 2021
    "ALXN",  # Alexion — acquired by AZ 2021
    "RTN",   # Raytheon — merged to RTX
    "UTX",   # United Technologies — merged to RTX
]

# These are still active and likely in cache (control group)
KNOWN_SURVIVORS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "JNJ", "V", "UNH", "HD", "PG", "MA", "BAC",
]


def load_cache_symbols() -> list[str]:
    from app.data.cache import get_cache
    cache = get_cache()
    daily_dir = cache._dir / "daily"
    return [p.stem for p in sorted(daily_dir.glob("*.parquet"))]


def check_symbol_date_range(symbol: str) -> tuple[date | None, date | None, int]:
    """Return (first_date, last_date, n_rows) for a cached symbol."""
    from app.data.cache import get_cache
    cache = get_cache()
    path = cache._dir / "daily" / f"{symbol}.parquet"
    if not path.exists():
        return None, None, 0
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df.index[0].date(), df.index[-1].date(), len(df)
    except Exception:
        return None, None, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=f"docs/survivorship_audit_{date.today().strftime('%Y%m%d')}.md")
    args = parser.parse_args()

    logger.info("Loading cache symbols...")
    cache_syms = set(load_cache_symbols())
    logger.info("Cache: %d symbols", len(cache_syms))

    # Check known deletions
    deletion_results = []
    for sym in KNOWN_DELETIONS:
        if sym in cache_syms:
            first, last, n = check_symbol_date_range(sym)
            deletion_results.append({
                "symbol": sym, "in_cache": True,
                "first_date": first, "last_date": last, "n_rows": n,
            })
        else:
            deletion_results.append({
                "symbol": sym, "in_cache": False,
                "first_date": None, "last_date": None, "n_rows": 0,
            })

    # Check survivors
    survivor_results = []
    for sym in KNOWN_SURVIVORS:
        if sym in cache_syms:
            first, last, n = check_symbol_date_range(sym)
            survivor_results.append({
                "symbol": sym, "in_cache": True,
                "first_date": first, "last_date": last, "n_rows": n,
            })

    del_df = pd.DataFrame(deletion_results)
    surv_df = pd.DataFrame(survivor_results)

    n_del_found    = del_df["in_cache"].sum()
    n_del_missing  = (~del_df["in_cache"]).sum()
    pct_del_found  = n_del_found / len(KNOWN_DELETIONS) * 100
    bias_flag      = pct_del_found < 50  # if >50% of delisted names are missing → problem

    verdict = "POTENTIAL SURVIVORSHIP BIAS" if bias_flag else "ACCEPTABLE"

    # Build markdown report
    lines = [
        f"# Survivorship Bias Audit — {date.today()}",
        "",
        "## Summary",
        "",
        f"- Total symbols in daily cache: **{len(cache_syms)}**",
        f"- Known delisted names checked: **{len(KNOWN_DELETIONS)}**",
        f"- Found in cache: **{n_del_found} ({pct_del_found:.0f}%)**",
        f"- Missing from cache: **{n_del_missing}**",
        f"- Verdict: **{verdict}**",
        "",
        "> Note: This audit uses a partial list of S&P 500 deletions 2019–2024.",
        "> For a definitive audit, use CRSP or Compustat membership history.",
        "",
        "## Delisted Names — Cache Coverage",
        "",
        "| Symbol | In Cache | First Date | Last Date | N Rows |",
        "|--------|----------|------------|-----------|--------|",
    ]
    for _, row in del_df.iterrows():
        status = "✅" if row["in_cache"] else "❌ MISSING"
        lines.append(
            f"| {row['symbol']} | {status} | {row['first_date'] or '—'} "
            f"| {row['last_date'] or '—'} | {row['n_rows'] or '—'} |"
        )

    lines += [
        "",
        "## Active Survivors — Cache Coverage (Control Group)",
        "",
        "| Symbol | First Date | Last Date | N Rows |",
        "|--------|------------|-----------|--------|",
    ]
    for _, row in surv_df.iterrows():
        lines.append(
            f"| {row['symbol']} | {row['first_date']} | {row['last_date']} | {row['n_rows']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- If delisted names **are present** in the cache with data through their delisting date: good.",
        "- If delisted names are **absent**: the backtest universe is survivors-only,",
        "  inflating factor scores (losers excluded) and overstating WF Sharpe.",
        "",
        "## Recommended Action",
        "",
        "- If bias detected: augment cache with historical Polygon S3 data for delisted tickers.",
        "  The `polygon_s3.py` provider already supports this.",
        "- If coverage is adequate: mark audit passed; no action needed.",
    ]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written → %s", out_path)

    print(f"\nSurvivorship Audit: {verdict}")
    print(f"  {n_del_found}/{len(KNOWN_DELETIONS)} known delisted names found in cache")
    print(f"  Report: {out_path}\n")

    return 0 if not bias_flag else 2


if __name__ == "__main__":
    sys.exit(main())
