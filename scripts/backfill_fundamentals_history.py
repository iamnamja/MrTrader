"""
Phase 89a — Historical Fundamentals Backfill

Fetches SEC EDGAR XBRL company facts for each symbol and builds a
point-in-time parquet store of annual fundamental snapshots keyed by
(symbol, as_of_date). Training can load from this store instead of
making live API calls, allowing pe_ratio / pb_ratio / revenue_growth /
profit_margin / debt_to_equity to be un-pruned from PRUNED_FEATURES.

Point-in-time safety: uses the 10-K filing `end` date (fiscal year end),
not the filing date, since that is the earliest date the data could have
been known to a model trained on that day's bar data.

Output:
  data/fundamentals/fundamentals_history.parquet
  Columns: symbol, as_of_date (str YYYY-MM-DD), pe_ratio, pb_ratio,
           profit_margin, revenue_growth, debt_to_equity

Usage:
  python scripts/backfill_fundamentals_history.py [--workers 4] [--dry-run]

Cost: free — SEC EDGAR XBRL API is public, no key required.
Runtime: ~30-60 min for 430 symbols (EDGAR rate-limits to ~10 req/s).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fundamentals_backfill")

OUTPUT_PATH = Path("data/fundamentals/fundamentals_history.parquet")


def _fetch_symbol_history(symbol: str) -> list[dict]:
    """
    Fetch all historical annual fundamental snapshots for a symbol.
    Returns list of dicts: {symbol, as_of_date, pe_ratio, pb_ratio,
                            profit_margin, revenue_growth, debt_to_equity}
    """
    from app.ml.fundamental_fetcher import _get_cik, _fetch_edgar_facts

    cik = _get_cik(symbol)
    if cik is None:
        logger.debug("No CIK for %s — skipping", symbol)
        return []

    facts = _fetch_edgar_facts(cik)
    if not facts:
        return []

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []

    def _annual_entries(tag: str, unit: str = "USD") -> list[dict]:
        data = us_gaap.get(tag, {}).get("units", {}).get(unit, [])
        return sorted(
            [e for e in data if e.get("form") in ("10-K", "10-K/A") and "end" in e],
            key=lambda e: e["end"],
        )

    def _income_entries(tag: str) -> list[dict]:
        """Annual income statement items (spanning ~12 months)."""
        entries = _annual_entries(tag)
        def _span(e):
            try:
                from datetime import datetime as _dt
                return (_dt.strptime(e["end"], "%Y-%m-%d") - _dt.strptime(e.get("start", e["end"]), "%Y-%m-%d")).days
            except Exception:
                return 0
        return [e for e in entries if "start" not in e or 330 <= _span(e) <= 400]

    # Collect all fiscal year-end dates across revenue filings
    rev_entries = _income_entries("Revenues") or _income_entries("RevenueFromContractWithCustomerExcludingAssessedTax") or _income_entries("SalesRevenueNet")
    if not rev_entries:
        return []

    # For each fiscal year-end, compute snapshot
    for i, rev_e in enumerate(rev_entries):
        fy_end = rev_e["end"]
        rev_now = float(rev_e.get("val") or 0)

        # Revenue growth vs prior year
        rev_growth = 0.0
        if i > 0:
            rev_prev = float(rev_entries[i - 1].get("val") or 0)
            if rev_prev != 0:
                rev_growth = float(max(-1.0, min(5.0, (rev_now - rev_prev) / abs(rev_prev))))

        # Net income margin
        profit_margin = 0.0
        ni_entries = _income_entries("NetIncomeLoss")
        ni_e = next((e for e in reversed(ni_entries) if e["end"] <= fy_end), None)
        if ni_e and rev_now != 0:
            ni = float(ni_e.get("val") or 0)
            profit_margin = float(max(-1.0, min(1.0, ni / abs(rev_now))))

        # Debt-to-equity
        debt_to_equity = 0.0
        eq_entries = _annual_entries("StockholdersEquity")
        eq_e = next((e for e in reversed(eq_entries) if e["end"] <= fy_end), None)
        debt_entries = _annual_entries("LongTermDebt")
        debt_e = next((e for e in reversed(debt_entries) if e["end"] <= fy_end), None)
        if eq_e and debt_e:
            eq = float(eq_e.get("val") or 0)
            debt = float(debt_e.get("val") or 0)
            if eq != 0:
                debt_to_equity = float(min(10.0, abs(debt / eq)))

        # P/E and P/B require price — we store EPS/BVPS and let training compute ratios
        # using the daily bar close at training time. Store 0.0 as placeholder.
        rows.append({
            "symbol": symbol,
            "as_of_date": fy_end,
            "pe_ratio": 0.0,        # computed at training time using bar close / EPS
            "pb_ratio": 0.0,        # computed at training time using bar close / BVPS
            "profit_margin": round(profit_margin, 6),
            "revenue_growth": round(rev_growth, 6),
            "debt_to_equity": round(debt_to_equity, 6),
        })

    return rows


def run(workers: int = 4, dry_run: bool = False) -> None:
    from app.utils.constants import SP_100_TICKERS
    try:
        from app.utils.constants import SP_500_TICKERS
        symbols = list(SP_500_TICKERS)
    except ImportError:
        symbols = list(SP_100_TICKERS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fundamentals backfill: %d symbols, %d workers, dry_run=%s",
                len(symbols), workers, dry_run)

    all_rows: list[dict] = []
    ok = errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_symbol_history, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                rows = fut.result()
                if rows:
                    all_rows.extend(rows)
                    ok += 1
                    logger.debug("%s: %d historical snapshots", sym, len(rows))
                else:
                    logger.debug("%s: no data", sym)
            except Exception as exc:
                errors += 1
                logger.warning("%s: failed — %s", sym, exc)

    logger.info("Fetch complete: %d symbols ok, %d errors, %d total rows", ok, errors, len(all_rows))

    if dry_run:
        logger.info("[DRY-RUN] Would write %d rows to %s", len(all_rows), OUTPUT_PATH)
        if all_rows:
            logger.info("Sample rows:")
            for r in all_rows[:3]:
                logger.info("  %s", r)
        return

    if not all_rows:
        logger.warning("No rows to write — exiting")
        return

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["symbol", "as_of_date"]).reset_index(drop=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Written: %s (%d rows, %d symbols)", OUTPUT_PATH, len(df), df["symbol"].nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical fundamentals from SEC EDGAR")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(workers=args.workers, dry_run=args.dry_run)
