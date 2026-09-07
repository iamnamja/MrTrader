"""Phase R1/R7 — Backfill regime_snapshots table.

Iterates trading days from START_DATE to yesterday, computes V2 regime features
for each, and writes a row to regime_snapshots with:
  - snapshot_trigger = 'backfill'
  - model_version = NULL  (no trained model yet)
  - regime_score = NULL
  - regime_label = 'UNKNOWN'
  - regime_label_rule = V2 rule-based label (RISK_OFF/RISK_CAUTION/RISK_ON)
  - all raw feature columns populated

R7: Uses RegimeFeatureBuilder.fetch_all_prefetched() to batch-download all 15
tickers in one yfinance call (SPY, RSP, ^VIX, ^VIX3M, HYG, IEF, sector ETFs).

Usage:
    python scripts/backfill_regime_snapshots.py
    python scripts/backfill_regime_snapshots.py --start 2018-01-01
    python scripts/backfill_regime_snapshots.py --dry-run
    python scripts/backfill_regime_snapshots.py --rewrite   # overwrite existing rows
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

START_DATE_DEFAULT = date(2018, 1, 1)


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5


def _trading_days_between(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        if _is_trading_day(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _label_name(label_int: int) -> str:
    return {0: "RISK_OFF", 1: "RISK_CAUTION", 2: "RISK_ON"}.get(label_int, "UNKNOWN")


# The reusable helpers now live in app/ml/regime_backfill.py — app code must not depend
# on `scripts.*` being importable (see that module's docstring). Re-exported here so the
# CLI and any existing callers keep working.
from app.ml.regime_backfill import (            # noqa: E402
    _upsert_snapshot,
    _usable,
    extend_backfill,
    last_backfill_date,
    trading_days_between,
)

__all__ = ["extend_backfill", "last_backfill_date", "_upsert_snapshot", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill regime_snapshots table (V2)")
    parser.add_argument("--start", default=START_DATE_DEFAULT.isoformat(),
                        help="Start date YYYY-MM-DD (default 2018-01-01)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print features without writing to DB")
    parser.add_argument("--rewrite", action="store_true",
                        help="Overwrite existing backfill rows (re-compute all features)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.today() - timedelta(days=1)
    logger.info("Backfill range: %s → %s  rewrite=%s", start, end, args.rewrite)

    if args.dry_run:
        from app.ml.regime_features import (
            RegimeFeatureBuilder, label_regime_day, label_name)

        trading_days = trading_days_between(start, end)
        logger.info("Trading days to process: %d", len(trading_days))
        builder = RegimeFeatureBuilder()
        prefetched = builder.fetch_all_prefetched(start - timedelta(days=400),
                                                  end + timedelta(days=1))
        logger.info("Prefetch complete: %d tickers loaded",
                    sum(1 for v in prefetched.values()
                        if v is not None and not v.empty))
        # Mirror the real path: same _usable() verdict, same per-day error containment.
        # A dry run that does not apply the write-guard does not predict what the real run
        # writes, and one without try/except aborts the whole preview on a single bad day.
        writable = unusable = errors = 0
        for i, d in enumerate(trading_days):
            try:
                feats = builder.build(as_of_date=d, _prefetched=prefetched)
                if feats is None or not _usable(feats):
                    unusable += 1
                    continue
                feats["regime_label_rule"] = label_name(label_regime_day(feats))
                if i < 5 or d >= end - timedelta(days=7):
                    logger.info(
                        "[DRY RUN] %s  vix=%s  vix_term=%s  credit_20d=%s  label=%s",
                        d, feats.get("vix_level"), feats.get("vix_term_ratio"),
                        feats.get("credit_hyg_ief_20d"),
                        feats.get("regime_label_rule", "?"))
                writable += 1
            except Exception as exc:
                errors += 1
                logger.warning("Error on %s: %s", d, exc)
        logger.info("Done (dry run). writable=%d  unusable=%d  errors=%d",
                    writable, unusable, errors)
        return

    # Route ALL writes through extend_backfill so the CLI gets the SAME guards as the
    # weekly retrain: the empty-prefetch bail-out and the never-degrade-a-row upsert.
    # Without this, `--rewrite` during a yfinance outage would NULL every row in the table
    # — build() returns an all-NaN dict, not None, so a `feats is None` check does not
    # catch it — and this is the tool the new staleness warnings tell operators to run.
    counts = extend_backfill(start, end, rewrite=args.rewrite)
    logger.info("Done. ok=%d  skipped=%d  errors=%d  unusable=%d",
                counts["ok"], counts["skipped"], counts["errors"],
                counts.get("unusable", 0))
    if counts.get("unusable"):
        logger.warning("%d day(s) lacked required features and were NOT written",
                       counts["unusable"])

    from app.database.session import get_session
    from app.database.models import RegimeSnapshot
    db2 = get_session()
    try:
        total = db2.query(RegimeSnapshot).filter(
            RegimeSnapshot.snapshot_trigger == "backfill"
        ).count()
        logger.info("Total backfill rows in regime_snapshots: %d", total)
        gate = 1500  # expect ~2000 rows from 2018
        if total < gate:
            logger.warning("Gate: expected >= %d rows, got %d", gate, total)
        else:
            logger.info("Gate PASSED: >= %d backfill rows", gate)
    finally:
        db2.close()


if __name__ == "__main__":
    main()
