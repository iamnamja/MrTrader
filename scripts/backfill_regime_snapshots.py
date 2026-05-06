"""Phase R1 — Backfill regime_snapshots table.

Iterates trading days from START_DATE to today, computes raw regime features
for each, and writes a row to regime_snapshots with:
  - snapshot_trigger = 'backfill'
  - model_version = NULL  (no trained model yet)
  - regime_score = NULL
  - regime_label = 'UNKNOWN'
  - all raw feature columns populated

This provides the training dataset for Phase R2 (regime model training).

Usage:
    python scripts/backfill_regime_snapshots.py
    python scripts/backfill_regime_snapshots.py --start 2024-01-01
    python scripts/backfill_regime_snapshots.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

START_DATE_DEFAULT = date(2023, 1, 1)


def _is_trading_day(d: date) -> bool:
    """Rough check — excludes weekends. Holidays may still slip through
    but regime_features.build() returns NaN-heavy rows which is acceptable."""
    return d.weekday() < 5  # Mon-Fri


def _trading_days_between(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        if _is_trading_day(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _upsert_snapshot(db, snap_cls, feats: dict, d: date) -> None:
    """Insert a new backfill row, or skip if one already exists for this date+trigger."""
    existing = (
        db.query(snap_cls)
        .filter(
            snap_cls.snapshot_date == d,
            snap_cls.snapshot_trigger == "backfill",
        )
        .first()
    )
    if existing is not None:
        return  # idempotent

    row = snap_cls(
        snapshot_date=d,
        snapshot_trigger="backfill",
        regime_label="UNKNOWN",
        **{k: (None if (v != v) else v) for k, v in feats.items()},  # NaN → None for DB
    )
    db.add(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill regime_snapshots table")
    parser.add_argument("--start", default=START_DATE_DEFAULT.isoformat(),
                        help="Start date YYYY-MM-DD (default 2023-01-01)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print features without writing to DB")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.today() - timedelta(days=1)  # yesterday (today's data may be incomplete)

    logger.info("Backfill range: %s → %s", start, end)

    trading_days = _trading_days_between(start, end)
    logger.info("Trading days to process: %d", len(trading_days))

    from app.database.session import init_db, get_session
    from app.database.models import RegimeSnapshot
    from app.ml.regime_features import RegimeFeatureBuilder

    if not args.dry_run:
        init_db()

    builder = RegimeFeatureBuilder()

    # Pre-fetch full SPY and VIX history once to avoid 750 individual downloads
    logger.info("Pre-fetching SPY history (%d calendar days)...", (end - start).days + 60)
    import yfinance as yf
    spy_full = yf.download(
        "SPY",
        start=(start - timedelta(days=365)).isoformat(),  # extra year for MA200
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if isinstance(spy_full.columns, pd.MultiIndex):
        spy_full.columns = spy_full.columns.get_level_values(0)
    spy_full.columns = [c.lower() for c in spy_full.columns]
    logger.info("SPY: %d rows fetched", len(spy_full))

    logger.info("Pre-fetching VIX history...")
    vix_full = yf.download(
        "^VIX",
        start=(start - timedelta(days=365)).isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if isinstance(vix_full.columns, pd.MultiIndex):
        vix_full.columns = vix_full.columns.get_level_values(0)
    vix_full.columns = [c.lower() for c in vix_full.columns]
    vix_series = vix_full["close"]
    logger.info("VIX: %d rows fetched", len(vix_series))

    ok = 0
    skipped = 0
    errors = 0

    db = None if args.dry_run else get_session()
    try:
        for i, d in enumerate(trading_days):
            try:
                # Slice pre-fetched data up to d (PIT-correct)
                spy_slice = spy_full[spy_full.index.date <= d]
                vix_slice = vix_series[vix_series.index.date <= d]

                if spy_slice.empty or vix_slice.empty:
                    skipped += 1
                    continue

                feats = builder.build(
                    as_of_date=d,
                    _spy_df=spy_slice,
                    _vix_df=vix_slice,
                )

                if args.dry_run:
                    if i < 3 or d >= end - timedelta(days=7):
                        logger.info(
                            "[DRY RUN] %s  vix=%.1f  spy_ma20=%.3f  days_to_fomc=%.0f",
                            d,
                            feats.get("vix_level") or float("nan"),
                            feats.get("spy_ma20_dist") or float("nan"),
                            feats.get("days_to_fomc") or float("nan"),
                        )
                    ok += 1
                else:
                    _upsert_snapshot(db, RegimeSnapshot, feats, d)
                    ok += 1
                    if (i + 1) % 50 == 0:
                        db.commit()
                        logger.info("Progress: %d / %d  (committed)", i + 1, len(trading_days))

            except Exception as exc:
                errors += 1
                logger.warning("Error on %s: %s", d, exc)

        if not args.dry_run:
            db.commit()

    finally:
        if db is not None:
            db.close()

    logger.info("Done. ok=%d  skipped=%d  errors=%d", ok, skipped, errors)

    if not args.dry_run:
        # Verify row count
        db2 = get_session()
        try:
            count = db2.query(RegimeSnapshot).filter(
                RegimeSnapshot.snapshot_trigger == "backfill"
            ).count()
            logger.info("regime_snapshots backfill rows: %d", count)
            if count < 500:
                logger.warning(
                    "Gate check: expected >= 750 rows, got %d — re-check date range", count
                )
            else:
                logger.info("Gate check PASSED: >= 500 backfill rows in regime_snapshots")
        finally:
            db2.close()


if __name__ == "__main__":
    main()
