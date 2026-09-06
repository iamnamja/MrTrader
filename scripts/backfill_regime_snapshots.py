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


def _upsert_snapshot(db, snap_cls, feats: dict, d: date, rewrite: bool) -> bool:
    """Insert or update a backfill row. Returns True if written."""
    existing = (
        db.query(snap_cls)
        .filter(
            snap_cls.snapshot_date == d,
            snap_cls.snapshot_trigger == "backfill",
        )
        .first()
    )
    if existing is not None and not rewrite:
        return False

    clean = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in feats.items()}

    if existing is not None:
        for k, v in clean.items():
            if hasattr(existing, k):
                setattr(existing, k, v)
        if hasattr(existing, "regime_label_rule") and "regime_label_rule" in clean:
            existing.regime_label_rule = clean["regime_label_rule"]
    else:
        row = snap_cls(
            snapshot_date=d,
            snapshot_trigger="backfill",
            regime_label="UNKNOWN",
            **{k: v for k, v in clean.items() if hasattr(snap_cls, k)},
        )
        db.add(row)
    return True


def last_backfill_date() -> "date | None":
    """Latest `backfill`-trigger snapshot_date, or None when the table is empty."""
    from sqlalchemy import func
    from app.database.session import init_db, get_session
    from app.database.models import RegimeSnapshot

    init_db()
    with get_session() as db:
        return (
            db.query(func.max(RegimeSnapshot.snapshot_date))
            .filter(RegimeSnapshot.snapshot_trigger == "backfill")
            .scalar()
        )


def extend_backfill(
    start: "date | None",
    end: date,
    rewrite: bool = False,
    rewrite_recent_days: int = 0,
) -> dict:
    """Write `backfill`-trigger snapshots for [start, end]. Idempotent; returns counts.

    Extracted from main() so the weekly regime retrain can KEEP THE DATASET CURRENT
    ITSELF. Nothing scheduled this script, so when the backfill stopped on 2026-05-07 the
    regime training set silently froze for four months while the retrain kept "succeeding"
    (DECISIONS 2026-09-06). A gate that detects staleness is necessary but not sufficient —
    something has to actually advance the data, or the gate just fails every week instead.

    `start=None` RESUMES FROM THE LAST EXISTING ROW. A fixed lookback (the first cut used
    today-30) cannot close a gap longer than the lookback: it writes only the recent tail,
    leaves the older hole empty forever, and — worse — drags `max(snapshot_date)` up to
    today so the staleness gate reads the dataset as CURRENT and passes over a hole. A
    false green is worse than the stale red it replaces.

    `rewrite_recent_days` re-computes the last N days even where rows exist. A day written
    while the feed was degraded carries a NULL VIX block, and `load_dataset`'s
    dropna(subset=core) then drops it from training permanently; nothing else would ever
    revisit it. The staleness guard makes such writes more likely, not less, because it
    NULLs rather than carrying a stale value forward.
    """
    from app.database.session import init_db, get_session
    from app.database.models import RegimeSnapshot
    from app.ml.regime_features import RegimeFeatureBuilder, label_regime_day, label_name

    if start is None:
        last = last_backfill_date()
        start = (last + timedelta(days=1)) if last else (end - timedelta(days=30))

    trading_days = _trading_days_between(start, end)
    rewrite_from = (end - timedelta(days=rewrite_recent_days)
                    if rewrite_recent_days > 0 else None)
    if rewrite_from is not None:
        # widen the range so degraded recent rows are revisited even when start > them
        extra = [d for d in _trading_days_between(rewrite_from, end)
                 if d not in set(trading_days)]
        trading_days = sorted(set(trading_days) | set(extra))
    if not trading_days:
        return {"ok": 0, "skipped": 0, "errors": 0, "days": 0}

    if not trading_days:
        return {"ok": 0, "skipped": 0, "errors": 0, "days": 0, "start": None}

    builder = RegimeFeatureBuilder()
    prefetched = builder.fetch_all_prefetched(min(trading_days) - timedelta(days=400),
                                              end + timedelta(days=1))
    init_db()
    db = get_session()
    ok = skipped = errors = 0
    try:
        for i, d in enumerate(trading_days):
            try:
                feats = builder.build(as_of_date=d, _prefetched=prefetched)
                if feats is None:
                    skipped += 1
                    continue
                feats["regime_label_rule"] = label_name(label_regime_day(feats))
                _rw = rewrite or (rewrite_from is not None and d >= rewrite_from)
                if _upsert_snapshot(db, RegimeSnapshot, feats, d, _rw):
                    ok += 1
                else:
                    skipped += 1
                if (i + 1) % 100 == 0:
                    db.commit()
            except Exception as exc:
                errors += 1
                logger.warning("Error on %s: %s", d, exc)
        db.commit()
    finally:
        db.close()
    return {"ok": ok, "skipped": skipped, "errors": errors,
            "days": len(trading_days), "start": trading_days[0].isoformat()}


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

    trading_days = _trading_days_between(start, end)
    logger.info("Trading days to process: %d", len(trading_days))

    from app.ml.regime_features import RegimeFeatureBuilder, label_regime_day, label_name

    builder = RegimeFeatureBuilder()

    # Batch-fetch all 15 tickers (SPY, RSP, ^VIX, ^VIX3M, HYG, IEF, 9 sector ETFs)
    # Extra year of history before start for MA200 and rolling windows
    fetch_start = start - timedelta(days=400)
    fetch_end = end + timedelta(days=1)
    logger.info("Batch-fetching all regime tickers from %s to %s...", fetch_start, fetch_end)
    prefetched = builder.fetch_all_prefetched(fetch_start, fetch_end)
    logger.info(
        "Prefetch complete: %d tickers loaded",
        sum(1 for v in prefetched.values() if v is not None and not v.empty),
    )

    if not args.dry_run:
        from app.database.session import init_db, get_session
        from app.database.models import RegimeSnapshot
        init_db()

    ok = 0
    skipped = 0
    errors = 0

    db = None if args.dry_run else get_session()
    try:
        for i, d in enumerate(trading_days):
            try:
                feats = builder.build(as_of_date=d, _prefetched=prefetched)
                if feats is None:
                    skipped += 1
                    continue

                # Attach V2 rule label
                label_int = label_regime_day(feats)
                feats["regime_label_rule"] = label_name(label_int)

                if args.dry_run:
                    if i < 5 or d >= end - timedelta(days=7):
                        logger.info(
                            "[DRY RUN] %s  vix=%.1f  vix_term=%.3f  credit_20d=%.4f  "
                            "breadth_20d=%.4f  label=%s",
                            d,
                            feats.get("vix_level") or float("nan"),
                            feats.get("vix_term_ratio") or float("nan"),
                            feats.get("credit_hyg_ief_20d") or float("nan"),
                            feats.get("breadth_rsp_spy_ratio_20d") or float("nan"),
                            feats.get("regime_label_rule", "?"),
                        )
                    ok += 1
                else:
                    written = _upsert_snapshot(db, RegimeSnapshot, feats, d, args.rewrite)
                    if written:
                        ok += 1
                    else:
                        skipped += 1

                    if (i + 1) % 100 == 0:
                        db.commit()
                        logger.info(
                            "Progress: %d / %d  ok=%d skipped=%d errors=%d",
                            i + 1, len(trading_days), ok, skipped, errors,
                        )

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
