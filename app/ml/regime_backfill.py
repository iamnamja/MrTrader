"""Keep the `backfill`-trigger regime snapshots current.

WHY THIS LIVES IN app/ AND NOT scripts/. The weekly regime retrain calls this, and app
code importing `scripts.*` depends on the repo root being on sys.path — true when a
script is run from the root, not guaranteed for a service started elsewhere. That import
sat inside a try/except that downgraded failure to a single warning line, so an
ImportError would have meant: the data never advances, the staleness gate fails every
week forever, and the only symptom is one WARNING. Exactly the class of silent decay this
whole change exists to end. `scripts/backfill_regime_snapshots.py` now imports FROM here.

Background: nothing ever scheduled the backfill script, so when it stopped on 2026-05-07
the regime training set silently froze for four months while the weekly retrain kept
re-fitting the same 2179 rows and reporting success (DECISIONS 2026-09-06).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from app.ml.regime_features import RegimeFeatureBuilder, label_regime_day, label_name

logger = logging.getLogger(__name__)

# A row is worth writing only if the features the trainer REQUIRES are present. These are
# load_dataset's `core` dropna set in miniature: a row missing them is dropped from
# training anyway, so writing it buys nothing and — under rewrite — destroys a good row.
_REQUIRED_FEATURES = ("vix_level", "spy_ma50_dist", "spy_ma200_dist")


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5


def trading_days_between(start: date, end: date) -> list:
    days = []
    cur = start
    while cur <= end:
        if _is_trading_day(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def last_backfill_date() -> Optional[date]:
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


def _usable(feats: dict) -> bool:
    """True when the row carries the features training actually requires."""
    for f in _REQUIRED_FEATURES:
        v = feats.get(f)
        if v is None or v != v:      # None or NaN
            return False
    return True


def extend_backfill(
    start: Optional[date],
    end: date,
    rewrite: bool = False,
    rewrite_recent_days: int = 0,
    upsert=None,
) -> dict:
    """Write `backfill`-trigger snapshots for [start, end]. Idempotent; returns counts.

    `start=None` RESUMES FROM THE LAST EXISTING ROW. A fixed lookback cannot close a gap
    longer than the lookback: it writes only the recent tail, leaves the older hole empty
    forever, and — worse — drags `max(snapshot_date)` up to today so the staleness gate
    reads the dataset as CURRENT and passes over the hole. A false green is worse than the
    stale red it replaces.

    `rewrite_recent_days` re-computes the last N days even where rows exist, so a day
    written while the feed was degraded heals instead of being dropped from training
    forever by `load_dataset`'s dropna.

    ROWS WITHOUT THE REQUIRED FEATURES ARE NEVER WRITTEN. `fetch_all_prefetched` returns
    `{}` on a batch-download failure; `{}` still takes the prefetched branch, every lookup
    misses, and `build()` returns a full dict of NaN — NOT None, so a `feats is None`
    check does not catch it. Combined with a forced rewrite that would NULL a fortnight of
    previously-good rows and then advance `last_backfill_date()` past them, making the
    damage permanent and invisible. `_usable()` is the guard.
    """
    from app.database.session import init_db, get_session
    from app.database.models import RegimeSnapshot

    if upsert is None:
        from scripts.backfill_regime_snapshots import _upsert_snapshot as upsert

    if start is None:
        last = last_backfill_date()
        start = (last + timedelta(days=1)) if last else (end - timedelta(days=30))

    trading_days = trading_days_between(start, end)
    rewrite_from = (end - timedelta(days=rewrite_recent_days)
                    if rewrite_recent_days > 0 else None)
    if rewrite_from is not None:
        extra = [d for d in trading_days_between(rewrite_from, end)
                 if d not in set(trading_days)]
        trading_days = sorted(set(trading_days) | set(extra))

    if not trading_days:
        return {"ok": 0, "skipped": 0, "errors": 0, "days": 0, "unusable": 0,
                "start": None}

    builder = RegimeFeatureBuilder()
    prefetched = builder.fetch_all_prefetched(min(trading_days) - timedelta(days=400),
                                              end + timedelta(days=1))
    if not prefetched:
        # Distinguish "the feed is down" from "there was nothing to do". Writing here
        # would produce all-NULL rows; under rewrite it would also destroy good ones.
        logger.error("Regime backfill: price prefetch returned nothing — writing no rows "
                     "(a forced rewrite here would NULL previously-good days)")
        return {"ok": 0, "skipped": 0, "errors": 0, "days": len(trading_days),
                "unusable": len(trading_days), "start": trading_days[0].isoformat()}

    init_db()
    db = get_session()
    ok = skipped = errors = unusable = 0
    try:
        for i, d in enumerate(trading_days):
            try:
                feats = builder.build(as_of_date=d, _prefetched=prefetched)
                if feats is None or not _usable(feats):
                    unusable += 1
                    continue
                feats["regime_label_rule"] = label_name(label_regime_day(feats))
                rw = rewrite or (rewrite_from is not None and d >= rewrite_from)
                if upsert(db, RegimeSnapshot, feats, d, rw):
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

    if unusable:
        logger.warning("Regime backfill: %d/%d day(s) lacked required features and were "
                       "NOT written", unusable, len(trading_days))
    return {"ok": ok, "skipped": skipped, "errors": errors, "unusable": unusable,
            "days": len(trading_days), "start": trading_days[0].isoformat()}
