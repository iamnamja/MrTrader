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

from app.ml.regime_features import (
    CORE_FEATURE_NAMES,
    RegimeFeatureBuilder,
    label_regime_day,
    label_name,
)

logger = logging.getLogger(__name__)

# A row is worth writing only if the features the trainer REQUIRES are present — the FULL
# set, imported from regime_features so it cannot drift from load_dataset's dropna.
#
# Guarding a hand-picked three was not enough. `rewrite_recent_days` overwrites EVERY
# column, so one missing ticker in the batch download (HYG, say) would NULL 14 days of
# previously-good credit features and relabel those days, and a MacroCalendar failure would
# NULL days_to_fomc/cpi/nfp — which ARE in the core set, making those 14 rows permanently
# untrainable. Neither would ever heal, because the rewrite window moves on.
_REQUIRED_FEATURES = CORE_FEATURE_NAMES

# Where an EMPTY table starts from. Falling back to a 30-day tail here would be the exact
# fixed-lookback behaviour this module's docstring forbids: the 2018-to-today history would
# then never be written by any automated path, and `max(snapshot_date)` would read current
# over a table containing one month. Matches the CLI's START_DATE_DEFAULT.
INITIAL_START = date(2018, 1, 1)


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
        # A REWRITE MAY IMPROVE A ROW, NEVER DEGRADE IT. Blanket-overwriting means one
        # missing ticker in the batch download (HYG, say) NULLs previously-good credit
        # features on days that already had them, and relabels any RISK_OFF day that came
        # from `credit_20d < -0.03`. A required-feature allowlist cannot cover this,
        # because the columns at risk (credit, breadth, sector, vix_term) are legitimately
        # NULL on some days and so cannot be required. The invariant that DOES hold is
        # directional: never replace a value with nothing.
        for k, v in clean.items():
            if not hasattr(existing, k):
                continue
            if v is None and getattr(existing, k) is not None:
                continue          # keep the good value we already have
            setattr(existing, k, v)
        if (hasattr(existing, "regime_label_rule")
                and clean.get("regime_label_rule") is not None):
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
        upsert = _upsert_snapshot

    if start is None:
        last = last_backfill_date()
        if last:
            start = last + timedelta(days=1)
        else:
            logger.warning("Regime snapshots table is EMPTY — backfilling from %s",
                           INITIAL_START)
            start = INITIAL_START

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
                # WITHOUT the rollback, one failed periodic commit leaves the Session in
                # pending-rollback state and EVERY remaining day raises
                # PendingRollbackError, is counted as an error, and the final commit
                # raises out into _retrain_regime — which swallows it as a single WARNING.
                # That is the silent-decay mode this whole change exists to end.
                errors += 1
                try:
                    db.rollback()
                except Exception:
                    pass
                logger.warning("Error on %s: %s", d, exc)
        db.commit()
    finally:
        db.close()

    if unusable:
        logger.warning("Regime backfill: %d/%d day(s) lacked required features and were "
                       "NOT written", unusable, len(trading_days))
    return {"ok": ok, "skipped": skipped, "errors": errors, "unusable": unusable,
            "days": len(trading_days), "start": trading_days[0].isoformat()}
