"""
oos_guard.py — enforce the OOS invariant for pre-trained model evaluation.

## Design contract

Every WF/CPCV run loads ONE pre-trained model and scores it across multiple
test windows. This is a GENERALIZATION TEST, not a true per-fold retrain.
It is only statistically valid when:

    te_start > model.trained_through + purge_days   (for every test fold)

assert_model_oos() enforces this. It is called at every evaluation entry point:
  - run_cpcv()                    (scripts/walkforward/cpcv.py)
  - FoldEngine.run()              (scripts/walkforward/engine.py)
  - run_swing_walkforward()       (scripts/walkforward_tier3.py) — skipped when model is None
  - run_intraday_walkforward()    (scripts/walkforward_tier3.py)

## trained_through persistence

model.trained_through (datetime.date) is set by trainers before save():
  - IntradayTrainer: max of day_ordinal column in raw_train matrix
  - ModelTrainer (swing): max of _last_all_dates

On load(), meta.get("trained_through") restores it. Absent → None → guard raises.

## Rules-based strategies (no ML model)

Strategies with no training cutoff (e.g. PEADStrategy) set
model.trained_through = date.min so every fold trivially passes.
run_swing_walkforward skips the guard entirely when model is None.

## Escape hatch

--allow-in-sample / allow_in_sample=True: guard logs a warning instead of
raising. The result's in_sample_override=True is checked FIRST in every
gate_passed() implementation, ensuring in-sample runs can never promote.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class OOSViolation(AssertionError):
    """Raised when a test fold overlaps the model's training period."""


def _trading_day_gap(from_date: date, to_date: date, trading_day_set: Set[date]) -> int:
    """Count trading days strictly between from_date and to_date (exclusive both ends)."""
    if not trading_day_set or to_date <= from_date:
        return 0
    return sum(1 for d in trading_day_set if from_date < d < to_date)


def assert_model_oos(
    trained_through: Optional[date],
    fold_boundaries: Iterable[Tuple[date, date, date, date]],
    purge_days: int = 0,
    model_label: str = "model",
    allow_in_sample: bool = False,
    trading_day_set: Optional[Set[date]] = None,
) -> None:
    """
    Verify trained_through + purge_days < te_start for every fold.

    fold_boundaries: iterable of (tr_start, tr_end, te_start, te_end) tuples.
    purge_days: minimum gap required between training cutoff and te_start.
    allow_in_sample: escape hatch for explicit in-sample diagnostic runs;
        logs a loud warning but does not raise. Results cannot promote past
        gates when this flag is True.
    trading_day_set: when provided, purge_days is interpreted as TRADING days
        (not calendar days). Required for intraday where purge_days=2 means
        2 trading days — calendar-day counting would under-count across weekends.
        BUG-8 fix: without this, a Friday trained_through + purge_days=2 gives
        a Sunday cutoff that Monday te_start trivially clears with only 1 trading
        day of actual gap.

    Raises OOSViolation if any test fold te_start <= trained_through + purge.
    Raises OOSViolation if trained_through is None (unknown training cutoff).
    """
    if trained_through is not None and not isinstance(trained_through, date):
        logger.warning(
            "OOS guard: %s.trained_through has unexpected type %s (expected date or None) — "
            "treating as None. Ensure the model was saved after the trained_through fix.",
            model_label, type(trained_through).__name__,
        )
        trained_through = None
    # Normalize datetime → date so comparison with te_start (always date) never raises TypeError.
    # (datetime subclasses date so isinstance passes, but date <= datetime raises TypeError.)
    if trained_through is not None and type(trained_through) is not date:
        try:
            trained_through = trained_through.date()
        except AttributeError:
            trained_through = None

    if trained_through is None:
        msg = (
            f"OOS guard: {model_label}.trained_through is None — cannot verify "
            "out-of-sample validity. Retrain the model so trained_through is "
            "persisted, or pass --allow-in-sample to override (results will be "
            "labeled in-sample and cannot promote past gates)."
        )
        if allow_in_sample:
            logger.warning(msg)
            return
        raise OOSViolation(msg)

    bad: list = []
    for tr_start, tr_end, te_start, te_end in fold_boundaries:
        if trading_day_set is not None:
            # BUG-8: count gap in trading days, not calendar days.
            # C13-4: always require te_start > trained_through even when purge_days=0,
            # preventing same-day overlap (te_start on trained_through date).
            gap = _trading_day_gap(trained_through, te_start, trading_day_set)
            violates = (te_start <= trained_through) or (gap < purge_days)
        else:
            cutoff = trained_through + timedelta(days=max(purge_days, 0))
            violates = te_start <= cutoff
        if violates:
            bad.append((te_start, te_end))

    if bad:
        purge_unit = "trading days" if trading_day_set is not None else "calendar days"
        msg = (
            f"OOS guard: {model_label} trained_through={trained_through} but "
            f"{len(bad)} test fold(s) have insufficient purge gap "
            f"(required {purge_days} {purge_unit}). "
            f"Earliest violating fold: te_start={bad[0][0]} te_end={bad[0][1]}. "
            "Re-train the model with an earlier cutoff, or pass --allow-in-sample "
            "to label this run as in-sample (cannot promote past gates)."
        )
        if allow_in_sample:
            logger.warning(msg)
            return
        raise OOSViolation(msg)
