"""
oos_guard.py — assert that a pre-trained model is genuinely out-of-sample for
a given set of test fold boundaries.

The "generalization test" design (one model loaded, scored across many test
windows) is only valid when every test window starts strictly after the
model's last training observation (trained_through). This module enforces
that invariant.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


class OOSViolation(AssertionError):
    """Raised when a test fold overlaps the model's training period."""


def assert_model_oos(
    trained_through: Optional[date],
    fold_boundaries: Iterable[Tuple[date, date, date, date]],
    purge_days: int = 0,
    model_label: str = "model",
    allow_in_sample: bool = False,
) -> None:
    """
    Verify trained_through + purge_days < te_start for every fold.

    fold_boundaries: iterable of (tr_start, tr_end, te_start, te_end) tuples.
    purge_days: minimum gap required between training cutoff and te_start.
    allow_in_sample: escape hatch for explicit in-sample diagnostic runs;
        logs a loud warning but does not raise. Results cannot promote past
        gates when this flag is True.

    Raises OOSViolation if any test fold te_start <= trained_through + purge.
    Raises OOSViolation if trained_through is None (unknown training cutoff).
    """
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

    cutoff = trained_through + timedelta(days=max(purge_days, 0))
    bad: list = []
    for tr_start, tr_end, te_start, te_end in fold_boundaries:
        if te_start <= cutoff:
            bad.append((te_start, te_end))

    if bad:
        msg = (
            f"OOS guard: {model_label} trained_through={trained_through} but "
            f"{len(bad)} test fold(s) start on/before cutoff "
            f"{cutoff} (purge_days={purge_days}). "
            f"Earliest violating fold: te_start={bad[0][0]} te_end={bad[0][1]}. "
            "Re-train the model with an earlier cutoff, or pass --allow-in-sample "
            "to label this run as in-sample (cannot promote past gates)."
        )
        if allow_in_sample:
            logger.warning(msg)
            return
        raise OOSViolation(msg)
