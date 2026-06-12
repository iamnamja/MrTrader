"""
options_quality.py — the R1K options-quality coverage filter (Alpha-v6 P2/P4).

`filter_options_universe(as_of, candidate_symbols, features)` returns the subset
of `candidate_symbols` that, as of `as_of`, have a tradeable options surface in
the daily feature table (app/data/options_features.py). This is the universe the
Phase-4 cross-sectional L/S hypotheses (H4a-H4e) sort within: a name with a thin
or stale chain has unreliable IV/skew/term features and must not enter the sort.

A name QUALIFIES on `as_of` if its LATEST feature row knowable by `as_of` (i.e.
the row with the largest knowable_date <= as_of — PIT) satisfies ALL of:
  * n_valid_contracts >= MIN_VALID_CONTRACTS  (enough "ok"+fresh contracts to
    form a surface; matches the builder's drop floor),
  * atm_iv_30d is non-NaN  (the ~30-DTE ATM point exists — the anchor every
    H4 feature is relative to),
  * total_opt_volume >= MIN_OPT_VOLUME  (the chain actually trades — a chain that
    prices but barely trades has unreliable volume features and stale IVs).

The thresholds are MODULE CONSTANTS, deliberately CONSERVATIVE, and TUNABLE —
they are an operational coverage floor, NOT part of the frozen statistical
acceptance spec (the H4 criteria are frozen in
scripts/preregister_options_xs_features.py; this filter only decides who is in
the candidate pool).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── coverage-floor thresholds (operational, tunable — NOT the frozen stat spec) ─
# Mirror options_features.MIN_VALID_CONTRACTS (the builder's drop floor); a row
# that exists already cleared it, but a name can be admitted from a STALE row, so
# we re-check here.
MIN_VALID_CONTRACTS = 6
# Min total option contracts traded on the latest row (a near-dead chain prices
# but does not trade) — a conservative liquidity floor, well below R1K medians.
MIN_OPT_VOLUME = 100.0


def _latest_known_row(name_rows: pd.DataFrame,
                      as_of_ts: pd.Timestamp) -> Optional[pd.Series]:
    """The single feature row with the largest knowable_date <= as_of (PIT).
    None if the name has no row knowable by as_of."""
    known = name_rows[name_rows["knowable_date"] <= as_of_ts]
    if known.empty:
        return None
    return known.loc[known["knowable_date"].idxmax()]


def _row_qualifies(row: pd.Series) -> bool:
    """Apply the coverage floor to one (latest-known) feature row."""
    n_valid = row.get("n_valid_contracts")
    if n_valid is None or pd.isna(n_valid) or int(n_valid) < MIN_VALID_CONTRACTS:
        return False
    atm_iv = row.get("atm_iv_30d")
    if atm_iv is None or pd.isna(atm_iv):
        return False

    # Option-volume floor — the chain must actually trade.
    total_vol = row.get("total_opt_volume")
    if total_vol is None or pd.isna(total_vol) or float(total_vol) < MIN_OPT_VOLUME:
        return False
    return True


def filter_options_universe(
    as_of: date,
    candidate_symbols: Iterable[str],
    features: pd.DataFrame,
) -> List[str]:
    """Return the subset of `candidate_symbols` whose latest feature row knowable
    by `as_of` clears the coverage floor (see module docstring). Order-preserving
    on `candidate_symbols`; symbols absent from `features` are dropped.

    `features` is the (whole) daily feature table from
    app/data/options_features.py — columns include underlying, knowable_date,
    n_valid_contracts, atm_iv_30d, total_opt_volume.
    """
    candidates = list(candidate_symbols)
    if features is None or features.empty or not candidates:
        return []
    f = features
    # Be tolerant of a not-yet-typed knowable_date column.
    if not np.issubdtype(f["knowable_date"].dtype, np.datetime64):
        f = f.copy()
        f["knowable_date"] = pd.to_datetime(f["knowable_date"])
    as_of_ts = pd.Timestamp(as_of)

    by_name = {u: grp for u, grp in f.groupby("underlying")}
    qualified: List[str] = []
    for sym in candidates:
        rows = by_name.get(sym)
        if rows is None:
            continue
        row = _latest_known_row(rows, as_of_ts)
        if row is not None and _row_qualifies(row):
            qualified.append(sym)
    logger.debug("options-quality filter @ %s: %d/%d candidates qualified",
                 as_of, len(qualified), len(candidates))
    return qualified
