"""
Phase 79 — Point-in-Time Index Membership.

Provides members_at(index, date) returning the set of tickers that belonged
to the index at the given historical date, eliminating survivorship bias in
back-testing and walk-forward evaluation.

Parquet files live at data/universe/{index}_membership.parquet.
Schema: columns = ['ticker', 'added', 'removed']
  added   — date the ticker entered the index (YYYY-MM-DD string)
  removed — date the ticker left the index, or '' / NaN if still in
"""
from __future__ import annotations

import logging
import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_UNIVERSE_DIR = Path(__file__).parent.parent.parent / "data" / "universe"

# Fallback static lists when parquet not available
def _fallback(index: str) -> List[str]:
    if index in ("sp500", "sp_500"):
        from app.utils.constants import SP_500_TICKERS
        return list(SP_500_TICKERS)
    if index in ("sp100", "sp_100"):
        from app.utils.constants import SP_100_TICKERS
        return list(SP_100_TICKERS)
    if index in ("russell1000", "russell_1000"):
        from app.utils.constants import RUSSELL_1000_TICKERS
        return list(RUSSELL_1000_TICKERS)
    raise ValueError(f"Unknown index: {index!r}")


@lru_cache(maxsize=8)
def _load_membership(index: str):
    """Load and cache membership parquet for the given index name."""
    try:
        import pandas as pd
    except ImportError:
        return None

    path = _UNIVERSE_DIR / f"{index}_membership.parquet"
    if not path.exists():
        logger.debug("No membership file for %s at %s — using static list", index, path)
        return None

    df = pd.read_parquet(path)
    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    required = {"ticker", "added"}
    if not required.issubset(df.columns):
        logger.warning("Membership file %s missing columns %s", path, required - set(df.columns))
        return None

    df["added"] = pd.to_datetime(df["added"])
    if "removed" in df.columns:
        df["removed"] = pd.to_datetime(df["removed"], errors="coerce")
    else:
        df["removed"] = pd.NaT
    return df


def members_at(index: str, as_of: date) -> List[str]:
    """
    Return the list of tickers that were members of *index* on *as_of*.

    Falls back to the current static constant list if no parquet file is found,
    which preserves the old behaviour for live trading (where only current
    members matter).
    """
    df = _load_membership(index)
    if df is None:
        return _fallback(index)

    import pandas as pd
    as_of_ts = pd.Timestamp(as_of)
    mask_added = df["added"] <= as_of_ts
    mask_removed = df["removed"].isna() | (df["removed"] > as_of_ts)
    members = df.loc[mask_added & mask_removed, "ticker"].tolist()

    return members


def invalidate_cache() -> None:
    """Clear the in-process LRU cache (useful in tests)."""
    _load_membership.cache_clear()
