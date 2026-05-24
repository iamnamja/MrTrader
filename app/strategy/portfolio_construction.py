"""Phase RA — portfolio construction primitives for REBALANCE execution mode.

Pure deterministic helpers: no model loading, no I/O, no global state, no time
queries. Callers pass in the already-scored ranking and current state.

The execution model:
  1. Score all eligible symbols with the swing model (LambdaRank → ordinal rank).
  2. Apply liquidity_filter to exclude illiquid names (PIT-safe: bars strictly
     before as_of).
  3. Apply apply_sector_cap to avoid sector concentration.
  4. compute_target_portfolio with two-band hysteresis to reduce turnover.
  5. compute_equal_weights (or with regime multiplier) to get dollar allocations.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Liquidity filter
# ---------------------------------------------------------------------------

def liquidity_filter(
    bars_map: Mapping[str, pd.DataFrame],
    as_of: date,
    min_avg_daily_dollar_vol: float = 20_000_000.0,
    lookback_days: int = 60,
) -> Set[str]:
    """Return symbols with sufficient trailing liquidity.

    Eligibility: trailing avg(close * volume) over the `lookback_days` bars
    strictly BEFORE `as_of` must be >= `min_avg_daily_dollar_vol`.

    Symbols with fewer than lookback_days // 2 valid bars are excluded to
    avoid a thin-history bias.
    """
    as_of_ts = pd.Timestamp(as_of)
    eligible: Set[str] = set()
    min_bars = max(1, lookback_days // 2)

    for sym, df in bars_map.items():
        if df is None or df.empty:
            continue
        close_col = "close" if "close" in df.columns else "Close"
        vol_col = "volume" if "volume" in df.columns else "Volume"
        if close_col not in df.columns or vol_col not in df.columns:
            continue

        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx)
            except Exception:
                continue

        mask = idx < as_of_ts
        prior = df.loc[mask].tail(lookback_days)
        if len(prior) < min_bars:
            continue

        dollar_vol = prior[close_col] * prior[vol_col]
        if float(dollar_vol.mean()) >= min_avg_daily_dollar_vol:
            eligible.add(sym)

    return eligible


# ---------------------------------------------------------------------------
# Sector cap
# ---------------------------------------------------------------------------

def apply_sector_cap(
    ranked_symbols: Sequence[str],
    sector_map: Mapping[str, str],
    cap: float = 0.30,
    n_target: int = 30,
) -> List[str]:
    """Walk the ranking top-down, accepting symbols while sector share <= cap.

    The cap is computed as accepted_in_sector / n_target (so the denominator
    is always n_target, not the running accepted count — this is stable and
    deterministic). Symbols with unknown sector are bucketed as 'UNKNOWN' and
    capped identically. Returns an ordered list of length <= n_target.
    """
    sector_counts: Dict[str, int] = {}
    accepted: List[str] = []
    max_per_sector = int(np.floor(cap * n_target))
    max_per_sector = max(1, max_per_sector)

    for sym in ranked_symbols:
        if len(accepted) >= n_target:
            break
        sector = sector_map.get(sym, "UNKNOWN")
        count = sector_counts.get(sector, 0)
        if count < max_per_sector:
            accepted.append(sym)
            sector_counts[sector] = count + 1

    return accepted


# ---------------------------------------------------------------------------
# Hysteresis target portfolio
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RebalanceDelta:
    target: List[str]    # full target set after rebalance
    to_add: List[str]    # newly-entering symbols
    to_drop: List[str]   # symbols to close
    held: List[str]      # symbols kept without action


def compute_target_portfolio(
    ranked_symbols: Sequence[str],
    current_holdings: Iterable[str],
    n_target: int = 30,
    add_rank_threshold: int = 15,
    drop_rank_threshold: int = 30,
) -> RebalanceDelta:
    """Two-band hysteresis to reduce unnecessary turnover.

    - A held symbol is KEPT if its rank (1-indexed) <= drop_rank_threshold
      AND it is still in ranked_symbols.
    - A held symbol is DROPPED if its rank > drop_rank_threshold OR it is
      absent from ranked_symbols.
    - New symbols enter only if their rank <= add_rank_threshold AND we still
      have room (len(target) < n_target).
    - After computing keeps, fill remaining slots from ranked_symbols not
      currently held, in rank order, up to n_target.

    `ranked_symbols` should already have liquidity and sector cap applied.
    """
    rank_of: Dict[str, int] = {sym: i + 1 for i, sym in enumerate(ranked_symbols)}
    holdings = set(current_holdings)

    # Step 1: decide which holdings to keep
    kept: List[str] = []
    dropped: List[str] = []
    for sym in holdings:
        r = rank_of.get(sym)
        if r is not None and r <= drop_rank_threshold:
            kept.append(sym)
        else:
            dropped.append(sym)

    # Step 2: fill open slots with new top-ranked names
    added: List[str] = []
    for sym, r in rank_of.items():
        if len(kept) + len(added) >= n_target:
            break
        if sym in holdings:
            continue  # already handled above
        if r <= add_rank_threshold:
            added.append(sym)

    # Step 3: if still under n_target, open slots with next-best (beyond threshold)
    if len(kept) + len(added) < n_target:
        for sym in ranked_symbols:
            if len(kept) + len(added) >= n_target:
                break
            if sym not in holdings and sym not in added:
                added.append(sym)

    target = kept + added
    return RebalanceDelta(
        target=target,
        to_add=added,
        to_drop=dropped,
        held=kept,
    )


# ---------------------------------------------------------------------------
# Equal-weight sizing
# ---------------------------------------------------------------------------

def compute_equal_weights(
    symbols: Sequence[str],
    total_equity: float,
    gross_exposure_multiplier: float = 1.0,
) -> Dict[str, float]:
    """Equal-dollar allocation within the regime-adjusted gross exposure.

    Each symbol receives (total_equity * gross_exposure_multiplier) / N dollars.
    Returns {} for empty input. Does NOT convert to share counts — the
    Trader/Simulator layer handles dollars→shares at fill price.
    """
    symbols = list(symbols)
    if not symbols or total_equity <= 0:
        return {}
    per_position = (total_equity * gross_exposure_multiplier) / len(symbols)
    return {sym: per_position for sym in symbols}
