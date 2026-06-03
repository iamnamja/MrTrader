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


# ---------------------------------------------------------------------------
# L/S helpers
# ---------------------------------------------------------------------------

def apply_sector_cap_shorts(
    worst_first_symbols: Sequence[str],
    sector_map: Mapping[str, str],
    cap: float = 0.30,
    n_target: int = 30,
) -> List[str]:
    """apply_sector_cap for the short book: walk worst-first (same greedy logic).

    NOTE (RANKER v2 §3.1, Failure B): this PER-SIDE cap bounds the short *count*
    per sector at floor(cap * n_target) regardless of how the long book is
    distributed. On R1K the bottom-of-momentum short tail is sector-concentrated
    (+ illiquid), so this cap structurally starves the short leg (realized short
    gross ~0.13 vs 0.40). Use `apply_net_sector_cap` instead to control NET sector
    exposure (long − short) — it lets the short tail concentrate where the longs
    do NOT, which is exactly the momentum-ranker case. This per-side helper is kept
    as the default so existing rebalance/L-S runs are byte-identical.
    """
    return apply_sector_cap(worst_first_symbols, sector_map, cap=cap, n_target=n_target)


def apply_net_sector_cap(
    worst_first_symbols: Sequence[str],
    long_book: Sequence[str],
    sector_map: Mapping[str, str],
    cap: float = 0.30,
    n_target: int = 30,
) -> List[str]:
    """Admit shorts greedily (worst-first) subject to a NET-sector-exposure cap.

    Replaces the per-side short-gross sector cap (RANKER v2 §3.1, Failure B). The
    short tail is allowed to CONCENTRATE in a sector — what is bounded is the book's
    NET exposure per sector, measured (as a fraction of `n_target`) by

        net_share[s] = (long_count_in_s - short_count_in_s) / n_target

    A short is admitted only if, after admission, |net_share[sector]| <= cap. Because
    the long book occupies its own sectors (winners) and the short tail occupies the
    losers' sectors, the net per-sector exposure is naturally bounded even when the
    raw short tail is concentrated — so shorts that the old per-side cap would have
    refused (purely because too many land in one sector) are now admitted, letting
    the short leg reach its gross target. Residual net beta the single-name shorts
    cannot neutralize is handled by the SPY beta-hedge overlay.

    Deterministic and order-stable: walks `worst_first_symbols` in rank order and
    accepts up to `n_target`. `long_book` is the already-selected long target set
    (its sector counts seed the net tally). Unknown sectors bucket as 'UNKNOWN'.
    """
    n_target = max(1, int(n_target))
    # Long sector counts seed the net tally (longs push net POSITIVE per sector).
    long_count: Dict[str, int] = {}
    for sym in long_book:
        sec = sector_map.get(sym, "UNKNOWN")
        long_count[sec] = long_count.get(sec, 0) + 1

    short_count: Dict[str, int] = {}
    accepted: List[str] = []
    # |net_share| <= cap  ⇔  |long_count - short_count| <= cap * n_target.
    max_net = max(1, int(np.floor(cap * n_target)))

    long_set = set(long_book)
    for sym in worst_first_symbols:
        if len(accepted) >= n_target:
            break
        if sym in long_set:
            continue  # never short a name we are long (avoid intra-book wash)
        sec = sector_map.get(sym, "UNKNOWN")
        cur_short = short_count.get(sec, 0)
        cur_net = long_count.get(sec, 0) - cur_short
        new_net = cur_net - 1  # one more short in this sector
        # Admit if the result is within the cap band OR the short moves the sector
        # net TOWARD neutral (so a long-over-concentrated sector can be offset by its
        # shorts even when it starts outside the band — the key difference from the
        # per-side cap, which would just refuse the concentrated tail outright).
        if abs(new_net) <= max_net or abs(new_net) < abs(cur_net):
            accepted.append(sym)
            short_count[sec] = cur_short + 1

    # Phase 2 (§3.1) breadth pass: the net-sector cap above can leave the short book
    # far below n_target when the loser tail concentrates in a few sectors — starving
    # the COUNT, which both under-funds the leg and kills the breadth the thesis needs
    # (IR ~ IC·sqrt(breadth)). Dollar-neutrality is now enforced by per-leg SIZING (the
    # rebalance resize pass), and residual beta by the SPY hedge, so it is safe to fill
    # the remainder by rank — relaxing the net-sector bound — until n_target is reached.
    if len(accepted) < n_target:
        accepted_set = set(accepted)
        for sym in worst_first_symbols:
            if len(accepted) >= n_target:
                break
            if sym in long_set or sym in accepted_set:
                continue
            accepted.append(sym)
            accepted_set.add(sym)

    return accepted


def compute_target_portfolio_shorts(
    worst_first_symbols: Sequence[str],
    current_short_holdings: Iterable[str],
    n_target: int = 30,
    add_rank_threshold: int = 15,
    drop_rank_threshold: int = 30,
) -> "RebalanceDelta":
    """Hysteresis target for the short book.

    Works identically to compute_target_portfolio but the input list is already
    worst-first (rank 1 = most-shortable). The caller reverses the IC scored list
    before passing here.
    """
    return compute_target_portfolio(
        worst_first_symbols,
        current_short_holdings,
        n_target=n_target,
        add_rank_threshold=add_rank_threshold,
        drop_rank_threshold=drop_rank_threshold,
    )


def split_gross_budgets(
    equity: float,
    net_target: float = 0.40,
    gross_target: float = 1.50,
    long_regime_mult: float = 1.0,
    short_regime_mult: float = 1.0,
) -> tuple:
    """Compute per-side dollar budgets for L/S sizing.

    Net exposure = long_gross - short_gross = net_target × equity
    Gross exposure = long_gross + short_gross = gross_target × equity

    Solving:
        long_gross  = equity × (gross_target + net_target) / 2
        short_gross = equity × (gross_target - net_target) / 2

    Returns (long_budget, short_budget) after applying per-side regime multipliers.
    """
    long_base = equity * (gross_target + net_target) / 2.0
    short_base = equity * (gross_target - net_target) / 2.0
    return long_base * long_regime_mult, short_base * short_regime_mult


def compute_inverse_vol_weights(
    symbols: Sequence[str],
    bars_map: Mapping[str, "pd.DataFrame"],
    as_of: date,
    total_equity: float,
    gross_exposure_multiplier: float = 1.0,
    vol_lookback_days: int = 20,
    min_weight_mult: float = 0.5,
    max_weight_mult: float = 2.0,
) -> Dict[str, float]:
    """Inverse-volatility dollar allocation capped at min/max multiples of equal weight.

    Steps (all PIT-safe — uses bars strictly before as_of):
      1. Compute realized vol = std(daily returns) over vol_lookback_days per symbol.
      2. Invert: raw_weight[i] = 1 / vol[i].
      3. Normalize so weights sum to 1.
      4. Cap each weight at [equal_weight * min_weight_mult, equal_weight * max_weight_mult].
      5. Re-normalize after capping.
      6. Multiply by total_equity * gross_exposure_multiplier for dollar allocations.

    Falls back to equal-weight for symbols with insufficient history.
    """
    import pandas as _pd

    symbols = list(symbols)
    if not symbols or total_equity <= 0:
        return {}

    equal_w = 1.0 / len(symbols)
    _day_ts = _pd.Timestamp(as_of)

    vol_by_sym: Dict[str, float] = {}
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None:
            continue
        close_col = "close" if "close" in df.columns else "Close"
        if close_col not in df.columns:
            continue
        hist = df[df.index < _day_ts][close_col]
        if len(hist) < vol_lookback_days + 1:
            continue
        rets = hist.iloc[-(vol_lookback_days + 1):].pct_change().dropna()
        if len(rets) < 2:
            continue
        v = float(rets.std())
        if v > 0:
            vol_by_sym[sym] = v

    # Fall back to equal weight if vol unavailable for any symbol
    if not vol_by_sym or len(vol_by_sym) < len(symbols):
        return compute_equal_weights(symbols, total_equity, gross_exposure_multiplier)

    # Inverse-vol raw weights
    raw = {s: 1.0 / vol_by_sym[s] for s in symbols}
    total_raw = sum(raw.values())
    norm_w = {s: raw[s] / total_raw for s in symbols}

    # Cap at [equal_w * min_mult, equal_w * max_mult]
    lo = equal_w * min_weight_mult
    hi = equal_w * max_weight_mult
    capped = {s: max(lo, min(hi, norm_w[s])) for s in symbols}

    # Re-normalize after capping
    total_capped = sum(capped.values())
    final_w = {s: capped[s] / total_capped for s in symbols}

    gross = total_equity * gross_exposure_multiplier
    return {s: final_w[s] * gross for s in symbols}
