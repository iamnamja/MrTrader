"""
options_xs_ls.py — Alpha-v6 P4 options-as-signal cross-sectional L/S core.

The PURE logic behind the H4a-H4e confirmatory runs (scripts/run_options_xs_cpcv.py
drives the I/O): the weekly dollar-neutral DECILE long/short construction, the
multi-factor residual-alpha test, and the factor-return frame. Kept pure (no I/O,
no look-ahead) so the math is unit-testable in isolation; the runner supplies the
per-name feature values and forward returns.

The five hypotheses (scripts/preregister_options_xs_features.py, frozen
2026-06-12T12:00Z) each test whether an options-derived feature carries a
cross-sectional EQUITY edge, executed as a weekly dollar-neutral L/S sleeve
(information harvested at EQUITY cost — NOT an options trade). The line is a
SIMPLE per-feature decile sort on purpose (the kill rule forbids escalating to ML
combinations).

DIRECTION (frozen): each feature's `FEATURE_DIRECTION` is the sign of the
hypothesized feature->forward-return relationship. +1 = long the HIGH-feature
decile / short the LOW (the hypothesis says high feature -> high return); -1 =
long LOW / short HIGH. The L/S book is always built in the hypothesized direction
so the one-sided week-clustered t-stat tests the pre-committed sign.

H4c is the one CONDITIONED feature: its hypothesis is "high O/S WITH put-heavy
flow -> negative return", so its sort signal is the pre-registered composite
rank(opt_share_volume_ratio) + rank(put_call_volume_ratio) (high = both a high
option/stock volume ratio AND put-dominated flow), shorted at the top. This is
the conditioning the hypothesis specifies (conditioned_on=put_call_volume_ratio),
NOT a fitted/ML combination — see build_signal().
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

ANN = 252

# Per-feature hypothesized sign of the feature->forward-return relationship.
# +1: long the high-feature decile (high feature -> high return).
# -1: long the low-feature decile (high feature -> low return).
FEATURE_DIRECTION: Dict[str, int] = {
    "cpiv_matched_delta": +1,    # H4a Cremers-Weinbaum: high CPIV -> + return
    "skew_25d_put": -1,          # H4b steep put skew -> - return
    "opt_share_volume_ratio": -1,  # H4c high put-heavy O/S -> - return (composite)
    "term_slope_30_60": +1,      # H4d contango (high slope) -> + return
    "iv_rv_20d_ratio": -1,       # H4e rich IV/RV -> - return
}

# H4c is sorted on a composite of its feature + the conditioning column.
CONDITIONED_FEATURE = "opt_share_volume_ratio"
CONDITION_COLUMN = "put_call_volume_ratio"

# Minimum qualified names in a week to form clean deciles (need >= 2 per leg, and
# enough total that a decile is a meaningful slice). Below this the week is skipped.
MIN_NAMES_FOR_DECILES = 20


def build_signal(feats: pd.DataFrame, feature: str) -> pd.Series:
    """The cross-sectional sort signal for one rebalance, indexed by symbol.

    `feats` is the per-name latest-known feature frame for ONE rebalance date
    (one row per symbol; must carry `feature`, and for H4c also CONDITION_COLUMN).
    For four features the signal IS the feature column. For H4c
    (opt_share_volume_ratio) the signal is the pre-registered composite
    rank(O/S) + rank(put/call) — high when BOTH the option/stock volume ratio and
    the put/call ratio are high (the "put-heavy O/S" the hypothesis conditions on).
    Rows with a NaN signal input are dropped (cannot be sorted)."""
    if feature == CONDITIONED_FEATURE:
        cols = feats[[CONDITIONED_FEATURE, CONDITION_COLUMN]].dropna()
        if cols.empty:
            return pd.Series(dtype=float)
        # Average percentile rank of the two legs (monotone in both); 'first' so
        # ties don't collapse names. The composite is high <=> high O/S AND
        # put-heavy flow, exactly the conditioned signal H4c bets short.
        r_os = cols[CONDITIONED_FEATURE].rank(method="first", pct=True)
        r_pc = cols[CONDITION_COLUMN].rank(method="first", pct=True)
        return (r_os + r_pc) / 2.0
    return feats[feature].dropna()


def decile_ls_weights(signal: pd.Series, direction: int,
                      n_deciles: int = 10,
                      min_names: int = MIN_NAMES_FOR_DECILES) -> pd.Series:
    """Equal-weight DECILE long/short weights from one rebalance's sort `signal`
    (indexed by symbol) — the textbook HIGH-MINUS-LOW spread book: the long leg
    sums to +1, the short leg to -1 (net = 0, gross = 2), so sum_i w_i*r_i is the
    standard top-decile-mean minus bottom-decile-mean spread return. Long the
    decile the hypothesis expects to outperform, short the opposite.

    direction=+1 -> long the TOP (high-signal) decile, short the BOTTOM.
    direction=-1 -> long the BOTTOM (low-signal) decile, short the TOP.
    Returns an EMPTY Series if fewer than `min_names` names (week skipped) — never
    a one-sided or degenerate book."""
    s = signal.dropna()
    if len(s) < min_names:
        return pd.Series(dtype=float)
    # Decile bucket by rank (0 = lowest signal .. n_deciles-1 = highest). 'first'
    # ranking avoids tie-induced empty/overfull buckets.
    ranks = s.rank(method="first")
    buckets = np.minimum((ranks.to_numpy() - 1) //
                         (len(s) / n_deciles), n_deciles - 1).astype(int)
    bucket = pd.Series(buckets, index=s.index)
    top = s.index[bucket == n_deciles - 1]
    bottom = s.index[bucket == 0]
    if len(top) == 0 or len(bottom) == 0:
        return pd.Series(dtype=float)
    long_names = top if direction > 0 else bottom
    short_names = bottom if direction > 0 else top
    w = pd.Series(0.0, index=s.index)
    w[long_names] = 1.0 / len(long_names)
    w[short_names] = -1.0 / len(short_names)
    return w[w != 0.0]


def ls_spread_return(weights: pd.Series, fwd_returns: pd.Series) -> float:
    """The realized L/S spread return for one held week: sum_i w_i * r_i over the
    names with BOTH a weight and a forward return. Names missing a forward return
    (delisted/halted mid-week) are dropped and the surviving legs renormalized to
    keep the high-minus-low convention (long leg sums +1, short leg -1, net 0), so
    a missing name can't silently net-bias the book. Returns 0.0 if either leg
    fully vanishes."""
    common = weights.index.intersection(fwd_returns.dropna().index)
    if len(common) == 0:
        return 0.0
    w = weights.loc[common].copy()
    longs = w[w > 0]
    shorts = w[w < 0]
    if longs.empty or shorts.empty:
        return 0.0
    # Renormalize each leg after any drops: longs sum to +1; shorts (already
    # negative) scale to sum to -1 (preserve the net-0, high-minus-low book).
    w.loc[longs.index] = longs * (1.0 / longs.sum())
    w.loc[shorts.index] = shorts * (1.0 / shorts.abs().sum())
    return float((w * fwd_returns.loc[common]).sum())


def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    """One-way turnover sum|w_new - w_prev| between two consecutive weekly books
    (union of names; absent = 0 weight). Used to charge per-rebalance costs."""
    idx = prev_w.index.union(new_w.index)
    return float((new_w.reindex(idx).fillna(0.0)
                  - prev_w.reindex(idx).fillna(0.0)).abs().sum())


def build_factor_frame(etf_closes: Dict[str, pd.Series]) -> pd.DataFrame:
    """The 5-column DAILY factor-RETURN matrix for the multi-factor residual-alpha
    test, from raw ETF close series (each a date-indexed close Series):
      SPY        — market.
      IWM_SPY    — size  (small-minus-big: IWM return - SPY return).
      MTUM_SPY   — momentum tilt (MTUM - SPY).
      VLUE_SPY   — value tilt (VLUE - SPY).
      VIXY       — vol/convexity.
    The three style legs are LONG-SHORT spreads vs SPY (so the design isn't
    collinear with raw SPY). Pure: returns are computed from the supplied closes,
    aligned on the common dates, NaNs dropped. Missing ETFs are omitted (the
    runner warns); SPY is required (returns empty if absent)."""
    need = {"SPY", "IWM", "MTUM", "VLUE", "VIXY"}
    rets = {}
    for sym, cl in etf_closes.items():
        if cl is None or len(cl) < 2:
            continue
        s = pd.Series(cl).sort_index()
        rets[sym] = s.pct_change()
    if "SPY" not in rets:
        return pd.DataFrame()
    df = pd.DataFrame(rets).dropna(how="all")
    out = pd.DataFrame(index=df.index)
    out["SPY"] = df["SPY"]
    for style in ("IWM", "MTUM", "VLUE"):
        if style in df.columns:
            out[f"{style}_SPY"] = df[style] - df["SPY"]
    if "VIXY" in df.columns:
        out["VIXY"] = df["VIXY"]
    missing = need - set(etf_closes)
    if missing:
        out.attrs["missing_factors"] = sorted(missing)
    return out.dropna()


# multifactor_alpha moved to the canonical inference core (Ruler v2); re-exported
# below so existing callers (xs.multifactor_alpha) keep working.
from app.research.inference import multifactor_alpha  # noqa: E402,F401


def weekly_spread_panel(weekly: List[dict]) -> pd.DataFrame:
    """Assemble the per-week records (each {week, ls_spread, n_long, n_short, ...})
    into the panel the week-clustered inference consumes. `week` is the rebalance
    date; the panel's one row per week IS the cluster unit (WEEKS as clusters)."""
    if not weekly:
        return pd.DataFrame(columns=["week", "ls_spread"])
    return pd.DataFrame(weekly).sort_values("week").reset_index(drop=True)
