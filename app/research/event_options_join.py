"""
event_options_join.py — PIT event-time join of the daily options feature table
onto each earnings event (Alpha-v6 P3 enrichment; prerequisite for H2/H3).

Populates the event panel's OPTION_COLUMNS (app/research/event_panel.OPTION_COLUMNS)
from app/data/options_features.parquet at the PRE-EVENT snapshot — the last options
chain knowable strictly BEFORE the announcement (announce_ts_flag is UNK, so we
never assume the announce-day chain is pre-release). Pure: the runner does I/O.

For event (symbol, announce_date, announce_gap_pct), given that symbol's daily
feature rows, computes:
  cpiv_pre               = cpiv_matched_delta at the pre-event date
  skew_25d_pre           = skew_25d_put         "
  term_kink_pre          = term_slope_30_60     "  (the 30->60d slope; documented)
  opt_volume_z_pre       = opt_volume_z          "
  pre_event_implied_move = implied_move_front    "
  iv_runup_t10_t1        = atm_iv_30d[pre] / atm_iv_30d[~10 trading days prior] - 1
  reaction_ratio         = |announce_gap_pct| / pre_event_implied_move  (under-
                           reaction proxy; small realized vs implied => more drift)
  post_iv_retention_t1   = atm_iv_30d[first chain AFTER announce] / atm_iv_30d[pre]
                           (IV-crush retention; knowable at t+1, used post-event)
  options_coverage_flag  = a usable, non-stale pre-event snapshot existed.

STALENESS: a pre-event row counts only if it is within MAX_PRE_STALE_DAYS calendar
days before the announcement (a name whose options stopped trading has no usable
pre-event chain). Any leg whose inputs are missing/non-positive is NaN — never a
fabricated value.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

# A pre-event chain must be this recent (calendar days before the announce) to use.
MAX_PRE_STALE_DAYS = 7
# Target lookback (trading rows) for the IV run-up's earlier leg.
IV_RUNUP_LOOKBACK = 10

_NAN_RESULT = {
    "pre_event_implied_move": np.nan, "iv_runup_t10_t1": np.nan,
    "reaction_ratio": np.nan, "cpiv_pre": np.nan, "skew_25d_pre": np.nan,
    "term_kink_pre": np.nan, "opt_volume_z_pre": np.nan,
    "post_iv_retention_t1": np.nan, "options_coverage_flag": False,
}


def _val(row: Optional[pd.Series], col: str) -> float:
    if row is None or col not in row or pd.isna(row[col]):
        return np.nan
    return float(row[col])


def compute_event_option_features(announce_date, announce_gap_pct,
                                  feat_rows: pd.DataFrame) -> dict:
    """The OPTION_COLUMNS for one event. `feat_rows` is THIS symbol's slice of the
    options feature table (columns: date, atm_iv_30d, implied_move_front,
    cpiv_matched_delta, skew_25d_put, term_slope_30_60, opt_volume_z), any order.
    PIT: the pre-event snapshot uses only chain dates STRICTLY before
    `announce_date`; post_iv_retention uses the first chain strictly AFTER (a
    post-event input, knowable at t+1)."""
    if feat_rows is None or feat_rows.empty:
        return dict(_NAN_RESULT)
    f = feat_rows.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date").reset_index(drop=True)
    ann = pd.Timestamp(announce_date)

    # PIT pre-event snapshot: the last chain PUBLICLY KNOWABLE strictly before the
    # announce day. With UNK BMO/AMC timing, gate on the feature table's
    # holiday-aware knowable_date (= chain date + 1 NYSE session) when present, so
    # a chain whose data only becomes public ON the announce day is NOT used (the
    # conservative choice); fall back to the chain `date` if knowable_date is absent.
    asof = (pd.to_datetime(f["knowable_date"]) if "knowable_date" in f.columns
            else f["date"])
    pre_mask = asof < ann
    if not pre_mask.any():
        return dict(_NAN_RESULT)
    pre_idx = f.index[pre_mask][-1]
    pre = f.loc[pre_idx]
    # Staleness guard: the pre-event chain must be recent enough to be the event's.
    if (ann - pre["date"]) > timedelta(days=MAX_PRE_STALE_DAYS):
        return dict(_NAN_RESULT)

    out = dict(_NAN_RESULT)
    out["options_coverage_flag"] = True
    out["cpiv_pre"] = _val(pre, "cpiv_matched_delta")
    out["skew_25d_pre"] = _val(pre, "skew_25d_put")
    out["term_kink_pre"] = _val(pre, "term_slope_30_60")
    out["opt_volume_z_pre"] = _val(pre, "opt_volume_z")
    im = _val(pre, "implied_move_front")
    out["pre_event_implied_move"] = im if (np.isfinite(im) and im > 0) else np.nan

    # IV run-up: ATM IV at pre vs ~IV_RUNUP_LOOKBACK trading rows earlier.
    atm_pre = _val(pre, "atm_iv_30d")
    pos = f.index.get_loc(pre_idx)
    if pos - IV_RUNUP_LOOKBACK >= 0 and np.isfinite(atm_pre) and atm_pre > 0:
        atm_ref = _val(f.loc[f.index[pos - IV_RUNUP_LOOKBACK]], "atm_iv_30d")
        if np.isfinite(atm_ref) and atm_ref > 0:
            out["iv_runup_t10_t1"] = atm_pre / atm_ref - 1.0

    # Reaction ratio: realized announce-day |move| relative to the implied move.
    if (np.isfinite(out["pre_event_implied_move"])
            and announce_gap_pct is not None and np.isfinite(announce_gap_pct)):
        out["reaction_ratio"] = abs(float(announce_gap_pct)) / out["pre_event_implied_move"]

    # Post-event IV retention: first chain AFTER the announce vs the pre chain.
    post_mask = f["date"] > ann
    if post_mask.any() and np.isfinite(atm_pre) and atm_pre > 0:
        atm_post = _val(f.loc[f.index[post_mask][0]], "atm_iv_30d")
        if np.isfinite(atm_post) and atm_post > 0:
            out["post_iv_retention_t1"] = atm_post / atm_pre
    return out
