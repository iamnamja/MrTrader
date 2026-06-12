"""
options_features.py — the daily options FEATURE TABLE (Alpha-v6 Phase 4, PR4a).

One row per (underlying, date), computed from the computed-greeks store
(data/options_greeks/underlying={U}/part-0.parquet, built by
scripts/backfill_computed_greeks.py) chain snapshot ON `date`. Each row carries
the store's HOLIDAY-AWARE `knowable_date` (= date + 1 NYSE session, carried
straight through from the greeks store — NOT recomputed with a plain business-day
offset, which would land on a holiday) so PIT consumers filter
`knowable_date <= as_of`.

These are the five cross-sectional EQUITY-signal features the P4 confirmatory
hypotheses H4a-H4e (scripts/preregister_options_xs_features.py) are adjudicated
on, plus the volume/coverage columns the options-quality filter
(app/data/options_quality.py) needs:

  atm_iv_30d            IV of the ATM contract (|delta| nearest 0.5) at the
                        ~30-DTE expiry (the front analysis tenor).
  implied_move_front    front-expiry ATM straddle (nearest-ATM call + put)
                        / underlying_close — generalizes
                        ImpliedMoveProvider.implied_move to ANY date (NOT
                        event-timed). "Front" = the nearest expiry >= MIN_DTE.
  cpiv_matched_delta    IV(call) - IV(put) at matched |delta|~=0.5 (ATM),
                        ~30 DTE (Cremers-Weinbaum). Call and put are the two
                        whose |delta| is closest to 0.5 in the SAME ~30-DTE
                        expiry, and their |delta| must agree within
                        CPIV_DELTA_MATCH_TOL or the value is NaN (coverage flag).
  skew_25d_put          IV(put, |delta|~=0.25) - IV(ATM, |delta|~=0.5), ~30 DTE
                        (Xing-Zhang-Zhao). NaN if no put within the 0.25 band.
  term_slope_30_60      IV(ATM ~60 DTE) - IV(ATM ~30 DTE).
  iv_rv_20d_ratio       atm_iv_30d / (20d realized vol, annualized
                        = std(20 daily log-returns) * sqrt(252)). RV uses the
                        SPLIT-ADJUSTED equity closes (event_panel loader) STRICTLY
                        BEFORE `date` (PIT) when the builder supplies them, falling
                        back to the store's as-traded underlying_close (which is
                        split-jump-guarded — an unadjusted series steps on a split
                        and a raw return-vol would spike ~20x). Split-adjusted
                        closes are the correct return basis; the store close is
                        as-traded by design (right for greeks pricing, wrong for RV).
  opt_share_volume_ratio  (sum option contract volume across the valid chain
                        * 100) / equity share volume that day. Equity share
                        volume is read from the daily-bars loader
                        (event_panel._get_daily_bars; PIT, the `date` session's
                        own share volume — a SAME-DAY input, like option volume).
  put_call_volume_ratio sum(put volume) / sum(call volume) across the valid
                        chain.
  opt_volume_z          (total option volume on date - trailing-20d mean) /
                        trailing-20d std, per underlying (PIT, STRICTLY-prior
                        window of the underlying's own daily option volume).

QUALITY (the frozen contract): a contract is "valid" only if
solver_status == "ok" AND stale_flag is False. A name-date with fewer than
MIN_VALID_CONTRACTS valid contracts is dropped entirely (no row emitted).

TENOR SELECTION (documented, frozen): the "~30 DTE" / "~60 DTE" analysis tenors
are the expiry whose DTE is NEAREST the target, AS LONG AS it lands inside the
tolerance band [target - TENOR_TOL_DAYS, target + TENOR_TOL_DAYS]. No expiry in
band -> that tenor's features are NaN + a coverage flag. The "front" tenor (for
implied_move_front) is the nearest expiry with DTE >= FRONT_MIN_DTE (the
0-DTE-pin guard ImpliedMoveProvider uses), capped at FRONT_MAX_DTE.

DELTA MATCHING (documented, frozen): inside a chosen expiry, "ATM" = the
contract whose |delta| is nearest 0.5; "25-delta put" = the put whose |delta| is
nearest 0.25, accepted only if within DELTA_25_TOL of 0.25. CPIV pairs the ATM
call and ATM put and requires |delta_call| and |delta_put| to AGREE within
CPIV_DELTA_MATCH_TOL (else NaN) — a one-sided ATM is not a matched-delta spread.

coverage_flags is a bitfield (CF_*) recording which optional legs were missing;
n_valid_contracts is the count that survived the quality filter.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── coverage_flags bitfield ──────────────────────────────────────────────────
CF_MISSING_ATM = 1        # no ~30-DTE expiry in band, or no ATM contract there
CF_MISSING_25D_PUT = 2    # no put within the 0.25-delta band at ~30 DTE
CF_MISSING_60D_TENOR = 4  # no ~60-DTE expiry in band (term_slope NaN)
CF_NO_EQUITY_VOLUME = 8   # equity share volume unavailable -> O/S ratio NaN
CF_THIN_CHAIN = 16        # n_valid_contracts < THIN_CHAIN_WARN (row still emitted)
CF_MISSING_CPIV = 32      # ATM call/put deltas don't match within tol -> CPIV NaN
CF_MISSING_FRONT = 64     # no front expiry >= FRONT_MIN_DTE -> implied_move NaN

# ── quality / selection constants (frozen contract; tunable only via a new PR) ─
MIN_VALID_CONTRACTS = 6   # drop a name-date with fewer valid ("ok"+fresh) rows
THIN_CHAIN_WARN = 12      # below this, emit the row but set CF_THIN_CHAIN

TARGET_DTE_30 = 30        # primary analysis tenor (the "~30 DTE" surface point)
TARGET_DTE_60 = 60        # term-structure long leg
# An expiry qualifies for a target if |DTE - target| <= TENOR_TOL_DAYS (covers
# monthly + weekly listing gaps around the 30/60d points).
TENOR_TOL_DAYS = 12

# Front-expiry floor (skip same-day/0-DTE pins — the ImpliedMoveProvider min_dte
# spirit) and ceiling (ignore a name with no near expiry at all).
FRONT_MIN_DTE = 2
FRONT_MAX_DTE = 75

ATM_DELTA = 0.50          # ATM = |delta| nearest this
TARGET_25D = 0.25         # 25-delta put target
DELTA_25_TOL = 0.10       # accept a "25d put" only if |delta| in [0.15, 0.35]
CPIV_DELTA_MATCH_TOL = 0.10  # ATM call/put |delta| must agree within this for CPIV

RV_WINDOW = 20            # realized-vol window (daily log-returns)
RV_ANNUALIZE = float(np.sqrt(252.0))
OPT_VOL_Z_WINDOW = 20     # trailing window for opt_volume_z (strictly prior)
# Single-day log-returns whose |value| exceeds this are treated as split steps,
# NOT volatility, and dropped from the RV window. An UNADJUSTED close series
# steps on a split (a 10:1 leaves a ~-2.30 log-return; the smallest common
# forward split, 2:1, is 0.69) and even ONE such jump inflates a 20d RV ~20x —
# non-random (only split names), which would bias the H4e (IV/RV) sort. Real
# R1K single-day moves sit well below 0.50 (~65%). This is defense-in-depth for
# the store-close fallback; the PRIMARY RV close path (split-adjusted equity
# bars, see assemble_underlying_features) carries no split steps at all.
SPLIT_LOGRET_THRESHOLD = 0.50

# Output columns (single-file parquet; `underlying`/`date`/`knowable_date` carried
# IN-file here — unlike the greeks store, this table is small enough to be one file
# and the scorers read it whole + filter by knowable_date).
FEATURE_COLS = [
    "underlying", "date", "knowable_date",
    "atm_iv_30d", "implied_move_front", "cpiv_matched_delta", "skew_25d_put",
    "term_slope_30_60", "iv_rv_20d_ratio", "opt_share_volume_ratio",
    "put_call_volume_ratio", "opt_volume_z",
    "total_opt_volume", "n_valid_contracts", "coverage_flags",
]


# ═════════════════════════════════════════════════════════════════════════════
# Tenor + delta selection helpers (pure; operate on a single-expiry / single-date
# slice of valid contracts)
# ═════════════════════════════════════════════════════════════════════════════

def select_tenor(expiries_dte: Dict[float, int], target: int,
                 tol: int = TENOR_TOL_DAYS) -> Optional[float]:
    """Return the expiration key whose DTE is nearest `target` within +-tol, or
    None. `expiries_dte` maps an expiry id (e.g. a pd.Timestamp) -> integer DTE."""
    best, best_gap = None, None
    for exp, dte in expiries_dte.items():
        gap = abs(int(dte) - target)
        if gap <= tol and (best_gap is None or gap < best_gap):
            best, best_gap = exp, gap
    return best


def select_front_tenor(expiries_dte: Dict[float, int],
                       min_dte: int = FRONT_MIN_DTE,
                       max_dte: int = FRONT_MAX_DTE) -> Optional[float]:
    """Front analysis expiry = the NEAREST expiry with DTE in [min_dte, max_dte]
    (skips 0/1-DTE pins). Returns the expiry key or None."""
    candidates = {e: d for e, d in expiries_dte.items() if min_dte <= d <= max_dte}
    if not candidates:
        return None
    return min(candidates, key=lambda e: candidates[e])


def _nearest_by_abs_delta(leg: pd.DataFrame, target_abs: float) -> Optional[pd.Series]:
    """Row of `leg` whose |delta| is nearest `target_abs` (NaN-delta rows excluded)."""
    cand = leg[leg["delta"].notna()]
    if cand.empty:
        return None
    gaps = (cand["delta"].abs() - target_abs).abs()
    return cand.loc[gaps.idxmin()]


# ═════════════════════════════════════════════════════════════════════════════
# Per-(underlying, date) feature computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_iv_term_features(valid: pd.DataFrame) -> dict:
    """The IV-surface features (atm_iv_30d, cpiv_matched_delta, skew_25d_put,
    term_slope_30_60) + their coverage flags, from ONE date's VALID chain rows.

    `valid` must already be quality-filtered (solver_status=="ok", not stale) and
    carry numeric `delta`, `iv`, `expiration` (datetime), and an integer `dte`
    column. Pure: no I/O, no look-ahead (single-date snapshot)."""
    out = {
        "atm_iv_30d": np.nan, "cpiv_matched_delta": np.nan,
        "skew_25d_put": np.nan, "term_slope_30_60": np.nan, "flags": 0,
    }
    if valid.empty:
        out["flags"] |= CF_MISSING_ATM | CF_MISSING_25D_PUT | CF_MISSING_60D_TENOR
        return out

    dte_by_exp = valid.groupby("expiration")["dte"].first().to_dict()

    exp30 = select_tenor(dte_by_exp, TARGET_DTE_30)
    atm30 = None
    if exp30 is None:
        out["flags"] |= CF_MISSING_ATM | CF_MISSING_25D_PUT
    else:
        leg30 = valid[valid["expiration"] == exp30]
        atm30 = _nearest_by_abs_delta(leg30, ATM_DELTA)
        if atm30 is None:
            out["flags"] |= CF_MISSING_ATM
        else:
            out["atm_iv_30d"] = float(atm30["iv"])

        # CPIV — matched-delta ATM call minus ATM put in the SAME 30d expiry.
        calls30 = leg30[leg30["contract_type"] == "call"]
        puts30 = leg30[leg30["contract_type"] == "put"]
        atm_c = _nearest_by_abs_delta(calls30, ATM_DELTA)
        atm_p = _nearest_by_abs_delta(puts30, ATM_DELTA)
        if atm_c is not None and atm_p is not None and \
                abs(abs(float(atm_c["delta"])) - abs(float(atm_p["delta"]))) \
                <= CPIV_DELTA_MATCH_TOL:
            out["cpiv_matched_delta"] = float(atm_c["iv"]) - float(atm_p["iv"])
        else:
            out["flags"] |= CF_MISSING_CPIV

        # 25-delta put skew — IV(put, |delta|~=0.25) - IV(ATM, |delta|~=0.5).
        p25 = _nearest_by_abs_delta(puts30, TARGET_25D)
        if p25 is not None and abs(abs(float(p25["delta"])) - TARGET_25D) \
                <= DELTA_25_TOL and atm30 is not None:
            out["skew_25d_put"] = float(p25["iv"]) - float(atm30["iv"])
        else:
            out["flags"] |= CF_MISSING_25D_PUT

    # Term slope — ATM IV at ~60 DTE minus ATM IV at ~30 DTE.
    exp60 = select_tenor(dte_by_exp, TARGET_DTE_60)
    if exp60 is not None and atm30 is not None:
        atm60 = _nearest_by_abs_delta(valid[valid["expiration"] == exp60], ATM_DELTA)
        if atm60 is not None:
            out["term_slope_30_60"] = float(atm60["iv"]) - float(atm30["iv"])
        else:
            out["flags"] |= CF_MISSING_60D_TENOR
    else:
        out["flags"] |= CF_MISSING_60D_TENOR
    return out


def compute_implied_move_front(valid: pd.DataFrame) -> dict:
    """Front-expiry ATM straddle / underlying_close — generalized (any-date)
    ImpliedMoveProvider.implied_move. ATM call/put are each the strike nearest
    the underlying close in the front expiry. NaN + CF_MISSING_FRONT if no front
    expiry / no two-sided ATM / non-positive spot."""
    out = {"implied_move_front": np.nan, "flags": 0}
    if valid.empty:
        out["flags"] |= CF_MISSING_FRONT
        return out
    dte_by_exp = valid.groupby("expiration")["dte"].first().to_dict()
    front = select_front_tenor(dte_by_exp)
    if front is None:
        out["flags"] |= CF_MISSING_FRONT
        return out
    leg = valid[valid["expiration"] == front]
    spot = float(leg["underlying_close"].iloc[0])
    calls = leg[leg["contract_type"] == "call"]
    puts = leg[leg["contract_type"] == "put"]
    if spot <= 0 or calls.empty or puts.empty:
        out["flags"] |= CF_MISSING_FRONT
        return out
    c_atm = calls.loc[(calls["strike"] - spot).abs().idxmin()]
    p_atm = puts.loc[(puts["strike"] - spot).abs().idxmin()]
    straddle = float(c_atm["close"]) + float(p_atm["close"])
    if straddle <= 0:
        out["flags"] |= CF_MISSING_FRONT
        return out
    out["implied_move_front"] = straddle / spot
    return out


def compute_volume_features(valid: pd.DataFrame,
                            equity_share_volume: Optional[float]) -> dict:
    """opt_share_volume_ratio + put_call_volume_ratio + total option volume
    (for opt_volume_z, computed across dates by the builder). All same-day.

    Note on the O/S adjustment basis: `equity_share_volume` is the loader's
    split-ADJUSTED share volume while option volume is raw contracts. Because the
    ratio is purely SAME-DAY and cross-sectional (never compared across a split),
    both legs are in that day's share terms and the ratio is internally
    consistent; the split-day basis is a documented non-issue, not a trailing
    distortion (opt_volume_z trails option volume only, not the ratio).

    `valid` is the QUALITY-filtered (ok + fresh) chain, so total_opt_volume and
    the O/S ratio are on the valid subset — consistent across names, by design."""
    out = {
        "opt_share_volume_ratio": np.nan, "put_call_volume_ratio": np.nan,
        "total_opt_volume": 0.0, "flags": 0,
    }
    if valid.empty:
        return out
    vol = valid["volume"].fillna(0.0)
    total = float(vol.sum())
    out["total_opt_volume"] = total
    call_vol = float(vol[valid["contract_type"] == "call"].sum())
    put_vol = float(vol[valid["contract_type"] == "put"].sum())
    if call_vol > 0:
        out["put_call_volume_ratio"] = put_vol / call_vol
    if equity_share_volume is not None and equity_share_volume > 0:
        # option volume is in CONTRACTS (x100 shares) — scale to share-equivalents.
        out["opt_share_volume_ratio"] = (total * 100.0) / float(equity_share_volume)
    else:
        out["flags"] |= CF_NO_EQUITY_VOLUME
    return out


def compute_realized_vol_pit(underlying_close_history: pd.Series,
                             as_of: pd.Timestamp,
                             window: int = RV_WINDOW) -> Optional[float]:
    """Annualized realized vol from the underlying's daily closes STRICTLY BEFORE
    `as_of` (PIT). `underlying_close_history` is a date-indexed close Series (the
    distinct daily underlying_close already present in the greeks store, or any
    equity close series). std of the last `window` daily log-returns * sqrt(252).
    None if fewer than `window` prior returns exist or the series is degenerate."""
    if underlying_close_history is None or underlying_close_history.empty:
        return None
    prior = underlying_close_history[underlying_close_history.index < as_of]
    prior = prior[prior > 0].sort_index()
    if len(prior) < window + 1:
        return None
    logret = np.diff(np.log(prior.to_numpy(dtype=float)))
    # Drop split-step jumps (see SPLIT_LOGRET_THRESHOLD) so an unadjusted close
    # fallback cannot inflate RV ~20x for the ~20 sessions after a split. The
    # last `window` SURVIVING returns form the estimate (reaching slightly
    # further back to backfill a dropped jump); harmless on already-adjusted
    # closes (no return that large survives in a large-cap name).
    logret = logret[np.abs(logret) <= SPLIT_LOGRET_THRESHOLD]
    if logret.size < window:
        return None
    rv = float(np.std(logret[-window:], ddof=1) * RV_ANNUALIZE)
    return rv if np.isfinite(rv) and rv > 0 else None


def assemble_underlying_features(
    greeks: pd.DataFrame,
    equity_bars: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the per-date feature rows for ONE underlying from its greeks-store
    slice. `greeks` is the underlying's part-0.parquet content (the GREEKS_COLS
    schema). `equity_bars` (optional) is a date-indexed daily-bars frame with
    `close` + `volume` columns (event_panel._get_daily_bars output) supplying
    (a) the SPLIT-ADJUSTED close series for RV and (b) equity share volume for
    opt_share_volume_ratio; when None, RV falls back to the store's as-traded
    close (split-jump-guarded) and the O/S ratio is NaN + flag.

    Returns a DataFrame with FEATURE_COLS, sorted by date. PIT-correct:
      * IV/skew/term/implied-move use only the chain rows dated == `date`;
      * iv_rv uses (split-adjusted) closes STRICTLY before `date`;
      * opt_volume_z uses the underlying's own option volume STRICTLY before
        `date` (trailing 20d);
      * knowable_date is carried through from the store (holiday-aware, = date +
        1 NYSE session) — emitted so downstream filters `knowable_date <= as_of`.
    """
    if greeks is None or greeks.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    g = greeks.copy()
    g["date"] = pd.to_datetime(g["date"])
    g["expiration"] = pd.to_datetime(g["expiration"])
    underlying = (str(g["underlying"].iloc[0]) if "underlying" in g.columns
                  else str(g.get("symbol", pd.Series(["?"])).iloc[0]))

    # knowable_date: carry the store's HOLIDAY-AWARE value through (date + 1 NYSE
    # session, stamped by the greeks backfill). Recomputing it here with a plain
    # business-day offset would land on a market holiday the session before every
    # holiday (e.g. 07-03 -> 07-04), publishing a name into the PIT sort one
    # trading day early. Map date -> knowable_date (unique per date in the store).
    knowable_by_date = None
    if "knowable_date" in g.columns:
        g["knowable_date"] = pd.to_datetime(g["knowable_date"])
        knowable_by_date = g.groupby("date")["knowable_date"].first()

    # Quality filter: only "ok" solver rows that are not stale.
    valid_all = g[(g["solver_status"] == "ok") & (~g["stale_flag"].astype(bool))].copy()
    valid_all["dte"] = (valid_all["expiration"] - valid_all["date"]).dt.days

    # PIT underlying-close history for RV. PREFER the SPLIT-ADJUSTED equity closes
    # (event_panel loader, auto_adjust) — the store's underlying_close is
    # AS-TRADED (unadjusted, correct for greeks pricing) and STEPS on a split,
    # which would corrupt a return-based RV for ~20 sessions per split (only split
    # names — a non-random bias into the H4e sort). Fall back to the store close
    # (split-jump-guarded in compute_realized_vol_pit) when equity bars are
    # unavailable. PIT is preserved downstream: compute_realized_vol_pit uses only
    # closes STRICTLY before each `date`.
    close_hist = None
    if equity_bars is not None and "close" in equity_bars.columns:
        ec = equity_bars.copy()
        ec.index = pd.to_datetime(ec.index)
        close_hist = ec["close"].dropna().sort_index()
        close_hist = close_hist[close_hist > 0]
    if close_hist is None or close_hist.empty:
        close_hist = (g.dropna(subset=["underlying_close"])
                      .groupby("date")["underlying_close"].first().sort_index())
        close_hist = close_hist[close_hist > 0]

    # Equity share volume per date (same-day input). Prefer the explicit equity
    # bars; otherwise no O/S ratio.
    eq_vol_by_date: Dict[pd.Timestamp, float] = {}
    if equity_bars is not None and "volume" in equity_bars.columns:
        ev = equity_bars.copy()
        ev.index = pd.to_datetime(ev.index)
        eq_vol_by_date = {d: float(v) for d, v in ev["volume"].items()
                          if pd.notna(v)}

    rows: List[dict] = []
    # Trailing option-volume history for the z-score (strictly-prior window).
    vol_hist: List[float] = []
    for d in sorted(valid_all["date"].unique()):
        day = valid_all[valid_all["date"] == d]
        n_valid = int(len(day))
        if n_valid < MIN_VALID_CONTRACTS:
            continue  # drop the name-date entirely (frozen quality contract)

        flags = 0
        if n_valid < THIN_CHAIN_WARN:
            flags |= CF_THIN_CHAIN

        iv_feats = compute_iv_term_features(day)
        im_feats = compute_implied_move_front(day)
        eq_vol = eq_vol_by_date.get(pd.Timestamp(d))
        vol_feats = compute_volume_features(day, eq_vol)
        flags |= iv_feats["flags"] | im_feats["flags"] | vol_feats["flags"]

        rv = compute_realized_vol_pit(close_hist, pd.Timestamp(d))
        atm_iv = iv_feats["atm_iv_30d"]
        iv_rv = (atm_iv / rv) if (rv is not None and np.isfinite(atm_iv)) else np.nan

        # opt_volume_z — strictly-prior trailing window of THIS underlying's
        # total daily option volume (vol_hist holds dates < d only at this point).
        total_vol = vol_feats["total_opt_volume"]
        if len(vol_hist) >= OPT_VOL_Z_WINDOW:
            window = np.asarray(vol_hist[-OPT_VOL_Z_WINDOW:], dtype=float)
            mu, sigma = float(window.mean()), float(window.std(ddof=1))
            z = (total_vol - mu) / sigma if sigma > 1e-9 else np.nan
        else:
            z = np.nan

        knowable = (knowable_by_date.get(pd.Timestamp(d))
                    if knowable_by_date is not None else None)
        if knowable is None or pd.isna(knowable):
            knowable = pd.Timestamp(d) + pd.offsets.BDay(1)  # defensive only

        rows.append({
            "underlying": underlying,
            "date": pd.Timestamp(d),
            "knowable_date": knowable,
            "atm_iv_30d": atm_iv,
            "implied_move_front": im_feats["implied_move_front"],
            "cpiv_matched_delta": iv_feats["cpiv_matched_delta"],
            "skew_25d_put": iv_feats["skew_25d_put"],
            "term_slope_30_60": iv_feats["term_slope_30_60"],
            "iv_rv_20d_ratio": iv_rv,
            "opt_share_volume_ratio": vol_feats["opt_share_volume_ratio"],
            "put_call_volume_ratio": vol_feats["put_call_volume_ratio"],
            "opt_volume_z": z,
            "total_opt_volume": total_vol,
            "n_valid_contracts": n_valid,
            "coverage_flags": int(flags),
        })
        vol_hist.append(total_vol)

    return pd.DataFrame(rows, columns=FEATURE_COLS)


def filter_dates(features: pd.DataFrame, start: Optional[date],
                 end: Optional[date]) -> pd.DataFrame:
    """Restrict a feature frame to date in [start, end] (inclusive); used by the
    builder to honor --start/--end after the per-date PIT history is built."""
    if features.empty:
        return features
    out = features
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)
