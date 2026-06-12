"""
event_panel.py — the earnings-event research panel (Alpha-v6 Phase 3, PR3).

One row per (symbol, announce_date) earnings event, Russell-1000 PIT,
2019 -> 2026, EQUITY columns only in this PR (every option column from the
blueprint schema is reserved and emitted as all-NaN with
options_coverage_flag=False; Phase 2's quality reader populates them later).
This panel is the population H1-PEAD-EVENTLEVEL-20260611 (registry R4,
ONE confirmatory shot) is adjudicated on — correctness over cleverness.

════════════════════════════════════════════════════════════════════════════
FROZEN METHODOLOGICAL DECISIONS (fixed BEFORE any number was computed — they
preserve the pre-registration integrity of the H1 one-shot run)
════════════════════════════════════════════════════════════════════════════
1. "PEAD-qualified" = the COMMITTED +0.546 CPCV config
   (scripts/run_pead_cpcv.build_pead_scorer): long-only,
   `fmp_surprise_1q >= 0.05`, `1 <= fmp_days_since_earnings <= 3`,
   VIX-close-at-signal <= 30 (vix_block_all=30). Store `vix_at_signal` + an
   SPY<200d trend flag so the trend-gated (live B5) slice is a REPORTED,
   non-deciding robustness cut.
2. `entry_open_next` = next session OPEN strictly AFTER announce_date
   (announce+1 open) = PRIMARY. Also store an `entry_open_next2`
   (announce+2 open) alternate column, reported once (the gap = the day-1
   momentum the live book forfeits). Forward returns run from the primary
   entry open.
3. SPY-hedge = UNIT hedge (raw − SPY same-window) = PRIMARY for H1. Store
   `beta_60d` separately; a beta-adjusted hedged return may be reported
   alongside but H1 decides on the unit-hedged 10d.
4. announce_ts_flag = "UNK" everywhere in PR3 (no BMO/AMC source on-plan);
   the announce+1-open convention is the conservative guard (never uses the
   announce-day bar for entry).
════════════════════════════════════════════════════════════════════════════

PIT DISCIPLINE (the blueprint:254 contract, enforced by validate_panel_pit):
  * Every FEATURE column uses only data with timestamp <= announce_date
    (the announce-day session close is the last knowable print before the
    announce+1-open entry; with announce_ts_flag=UNK this matches the
    committed scorer's daily-bar convention).
  * Every OUTCOME column (fwd_ret_*) runs strictly FORWARD from
    entry_open_next, with the SPY/sector hedge leg measured on the IDENTICAL
    calendar window (entry-session open -> h-th-session close, same dates).
  * vix_at_signal = the last ^VIX close at-or-before announce_date — strictly
    before the entry open (never the entry day's own close).
  * Sacred holdout (retrain_config.SACRED_HOLDOUT_START = 2026-11-09):
    events whose announce date OR any realized forward session reaches the
    holdout are DROPPED, and the builder asserts none remain.

Column-by-column definitions (equity features):
  sue              = FMP EPS surprise_pct, clip((actual-est)/|est|, -1, 1)
                     — byte-identical to fmp_provider's clip formula.
  sue_z_pit        = (sue - mu)/sigma over the expanding pool of panel events
                     with announce_date STRICTLY EARLIER (mirrors
                     AgentSimulator._sue_zscore_pit; <2 priors -> 0.0).
  revision_momentum= reserved (all-NaN in PR3; H3's analyst-revision feature
                     is wired in the H3 prep PR — keeps this build to ONE FMP
                     call per symbol).
  announce_gap_pct = close(announce session)/close(prior session) - 1, the
                     committed scorer's priced-in input (last two bars
                     at-or-before announce_date). With UNK timestamps an AMC
                     reaction lands in the NEXT session — known limitation,
                     inherited deliberately for scorer fidelity.
  gap_vs_vol20     = announce_gap_pct / std(daily returns of the 20 sessions
                     STRICTLY BEFORE announce_date).
  prior_qtr_drift  = close[t-1]/close[t-64] - 1 on sessions strictly before
                     announce (63-trading-day pre-announcement drift).
  pead_score_v1    = sue_z_pit (the v1 conviction driver: AgentSimulator's
                     conviction sizing is clip(SUE_z,0,3)/vol). The H1
                     decile-monotonicity robustness cut sorts on this column.
  beta_60d         = OLS slope of symbol vs SPY daily returns over the 60
                     sessions strictly before announce (>=40 aligned obs).
  mktcap_decile    = 1..10 size decile of dollar_vol20 (mean close*volume of
                     the 20 pre-announce sessions) WITHIN the event's calendar
                     quarter. PROXY: no PIT shares-outstanding source is
                     on-plan, so this is a PIT liquidity-based size decile,
                     not literal market cap. dollar_vol20 is kept as a column
                     so the proxy is auditable.
  spy_below_200d   = SPY close < its 200d SMA at announce (PIT; None when
                     <200 SPY closes exist) — the live-B5 trend-gate flag.

quality_flags bitfield (an event with ANY of INCOMPLETE_FWD20 / NO_SPY_HEDGE
/ SUSPECT_BARS is EXCLUDED from inference by the runner):
  QF_INCOMPLETE_FWD20 1   fewer than 20 forward sessions -> fwd_ret_20 missing
  QF_NO_SPY_HEDGE     2   SPY bar missing on an entry/horizon date
  QF_NO_SECTOR_HEDGE  4   sector ETF leg unavailable (unknown sector / no ETF
                          bar — e.g. events before the ETF history starts)
  QF_NO_VIX           8   no ^VIX close at-or-before announce
  QF_NO_BETA         16   <40 aligned pre-announce obs for beta_60d
  QF_NO_VOL20        32   <15 pre-announce returns for vol20
  QF_NO_PRIOR_QTR    64   <64 pre-announce sessions for prior_qtr_drift
  QF_SUSPECT_BARS   128   the symbol's series kept a suspected split-artifact
                          cliff that a fresh refetch could NOT heal (e.g.
                          delisted/renamed R1K names yfinance cannot serve) —
                          conservative: ALL of that symbol's events carry the
                          bit (a fake -75% cliff inside one forward window
                          would poison the H1 mean)
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── quality_flags bits ────────────────────────────────────────────────────────
QF_INCOMPLETE_FWD20 = 1
QF_NO_SPY_HEDGE = 2
QF_NO_SECTOR_HEDGE = 4
QF_NO_VIX = 8
QF_NO_BETA = 16
QF_NO_VOL20 = 32
QF_NO_PRIOR_QTR = 64
QF_SUSPECT_BARS = 128

# Bits that EXCLUDE an event from inference (the runner enforces this).
QF_EXCLUDE_FROM_INFERENCE = QF_INCOMPLETE_FWD20 | QF_NO_SPY_HEDGE | QF_SUSPECT_BARS

# ── committed PEAD-qualification constants (decision 1 — do NOT re-tune) ─────
PEAD_LONG_THRESHOLD = 0.05      # fmp_surprise_1q >= 0.05 (long-only)
PEAD_MAX_DAYS_AFTER = 3         # 1 <= calendar days to first session <= 3
PEAD_VIX_BLOCK_ALL = 30.0       # VIX-close-at-signal <= 30

FWD_HORIZONS = (1, 3, 5, 10, 20)
SECTOR_HEDGE_HORIZONS = (5, 10, 20)

# Option columns reserved for Phase 2 (all-NaN in PR3).
OPTION_COLUMNS = (
    "pre_event_implied_move", "iv_runup_t10_t1", "reaction_ratio", "cpiv_pre",
    "skew_25d_pre", "term_kink_pre", "opt_volume_z_pre", "post_iv_retention_t1",
)

# yfinance-style sector names (data/sector_map.parquet values) -> SPDR ETF.
# Superset of app.ml.fundamental_fetcher.SECTOR_ETF_MAP: that map uses GICS
# names ("Health Care", "Consumer Discretionary"); the sector_map cache stores
# yfinance names ("Healthcare", "Consumer Cyclical") — both spellings resolve.
SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

_FMP_BASE = "https://financialmodelingprep.com/stable"
_EARNINGS_LIMIT = 120            # >= 60 (spec): 120 quarters comfortably covers 2019->
_EARNINGS_CACHE_TTL = 7 * 86_400  # on-disk cache TTL (7 days)

# PIT feature columns recomputed by validate_panel_pit from truncated inputs.
PIT_FEATURE_COLUMNS = (
    "sue", "announce_gap_pct", "gap_vs_vol20", "prior_qtr_drift", "beta_60d",
    "dollar_vol20", "vix_at_signal", "spy_below_200d",
)


# ═════════════════════════════════════════════════════════════════════════════
# Earnings fetch (NEW /stable/earnings call — the live fmp_provider is NOT
# mutated; only its clip formula is reused, byte-identically)
# ═════════════════════════════════════════════════════════════════════════════

def _fmp_api_key() -> str:
    from app.config import settings
    return settings.fmp_api_key or ""


def parse_earnings_rows(rows: List[dict]) -> List[dict]:
    """FMP /stable/earnings rows -> panel records, REUSING fmp_provider's clip
    formula verbatim: surprise = clip((actual-est)/|est|, -1, 1), |est|>0.001.
    Rows without a reported actual (future scheduled prints) are dropped.
    Output is sorted by date descending (most recent first), matching
    get_earnings_history_fmp's ordering contract."""
    records: List[dict] = []
    for row in rows:
        actual = row.get("epsActual")
        est = row.get("epsEstimated")
        if actual is None or not row.get("date"):
            continue
        surprise = None
        if est is not None:
            try:
                a, e = float(actual), float(est)
                if abs(e) > 0.001:
                    surprise = float(max(-1.0, min(1.0, (a - e) / abs(e))))
            except (TypeError, ValueError):
                pass
        records.append({
            "date": row.get("date", ""),
            "epsActual": actual,
            "epsEstimated": est,
            "surprise_pct": surprise,
        })
    records.sort(key=lambda r: r["date"], reverse=True)
    return records


def fetch_earnings_history(symbol: str, *, limit: int = _EARNINGS_LIMIT,
                           cache=None, ttl: int = _EARNINGS_CACHE_TTL) -> List[dict]:
    """Deep earnings history for the panel (limit=120 vs the live provider's
    20 — 7 years needs ~30 quarters). On-disk cached via DataCache JSON
    (data/cache/misc/earnings_panel_{symbol}.json) so the full build is
    re-runnable without burning the FMP quota."""
    import requests
    from app.data.cache import get_cache

    cache = cache if cache is not None else get_cache()
    key = f"misc/earnings_panel_{symbol}"
    cached = cache.get_json(key, ttl=ttl)
    if cached is not None and "records" in cached:
        return list(cached["records"])

    records: List[dict] = []
    try:
        resp = requests.get(
            f"{_FMP_BASE}/earnings",
            params={"symbol": symbol, "limit": limit, "apikey": _fmp_api_key()},
            timeout=15,
        )
        if resp.status_code == 200:
            records = parse_earnings_rows(resp.json())
        else:
            logger.warning("FMP earnings %s -> HTTP %s", symbol, resp.status_code)
    except Exception as exc:
        logger.warning("FMP earnings fetch failed for %s: %s", symbol, exc)
    if records:  # never cache an empty/failed fetch
        cache.put_json(key, {"records": records})
    return records


# ═════════════════════════════════════════════════════════════════════════════
# Pure per-event computations (shared by assemble_panel AND validate_panel_pit
# — recomputing from truncated inputs proves no feature reads past announce)
# ═════════════════════════════════════════════════════════════════════════════

def _last_close_at_or_before(series: pd.Series, ts: pd.Timestamp) -> Optional[float]:
    prior = series.loc[:ts]
    if len(prior) == 0:
        return None
    return float(prior.iloc[-1])


def compute_event_features(
    sym_bars: pd.DataFrame,
    spy_close: Optional[pd.Series],
    vix_close: Optional[pd.Series],
    announce_ts: pd.Timestamp,
) -> dict:
    """All PIT FEATURE values for one event. Inputs are sliced internally with
    .loc[:announce_ts] ONLY — feeding this function inputs truncated at
    announce_ts must reproduce the panel byte-for-byte (validate_panel_pit)."""
    out: dict = {
        "announce_gap_pct": np.nan, "gap_vs_vol20": np.nan, "vol20": np.nan,
        "prior_qtr_drift": np.nan, "beta_60d": np.nan, "dollar_vol20": np.nan,
        "vix_at_signal": np.nan, "spy_below_200d": None, "quality_bits": 0,
    }

    prior = sym_bars.loc[:announce_ts]                 # sessions <= announce
    pre = prior[prior.index < announce_ts]             # sessions STRICTLY before

    # announce_gap_pct — the committed scorer's definition (last two bars
    # at-or-before announce_date; see module docstring on the UNK caveat).
    if len(prior) >= 2:
        c0 = float(prior["close"].iloc[-2])
        c1 = float(prior["close"].iloc[-1])
        if c0 > 0:
            out["announce_gap_pct"] = c1 / c0 - 1.0

    # vol20 — std of the last 20 daily returns strictly before announce.
    # Guard the empty/short pre-window: pandas pct_change() raises on an empty
    # Series ("argmax of an empty sequence"), and the length check below is too
    # late — a name whose history starts at/after announce has no pre bars.
    pre_rets = (pre["close"].pct_change().dropna()
                if len(pre) >= 2 else pd.Series(dtype="float64"))
    if len(pre_rets) >= 15:
        vol = float(pre_rets.iloc[-20:].std(ddof=1))
        if np.isfinite(vol) and vol > 1e-9:
            out["vol20"] = vol
            if np.isfinite(out["announce_gap_pct"]):
                out["gap_vs_vol20"] = out["announce_gap_pct"] / vol
    else:
        out["quality_bits"] |= QF_NO_VOL20

    # prior_qtr_drift — 63-trading-day return ending the session before announce.
    if len(pre) >= 64:
        p_old = float(pre["close"].iloc[-64])
        p_new = float(pre["close"].iloc[-1])
        if p_old > 0:
            out["prior_qtr_drift"] = p_new / p_old - 1.0
    else:
        out["quality_bits"] |= QF_NO_PRIOR_QTR

    # dollar_vol20 — PIT size/liquidity proxy for mktcap_decile.
    if len(pre) >= 15 and "volume" in pre.columns:
        dv = (pre["close"] * pre["volume"]).iloc[-20:]
        if len(dv) > 0 and np.isfinite(dv.mean()):
            out["dollar_vol20"] = float(dv.mean())

    # beta_60d — OLS slope on SPY daily returns, 60 pre-announce sessions.
    if spy_close is not None:
        spy_pre = spy_close.loc[:announce_ts]
        spy_pre = spy_pre[spy_pre.index < announce_ts]
        # Same empty-Series pct_change guard as vol20 (both legs must be non-empty).
        if len(pre) >= 2 and len(spy_pre) >= 2:
            joined = pd.concat(
                [pre["close"].pct_change(), spy_pre.pct_change()], axis=1, join="inner"
            ).dropna()
        else:
            joined = pd.DataFrame()
        joined = joined.iloc[-60:]
        if len(joined) >= 40:
            x = joined.iloc[:, 1].to_numpy(dtype=float)
            yv = joined.iloc[:, 0].to_numpy(dtype=float)
            varx = float(np.var(x, ddof=1))
            if varx > 1e-12:
                out["beta_60d"] = float(np.cov(x, yv, ddof=1)[0, 1] / varx)
        if not np.isfinite(out["beta_60d"]):
            out["quality_bits"] |= QF_NO_BETA
    else:
        out["quality_bits"] |= QF_NO_BETA

    # vix_at_signal — last ^VIX close at-or-before announce (strictly before
    # the announce+1 entry open; decision 1).
    if vix_close is not None:
        v = _last_close_at_or_before(vix_close, announce_ts)
        if v is not None:
            out["vix_at_signal"] = v
    if not np.isfinite(out["vix_at_signal"]):
        out["quality_bits"] |= QF_NO_VIX

    # spy_below_200d — the live-B5 trend gate flag (PIT, closes <= announce).
    if spy_close is not None:
        spy_hist = spy_close.loc[:announce_ts]
        if len(spy_hist) >= 200:
            sma = float(spy_hist.iloc[-200:].mean())
            out["spy_below_200d"] = bool(float(spy_hist.iloc[-1]) < sma)
    return out


def compute_event_forward(
    sym_bars: pd.DataFrame,
    spy_bars: Optional[pd.DataFrame],
    etf_bars: Optional[pd.DataFrame],
    announce_ts: pd.Timestamp,
) -> dict:
    """OUTCOME legs for one event: entry opens + forward raw/hedged returns.

    Entry = OPEN of the first session STRICTLY AFTER announce_date
    (decision 2). fwd_ret_h_raw = close(s_h)/open(s_1) - 1 over the h forward
    sessions s_1..s_h. Hedge legs use the IDENTICAL calendar window:
    hedge_ret_h = hedge_close(date of s_h)/hedge_open(date of s_1) - 1, exact
    date match required (a missing hedge bar -> NaN + quality bit, never a
    nearest-neighbor fill).
    """
    out: dict = {
        "entry_open_next": np.nan, "entry_open_next2": np.nan,
        "cal_days_to_entry": np.nan, "quality_bits": 0, "last_fwd_date": None,
    }
    for h in FWD_HORIZONS:
        out[f"fwd_ret_{h}_raw"] = np.nan
        out[f"fwd_ret_{h}_spyhedged"] = np.nan
    for h in SECTOR_HEDGE_HORIZONS:
        out[f"fwd_ret_{h}_sectorhedged"] = np.nan

    fwd = sym_bars.loc[sym_bars.index > announce_ts]
    if len(fwd) == 0:
        out["quality_bits"] |= QF_INCOMPLETE_FWD20 | QF_NO_SPY_HEDGE
        return out

    s1 = fwd.index[0]
    entry_open = float(fwd["open"].iloc[0])
    out["entry_open_next"] = entry_open
    out["cal_days_to_entry"] = int((s1.date() - announce_ts.date()).days)
    if len(fwd) >= 2:
        out["entry_open_next2"] = float(fwd["open"].iloc[1])

    if entry_open <= 0:
        out["quality_bits"] |= QF_INCOMPLETE_FWD20 | QF_NO_SPY_HEDGE
        return out

    def _hedge_window_ret(bars: Optional[pd.DataFrame], d_end: pd.Timestamp) -> float:
        if bars is None:
            return np.nan
        try:
            h_open = float(bars.at[s1, "open"])
            h_close = float(bars.at[d_end, "close"])
        except (KeyError, ValueError):
            return np.nan
        if not np.isfinite(h_open) or h_open <= 0 or not np.isfinite(h_close):
            return np.nan
        return h_close / h_open - 1.0

    max_h_avail = 0
    for h in FWD_HORIZONS:
        if len(fwd) < h:
            continue
        d_end = fwd.index[h - 1]
        raw = float(fwd["close"].iloc[h - 1]) / entry_open - 1.0
        out[f"fwd_ret_{h}_raw"] = raw
        max_h_avail = h
        out["last_fwd_date"] = d_end.date()
        spy_leg = _hedge_window_ret(spy_bars, d_end)
        if np.isfinite(spy_leg):
            out[f"fwd_ret_{h}_spyhedged"] = raw - spy_leg
        else:
            out["quality_bits"] |= QF_NO_SPY_HEDGE
        if h in SECTOR_HEDGE_HORIZONS:
            etf_leg = _hedge_window_ret(etf_bars, d_end)
            if np.isfinite(etf_leg):
                out[f"fwd_ret_{h}_sectorhedged"] = raw - etf_leg
            else:
                out["quality_bits"] |= QF_NO_SECTOR_HEDGE
    if max_h_avail < max(FWD_HORIZONS):
        out["quality_bits"] |= QF_INCOMPLETE_FWD20
    return out


def qualify_event(sue: Optional[float], cal_days_to_entry, vix_at_signal,
                  entry_open_next) -> Tuple[bool, str]:
    """Decision 1 — the COMMITTED +0.546 long-only qualification, evaluated at
    the panel's announce+1-open entry:
      * long-only: sue >= 0.05 (fmp_surprise_1q at the first scoring day IS
        this event's surprise);
      * 1 <= calendar days from announce to the first session <= 3 (the
        scorer's fmp_days_since_earnings window; days_since=0 excluded by
        construction since the entry session is strictly after announce);
      * VIX-close-at-signal <= 30; a MISSING VIX never blocks (the scorer
        fails open when no VIX data is present).
    Returns (qualified, qual_reason); reasons are stable strings for audits.
    """
    if entry_open_next is None or not np.isfinite(entry_open_next):
        return False, "no_entry_bar"
    if sue is None or not np.isfinite(sue):
        return False, "no_surprise"
    if sue < PEAD_LONG_THRESHOLD:
        return False, "sue_below_long_threshold"
    if cal_days_to_entry is None or not np.isfinite(cal_days_to_entry) \
            or cal_days_to_entry > PEAD_MAX_DAYS_AFTER:
        return False, "first_session_gt_max_cal_days"
    if cal_days_to_entry < 1:
        return False, "entry_not_after_announce"
    if vix_at_signal is not None and np.isfinite(vix_at_signal) \
            and vix_at_signal > PEAD_VIX_BLOCK_ALL:
        return False, "vix_block"
    return True, "ok"


def _sue_z_pit_column(events: pd.DataFrame) -> pd.Series:
    """Expanding PIT standardization of sue (mirrors
    AgentSimulator._sue_zscore_pit): z against the pool of panel events with
    announce_date STRICTLY EARLIER; <2 priors (or zero dispersion) -> 0.0."""
    ordered = events.sort_values(["announce_date", "symbol"], kind="mergesort")
    sues = ordered["sue"].to_numpy(dtype=float)
    dates = ordered["announce_date"].to_numpy()
    z = np.zeros(len(ordered), dtype=float)
    pool: List[float] = []
    i = 0
    while i < len(ordered):
        # All events sharing one announce_date standardize against the SAME
        # strictly-prior pool (same-day peers are excluded — conservative).
        j = i
        while j < len(ordered) and dates[j] == dates[i]:
            j += 1
        if len(pool) >= 2:
            arr = np.asarray(pool, dtype=float)
            mu = float(arr.mean())
            sigma = float(arr.std(ddof=1))
            if np.isfinite(sigma) and sigma > 1e-9:
                z[i:j] = (sues[i:j] - mu) / sigma
        for v in sues[i:j]:
            if np.isfinite(v):
                pool.append(float(v))
        i = j
    return pd.Series(z, index=ordered.index).reindex(events.index)


def _mktcap_decile_column(events: pd.DataFrame) -> pd.Series:
    """1..10 size decile of dollar_vol20 WITHIN the event's calendar quarter
    (cross-sectional, PIT — only same-quarter peers, no future information
    beyond the quarter the event itself sits in... deciles for a quarter are
    only fully stable once the quarter completes; they are a REPORTED
    stratification feature, never a qualification input)."""
    from scripts.pead_significance import cluster_key
    quarters = events["announce_date"].map(lambda d: cluster_key(pd.Timestamp(d).date()))
    pct = events.groupby(quarters)["dollar_vol20"].rank(pct=True, method="average")
    decile = np.ceil(pct * 10.0).clip(1, 10)
    return decile.where(events["dollar_vol20"].notna())


# ═════════════════════════════════════════════════════════════════════════════
# Assembly (pure — all inputs in memory; the builder script does the loading)
# ═════════════════════════════════════════════════════════════════════════════

def assemble_panel(
    bars: Dict[str, pd.DataFrame],
    earnings: Dict[str, List[dict]],
    spy_bars: Optional[pd.DataFrame],
    vix_close: Optional[pd.Series],
    sector_etf_bars: Dict[str, pd.DataFrame],
    sectors: Dict[str, str],
    start: date,
    end: date,
    membership_fn: Optional[Callable[[str, date], bool]] = None,
    holdout_start: Optional[date] = None,
    suspect_symbols: Optional[set] = None,
) -> pd.DataFrame:
    """Build the event panel from in-memory inputs. One row per
    (symbol, announce_date) with announce_date in [start, end], symbol a PIT
    R1K member at announce (membership_fn), surprise computable. See the
    module docstring for every column definition and the FROZEN decisions.
    suspect_symbols = names whose suspect bar series could not be healed
    (loader-reported); EVERY event of such a symbol carries QF_SUSPECT_BARS."""
    if holdout_start is None:
        from app.ml.retrain_config import SACRED_HOLDOUT_START
        y, m, d = SACRED_HOLDOUT_START.split("-")
        holdout_start = date(int(y), int(m), int(d))
    assert end < holdout_start, (
        f"panel end {end} reaches the sacred holdout {holdout_start} — refuse to build"
    )

    spy_close = spy_bars["close"] if spy_bars is not None else None
    rows: List[dict] = []
    dropped_holdout = 0
    dropped_membership = 0

    for sym in sorted(bars.keys()):
        sym_bars = bars[sym]
        if sym_bars is None or sym_bars.empty:
            continue
        # Unhealed suspect series (loader-reported): conservative — flag ALL of
        # the symbol's events (suspect symbols are rare; one fake split-cliff
        # in a forward window would poison the H1 mean silently otherwise).
        sym_suspect_bit = (
            QF_SUSPECT_BARS if suspect_symbols and sym in suspect_symbols else 0
        )
        recs = earnings.get(sym) or []
        sector = sectors.get(sym, "UNKNOWN")
        etf = SECTOR_ETF_MAP.get(sector)
        etf_bars = sector_etf_bars.get(etf) if etf else None
        for rec in recs:
            try:
                ann = pd.Timestamp(rec["date"]).date()
            except (ValueError, TypeError):
                continue
            if ann < start or ann > end:
                continue
            if rec.get("surprise_pct") is None:
                continue  # no computable surprise -> not a panel event
            if membership_fn is not None and not membership_fn(sym, ann):
                dropped_membership += 1
                continue
            ann_ts = pd.Timestamp(ann)

            feats = compute_event_features(sym_bars, spy_close, vix_close, ann_ts)
            fwd = compute_event_forward(sym_bars, spy_bars, etf_bars, ann_ts)

            # Sacred-holdout guard: drop any event whose announce or realized
            # forward window touches the holdout (moot today — asserted below).
            last_fwd = fwd.get("last_fwd_date")
            if ann >= holdout_start or (last_fwd is not None and last_fwd >= holdout_start):
                dropped_holdout += 1
                continue

            sue = float(rec["surprise_pct"])
            qualified, reason = qualify_event(
                sue, fwd["cal_days_to_entry"], feats["vix_at_signal"],
                fwd["entry_open_next"],
            )
            row = {
                "event_id": f"{sym}|{ann.isoformat()}",
                "symbol": sym,
                "announce_date": ann,
                "announce_ts_flag": "UNK",     # decision 4
                "sector": sector,
                "sue": sue,
                "revision_momentum": np.nan,   # reserved (H3 prep PR)
                "announce_gap_pct": feats["announce_gap_pct"],
                "gap_vs_vol20": feats["gap_vs_vol20"],
                "prior_qtr_drift": feats["prior_qtr_drift"],
                "vix_at_signal": feats["vix_at_signal"],
                "spy_below_200d": feats["spy_below_200d"],
                "pead_qualified": qualified,
                "qual_reason": reason,
                "beta_60d": feats["beta_60d"],
                "dollar_vol20": feats["dollar_vol20"],
                "entry_open_next": fwd["entry_open_next"],
                "entry_open_next2": fwd["entry_open_next2"],
                "cal_days_to_entry": fwd["cal_days_to_entry"],
                "options_coverage_flag": False,
                "quality_flags": int(feats["quality_bits"] | fwd["quality_bits"]
                                     | sym_suspect_bit),
            }
            for h in FWD_HORIZONS:
                row[f"fwd_ret_{h}_raw"] = fwd[f"fwd_ret_{h}_raw"]
                row[f"fwd_ret_{h}_spyhedged"] = fwd[f"fwd_ret_{h}_spyhedged"]
            for h in SECTOR_HEDGE_HORIZONS:
                row[f"fwd_ret_{h}_sectorhedged"] = fwd[f"fwd_ret_{h}_sectorhedged"]
            for col in OPTION_COLUMNS:
                row[col] = np.nan
            row["_last_fwd_date"] = last_fwd
            rows.append(row)

    panel = pd.DataFrame(rows)
    if panel.empty:
        logger.warning("assemble_panel: no events produced")
        return panel

    # Cross-event columns (need the full event set).
    panel["sue_z_pit"] = _sue_z_pit_column(panel)
    panel["pead_score_v1"] = panel["sue_z_pit"]
    panel["mktcap_decile"] = _mktcap_decile_column(panel)

    # Hard holdout assertion (decision: the panel can NEVER touch the holdout).
    assert (panel["announce_date"] < holdout_start).all()
    fwd_dates = panel["_last_fwd_date"].dropna()
    assert (fwd_dates < holdout_start).all(), "forward window crosses the sacred holdout"
    panel = panel.drop(columns=["_last_fwd_date"])

    panel["spy_below_200d"] = panel["spy_below_200d"].astype("boolean")
    panel = panel.sort_values(["announce_date", "symbol"], kind="mergesort")
    panel = panel.reset_index(drop=True)

    ordered_cols = [
        "event_id", "symbol", "announce_date", "announce_ts_flag", "sector",
        "mktcap_decile", "sue", "sue_z_pit", "revision_momentum",
        "announce_gap_pct", "gap_vs_vol20", "prior_qtr_drift", "pead_score_v1",
        "vix_at_signal", "spy_below_200d", "pead_qualified", "qual_reason",
        *OPTION_COLUMNS,
        *[f"fwd_ret_{h}_raw" for h in FWD_HORIZONS],
        *[f"fwd_ret_{h}_spyhedged" for h in FWD_HORIZONS],
        *[f"fwd_ret_{h}_sectorhedged" for h in SECTOR_HEDGE_HORIZONS],
        "beta_60d", "dollar_vol20", "entry_open_next", "entry_open_next2",
        "cal_days_to_entry", "options_coverage_flag", "quality_flags",
    ]
    panel = panel[ordered_cols]
    logger.info(
        "assemble_panel: %d events (%d qualified, %d dropped-holdout, "
        "%d dropped-membership)",
        len(panel), int(panel["pead_qualified"].sum()), dropped_holdout,
        dropped_membership,
    )
    return panel


# ═════════════════════════════════════════════════════════════════════════════
# PIT validation (blueprint:254 — recompute features from TRUNCATED data)
# ═════════════════════════════════════════════════════════════════════════════

def validate_panel_pit(
    panel: pd.DataFrame,
    bars: Dict[str, pd.DataFrame],
    spy_bars: Optional[pd.DataFrame],
    vix_close: Optional[pd.Series],
    earnings: Dict[str, List[dict]],
    sample_n: int = 25,
    seed: int = 1303,
) -> dict:
    """For a random sample of events, TRUNCATE every input at announce_date
    and recompute each PIT feature; any inequality (excluding NaN==NaN) is a
    look-ahead leak. Returns {"checked", "ok", "mismatches": [...]}. The
    builder runs this on every build and refuses to ship a leaking panel."""
    rng = np.random.default_rng(seed)
    n = min(sample_n, len(panel))
    idx = rng.choice(panel.index.to_numpy(), size=n, replace=False)
    spy_close = spy_bars["close"] if spy_bars is not None else None
    mismatches: List[dict] = []

    for i in idx:
        row = panel.loc[i]
        sym = row["symbol"]
        ann_ts = pd.Timestamp(row["announce_date"])
        sym_bars = bars.get(sym)
        if sym_bars is None:
            mismatches.append({"event_id": row["event_id"], "field": "_bars_missing"})
            continue
        # TRUNCATED inputs: nothing after announce_date exists.
        t_bars = sym_bars.loc[:ann_ts]
        t_spy = spy_close.loc[:ann_ts] if spy_close is not None else None
        t_vix = vix_close.loc[:ann_ts] if vix_close is not None else None
        feats = compute_event_features(t_bars, t_spy, t_vix, ann_ts)
        feats["sue"] = _recompute_sue(earnings.get(sym) or [], ann_ts.date())

        for field in PIT_FEATURE_COLUMNS:
            want = row[field]
            got = feats.get(field)
            if field == "spy_below_200d":
                ok = (pd.isna(want) and got is None) or \
                     (not pd.isna(want) and got is not None and bool(want) == bool(got))
            else:
                ok = (pd.isna(want) and (got is None or not np.isfinite(got))) or (
                    not pd.isna(want) and got is not None
                    and np.isfinite(got) and math.isclose(float(want), float(got),
                                                          rel_tol=1e-9, abs_tol=1e-12)
                )
            if not ok:
                mismatches.append({
                    "event_id": row["event_id"], "field": field,
                    "panel": None if pd.isna(want) else float(want)
                    if field != "spy_below_200d" else bool(want),
                    "recomputed": got,
                })
    return {"checked": int(n), "ok": not mismatches, "mismatches": mismatches}


def _recompute_sue(records: List[dict], as_of: date) -> Optional[float]:
    """Most recent surprise with report date <= as_of (mirrors
    get_earnings_features_at's PIT filter). Used by the PIT validation."""
    past = [
        r for r in records
        if r.get("date") and r.get("surprise_pct") is not None
        and pd.Timestamp(r["date"]).date() <= as_of
    ]
    if not past:
        return None
    past.sort(key=lambda r: r["date"], reverse=True)
    return float(past[0]["surprise_pct"])


# ═════════════════════════════════════════════════════════════════════════════
# Input loading (DataCache read-through + yfinance fill; sector ETFs from the
# local parquet history; PIT membership via universe_history.members_at)
# ═════════════════════════════════════════════════════════════════════════════

# Pre-announce features need up to 200 SPY sessions (trend flag) and 64 symbol
# sessions; forward legs need 20 sessions + hedges. Calendar-day paddings:
_FEATURE_LOOKBACK_DAYS = 400
_FORWARD_PAD_DAYS = 45


def pit_membership_checker(index: str = "russell1000") -> Callable[[str, date], bool]:
    """(symbol, announce_date) -> was the symbol an index member that day.
    members_at() handles the membership parquet (incl. delisted names) and
    falls back to the static list when the parquet is absent — the fallback
    is survivorship-biased and members_at logs it; the builder reports which
    path is active. Per-date frozensets are memoized (one panel build touches
    ~1.5k distinct announce dates)."""
    from app.data.universe_history import members_at

    memo: Dict[date, frozenset] = {}

    def _check(symbol: str, as_of: date) -> bool:
        s = memo.get(as_of)
        if s is None:
            s = frozenset(members_at(index, as_of))
            memo[as_of] = s
        return symbol in s

    return _check


def load_sector_etf_bars(path=None) -> Dict[str, pd.DataFrame]:
    """data/sector_etf/sector_etf_history.parquet (long: etf/date/OHLCV) ->
    {etf: OHLCV DataFrame indexed by date}. History starts 2019-06-17; earlier
    events simply carry QF_NO_SECTOR_HEDGE."""
    from pathlib import Path
    p = Path(path) if path else Path("data/sector_etf/sector_etf_history.parquet")
    if not p.exists():
        logger.warning("sector ETF history missing at %s — sector hedge legs NaN", p)
        return {}
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    out: Dict[str, pd.DataFrame] = {}
    for etf, grp in df.groupby("etf"):
        g = grp.set_index("date").sort_index()
        out[str(etf)] = g[["open", "high", "low", "close", "volume"]]
    return out


def _fetch_daily_yf(symbol: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """One yfinance daily fetch, normalized to the cache's lowercase-OHLCV
    schema. Flattens the MultiIndex columns newer yfinance returns (the
    event_edge.fetch_data pattern — YFinanceProvider._normalise predates the
    MultiIndex change and fails on it, so we do NOT route through it)."""
    import yfinance as yf
    try:
        df = yf.download(symbol, start=start.isoformat(),
                         end=(end + timedelta(days=1)).isoformat(),  # yf end-exclusive
                         interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        if df.empty or "close" not in df.columns:
            return None
        return df[["open", "high", "low", "close", "volume"]].astype(float) \
            .dropna(subset=["close"])
    except Exception as exc:
        logger.debug("yfinance daily fetch failed for %s: %s", symbol, exc)
        return None


# Overnight |close/close - 1| above this is treated as a stale-adjustment
# artifact (a split rebased the fresh rows but not the older cached rows —
# observed in the real cache: NVDA 2021 4:1 and 2024 10:1 splits sit as raw
# 751->186 / 1209->121 cliffs). Splits are >=2:1 (=> moves >=50%/100%); the
# rare GENUINE >45% R1K day costs one redundant refetch and is then kept.
_SPLIT_ARTIFACT_MOVE = 0.45


def _split_artifact_dates(df: pd.DataFrame, thresh: float = _SPLIT_ARTIFACT_MOVE) -> list:
    """Dates whose overnight close-to-close move exceeds `thresh` (suspected
    un-readjusted split boundaries in the incremental cache)."""
    rets = df["close"].pct_change().abs()
    return [ts.date() for ts in rets[rets > thresh].index]


def _off_calendar_dates(df: pd.DataFrame, calendar) -> list:
    """Cached bar dates that fall INSIDE the calendar's span but are not US
    sessions (observed in the real cache: bogus foreign-session bars on US
    holidays — AAPL 2023-02-20 Presidents Day priced ~0.66x). `calendar` is
    the SPY DatetimeIndex (SPY's cache verified clean of both defects)."""
    if calendar is None or len(calendar) == 0:
        return []
    cal = pd.DatetimeIndex(calendar)
    in_span = df.index[(df.index >= cal.min()) & (df.index <= cal.max())]
    return [ts.date() for ts in in_span.difference(cal)]


def _replace_cached_daily(cache, symbol: str, df: pd.DataFrame) -> None:
    """REPLACE the symbol's cached daily series. put_daily merge-keeps orphan
    rows (a bogus holiday bar absent from fresh data would survive a merge),
    so healing must overwrite the file. Uses DataCache's own atomic writer on
    its own path (cache-internal layout, kept in one place here) and clears
    the in-memory entry."""
    path = cache._dir / "daily" / f"{symbol}.parquet"
    cache._write_df(df.sort_index(), path)
    cache.invalidate(symbol)


def _get_daily_bars(cache, symbol: str, start: date, end: date,
                    calendar=None,
                    suspect_out: Optional[set] = None) -> Optional[pd.DataFrame]:
    """DataCache read-through + yfinance fill (the YFinanceProvider read-path
    contract, with the MultiIndex-safe fetch above), hardened three ways —
    all three defects were OBSERVED in the real cache during the smoke build:

    1. RETRY + LOUD failure: a missing head/tail that cannot be fetched is a
       WARNING, never silence (a silently stale SPY tail would NaN the hedge
       legs of every recent event).
    2. SPLIT-ARTIFACT GUARD: the incremental cache stores auto-adjusted rows
       appended at different times, so a split leaves an un-readjusted cliff
       between old and new rows (NVDA 2021 4:1 sat as 751->186; beta_60d=-68
       caught it). Any overnight |move| > 45% triggers ONE fresh full-window
       refetch; a move persisting in fresh data is genuine and kept (logged).
    3. CALENDAR GUARD: bars on non-US-session dates (foreign-exchange holiday
       rows) trigger the same heal; the heal REPLACES the cached file (merge
       cannot delete orphan rows) and any survivors are dropped in-memory.

    suspect_out: when the refetch for a split-artifact cliff FAILS or comes
    back too thin (guaranteed for delisted/renamed names yfinance cannot
    serve), the suspect series is kept BUT the symbol is added to this set so
    the panel can stamp QF_SUSPECT_BARS on its events — never a silent keep.
    """
    fetch_start, fetch_end = cache.missing_daily_range(symbol, start, end)
    # The requested window is padded _FORWARD_PAD_DAYS past `end` for forward
    # legs — sessions that may not EXIST yet. Clamp the fetch at today so a
    # purely-future "missing" range is not a failure (Yahoo errors on it).
    today = date.today()
    if fetch_end is not None and fetch_end > today:
        fetch_end = today
    if fetch_start is not None and fetch_start <= fetch_end:
        fetched = None
        for attempt in (1, 2):
            fetched = _fetch_daily_yf(symbol, fetch_start, fetch_end)
            if fetched is not None and not fetched.empty:
                break
            time.sleep(0.5 * attempt)
        if fetched is not None and not fetched.empty:
            cache.put_daily(symbol, fetched)
        else:
            logger.warning(
                "%s: missing bar range %s -> %s could not be fetched after "
                "retry — bars may be STALE (recent forward legs would be NaN)",
                symbol, fetch_start, fetch_end)

    df = cache.get_daily(symbol, start, end)
    if df is None or df.empty:
        return df

    # Vol/index tickers (^VIX) legitimately gap >45% overnight and never
    # split — exempt them so every build doesn't burn a pointless refetch.
    if symbol.startswith("^"):
        return df

    suspects = _split_artifact_dates(df)
    off_cal = _off_calendar_dates(df, calendar)
    if suspects or off_cal:
        logger.warning(
            "%s: cache quality defects — %d suspect >%.0f%% move(s) %s, "
            "%d off-calendar bar(s) %s; refetching the window fresh",
            symbol, len(suspects), _SPLIT_ARTIFACT_MOVE * 100, suspects[:3],
            len(off_cal), off_cal[:3])
        fresh = _fetch_daily_yf(symbol, start, end)
        if fresh is not None and len(fresh) >= 0.9 * len(df):
            _replace_cached_daily(cache, symbol, fresh)
            df = cache.get_daily(symbol, start, end)
            remaining = _split_artifact_dates(df) if df is not None else []
            if remaining:
                logger.warning("%s: extreme move(s) persist in FRESH data at %s "
                               "— genuine, kept", symbol, remaining[:3])
            else:
                logger.info("%s: cache healed (consistent adjusted series)", symbol)
        else:
            logger.warning("%s: fresh refetch unavailable — keeping suspect "
                           "cached series (quality risk)", symbol)
            if suspects and suspect_out is not None:
                suspect_out.add(symbol)
                logger.warning("%s: persistent UNHEALED split-artifact bars — "
                               "marked suspect (events get QF_SUSPECT_BARS and "
                               "are excluded from inference)", symbol)

    # Belt-and-braces: whatever survived the heal, never hand back a bar on a
    # non-session date (it would corrupt forward-session indexing).
    if df is not None and calendar is not None:
        still_off = _off_calendar_dates(df, calendar)
        if still_off:
            logger.warning("%s: dropping %d off-calendar bar(s) in-memory: %s",
                           symbol, len(still_off), still_off[:5])
            off_set = {pd.Timestamp(d) for d in still_off}
            df = df[~df.index.isin(off_set)]
    return df


def load_panel_inputs(symbols: List[str], start: date, end: date) -> dict:
    """Fetch/assemble every input assemble_panel needs (DataCache read-through
    + yfinance fill, auto_adjust=True matching the cache's existing contents)."""
    from app.data.cache import get_cache
    from app.data.sector_map import get_sector_map

    cache = get_cache()
    bar_start = start - timedelta(days=_FEATURE_LOOKBACK_DAYS)
    bar_end = end + timedelta(days=_FORWARD_PAD_DAYS)

    t0 = time.time()
    # SPY first: it is both the hedge leg and the US-session CALENDAR the
    # per-symbol guards check against (SPY's cache verified clean).
    spy_bars = _get_daily_bars(cache, "SPY", bar_start, bar_end)
    calendar = spy_bars.index if spy_bars is not None else None
    bars: Dict[str, pd.DataFrame] = {}
    suspect_symbols: set = set()
    for sym in symbols:
        try:
            df = _get_daily_bars(cache, sym, bar_start, bar_end,
                                 calendar=calendar, suspect_out=suspect_symbols)
        except Exception as exc:
            logger.warning("bars failed for %s: %s", sym, exc)
            df = None
        if df is not None and len(df) >= 30:
            bars[sym] = df
    if suspect_symbols:
        logger.warning("%d symbol(s) kept UNHEALED suspect bar series — their "
                       "events will carry QF_SUSPECT_BARS: %s",
                       len(suspect_symbols), sorted(suspect_symbols)[:10])
    vix_bars = _get_daily_bars(cache, "^VIX", bar_start, bar_end)
    vix_close = vix_bars["close"] if vix_bars is not None else None
    if vix_close is None:
        logger.warning("^VIX unavailable — vix_at_signal will be NaN and the "
                       "VIX>30 qualification block FAILS OPEN (scorer parity)")
    logger.info("bars loaded: %d/%d symbols (+SPY%s) in %.1fs",
                len(bars), len(symbols), "/VIX" if vix_close is not None else "",
                time.time() - t0)

    t0 = time.time()
    earnings: Dict[str, List[dict]] = {}
    for sym in symbols:
        if sym in bars:
            earnings[sym] = fetch_earnings_history(sym)
    logger.info("earnings histories loaded for %d symbols in %.1fs",
                len(earnings), time.time() - t0)

    sectors = get_sector_map([s for s in symbols if s in bars])
    sector_etf_bars = load_sector_etf_bars()
    return {
        "bars": bars, "earnings": earnings, "spy_bars": spy_bars,
        "vix_close": vix_close, "sector_etf_bars": sector_etf_bars,
        "sectors": sectors, "suspect_symbols": suspect_symbols,
    }


def build_event_panel(symbols: List[str], start: date, end: date,
                      membership: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Load inputs and assemble the panel. Returns (panel, inputs) — inputs are
    handed back so the builder script can run validate_panel_pit on the SAME
    in-memory data the panel was computed from."""
    inputs = load_panel_inputs(symbols, start, end)
    membership_fn = pit_membership_checker() if membership else None
    panel = assemble_panel(
        bars=inputs["bars"], earnings=inputs["earnings"],
        spy_bars=inputs["spy_bars"], vix_close=inputs["vix_close"],
        sector_etf_bars=inputs["sector_etf_bars"], sectors=inputs["sectors"],
        start=start, end=end, membership_fn=membership_fn,
        suspect_symbols=inputs["suspect_symbols"],
    )
    return panel, inputs
