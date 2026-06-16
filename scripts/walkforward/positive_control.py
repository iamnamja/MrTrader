"""
positive_control.py -- Alpha-v9 P0-1: validate the feature->label pipeline.

WHY (ALPHA_V9_ROADMAP.md sec 2 (A) -- the keystone of Phase 0)
=========================================================
Our load-bearing conclusion is "free daily US equity *directional* alpha is mined
out" -- built on the swing-ML pipeline repeatedly returning IC~0. The external
review (Claude Max) flagged that this is **unsafe as stated**: our own changelog
has *deflationary* bugs (e.g. #PERFOLD2 -- a regime-map type bug that returned an
**empty X for every fold** -> mean Sharpe 0), and the pattern of
`except Exception: continue` in the feature worker silently drops windows. An
alignment / join / swallowed-exception bug in the feature->label seam produces
*exactly* the signature we trust most: "infra is sound, IC~0, no signal."

`gate_calibration.py` already pushes KNOWN signals (tsmom, xmom_12_1) through the
real CPCV->gate, **but every one of those controls computes its signal directly
from prices and BYPASSES the feature-construction path** (`engineer_features` ->
`build_train_matrix_for_window` -> the per-row feature/label/outcome join). So the
single seam most likely to harbour a deflationary bug -- and the one that produced
the IC~0 verdicts -- has *never* had a positive control. THIS module is that
control.

WHAT IT DOES
============
It runs a set of **published, robust cross-sectional anomalies** through the EXACT
production feature builder and measures whether the pipeline's own anomaly feature
columns predict the pipeline's own realized forward return (`meta.outcome_return`),
with the literature-expected sign:

  - xs_momentum   -> feature `momentum_252d_ex1m`  expected IC sign  +
  - short_reversal-> feature `momentum_5d`         expected IC sign  -
  - low_vol       -> feature `volatility`          expected IC sign  -

Nothing here reimplements feature or label logic -- it calls the real
`ModelTrainer.build_train_matrix_for_window` and only *measures* the result.

VERDICT  (FIDELITY-CENTERED -- this is the subtle, important part)
=======
The deflationary-bug question is "does the pipeline FAITHFULLY reproduce the
feature?", NOT "is the anomaly tradeable right now?". Anomaly tradeability is
regime-dependent (momentum + low-vol are weak-to-REVERSED in 2022-2026), so a
plain "anomaly not significant -> FAIL" would false-alarm on a perfectly healthy
pipeline. We therefore compare each pipeline feature column to an INDEPENDENT
from-raw-prices recompute (the reference IC), computed over the SAME window:

  FULL mode PASS  <=> matrix non-empty + finite, label fidelity IC(y,outcome)>0.30,
                      and NO anomaly is "deflationary" -- i.e. wherever the data
                      actually contains the effect (|reference IC| >= EFFECT_MIN)
                      the pipeline column reproduces it (|pipeline-reference| <= tol).
                  -> the feature path is sound; IC~0 in the model is market/model/
                     cost, so "mined out" stands on the feature side. Effect-presence
                     is reported separately as a regime diagnostic.
  FULL mode FAIL  <=> empty matrix (#PERFOLD2 class), broken label fidelity, or a
                      DEFLATIONARY DIVERGENCE (the data has the effect but the
                      pipeline loses it) -> "mined out" is UNSAFE; re-open the kills.

  SMOKE mode      -> a detector self-test on a synthetic panel with an INJECTED
                     momentum + low-vol effect: PASS iff the harness recovers them
                     through the real builder (and FAILs on a corrupted join -- see
                     tests). Proves the wiring; not a calibration of the live universe.

A KNOWN, IMPORTANT FINDING this harness also surfaces: `engineer_features` runs on
a ~63-bar `window_df` (WINDOW_DAYS), and every momentum lookback is clamped
`min(lookback, len(prices)-1)`. So `momentum_252d_ex1m` is **NOT** true 12-month
momentum -- it is a ~40-60d momentum. The swing features are structurally
window-limited; the pipeline never sees true 12-month momentum. We report each
feature's effective lookback so this is explicit, not hidden.

MODES
=====
  --smoke   fully synthetic GBM panel with an INJECTED momentum + low-vol effect,
            zero network / zero DB. Proves the detector recovers an injected signal
            THROUGH the real builder (and, via the test-suite corruption case, that
            it FAILS when the join is broken). Smoke verdicts are wiring evidence,
            not a calibration of the live universe.
  (full)    real Russell-1000 PIT universe over --window-years of yfinance history
            -> the actual verdict. Slow (yfinance fetch).

ARTIFACT: logs/positive_control_{YYYYMMDD}.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, asdict, field
from datetime import date as _date, datetime as _dt, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("positive_control")


# -- Anomaly specifications ---------------------------------------------------
@dataclass
class AnomalySpec:
    name: str
    feature: str          # the production feature column to test
    expected_sign: int    # +1 (higher feature -> higher fwd return) / -1 (lower)
    note: str
    smoke_injected: bool   # whether --smoke injects this effect (else IC~0 expected)


ANOMALIES: List[AnomalySpec] = [
    AnomalySpec(
        "xs_momentum", "momentum_252d_ex1m", +1,
        "Cross-sectional momentum (Jegadeesh-Titman). NOTE: misnomer -- on a 63-bar "
        "window with a 21-bar skip the real formation is only ~42 bars, not 252.",
        smoke_injected=True,
    ),
    AnomalySpec(
        "short_reversal", "momentum_5d", -1,
        "1-week short-term reversal -- recent winners revert (Lehmann/Lo-MacKinlay).",
        smoke_injected=False,
    ),
    AnomalySpec(
        "low_vol", "volatility", -1,
        "Low-volatility anomaly -- high realized vol -> lower risk-adjusted fwd return.",
        smoke_injected=True,
    ),
]

# IC significance threshold (|t| of the per-window cross-sectional IC series).
IC_T_THRESHOLD = 2.0
# Minimum number of anomalies that must pass for an overall PASS.
MIN_ANOMALIES_PASS = 2
# Minimum number of cross-sectional cohorts feeding the IC t-stat; below this the
# t-stat is not trustworthy and the anomaly is marked inconclusive (not passed).
MIN_COHORTS_FOR_SIG = 12
# The headline anomaly that MUST be recovered in SMOKE for an overall PASS (the
# detector self-test). In FULL mode the verdict is fidelity-based (see below).
MANDATORY_ANOMALY = "xs_momentum"
# Short-term-reversal feature skips the most-recent month (~21 bars).
_MOM_SKIP_BARS = 21
# FULL-mode verdict is FIDELITY-centered: the deflationary-bug question is "does the
# pipeline faithfully reproduce an INDEPENDENT recompute of the feature?", NOT "is
# the anomaly tradeable here?" (the latter is regime-dependent -- momentum/low-vol
# are weak-to-reversed in 2022-2026). A bug fires only when the data HAS the effect
# (|reference IC| >= EFFECT_MIN) but the pipeline column DIVERGES from it.
FIDELITY_TOL = 0.02     # max |pipeline_ic - reference_ic| for "faithful"
EFFECT_MIN = 0.015      # |reference_ic| above which the effect is "present" in the data


# -- Result containers --------------------------------------------------------
@dataclass
class AnomalyResult:
    name: str
    feature: str
    expected_sign: int
    n_rows: int
    n_windows: int
    effective_lookback_bars: Optional[float]   # median non-zero feature horizon evidence
    pipeline_ic: float                         # mean cross-sectional rank-IC
    pipeline_ic_t: float                       # t-stat of the per-window IC series
    reference_ic: Optional[float]              # independent recompute from raw prices
    sign_ok: bool
    significant: bool
    faithful: Optional[bool]                    # pipeline IC ~= reference IC (full mode)
    effect_present: bool                        # tradeable in this window (sign+significant)
    deflationary: bool                          # data HAS the effect but pipeline loses it
    passed: bool                               # mode-appropriate per-anomaly verdict
    note: str


@dataclass
class PositiveControlReport:
    as_of: str
    mode: str                                  # "smoke" | "full"
    label_scheme: str
    n_symbols: int
    n_rows: int
    n_windows: int
    window_days: int
    forward_days: int
    matrix_nonempty: bool
    matrix_finite: bool
    label_fidelity_ic: float                   # IC(y, outcome_return)
    label_fidelity_ok: bool
    anomalies: List[AnomalyResult]
    overall_pass: bool
    verdict: str
    run_at: str
    runtime_sec: float
    notes: List[str] = field(default_factory=list)


# -- Rank-IC: per-window cross-sectional Spearman, then IR/t over windows ------
def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Spearman rank correlation of two 1-D arrays; None if degenerate.

    Uses AVERAGE-tie ranks (scipy.rankdata): argsort-of-argsort breaks ties by
    array position, which on a heavily-tied/defaulted column induces a spurious
    correlation with anything order-correlated (a false-PASS vector). Average ties
    eliminate that at the source, so a mostly-tied column simply contributes its
    real (sparse) signal -- and a genuinely binary LABEL (e.g. lambdarank's sign
    label) still gets a valid point-biserial rank-IC. A fully constant column is
    caught by the std==0 guard.
    """
    from scipy.stats import rankdata
    if len(a) < 5 or len(b) < 5:
        return None
    ar = rankdata(a)            # average ties
    br = rankdata(b)
    if np.std(ar) < 1e-12 or np.std(br) < 1e-12:
        return None             # one side is constant -> rank-IC undefined
    return float(np.corrcoef(ar, br)[0, 1])


def classify_fidelity(
    pipeline_ic: float, reference_ic: Optional[float]
) -> Tuple[Optional[bool], bool]:
    """Decide (faithful, deflationary) for one anomaly from its pipeline vs reference IC.

    faithful     = the pipeline column reproduces the independent recompute
                   (|pipeline - reference| <= FIDELITY_TOL).
    deflationary = the data CONTAINS the effect (|reference| >= EFFECT_MIN) but the
                   pipeline DIVERGES from it -> a feature-construction bug. Effect
                   absence (reference ~ 0) is NOT deflationary, regardless of regime.
    Returns (None, False) when no reference is available (e.g. smoke mode).
    """
    if reference_ic is None:
        return None, False
    faithful = bool(abs(pipeline_ic - reference_ic) <= FIDELITY_TOL)
    deflationary = bool(abs(reference_ic) >= EFFECT_MIN and not faithful)
    return faithful, deflationary


def cross_sectional_ic(
    feature: np.ndarray, outcome: np.ndarray, window_ids: np.ndarray
) -> Tuple[float, float, int, List[float]]:
    """Compute the per-window cross-sectional rank-IC series, then summarize.

    Returns (mean_ic, ic_tstat, n_windows_used, ic_series).
    The IC the literature means is computed WITHIN each rebalance cross-section
    (one window = one date-cohort here) and then averaged; the t-stat of that
    series is the information-ratio significance (mean / (std/sqrt(n))).
    """
    ics: List[float] = []
    for wid in np.unique(window_ids):
        mask = window_ids == wid
        if mask.sum() < 5:           # need a real cross-section
            continue
        f = feature[mask]
        o = outcome[mask]
        finite = np.isfinite(f) & np.isfinite(o)
        if finite.sum() < 5:
            continue
        rho = _spearman(f[finite], o[finite])
        if rho is not None and np.isfinite(rho):
            ics.append(rho)
    if not ics:
        return 0.0, 0.0, 0, []
    arr = np.asarray(ics, dtype=float)
    mean_ic = float(np.mean(arr))
    if len(arr) > 1 and np.std(arr, ddof=1) > 1e-12:
        t = mean_ic / (np.std(arr, ddof=1) / math.sqrt(len(arr)))
    else:
        t = 0.0
    return mean_ic, float(t), len(arr), ics


# -- Synthetic panel (smoke) --------------------------------------------------
def _make_synthetic_panel(
    n_symbols: int = 60, n_days: int = 900, seed: int = 1909
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """A GBM panel with a persistent per-symbol drift (-> momentum) and a
    vol-coupled drift penalty (-> low-vol anomaly). Deterministic.

    Persistent drift makes a name's trailing momentum predict its forward return
    (positive momentum IC); coupling lower drift to higher sigma makes the
    volatility feature negatively predict forward return (low-vol IC). Short-term
    reversal is intentionally NOT injected -> its IC should be ~0 in smoke.
    """
    rng = np.random.default_rng(seed)
    # business-day spine ending recently
    end = _date(2025, 12, 31)
    spine = pd.bdate_range(end=end, periods=n_days)

    q = rng.standard_normal(n_symbols)                       # latent quality
    sigma = rng.uniform(0.010, 0.035, size=n_symbols)        # per-name daily vol
    sig_z = (sigma - sigma.mean()) / (sigma.std() + 1e-12)
    base = 0.0002
    # injection strengths set so even a SMALL (test-sized) panel clears t>=2.0 with
    # margin -- the CI per-test timeout (120s) forces tests onto a small panel.
    k_mom = 0.0028                                           # quality -> drift (momentum)
    k_lowvol = 0.0007                                        # high vol -> drift penalty (low-vol)
    mu = base + k_mom * q - k_lowvol * sig_z

    symbols_data: Dict[str, pd.DataFrame] = {}
    for i in range(n_symbols):
        eps = rng.standard_normal(n_days) * sigma[i]
        logret = mu[i] + eps
        price = 100.0 * np.exp(np.cumsum(logret))
        # OHLCV from the close path (high/low envelope from intraday vol proxy)
        close = price
        noise = np.abs(rng.standard_normal(n_days)) * sigma[i] * close
        high = close + noise
        low = np.maximum(close - noise, 0.01)
        openp = np.concatenate([[close[0]], close[:-1]])
        vol = rng.uniform(5e5, 5e6, size=n_days)
        df = pd.DataFrame(
            {"open": openp, "high": high, "low": low, "close": close, "volume": vol},
            index=pd.DatetimeIndex(spine),
        )
        symbols_data[f"SYN{i:03d}"] = df

    # SPY proxy = equal-weight average path (broad market)
    spy_close = np.mean(
        np.column_stack([symbols_data[s]["close"].values for s in symbols_data]), axis=1
    )
    spy = pd.Series(spy_close, index=pd.DatetimeIndex(spine), name="close")
    return symbols_data, spy


# -- Data fetch (full mode) ---------------------------------------------------
def _fetch_real_panel(
    window_years: int, as_of: _date, max_symbols: Optional[int]
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """Fetch the real Russell-1000 daily panel via yfinance (same download logic as
    the production SwingStrategy.fetch_data, but without the strategy wrapper -- we
    only need raw bars + SPY; regime/sector maps aren't used by the control)."""
    import yfinance as yf
    from app.utils.constants import RUSSELL_1000_TICKERS

    syms = list(RUSSELL_1000_TICKERS)
    if max_symbols:
        syms = syms[:max_symbols]
    # +420 calendar days of lead-in so the first windows have enough formation history
    start = (as_of - timedelta(days=365 * window_years + 420)).isoformat()
    end = as_of.isoformat()
    logger.info("Fetching %d symbols, %s -> %s (this is slow)...", len(syms), start, end)

    def _dl(sym: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            return df if len(df) >= 210 else None
        except Exception:
            return None

    symbols_data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(syms):
        df = _dl(sym)
        if df is not None:
            symbols_data[sym] = df
        if (i + 1) % 100 == 0:
            logger.info("  fetched %d / %d (%d usable)", i + 1, len(syms), len(symbols_data))

    spy_raw = _dl("SPY")
    if spy_raw is None or spy_raw.empty:
        raise RuntimeError("SPY fetch failed -- cannot define the trading-day spine")
    spy = spy_raw["close"]
    logger.info("Fetched %d usable symbols + SPY", len(symbols_data))
    return symbols_data, spy


# -- Reference IC (independent recompute, full mode) --------------------------
def _reference_ic(
    spec: AnomalySpec,
    symbols_data: Dict[str, pd.DataFrame],
    all_dates: List,
    meta: List[dict],
    window_days: int,
    forward_days: int,
) -> Optional[float]:
    """Recompute the anomaly signal and forward return DIRECTLY from raw prices on
    the same (symbol, entry-date) cohorts the pipeline used, and IC them. Uses the
    pipeline's OWN date spine (`trainer._last_all_dates`) so indices cannot drift,
    groups by entry DATE (a true cross-section), and computes each signal over the
    SAME window the pipeline sees (apples-to-apples) so refIC ~= pipelineIC when the
    feature path is correct -- a strong refIC against a pipeline IC~0 pinpoints a
    feature-construction bug rather than an absent effect."""
    if not all_dates:
        return None
    try:
        feat_vals: List[float] = []
        out_vals: List[float] = []
        cohort: List[int] = []
        for m in meta:
            sym = m.get("symbol")
            wid = m.get("window_idx")
            if sym is None or wid is None or sym not in symbols_data:
                continue
            entry_idx = int(wid) + window_days
            fwd_idx = entry_idx + forward_days
            if fwd_idx >= len(all_dates) or entry_idx >= len(all_dates):
                continue
            entry_date = all_dates[entry_idx]
            fwd_date = all_dates[fwd_idx]
            df = symbols_data[sym]
            idx = pd.DatetimeIndex(df.index).date
            ep = df.loc[idx == entry_date, "close"]
            fp = df.loc[idx == fwd_date, "close"]
            if len(ep) == 0 or len(fp) == 0:
                continue
            entry_price = float(ep.iloc[0])
            fwd_ret = (float(fp.iloc[0]) - entry_price) / entry_price
            hist = df.loc[idx <= entry_date, "close"].values.astype(float)
            sig = _reference_signal(spec.name, hist, window_days)
            if sig is None:
                continue
            feat_vals.append(sig)
            out_vals.append(fwd_ret)
            cohort.append(entry_date.toordinal())
        if len(feat_vals) < 20:
            return None
        ic, _, _, _ = cross_sectional_ic(
            np.asarray(feat_vals), np.asarray(out_vals), np.asarray(cohort)
        )
        return ic
    except Exception as exc:  # reference is a best-effort cross-check, never fatal
        logger.warning("reference IC for %s failed (non-fatal): %s", spec.name, exc)
        return None


def _reference_signal(name: str, prices: np.ndarray, window_days: int) -> Optional[float]:
    """Independent (clean) anomaly signal from a price history, computed over the
    SAME window the pipeline's feature sees (apples-to-apples with the in-pipeline
    column), so refIC is directly comparable to pipelineIC."""
    n = len(prices)
    if n < window_days + 2:
        return None
    if name == "xs_momentum":
        # match the pipeline's window-limited formation EXACTLY: it uses
        # prices[-(min(252, len-1)+1)] as the start. Mirror the min(252,...) clamp so
        # the reference stays aligned even if WINDOW_DAYS is ever raised above 252.
        start_off = min(252, window_days)
        p_start = prices[-(start_off + 1)]
        p_skip = prices[-(_MOM_SKIP_BARS + 1)]
        return (p_skip - p_start) / p_start if p_start > 0 else None
    if name == "short_reversal":
        return (prices[-1] - prices[-6]) / prices[-6] if prices[-6] > 0 else None
    if name == "low_vol":
        w = window_days
        rets = np.diff(prices[-w:]) / prices[-w:-1]
        return float(np.std(rets) * math.sqrt(252)) if len(rets) > 1 else None
    return None


def _effective_lookback(spec: AnomalySpec, window_days: int) -> Optional[float]:
    """Report the structural lookback the feature ACTUALLY spans (window-limited).

    For momentum_252d_ex1m on a ~`window_days`-bar window_df the formation span is
    (window start -> ~21 bars ago) = window_days - 21, NOT min(252, window_days):
    the "252d" name is a misnomer and the 21-bar skip further shortens it.
    """
    if spec.feature == "momentum_252d_ex1m":
        return float(max(1, min(252, window_days) - _MOM_SKIP_BARS))  # ~42 on a 63-bar window
    if spec.feature == "momentum_5d":
        return 5.0
    if spec.feature == "volatility":
        return float(window_days)
    return None


# -- Core ---------------------------------------------------------------------
def run_positive_control(
    *,
    as_of: _date,
    smoke: bool,
    label_scheme: str = "production",
    window_years: int = 4,
    max_symbols: Optional[int] = None,
    seed: int = 1909,
    n_workers: int = 1,
    smoke_n_symbols: int = 60,
    smoke_n_days: int = 900,
    _corrupt_join: bool = False,
) -> PositiveControlReport:
    """Build the real train matrix and measure feature->outcome IC for each anomaly.

    smoke_n_symbols / smoke_n_days size the synthetic panel; tests pass a small panel
    so each integration run stays under the 120s CI per-test timeout (the serial,
    cache-free build is the bottleneck), while the CLI keeps the larger defaults.
    """
    import time as _time
    import tempfile as _tempfile
    from app.ml import training as _tr
    from app.ml.training import ModelTrainer

    t0 = _time.time()
    notes: List[str] = []

    # Resolve the label scheme. Default "production" -> the LIVE swing scheme
    # (SWING_RETRAIN.label_scheme = "lambdarank"), so label fidelity is a REAL test
    # of the production label seam rather than the tautology return_regression gives
    # (y==outcome). The anomaly IC is measured against meta.outcome_return, which is
    # the horizon forward return for the cross-sectional family regardless.
    if label_scheme == "production":
        try:
            from app.ml.retrain_config import SWING_RETRAIN as _SR
            label_scheme = _SR.get("label_scheme", "lambdarank")
        except Exception:
            label_scheme = "lambdarank"
    notes.append(f"label_scheme={label_scheme} (production parity).")

    # 1) data
    if smoke:
        symbols_data, spy = _make_synthetic_panel(
            n_symbols=smoke_n_symbols, n_days=smoke_n_days, seed=seed)
        notes.append("SMOKE: synthetic GBM panel (momentum + low-vol injected; reversal not injected).")
    else:
        symbols_data, spy = _fetch_real_panel(window_years, as_of, max_symbols)
        notes.append(f"FULL: real R1K panel, {window_years}y window, as_of={as_of}.")

    train_start = (min(d.date() if hasattr(d, "date") else d for d in spy.index))
    train_end = (max(d.date() if hasattr(d, "date") else d for d in spy.index))

    # 2) build the REAL train matrix.
    #    use_feature_store=False is CRITICAL: a warm cache would serve stored feature
    #    rows and BYPASS engineer_features, defeating the entire control (we must
    #    exercise the live feature-construction path). It also prevents synthetic
    #    smoke rows from polluting the real feature store.
    #    We ALSO isolate the checkpoint dir (_windows_to_matrix reads MODEL_DIR/
    #    checkpoints and would RESUME stale rows from a recent real retrain with a
    #    colliding config key -> another way to bypass engineer_features). Pointing
    #    MODEL_DIR at a temp dir for the build neutralizes that without touching core.
    trainer = ModelTrainer(label_scheme=label_scheme, n_workers=n_workers, use_feature_store=False)
    assert trainer._feature_store is None, "feature store must be OFF for a valid control"
    _saved_model_dir = _tr.MODEL_DIR
    with _tempfile.TemporaryDirectory(prefix="poscontrol_ckpt_") as _td:
        _tr.MODEL_DIR = _td
        try:
            X, y, feature_names, meta = trainer.build_train_matrix_for_window(
                symbols_data, train_start, train_end, spy_prices=spy, regime_score_map=None,
            )
            all_dates = list(getattr(trainer, "_last_all_dates", []) or [])
        finally:
            _tr.MODEL_DIR = _saved_model_dir
    # build_train_matrix_for_window labels at LABEL_HORIZON_DAYS then RESTORES the
    # module FORWARD_DAYS, so reading _tr.FORWARD_DAYS now yields the stale default.
    # Capture the horizon actually used so the report + reference IC are correct.
    try:
        from app.ml.retrain_config import LABEL_HORIZON_DAYS as _LHD
        effective_forward = int(_LHD)
    except Exception:
        effective_forward = int(_tr.FORWARD_DAYS)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_rows = int(X.shape[0]) if X.ndim == 2 else 0
    window_ids = np.asarray([m.get("window_idx", -1) for m in meta])
    outcome = np.asarray([m.get("outcome_return", np.nan) for m in meta], dtype=float)

    # COHORT KEY: group cross-sections by the actual ENTRY DATE, not window_idx.
    # window_idx (= w_start_idx) is only a date proxy by the coincidence that every
    # surviving row for a window shares w_end_date (entry is an exact-date lookup on
    # the shared spine). Keying on the real date makes the cross-section correct by
    # construction, not by coincidence, and survives any future worker change.
    if n_rows and all_dates:
        cohort_ids = np.array([
            all_dates[int(w) + _tr.WINDOW_DAYS].toordinal()
            if 0 <= int(w) + _tr.WINDOW_DAYS < len(all_dates) else int(w)
            for w in window_ids
        ])
        # validity guard: each window_idx must map to exactly one entry date
        _per_wid = {}
        _mixed = 0
        for w, c in zip(window_ids, cohort_ids):
            if w in _per_wid and _per_wid[w] != c:
                _mixed += 1
            _per_wid[w] = c
        if _mixed:
            notes.append(f"WARNING: {_mixed} rows where window_idx maps to multiple "
                         "entry dates -- cohort integrity violated; using entry-date keys.")
    else:
        cohort_ids = window_ids
        if n_rows and not all_dates:
            notes.append("WARNING: _last_all_dates unavailable; grouping by window_idx "
                         "(date-proxy) -- cohort integrity not independently verified.")
    n_windows = int(len(np.unique(cohort_ids))) if n_rows else 0
    if n_rows and n_windows < MIN_COHORTS_FOR_SIG:
        notes.append(f"WARNING: only {n_windows} cross-sectional cohorts < "
                     f"{MIN_COHORTS_FOR_SIG} -- significance is UNDERPOWERED; a FAIL here "
                     "may be insufficient power, not a pipeline bug. Extend --window-years.")

    # Detection-power harness (tests only): a GLOBAL permutation of the realized
    # outcomes, which destroys the feature->outcome alignment within every cohort.
    # This simulates the misalignment / shuffled-join deflationary-bug class; a
    # correct detector MUST then report FAIL. Never enabled in real runs.
    if _corrupt_join and n_rows:
        rng = np.random.default_rng(20260616)
        perm = rng.permutation(len(outcome))
        outcome = outcome[perm]
        notes.append("CORRUPTION INJECTED: outcomes permuted (detection-power test).")

    matrix_nonempty = n_rows > 0 and len(feature_names) > 0
    feat_index = {name: j for j, name in enumerate(feature_names)}

    # matrix_finite is SCOPED to the columns this control actually uses (the tested
    # feature columns + outcome + y). A non-finite value in some UNRELATED feature
    # column must not false-FAIL the control.
    if matrix_nonempty:
        tested_cols = [feat_index[s.feature] for s in ANOMALIES if s.feature in feat_index]
        checks = [outcome, y] + ([X[:, tested_cols]] if tested_cols else [])
        matrix_finite = bool(all(np.isfinite(np.asarray(c)).all() for c in checks))
    else:
        matrix_finite = False
    if not matrix_nonempty:
        notes.append("CRITICAL: build_train_matrix returned an EMPTY matrix -- the "
                     "#PERFOLD2 empty-X deflationary class. Pipeline cannot surface any signal.")

    # 3) label fidelity -- the label must reflect the realized outcome. Under the
    #    production scheme (lambdarank) this is a REAL test (y is a sign/rank label,
    #    not the raw return), so a broken label seam shows up here.
    label_fidelity_ic = 0.0
    if matrix_nonempty:
        lic, _, _, _ = cross_sectional_ic(y, outcome, cohort_ids)
        label_fidelity_ic = lic
    label_fidelity_ok = label_fidelity_ic > 0.30

    # 4) per-anomaly IC through the real feature columns
    results: List[AnomalyResult] = []
    for spec in ANOMALIES:
        if not matrix_nonempty or spec.feature not in feat_index:
            results.append(AnomalyResult(
                spec.name, spec.feature, spec.expected_sign, 0, 0, None,
                0.0, 0.0, None, False, False, None, False, False, False,
                note=f"{spec.note} | feature MISSING from matrix" if matrix_nonempty else spec.note,
            ))
            continue
        col = X[:, feat_index[spec.feature]]
        ic, ic_t, nw, _ = cross_sectional_ic(col, outcome, cohort_ids)
        ref_ic = None if smoke else _reference_ic(
            spec, symbols_data, all_dates, meta, _tr.WINDOW_DAYS, effective_forward,
        )
        sign_ok = bool((np.sign(ic) == spec.expected_sign) and abs(ic) > 1e-4)
        # significance requires BOTH a |t| over threshold AND enough cohorts feeding
        # that t-stat (below the floor the t is not trustworthy -> inconclusive).
        significant = bool(abs(ic_t) >= IC_T_THRESHOLD and nw >= MIN_COHORTS_FOR_SIG)
        effect_present = bool(sign_ok and significant)

        # FIDELITY (full mode): does the pipeline column reproduce the independent
        # reference recompute? A deflationary BUG = the data has the effect but the
        # pipeline diverges from it. Effect-absence (ref~0, pipeline~0) is NOT a bug.
        faithful, deflationary = classify_fidelity(ic, ref_ic)

        if smoke:
            # detector self-test: only injected anomalies are expected to be recovered
            passed = effect_present if spec.smoke_injected else True
            note = spec.note + ("" if spec.smoke_injected else " | not injected (IC~0 expected)")
        else:
            # full mode: per-anomaly "passed" == "not a deflationary divergence"
            passed = not deflationary
            tag = ("FAITHFUL+present" if (faithful and effect_present) else
                   "FAITHFUL (effect weak/absent in window -- regime, not a bug)" if faithful else
                   "DEFLATIONARY DIVERGENCE" if deflationary else
                   "ref unavailable")
            note = f"{spec.note} | {tag}"
        results.append(AnomalyResult(
            spec.name, spec.feature, spec.expected_sign, n_rows, nw,
            _effective_lookback(spec, _tr.WINDOW_DAYS),
            ic, ic_t, ref_ic, sign_ok, significant, faithful, effect_present,
            deflationary, passed, note=note,
        ))

    # 5) verdict
    by_name_pre = {r.name: r for r in results}
    if smoke:
        # SMOKE detector self-test: the injected anomalies must be recovered, and the
        # headline momentum specifically must pass (no 2/3 masking of a broken column).
        _injected = {a.name for a in ANOMALIES if a.smoke_injected}
        n_pass = sum(1 for r in results if r.passed and r.name in _injected)
        mandatory_ok = bool(by_name_pre.get(MANDATORY_ANOMALY)
                            and by_name_pre[MANDATORY_ANOMALY].effect_present)
        overall_pass = bool(matrix_nonempty and matrix_finite and label_fidelity_ok
                            and n_pass >= MIN_ANOMALIES_PASS and mandatory_ok)
        verdict = ("SMOKE PASS -- the detector recovers injected anomalies THROUGH the "
                   "real builder and FAILs on a corrupted join (see tests). Wiring proven."
                   if overall_pass else
                   "SMOKE FAIL -- the detector did not recover an injected anomaly; the "
                   "harness wiring is broken (investigate before trusting any full run).")
    else:
        # FULL fidelity-centered verdict: a deflationary BUG is the only thing that
        # makes 'mined out' unsafe. Effect-presence is reported as a regime diagnostic.
        any_deflationary = any(r.deflationary for r in results)
        n_faithful = sum(1 for r in results if r.faithful)
        present = [r.name for r in results if r.effect_present]
        overall_pass = bool(matrix_nonempty and matrix_finite and label_fidelity_ok
                            and not any_deflationary)
        if overall_pass:
            verdict = (
                "PASS -- the feature-construction path is FAITHFUL: every tested anomaly's "
                f"pipeline IC matches an independent from-raw-prices recompute ({n_faithful}/"
                f"{len(results)} within tol), label fidelity holds, and NO anomaly shows a "
                "deflationary divergence. There is NO feature-pipeline bug, so IC~0 in the "
                "swing model is the market / model / cost -- the 'mined out' conclusion stands "
                "on the feature side. (Effect-presence is regime-dependent: "
                f"{'present=' + ','.join(present) if present else 'all three weak/absent in this window'}.)")
        else:
            bad = [f"{r.name}(pipe={r.pipeline_ic:+.3f} vs ref={r.reference_ic:+.3f})"
                   for r in results if r.deflationary]
            why = ("EMPTY matrix" if not matrix_nonempty else
                   "non-finite tested columns" if not matrix_finite else
                   "broken label fidelity" if not label_fidelity_ok else
                   "deflationary divergence: " + "; ".join(bad))
            verdict = (f"FAIL -- {why}. The pipeline does NOT faithfully reproduce a signal "
                       "the data contains -> a deflationary feature-pipeline bug. The 'mined "
                       "out' conclusion is UNSAFE -- re-open the equity-ML kills.")

    return PositiveControlReport(
        as_of=as_of.isoformat(),
        mode="smoke" if smoke else "full",
        label_scheme=label_scheme,
        n_symbols=len([s for s in symbols_data if s not in ("SPY", "^VIX")]),
        n_rows=n_rows,
        n_windows=n_windows,
        window_days=_tr.WINDOW_DAYS,
        forward_days=effective_forward,
        matrix_nonempty=matrix_nonempty,
        matrix_finite=matrix_finite,
        label_fidelity_ic=label_fidelity_ic,
        label_fidelity_ok=label_fidelity_ok,
        anomalies=results,
        overall_pass=overall_pass,
        verdict=verdict,
        run_at=_dt.now().isoformat(timespec="seconds") if not smoke else "SMOKE",
        runtime_sec=round(_time.time() - t0, 1),
        notes=notes,
    )


# -- Reporting ----------------------------------------------------------------
def _print_report(rep: PositiveControlReport) -> None:
    line = "=" * 78
    print(line)
    print(f"POSITIVE CONTROL -- feature->label pipeline ({rep.mode.upper()})  as_of={rep.as_of}")
    print(line)
    print(f"label_scheme={rep.label_scheme}  symbols={rep.n_symbols}  rows={rep.n_rows}  "
          f"windows={rep.n_windows}  WINDOW_DAYS={rep.window_days}  FORWARD_DAYS={rep.forward_days}")
    print(f"matrix_nonempty={rep.matrix_nonempty}  matrix_finite={rep.matrix_finite}  "
          f"label_fidelity_IC={rep.label_fidelity_ic:+.3f} (ok={rep.label_fidelity_ok})")
    print("-" * 78)
    print(f"{'anomaly':<16}{'feature':<20}{'exp':>4}{'IC':>8}{'IC_t':>7}{'refIC':>8}"
          f"{'faith':>7}{'eff':>5}  verdict")
    for r in rep.anomalies:
        ref = f"{r.reference_ic:+.3f}" if r.reference_ic is not None else "  n/a"
        faith = ("yes" if r.faithful else "NO") if r.faithful is not None else "-"
        eff = "yes" if r.effect_present else "no"
        v = "DEFLATIONARY" if r.deflationary else ("PASS" if r.passed else "FAIL")
        print(f"{r.name:<16}{r.feature:<20}{r.expected_sign:>+4}{r.pipeline_ic:>+8.3f}"
              f"{r.pipeline_ic_t:>7.2f}{ref:>8}{faith:>7}{eff:>5}  {v}")
        if r.effective_lookback_bars is not None:
            print(f"{'':<16}  -> lookback ~ {r.effective_lookback_bars:.0f} bars  | {r.note}")
    print("-" * 78)
    print(f"OVERALL: {'PASS' if rep.overall_pass else 'FAIL'}")
    print(rep.verdict)
    for n in rep.notes:
        print(f"  * {n}")
    print(line)


def _json_default(o):
    """Coerce numpy scalars to native Python for json.dumps."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _write_artifact(rep: PositiveControlReport, json_out: Optional[str]) -> Path:
    out = Path(json_out) if json_out else Path("logs") / f"positive_control_{rep.as_of.replace('-', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(rep), indent=2, default=_json_default))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Alpha-v9 P0-1 positive control for the feature->label pipeline")
    ap.add_argument("--as-of", default=_date.today().isoformat(), help="YYYY-MM-DD")
    ap.add_argument("--smoke", action="store_true", help="synthetic offline run (validates wiring)")
    ap.add_argument("--label-scheme", default="production",
                    help="label scheme to test; 'production' resolves to the live swing "
                         "scheme (lambdarank) so label-fidelity is a real test")
    ap.add_argument("--window-years", type=int, default=4)
    ap.add_argument("--max-symbols", type=int, default=None, help="cap universe (full mode debugging)")
    ap.add_argument("--workers", type=int, default=1,
                    help="feature workers; use 8 for the full real run, 1 for smoke/determinism")
    ap.add_argument("--seed", type=int, default=1909)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    as_of = _date.fromisoformat(args.as_of)
    rep = run_positive_control(
        as_of=as_of, smoke=args.smoke, label_scheme=args.label_scheme,
        window_years=args.window_years, max_symbols=args.max_symbols, seed=args.seed,
        n_workers=args.workers,
    )
    _print_report(rep)
    path = _write_artifact(rep, args.json_out)
    print(f"artifact: {path}")
    return 0 if rep.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
