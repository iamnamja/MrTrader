"""
walkforward_tier3.py — Rolling walk-forward validation using Tier 3 agent simulation.

Design (per PHASES_18_23_SPEC.md):
  Fold 1: train on Y1-Y2, test Tier 3 on Y3
  Fold 2: train on Y1-Y3, test Tier 3 on Y4
  Fold 3: train on Y1-Y4, test Tier 3 on Y5 (most recent)

Gate: avg OOS Tier 3 Sharpe > 0.8, no fold below -0.3

WF-1 additions (2026-05-07):
  - embargo_days parameter: clean gap on BOTH sides of every test window
      train | purge_days | TEST | embargo_days | next_train
  - New fold metrics: profit_factor, calmar_ratio, k_ratio
  - Extended gate: avg profit_factor > 1.10, avg calmar > 0.30

Usage:
    python scripts/walkforward_tier3.py --model swing --folds 3 --years 5
    python scripts/walkforward_tier3.py --model intraday --folds 3 --days 730
    python scripts/walkforward_tier3.py --model both --folds 3

Exit codes:
    0 — gate passed (avg Sharpe > 0.8, no fold < -0.3)
    1 — gate not passed or error
"""
from __future__ import annotations

import argparse
import atexit
import logging
import multiprocessing
import os
import signal
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.ml.retrain_config import MAX_WORKERS, MAX_FOLD_WORKERS, N_TRIALS_TESTED  # noqa: E402
from scripts.walkforward.gates import (  # noqa: E402
    SHARPE_GATE, MIN_FOLD_SHARPE, MIN_PROFIT_FACTOR, MIN_CALMAR,
    deflated_sharpe_ratio as _deflated_sharpe_ratio,
    compute_profit_factor as _compute_profit_factor,
    compute_calmar as _compute_calmar,
    compute_k_ratio as _compute_k_ratio,
    fold_years as _fold_years,
    FoldResult,
    WalkForwardReport,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _make_regime_gate_fn(
    symbols_data: dict,
    spy_ma_days: int = 200,
    vix_bull: float = 20.0,
    vix_bear: float = 30.0,
    bull_mult: float = 1.0,
    neutral_mult: float = 0.7,
    bear_mult: float = 0.3,
):
    """Return a callable(day: date) -> float that gives gross exposure multiplier.

    Regime rules (PIT-safe — uses data strictly before *day*):
      BULL:    SPY > SPY_MA(spy_ma_days) AND VIX < vix_bull  -> bull_mult
      BEAR:    SPY < SPY_MA(spy_ma_days) OR  VIX >= vix_bear -> bear_mult
      NEUTRAL: otherwise                                      -> neutral_mult
    """
    import pandas as _pd

    spy_df = symbols_data.get("SPY") if "SPY" in symbols_data else symbols_data.get("spy")
    vix_df = symbols_data.get("^VIX") if "^VIX" in symbols_data else symbols_data.get("VIX")

    spy_close: "_pd.Series | None" = spy_df["close"] if spy_df is not None else None
    vix_close: "_pd.Series | None" = vix_df["close"] if vix_df is not None else None

    spy_ma: "_pd.Series | None" = (
        spy_close.rolling(spy_ma_days, min_periods=max(spy_ma_days // 2, 1)).mean()
        if spy_close is not None else None
    )

    def _regime_fn(day: date) -> float:
        # PIT: use data strictly before the rebalance day
        _day_ts = _pd.Timestamp(day)

        spy_val = vix_val = spy_ma_val = None
        if spy_close is not None:
            _spy_hist = spy_close[spy_close.index < _day_ts]
            if len(_spy_hist) > 0:
                spy_val = float(_spy_hist.iloc[-1])
        if spy_ma is not None:
            _ma_hist = spy_ma[spy_ma.index < _day_ts]
            if len(_ma_hist) > 0:
                spy_ma_val = float(_ma_hist.iloc[-1])
        if vix_close is not None:
            _vix_hist = vix_close[vix_close.index < _day_ts]
            if len(_vix_hist) > 0:
                vix_val = float(_vix_hist.iloc[-1])

        # Determine regime
        above_ma = (spy_val is not None and spy_ma_val is not None
                    and spy_val > spy_ma_val)
        vix_low = vix_val is not None and vix_val < vix_bull
        vix_high = vix_val is not None and vix_val >= vix_bear
        spy_below = not above_ma if (spy_val is not None and spy_ma_val is not None) else False

        if above_ma and vix_low:
            return bull_mult      # BULL
        if spy_below or vix_high:
            return bear_mult      # BEAR
        return neutral_mult       # NEUTRAL

    return _regime_fn


def build_asymmetric_regime_fns(
    spy_df,
    vix_series,
    spy_ma_days: int = 200,
    vix_bull_threshold: float = 15.0,
    vix_bear_threshold: float = 25.0,
):
    """Build separate (long_fn, short_fn) with asymmetric regime logic.

    Long book multipliers  (same as symmetric gate):
      BULL (SPY>MA200 AND VIX<15) → 1.0
      BEAR (SPY<MA200 OR VIX>25)  → 0.30
      NEUTRAL                     → 0.70

    Short book multipliers  (inverse — shorts earn in bear, costly in bull):
      BULL (SPY>MA200 AND VIX<15) → 0.30  (shorts fight equity premium)
      BEAR (SPY<MA200 OR VIX>25)  → 1.0   (shorts earn, increase exposure)
      NEUTRAL                     → 0.70

    Both functions are PIT-safe: use data strictly before the query day.
    Pre-registered thresholds (not tuned on WF folds) to prevent overfitting.
    """
    import pandas as _pd

    spy_close = spy_df["close"] if spy_df is not None and "close" in spy_df.columns else None
    spy_ma = (
        spy_close.rolling(spy_ma_days, min_periods=max(spy_ma_days // 2, 1)).mean()
        if spy_close is not None else None
    )

    # Accept both Series and DataFrame for vix
    if isinstance(vix_series, _pd.DataFrame):
        vix_s = vix_series["close"] if "close" in vix_series.columns else vix_series.iloc[:, 0]
    else:
        vix_s = vix_series

    def _get_regime(day):
        _day_ts = _pd.Timestamp(day)
        spy_val = spy_ma_val = vix_val = None
        if spy_close is not None:
            h = spy_close[spy_close.index < _day_ts]
            if len(h):
                spy_val = float(h.iloc[-1])
        if spy_ma is not None:
            h = spy_ma[spy_ma.index < _day_ts]
            if len(h):
                spy_ma_val = float(h.iloc[-1])
        if vix_s is not None:
            h = vix_s[vix_s.index < _day_ts]
            if len(h):
                vix_val = float(h.iloc[-1])
        above_ma = (spy_val is not None and spy_ma_val is not None and spy_val > spy_ma_val)
        vix_ok = vix_val is not None and vix_val < vix_bull_threshold
        vix_hi = vix_val is not None and vix_val >= vix_bear_threshold
        spy_below = not above_ma if (spy_val is not None and spy_ma_val is not None) else False
        if above_ma and vix_ok:
            return "BULL"
        if spy_below or vix_hi:
            return "BEAR"
        return "NEUTRAL"

    def long_fn(day):
        r = _get_regime(day)
        return {"BULL": 1.0, "NEUTRAL": 0.70, "BEAR": 0.30}[r]

    def short_fn(day):
        r = _get_regime(day)
        return {"BULL": 0.30, "NEUTRAL": 0.70, "BEAR": 1.0}[r]

    return long_fn, short_fn


# BUG-12: _make_combined_regime_fn deleted — it was dead code.
# The only caller was guarded by a rebalance_factor_stability_gate flag that
# now raises unconditionally (line ~1285). The function also leaked test-window
# scores into the FactorStabilityGate by scoring all_dates (train+test together).
# Removed to prevent accidental future reactivation of this look-ahead path.


# ── Console helpers ───────────────────────────────────────────────────────────

def _ok(msg):
    print(f"  \033[32mOK\033[0m  {msg}")


def _warn(msg):
    print(f"  \033[33mWARN\033[0m  {msg}")


def _err(msg):
    print(f"  \033[31mFAIL\033[0m  {msg}")


def _header(msg):
    print(f"\n{'='*62}\n  {msg}\n{'='*62}")


def _subheader(msg):
    print(f"\n{'-'*62}\n  {msg}\n{'-'*62}")


# ── Model loading ─────────────────────────────────────────────────────────────

def _check_trained_through(obj, model_name, ver):
    """Log a clear error if a loaded ML model lacks trained_through (stale pre-#311 artifact).

    Does NOT raise — the object is still returned so the existing OOS guard
    produces the authoritative failure. This surfaces a clearer, earlier message
    pointing at the artifact (see docs/living/PIPELINE_ARCHITECTURE.md KL-10).
    """
    if getattr(obj, "trained_through", None) is None:
        logger.error(
            "%s v%d has no trained_through — stale pre-#311 artifact or DB-injected metadata lost. "
            "The OOS guard will reject it. Retrain the model to persist a training cutoff.",
            model_name, ver,
        )


def _load_model(model_name: str, version: Optional[int] = None):
    """Load a model by name. If version is given, load that specific version
    regardless of status — used for re-testing historical models.

    trained_through is artifact-sourced (carried in the pickled model object),
    NEVER DB-sourced. The DB path below loads the pickle directly and does not
    override trained_through from the ModelVersion record. See KL-10."""
    import pickle
    try:
        from app.database.models import ModelVersion as MV
        from app.database.session import get_session
        db = get_session()
        try:
            if version is not None:
                mv = db.query(MV).filter_by(
                    model_name=model_name, version=version
                ).first()
            else:
                mv = (
                    db.query(MV)
                    .filter_by(model_name=model_name, status="ACTIVE")
                    .order_by(MV.version.desc())
                    .first()
                )
            if mv and mv.model_path:
                path = Path(mv.model_path)
                # BUG-13 fix: require the .gate_passed sentinel on the DB path too
                # (when loading the ACTIVE model without a specific version). A manual
                # DB edit can flip status="ACTIVE" without the model having passed the
                # gate, bypassing all promotion safety. The sentinel file is the
                # tamper-evident physical record of a gate pass.
                if version is None and mv.status == "ACTIVE":
                    sentinel = path.parent / (path.stem + ".gate_passed")
                    if not sentinel.exists():
                        logger.error(
                            "BUG-13: DB says %s v%d is ACTIVE but no .gate_passed "
                            "sentinel at %s. Refusing to load — sentinel may have been "
                            "deleted or DB was manually edited. Touch the sentinel or "
                            "re-run the walk-forward gate.",
                            model_name, mv.version, sentinel,
                        )
                        return None, 0
                if path.exists():
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                    if hasattr(obj, "is_trained"):
                        logger.info("Loaded %s model v%d (status=%s)",
                                    model_name, mv.version, mv.status)
                        _check_trained_through(obj, model_name, mv.version)
                        return obj, mv.version
                from app.ml.model import PortfolioSelectorModel
                m = PortfolioSelectorModel(model_type="xgboost")
                m.load(str(path.parent), mv.version, model_name=model_name)
                logger.info("Loaded %s model v%d (status=%s)",
                            model_name, mv.version, mv.status)
                _check_trained_through(m, model_name, mv.version)
                return m, mv.version
        finally:
            db.close()
    except Exception as exc:
        logger.warning("DB model load failed: %s", exc)
    # Fallback: load from pkl files when DB is unreachable.
    # Only load a specific version if explicitly requested, OR if a .gate_passed
    # sentinel exists — prevents auto-promoting models that never passed the gate.
    model_dir = Path("app/ml/models")
    if version is not None:
        path = model_dir / f"{model_name}_v{version}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if hasattr(obj, "is_trained"):
                logger.info("Loaded %s model v%d from file", model_name, version)
                _check_trained_through(obj, model_name, version)
                return obj, version
            from app.ml.model import PortfolioSelectorModel
            m = PortfolioSelectorModel(model_type="xgboost")
            m.load(str(model_dir), version, model_name=model_name)
            _check_trained_through(m, model_name, version)
            return m, version

    # When no version specified: only consider pkls that have a .gate_passed sentinel.
    # This prevents a silently-retrained model from displacing the known-good version
    # just because its version number is higher.
    gated_files = sorted(
        [p for p in model_dir.glob(f"{model_name}_v*.pkl")
         if (p.parent / (p.stem + ".gate_passed")).exists()],
        key=lambda p: int(p.stem.split("_v")[-1]),
    )
    if not gated_files:
        # Warn loudly — no gated model means the system is misconfigured
        logger.error(
            "No gate_passed sentinel found for any %s model in %s — "
            "cannot safely load a fallback model. Run the walk-forward gate "
            "and touch %s_v<N>.gate_passed to certify a version.",
            model_name, model_dir, model_name,
        )
        return None, 0

    # Load the highest-versioned gated model
    latest_gated = gated_files[-1]
    ver = int(latest_gated.stem.split("_v")[-1])
    with open(latest_gated, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "is_trained"):
        logger.info("Loaded %s model v%d from file (gate_passed sentinel present)",
                    model_name, ver)
        _check_trained_through(obj, model_name, ver)
        return obj, ver
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel(model_type="xgboost")
    m.load(str(latest_gated.parent), ver, model_name=model_name)
    logger.info("Loaded %s model v%d from file (gate_passed sentinel present)",
                model_name, ver)
    _check_trained_through(m, model_name, ver)
    return m, ver


# ── Earnings calendar pre-fetch (Phase 2b) ────────────────────────────────────

def _fetch_earnings_calendar(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> Dict[str, set]:
    """Pre-fetch historical earnings dates for all symbols using yfinance.

    Returns dict of symbol → set of datetime.date when earnings were reported.
    Earnings within the simulation window are included. Symbols with no data
    get an empty set (no blackout applied — fail-open).

    Takes ~2-3 min for 700 symbols. Call once before the sim, not per-fold.
    """
    import yfinance as yf
    cal: Dict[str, set] = {}
    logger.info("Pre-fetching earnings calendar for %d symbols (yfinance)...", len(symbols))
    ok, fail = 0, 0
    for sym in symbols:
        try:
            ed = yf.Ticker(sym).earnings_dates
            if ed is not None and len(ed) > 0:
                dates: set = set()
                for idx in ed.index:
                    d = idx.date() if hasattr(idx, "date") else idx
                    if start_date <= d <= end_date:
                        dates.add(d)
                cal[sym] = dates
                ok += 1
            else:
                cal[sym] = set()
                ok += 1
        except Exception:
            cal[sym] = set()
            fail += 1
    logger.info("Earnings calendar: %d ok, %d failed (fail-open for failures)", ok, fail)
    return cal


# ── Bootstrap walk-forward (Phase 1d) ─────────────────────────────────────────

def _bootstrap_folds(
    run_fn,
    n_bootstrap: int = 100,
    perturb_days: int = 30,
    **kwargs,
) -> Dict:
    """Run the walk-forward n_bootstrap times with ±perturb_days randomised fold offsets.

    Returns dict with Sharpe distribution statistics for reporting.
    """
    sharpes: List[float] = []
    rng = np.random.default_rng(42)
    _subheader(f"Bootstrap walk-forward: {n_bootstrap} iterations × ±{perturb_days}d perturbation")
    for i in range(n_bootstrap):
        jitter = int(rng.integers(-perturb_days, perturb_days + 1))
        try:
            if "total_years" in kwargs:
                kw = dict(kwargs, total_years=kwargs["total_years"] + jitter / 365)
            elif "total_days" in kwargs:
                kw = dict(kwargs, total_days=max(180, kwargs["total_days"] + jitter))
            else:
                kw = kwargs
            report = run_fn(**kw)
            if report.folds:
                sharpes.append(report.avg_sharpe)
        except Exception as exc:
            logger.debug("Bootstrap iter %d failed: %s", i, exc)
    if not sharpes:
        return {}
    arr = np.array(sharpes)
    result = {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "pct_positive": float(np.mean(arr > 0)),
    }
    print(f"\n  Bootstrap Sharpe distribution ({len(arr)} iters):")
    print(f"    Mean:   {result['mean']:+.3f}   Median: {result['median']:+.3f}")
    print(f"    Std:    {result['std']:.3f}")
    print(f"    P5/P95: {result['p5']:+.3f} / {result['p95']:+.3f}")
    print(f"    % positive: {result['pct_positive']:.1%}")
    sig = result["pct_positive"] >= 0.75
    if sig:
        _ok(f"Bootstrap passes: {result['pct_positive']:.1%} of iterations positive")
    else:
        _warn(f"Bootstrap low confidence: only {result['pct_positive']:.1%} positive")
    return result


# ── Swing walk-forward ─────────────────────────────────────────────────────────

def run_swing_walkforward(
    n_folds: int = 3,
    total_years: int = 5,
    symbols: Optional[List[str]] = None,
    atr_stop_mult: float = 0.5,
    atr_target_mult: float = 1.5,
    meta_model=None,
    pm_abstention_vix: float = 0.0,
    pm_abstention_spy_ma_days: int = 0,
    pm_abstention_spy_5d: bool = False,
    model_version: Optional[int] = None,
    # M-10: Default 5bps is optimistic for $20k retail; realistic Russell 1000 round-trip =
    # 15-25bps at Alpaca (spread + slippage + fees). Increase this for live calibration.
    transaction_cost_pct: float = 0.0005,
    purge_days: int = 10,
    embargo_days: Optional[int] = None,  # WF-1: post-test gap before next fold trains (None = same as purge_days)
    use_opportunity_score: bool = False,
    no_prefilters: bool = False,
    train_years: Optional[int] = None,  # Phase 3b: rolling window (None = expanding)
    earnings_blackout: Optional[Dict[str, set]] = None,  # Phase 2b: pre-built calendar
    macro_blocked_dates: Optional[set] = None,  # WF-5a: FOMC/NFP/CPI/GDP blocked dates
    benign_blocked_dates: Optional[set] = None,  # P1: adverse-regime dates
    wf_max_symbols: Optional[int] = None,  # cap universe to top-N by avg dollar volume
    feature_cache_workers: int = 0,
    feature_cache_executor: str = "process",
    feature_cache_disable: bool = False,
    sim_scan_interval_days: int = 1,
    use_factor_portfolio: bool = False,  # Phase D: bypass ML model, use factor composite scorer
    scorer_instance=None,  # Phase G: inject any callable scorer directly (overrides use_factor_portfolio)
    max_hold_bars_override: Optional[int] = None,  # Phase H+: force per-position hold cap
    no_atr_stops: bool = False,  # Phase 4: disable ATR stops entirely
    # Phase RA — REBALANCE mode
    rebalance_mode: bool = False,
    rebalance_days: int = 20,
    rebalance_target_n: int = 30,
    rebalance_sector_cap: float = 0.30,
    rebalance_add_threshold: int = 15,
    rebalance_drop_threshold: int = 30,
    rebalance_min_adv: float = 20_000_000.0,
    rebalance_regime_gate: bool = False,
    rebalance_regime_vix_bull: float = 20.0,
    rebalance_regime_vix_bear: float = 30.0,
    rebalance_regime_spy_ma_days: int = 200,
    rebalance_inv_vol: bool = False,
    rebalance_inv_vol_lookback: int = 20,
    rebalance_inv_vol_min_mult: float = 0.5,
    rebalance_inv_vol_max_mult: float = 2.0,
    rebalance_spy_vol_damper: bool = False,
    rebalance_spy_vol_damper_scale: float = 0.50,
    rebalance_hard_exit_bear: bool = False,  # LX6b: force-close all longs when BEAR at rebalance
    rebalance_flat_stop_pct: float = 0.0,  # LX8: per-position trailing stop pct (0.0 = off)
    # Phase 2: L/S extension
    enable_shorts: bool = False,
    short_target_n: int = 30,
    short_bull_n: Optional[int] = None,  # BULL-regime short count (None = use short_target_n)
    long_gross: float = 0.95,
    short_gross: float = 0.55,
    short_min_adv: float = 50_000_000.0,
    short_add_threshold: int = 15,
    short_drop_threshold: int = 30,
    spy_beta_hedge: bool = False,
    spy_beta_lookback: int = 60,
    spy_hedge_max_gross: float = 0.30,
    spy_hedge_vix_lo: float = 0.0,
    spy_hedge_vix_hi: float = 0.0,
    rebalance_atr_stops: bool = False,
    # Phase 89: factor stability gate (rolling realized rank-IC filter) — DEPRECATED, use dispersion gate
    rebalance_factor_stability_gate: bool = False,
    rebalance_factor_stability_lookback: int = 63,
    rebalance_factor_stability_ic_threshold: float = 0.02,
    # Phase 89 v2: cross-sectional return dispersion gate
    rebalance_dispersion_gate: bool = False,
    rebalance_dispersion_k: int = 5,
    rebalance_dispersion_L: int = 126,
    as_of: Optional[date] = None,  # WF-C2: pin fold boundaries for reproducibility
    delisted_haircut: float = 0.0,  # R5b: survivorship-realistic force-close haircut
    per_fold_ic_weights: bool = False,  # R5b: compute Option-A per-fold IC weights
    daily_ic_parquet: Optional[str] = None,  # R5b: path to daily_ic.parquet
    allow_in_sample: bool = False,  # OOS-guard bypass: label run in-sample, cannot promote
) -> WalkForwardReport:
    import yfinance as yf
    from app.backtesting.agent_simulator import AgentSimulator

    report = WalkForwardReport(model_type="swing")
    # Design note (deep-review pass 3): one pre-trained model is loaded here and
    # reused across all n_folds test windows. This is a generalization test, not
    # a true expanding-window WF with per-fold retrain. `is_true_walkforward`
    # stays False; downstream reporting/DSR consumers should treat fold Sharpes
    # as OOS evaluations of the SAME model, not of independently-fit models.
    model, version = _load_model("swing", version=model_version)
    if model is None and not use_factor_portfolio and scorer_instance is None:
        # L-5 fix: previously this returned an empty WalkForwardReport with Sharpe=0,
        # which could silently pass a "no-failure" check in calling pipelines. Make it
        # explicit so misconfigured runs surface immediately instead of vacuously.
        raise ValueError(
            "No swing model found AND no factor scorer provided. "
            "Either retrain a swing model, pass --use-factor-portfolio, or supply a scorer_instance."
        )
    if model is None:
        _warn("No ML swing model — running with factor scorer / portfolio only.")
    _warn(
        f"NOTE: Using pre-trained swing model v{version} for all {n_folds} folds — "
        "this is a generalization test, not a true expanding-window walk-forward "
        "(no per-fold retrain). See WalkForwardReport.is_true_walkforward."
    )

    from app.utils.constants import RUSSELL_1000_TICKERS
    from app.data.universe_history import pit_union as _pit_union, historical_trade_symbols as _hist_syms
    symbols = symbols or list(RUSSELL_1000_TICKERS)

    # Download full history
    # WF-C2: pin fold boundaries via as_of for reproducibility.
    if as_of is not None:
        end_all = datetime.combine(as_of, datetime.min.time())
        logger.info("WF-C2: swing walk-forward pinned to --as-of=%s", as_of)
    else:
        end_all = datetime.now()
        _warn(
            "WF-C2: --as-of not specified; fold boundaries anchored to datetime.now() "
            f"({end_all.date()}). Results will drift on subsequent runs. "
            "Pass --as-of YYYY-MM-DD for reproducibility."
        )
    start_all = end_all - timedelta(days=total_years * 365 + 30)
    _subheader(f"Swing walk-forward: {n_folds} folds | {total_years}yr | "
               f"{len(symbols)} symbols | model v{version}")

    # WF-A2: augment download seed with symbols that appeared in the feature store
    # across the full backtest window (combats survivorship bias for delisted names).
    extra_seed = _hist_syms(start_all.date(), end_all.date(), trade_type="swing")
    if extra_seed:
        pre_len = len(symbols)
        symbols = sorted(set(symbols) | set(extra_seed))
        logger.info(
            "WF-A2: augmented download seed %d → %d (+%d historical)",
            pre_len, len(symbols), len(symbols) - pre_len,
        )

    logger.info("Downloading daily bars %s -> %s", start_all.date(), end_all.date())
    t0 = time.time()
    symbols_data: Dict[str, pd.DataFrame] = {}
    # WF-R4: count survivorship drops within the extra_seed augmentation so the
    # bias is visible even though we can't fully remove it.
    _extra_seed_set = set(extra_seed) if extra_seed else set()
    _extra_attempted = 0
    _extra_loaded = 0
    for sym in symbols:
        is_extra = sym in _extra_seed_set
        if is_extra:
            _extra_attempted += 1
        try:
            df = yf.download(sym, start=start_all.date().isoformat(),
                             end=end_all.date().isoformat(), progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
            if len(df) >= 210:
                symbols_data[sym] = df
                if is_extra:
                    _extra_loaded += 1
        except Exception:
            pass
    if _extra_attempted > 0:
        _dropped = _extra_attempted - _extra_loaded
        _pct = 100.0 * _dropped / _extra_attempted
        logger.info(
            "Survivorship augmentation: attempted %d extra_seed symbols, loaded %d, dropped %d (%.1f%%)",
            _extra_attempted, _extra_loaded, _dropped, _pct,
        )

    spy_raw = yf.download("SPY", start=start_all.date().isoformat(),
                          end=end_all.date().isoformat(), progress=False, auto_adjust=True)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_raw.columns = [c.lower() for c in spy_raw.columns]
    if spy_raw.columns.duplicated().any():
        spy_raw = spy_raw.loc[:, ~spy_raw.columns.duplicated()]
    spy_prices = spy_raw["close"]
    symbols_data["SPY"] = spy_raw  # make SPY available to factor_scorer regime gate

    # Download VIX for opportunity score (Phase 2a) and for external scorer regime gates.
    if use_opportunity_score or scorer_instance is not None:
        try:
            vix_raw = yf.download("^VIX", start=start_all.date().isoformat(),
                                  end=end_all.date().isoformat(), progress=False, auto_adjust=True)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.get_level_values(0)
            vix_raw.columns = [c.lower() for c in vix_raw.columns]
            if vix_raw.columns.duplicated().any():
                vix_raw = vix_raw.loc[:, ~vix_raw.columns.duplicated()]
            if len(vix_raw) >= 5:
                symbols_data["^VIX"] = vix_raw
                logger.info("VIX history loaded: %d rows (for opportunity score)", len(vix_raw))
        except Exception as exc:
            logger.warning("VIX download failed — opportunity score will use defaults: %s", exc)

    logger.info("Data loaded: %d symbols in %.1fs", len(symbols_data), time.time() - t0)

    # Speed-up: cap to top-N symbols by average dollar volume (price × volume).
    if wf_max_symbols and wf_max_symbols < len(symbols_data):
        _synthetic = {"^VIX", "SPY"}
        _ranked = []
        for sym, df in symbols_data.items():
            if sym in _synthetic:
                continue
            if "close" in df.columns and "volume" in df.columns:
                avg_dv = (df["close"] * df["volume"]).mean()
            else:
                avg_dv = 0.0
            _ranked.append((avg_dv, sym))
        _ranked.sort(reverse=True)
        _keep = {sym for _, sym in _ranked[:wf_max_symbols]} | _synthetic
        _before = len(symbols_data)
        symbols_data = {s: d for s, d in symbols_data.items() if s in _keep}
        logger.info("--wf-max-symbols %d: trimmed %d → %d symbols by avg dollar volume",
                    wf_max_symbols, _before, len(symbols_data))

    # WF-1: embargo_days defaults to purge_days when not specified.
    # Both sides of every test window now have a clean gap:
    #   train | purge_days | TEST | embargo_days | next_fold_train
    _embargo = embargo_days if embargo_days is not None else purge_days

    # Build fold boundaries.
    # train_years=None: expanding window (train always from start_all).
    # train_years=N: rolling window (train starts at train_end - N years per fold).
    # purge_days gap between train_end and test_start prevents 5-day label leakage.
    # embargo_days gap AFTER test_end prevents test rows leaking into next fold's training set.
    #
    # WF-C3 fix: previously test_end = train_end + segment_days - embargo, which
    # *shortened the test window* instead of placing the embargo after it. The
    # corrected layout is:
    #   train | purge | TEST | embargo | next_train
    # Each fold's full test window spans train_end + (purge, segment_days]; the
    # next fold's train_end is shifted forward by embargo_days so that no test
    # row (or its lookback features) appears in the next training set.
    segment_days = int(total_years * 365 / (n_folds + 1))

    # WF-R5 (FIX 1): verify the requested WF config actually fits in
    # `total_years`. Each fold needs `segment_days` of test + (n_folds-1)
    # cumulative embargo shifts + `purge_days` lead-in + a small buffer.
    # If it doesn't fit, the last fold's raw_test_end gets clamped to
    # end_all, which can collapse the embargo gap below the next-fold
    # threshold and trip the AssertionError below.
    _BUFFER_DAYS = 30
    _required_days = (
        n_folds * segment_days
        + (n_folds - 1) * _embargo
        + purge_days
        + _BUFFER_DAYS
    )
    _available_days = int(total_years * 365)
    if _required_days > _available_days:
        # Silently shrink segment_days so the layout fits. Log loudly so the
        # user can adjust --total-years if they want full-length test windows.
        _new_segment = max(
            30,
            (
                _available_days
                - (n_folds - 1) * _embargo
                - purge_days
                - _BUFFER_DAYS
            ) // n_folds,
        )
        _suggested_years = (
            (n_folds * segment_days + (n_folds - 1) * _embargo + purge_days + _BUFFER_DAYS)
            / 365.0
        )
        logger.info(
            "WF-R5 FIX 1: requested layout needs %dd but only %dd available "
            "(total_years=%g, n_folds=%d, embargo=%dd, purge=%dd). "
            "Shrinking segment_days %d → %d to fit. "
            "Pass --total-years %.1f or larger to preserve full-length test windows.",
            _required_days, _available_days, total_years, n_folds, _embargo,
            purge_days, segment_days, _new_segment, _suggested_years,
        )
        segment_days = _new_segment

    fold_boundaries = []
    for fold_idx in range(n_folds):
        # Shift each fold's train_end forward by fold_idx * embargo_days so that
        # successive folds preserve an embargo gap between consecutive test_end
        # and train_end (expanding window naturally excludes the gap region).
        train_end_dt = (
            end_all
            - timedelta(days=segment_days * (n_folds - fold_idx))
            + timedelta(days=fold_idx * _embargo)
        )
        test_start_dt = train_end_dt + timedelta(days=purge_days + 1)
        # Full-length test window (no longer truncated by embargo).
        raw_test_end_dt = train_end_dt + timedelta(days=segment_days)
        if train_years is not None:
            # Rolling window: shift train start to exclude rows within embargo of prior test
            fold_train_start = max(
                start_all.date(),
                (train_end_dt - timedelta(days=train_years * 365)).date(),
            )
        else:
            fold_train_start = start_all.date()
        fold_boundaries.append((
            fold_train_start,
            train_end_dt.date(),
            test_start_dt.date(),
            min(raw_test_end_dt.date(), end_all.date()),
            _embargo,  # carry embargo into the fold runner for logging
        ))

    # Guard: each fold's train_end must be >= prior fold's test_end + embargo_days.
    for _i in range(1, len(fold_boundaries)):
        _prev_te_end = fold_boundaries[_i - 1][3]
        _this_tr_end = fold_boundaries[_i][1]
        _gap = (_this_tr_end - _prev_te_end).days
        if _gap < _embargo:
            raise AssertionError(
                f"WF-C3: fold {_i} train_end ({_this_tr_end}) violates embargo "
                f"of {_embargo}d after prior test_end ({_prev_te_end}): gap={_gap}d"
            )

    # WF-R5 (FIX 1 cont.): warn if any fold's test window was clamped short.
    for _idx, (_, _tr_end, _te_start, _te_end, _) in enumerate(fold_boundaries, start=1):
        _test_len = (_te_end - _te_start).days
        if _test_len < 30:
            logger.warning(
                "WF-R5: fold %d test window clamped to %dd (%s -> %s) — "
                "Sharpe may be noisy. Consider --total-years larger than %g.",
                _idx, _test_len, _te_start, _te_end, total_years,
            )

    # OOS-guard: every test fold must start strictly after the model's training cutoff.
    # Skip when model is None (pure factor/scorer run — no ML training cutoff to enforce).
    if model is not None:
        from scripts.walkforward.oos_guard import assert_model_oos as _assert_oos
        _assert_oos(
            trained_through=getattr(model, "trained_through", None),
            fold_boundaries=[(tr, te, ts, te2) for tr, te, ts, te2, _ in fold_boundaries],
            purge_days=purge_days,
            model_label=f"swing v{version}",
            allow_in_sample=allow_in_sample,
        )
        if allow_in_sample:
            report.in_sample_override = True

    # WF-R5 (FIX 4): IC weights calibration cutoff must be strictly before every
    # fold's train_end, else "pre-fold" weights become in-sample. The constant
    # IC_WEIGHTS_CALIBRATION_END must mirror END_DATE in scripts/compute_factor_ic.py.
    try:
        from app.ml.factor_scorer import IC_WEIGHTS_CALIBRATION_END as _ic_cutoff
    except Exception:
        _ic_cutoff = date(2021, 4, 26)
    for (_tr_s, _tr_e, _te_s, _te_e, _emb) in fold_boundaries:
        if _tr_e <= _ic_cutoff:
            raise ValueError(
                f"WF-R5 FIX 4: fold train_end {_tr_e} <= IC calibration cutoff "
                f"{_ic_cutoff}. IC weights are in-sample for this WF config. "
                f"Recompute IC weights with END_DATE < {_tr_e} in "
                f"scripts/compute_factor_ic.py and update IC_WEIGHTS_CALIBRATION_END "
                f"in app/ml/factor_scorer.py."
            )

    # BUG-21 FIX: Load sector map once for all folds so sector cap (30%) actually fires.
    # Previously sector_map was never passed to sim.run() → apply_sector_cap was a no-op.
    _wf_sector_map: dict = {}
    try:
        from app.data.sector_map import get_sector_map as _get_sector_map
        _all_syms = [s for s in symbols_data if s not in {"^VIX", "VIX", "SPY"}]
        _wf_sector_map = _get_sector_map(_all_syms)
        logger.info("Walk-forward: sector map loaded for %d symbols", len(_wf_sector_map))
    except Exception as _exc_sm:
        logger.warning("Walk-forward: sector map unavailable — sector cap will be a no-op: %s", _exc_sm)

    # Load sector ETF bars once for all folds (PIT sector-neutral features)
    _sector_etf_bars_wf: Optional[dict] = None
    _etf_hist_path_wf = Path("data/sector_etf/sector_etf_history.parquet")
    if _etf_hist_path_wf.exists():
        try:
            import pandas as _pd_etf
            _ef_wf = _pd_etf.read_parquet(_etf_hist_path_wf)
            _sector_etf_bars_wf = {}
            for _etf, _grp in _ef_wf.groupby("etf"):
                _grp_s = _grp.sort_values("date")
                _sector_etf_bars_wf[_etf] = [
                    (row["date"], float(row["close"]))
                    for _, row in _grp_s.iterrows()
                ]
            logger.info("Walk-forward: loaded sector ETF bars for %d ETFs", len(_sector_etf_bars_wf))
        except Exception as _exc_etf:
            logger.warning("Walk-forward: could not load sector_etf_history.parquet — %s", _exc_etf)

    def _run_swing_fold(args):
        fold_idx, tr_start, tr_end, te_start, te_end, emb = args
        _subheader(f"Fold {fold_idx}/{n_folds}  train:{tr_start}->{tr_end}  "
                   f"test:{te_start}->{te_end}  purge={purge_days}d  embargo={emb}d")
        t_fold = time.time()
        # Point-in-time filter: only use symbols that were in the index at fold train start.
        # Synthetic symbols (^VIX, VIX, SPY) bypass the filter — they're needed for
        # regime gates and opportunity score regardless of index membership.
        # WF-A2/A3: PIT filter using Russell 1000 (matches training universe).
        # BUG-2 FIX: universe frozen at tr_end (training end), NOT te_end.
        # Using te_end leaked test-window index additions (high-momentum names joining
        # the index mid-test are included from day 1 → look-ahead survivorship bias).
        # extra also capped at tr_end for the same reason.
        # WF-M7 fix: previously pit_union("russell1000", tr_start, tr_end) with
        # tr_start = start_all included names that left the index years before
        # the fold began. Use a recency window bounded by te_start - (252 + purge_days)
        # so that historical adds/removes far outside the fold's lookback do not
        # pollute the universe.
        _pit_lookback_days = 252 + purge_days
        _pit_from = max(tr_start, te_start - timedelta(days=_pit_lookback_days))
        extra = _hist_syms(_pit_from, tr_end, trade_type="swing")
        pit_members = set(_pit_union("russell1000", _pit_from, tr_end, extra_symbols=extra))
        _synthetic = {"^VIX", "VIX", "SPY"}
        fold_symbols_data = {
            s: d for s, d in symbols_data.items()
            if s in pit_members or s in _synthetic
        }
        # Build per-fold feature cache (pre-computes all (sym, day) features in parallel)
        # Skip when scorer_instance is set: factor_scorer bypasses the cache in _pm_score,
        # so building it is wasteful AND causes (N, 2) shape corruption in fold_symbols_data
        # (BUG-LS3: feature cache ProcessPoolExecutor mutates shared DataFrame state).
        _feature_cache = None
        if not feature_cache_disable and scorer_instance is None:
            try:
                from app.backtesting.feature_cache import build_feature_cache as _build_fc
                import os as _os
                _test_days = sorted({
                    d.date() if hasattr(d, "date") else d
                    for df in fold_symbols_data.values()
                    for d in df.index
                    if te_start <= (d.date() if hasattr(d, "date") else d) <= te_end
                })
                _vix_df = fold_symbols_data.get("^VIX")
                if _vix_df is None:
                    _vix_df = fold_symbols_data.get("VIX")
                _vix_s = _vix_df["close"] if _vix_df is not None and "close" in _vix_df.columns else None
                _feat_names = getattr(model, "feature_names", None) or []
                _workers = feature_cache_workers or max(2, min(_os.cpu_count() or 4, MAX_WORKERS))
                logger.info("Fold %d: building feature cache (%d syms × %d days, %d %s workers)",
                            fold_idx, len(fold_symbols_data), len(_test_days), _workers, feature_cache_executor)
                _feature_cache = _build_fc(
                    symbols_data=fold_symbols_data,
                    trading_days=_test_days,
                    feature_names=_feat_names,
                    vix_history=_vix_s,
                    sector_etf_bars=_sector_etf_bars_wf,
                    workers=_workers,
                    executor=feature_cache_executor,
                )
            except Exception as _exc:
                logger.warning("Feature cache build failed, falling back to live compute: %s", _exc)
                _feature_cache = None

            # Lockstep diagnostic: verify cache breadth before sim runs.
            # If symbols_with(mid_day) << len(fold_symbols_data), cache is sparse
            # (date-key mismatch or worker crash) and WF results will be invalid.
            if _feature_cache is not None and _feature_cache.n_symbols > 0:
                _mid = _test_days[len(_test_days) // 2]
                _n_on_mid = len(_feature_cache.symbols_with(_mid))
                logger.info(
                    "Fold %d cache breadth: %d/%d symbols populated; mid-fold day %s has %d symbols scored",
                    fold_idx, _feature_cache.n_symbols, len(fold_symbols_data), _mid, _n_on_mid,
                )
                if _n_on_mid < len(fold_symbols_data) * 0.5:
                    logger.warning(
                        "Fold %d: only %d/%d symbols available on %s — "
                        "lockstep scoring will be degraded. Check for date-key mismatch or worker crashes.",
                        fold_idx, _n_on_mid, len(fold_symbols_data), _mid,
                    )

        # Phase RB.1: build regime gate fn if requested (PIT-safe, uses fold's SPY+VIX)
        _regime_gate_fn = None
        _long_regime_fn = None
        _short_regime_fn = None
        if rebalance_mode and rebalance_regime_gate:
            _regime_gate_fn = _make_regime_gate_fn(
                fold_symbols_data,
                spy_ma_days=rebalance_regime_spy_ma_days,
                vix_bull=rebalance_regime_vix_bull,
                vix_bear=rebalance_regime_vix_bear,
            )
            if enable_shorts:
                # Build asymmetric per-side regime fns (pre-registered thresholds, not tuned on folds)
                _spy_df = fold_symbols_data.get("SPY")
                if _spy_df is None:
                    _spy_df = fold_symbols_data.get("spy")
                _vix_df = fold_symbols_data.get("^VIX")
                if _vix_df is None:
                    _vix_df = fold_symbols_data.get("VIX")
                _vix_s = _vix_df["close"] if _vix_df is not None and "close" in _vix_df.columns else None
                if _spy_df is not None and _vix_s is not None:
                    _long_regime_fn, _short_regime_fn = build_asymmetric_regime_fns(
                        _spy_df, _vix_s,
                        spy_ma_days=rebalance_regime_spy_ma_days,
                    )
                else:
                    logger.warning("Fold %d: SPY/VIX unavailable — asymmetric regime fns will use defaults", fold_idx)

        # Phase 89: factor stability gate — DEPRECATED (trailing IC ~80d lag, test-window look-ahead).
        # H-1: hard-fail if anyone tries to re-enable this. Use rebalance_dispersion_gate instead.
        if rebalance_factor_stability_gate:
            raise ValueError(
                "rebalance_factor_stability_gate is deprecated and disabled due to test-window "
                "look-ahead in its FactorStabilityGate construction. Remove it from your call "
                "and use rebalance_dispersion_gate (Phase 89 v2) instead."
            )
        # BUG-12: _make_combined_regime_fn was deleted (look-ahead leak + dead code).

        # Phase 89 v2: cross-sectional dispersion gate (concurrent ~5d lag signal)
        if rebalance_mode and rebalance_dispersion_gate:
            from app.ml.dispersion_gate import make_combined_dispersion_regime_fn
            _regime_gate_fn = make_combined_dispersion_regime_fn(
                fold_symbols_data,
                spy_vix_fn=_regime_gate_fn,
                k=rebalance_dispersion_k,
                L=rebalance_dispersion_L,
            )

        # Deepcopy scorer so each fold gets independent mutable state (e.g. IcCompositeV220Scorer
        # tracks _in_momentum_regime + _breadth_ema across calls — sharing one instance across
        # ThreadPoolExecutor folds causes races and state contamination between folds).
        import copy
        _factor_scorer_inst = copy.deepcopy(scorer_instance) if scorer_instance is not None else None

        # R5b: Option-A per-fold IC weights. When enabled, recompute IC-IR weights from
        # daily_ic.parquet using data with date <= tr_end (PIT-safe) and inject them into
        # supported scorer classes (IcCompositeScorer, IcCompositeV221Scorer). Other scorer
        # classes log a notice and fall back to their static weights.
        if per_fold_ic_weights and _factor_scorer_inst is not None:
            from app.ml.ic_utils import compute_fold_ic_weights, find_latest_daily_ic_parquet
            _ic_path = daily_ic_parquet or find_latest_daily_ic_parquet()
            _scorer_cls_name = type(_factor_scorer_inst).__name__
            _supports = _scorer_cls_name in ("IcCompositeScorer", "IcCompositeV221Scorer")
            if _ic_path is None:
                logger.warning(
                    "WF R5b: --per-fold-ic-weights set but no daily_ic.parquet found; "
                    "fold %d using static pre-fold-1 weights.", fold_idx,
                )
            elif not _supports:
                logger.warning(
                    "WF R5b: --per-fold-ic-weights set but scorer %s does not accept "
                    "fold_ic_weights; fold %d using static weights. "
                    "TODO: extend Option-A support to %s.",
                    _scorer_cls_name, fold_idx, _scorer_cls_name,
                )
            else:
                _fold_weights = compute_fold_ic_weights(_ic_path, tr_end)
                if _fold_weights is None:
                    logger.warning(
                        "WF R5b: insufficient IC data for fold %d (tr_end=%s); using static weights.",
                        fold_idx, tr_end,
                    )
                else:
                    # Rebuild a fresh scorer with the per-fold weights so prior state is dropped.
                    _ScorerCls = type(_factor_scorer_inst)
                    try:
                        _factor_scorer_inst = _ScorerCls(fold_ic_weights=_fold_weights)
                        logger.info(
                            "WF R5b: fold %d per-fold IC weights (tr_end=%s, %d features): %s",
                            fold_idx, tr_end, len(_fold_weights),
                            ", ".join(f"{k}={v:.3f}" for k, v in sorted(
                                _fold_weights.items(), key=lambda kv: -kv[1])[:5]),
                        )
                    except Exception as _exc:
                        logger.warning(
                            "WF R5b: failed to rebuild %s with fold_ic_weights (%s); using static.",
                            _scorer_cls_name, _exc,
                        )
        # WF-R4: defensively reset stateful per-fold scorer state (EMA breadth,
        # regime flag) so a deepcopy from a pre-warmed scorer cannot leak state
        # across folds. No-op for scorers without reset().
        if _factor_scorer_inst is not None and hasattr(_factor_scorer_inst, "reset"):
            _factor_scorer_inst.reset()
            if hasattr(_factor_scorer_inst, "get_state"):
                _state_after_reset = _factor_scorer_inst.get_state()
                assert _state_after_reset == (False, None), (
                    f"WF-R4: scorer.reset() did not clear state, got {_state_after_reset}"
                )
        if _factor_scorer_inst is None and use_factor_portfolio:
            from app.ml.factor_scorer import FactorPortfolioScorer
            _factor_scorer_inst = FactorPortfolioScorer(
                top_n=20,
                top_n_short=15,
                long_short=True,  # Phase F: directional L/S, 40% net long
            )

        # Phase F: factor portfolio needs wider limits than the ML-signal defaults
        # (top-20 longs + top-15 shorts = 35 positions, 15% drawdown tolerance)
        if _factor_scorer_inst is not None:
            from app.agents.risk_rules import RiskLimits
            _sim_limits = RiskLimits(
                MAX_OPEN_POSITIONS=40,          # headroom for 20L + 15S + slack
                MAX_POSITION_SIZE_PCT=0.05,     # 5% per position (same)
                MAX_ACCOUNT_DRAWDOWN_PCT=0.15,  # 15% drawdown gate (vs 5% for single trades)
                MAX_DAILY_LOSS_PCT=0.05,        # 5% daily loss (vs 2%)
                MAX_PORTFOLIO_HEAT_PCT=0.30,    # 30% heat (diversified portfolio)
                MAX_SECTOR_CONCENTRATION_PCT=0.30,  # 30% sector cap
            )
        else:
            _sim_limits = None  # use AgentSimulator defaults

        # BUG-14: deep-copy model before passing to each fold's simulator.
        # On Linux/Mac, MAX_FOLD_WORKERS > 1 means folds run concurrently in threads.
        # XGBoost inference is GIL-safe but any mutable Python-layer state on the model
        # object (cached predictions, normalisation state) can race. Deep-copy is cheap
        # (~MB) and eliminates the risk. On Windows MAX_FOLD_WORKERS=1 so the copy is a
        # no-op in production but still documents the thread-safety contract.
        _fold_model = copy.deepcopy(model) if model is not None else None
        sim = AgentSimulator(
            model=_fold_model,
            atr_stop_mult=atr_stop_mult,
            atr_target_mult=atr_target_mult,
            meta_model=meta_model,
            pm_abstention_vix=pm_abstention_vix,
            pm_abstention_spy_ma_days=pm_abstention_spy_ma_days,
            pm_abstention_spy_5d=pm_abstention_spy_5d,
            transaction_cost_pct=transaction_cost_pct,
            use_opportunity_score=use_opportunity_score,
            no_prefilters=no_prefilters,
            earnings_blackout=earnings_blackout,
            macro_blocked_dates=macro_blocked_dates,
            benign_blocked_dates=benign_blocked_dates,
            feature_cache=_feature_cache,
            sim_scan_interval_days=sim_scan_interval_days,
            factor_scorer=_factor_scorer_inst,
            limits=_sim_limits,
            max_hold_bars_override=max_hold_bars_override,
            no_atr_stops=no_atr_stops,
            rebalance_mode=rebalance_mode,
            rebalance_days=rebalance_days,
            rebalance_target_n=rebalance_target_n,
            rebalance_sector_cap=rebalance_sector_cap,
            rebalance_add_threshold=rebalance_add_threshold,
            rebalance_drop_threshold=rebalance_drop_threshold,
            rebalance_min_adv=rebalance_min_adv,
            rebalance_regime_fn=_regime_gate_fn,
            rebalance_inv_vol=rebalance_inv_vol,
            rebalance_inv_vol_lookback=rebalance_inv_vol_lookback,
            rebalance_inv_vol_min_mult=rebalance_inv_vol_min_mult,
            rebalance_inv_vol_max_mult=rebalance_inv_vol_max_mult,
            rebalance_spy_vol_damper=rebalance_spy_vol_damper,
            rebalance_spy_vol_damper_scale=rebalance_spy_vol_damper_scale,
            rebalance_hard_exit_bear=rebalance_hard_exit_bear,
            rebalance_flat_stop_pct=rebalance_flat_stop_pct,
            enable_shorts=enable_shorts,
            short_target_n=short_target_n,
            short_bull_n=short_bull_n,
            long_gross=long_gross,
            short_gross=short_gross,
            short_min_adv=short_min_adv,
            short_add_threshold=short_add_threshold,
            short_drop_threshold=short_drop_threshold,
            short_regime_fn=_short_regime_fn if enable_shorts else None,
            long_regime_fn=_long_regime_fn if enable_shorts else None,
            spy_beta_hedge=spy_beta_hedge,
            spy_beta_lookback=spy_beta_lookback,
            spy_hedge_max_gross=spy_hedge_max_gross,
            spy_hedge_vix_lo=spy_hedge_vix_lo,
            spy_hedge_vix_hi=spy_hedge_vix_hi,
            rebalance_atr_stops=rebalance_atr_stops,
            delisted_haircut=delisted_haircut,
        )
        result = sim.run(
            fold_symbols_data,
            start_date=te_start,
            end_date=te_end,
            spy_prices=spy_prices,
            sector_map=_wf_sector_map,  # BUG-21 FIX: sector cap now active
        )
        elapsed = time.time() - t_fold
        stop_exits = result.exit_breakdown.get("STOP", 0)
        stop_rate = stop_exits / max(result.total_trades, 1)
        # WF-1: compute additional metrics
        trade_rets = (getattr(result, "trade_returns", None)
                      or [t.pnl_pct for t in (getattr(result, "trades", None) or [])
                          if t.exit_reason != "OPEN"])
        pf = _compute_profit_factor(trade_rets)
        fold_yrs = _fold_years(te_start, te_end)
        calmar = _compute_calmar(result.total_return_pct, result.max_drawdown_pct, fold_yrs)
        equity = getattr(result, "equity_curve", []) or []
        kr = _compute_k_ratio(equity)
        _ok(f"Fold {fold_idx} done in {elapsed:.1f}s — {result.total_trades} trades, "
            f"Sharpe {result.sharpe_ratio:.2f}  PF={pf:.2f}  Calmar={calmar:.2f}")
        return FoldResult(
            fold=fold_idx,
            train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=result.total_trades,
            win_rate=result.win_rate,
            sharpe=result.sharpe_ratio,
            max_drawdown=result.max_drawdown_pct,
            total_return=result.total_return_pct,
            stop_exit_rate=stop_rate,
            model_version=version,
            profit_factor=pf,
            calmar_ratio=calmar,
            k_ratio=kr,
            n_obs=max(len(equity) - 1, 0),  # daily returns count for DSR
        )

    from concurrent.futures import ThreadPoolExecutor
    fold_args = [
        (i + 1, tr_start, tr_end, te_start, te_end, emb)
        for i, (tr_start, tr_end, te_start, te_end, emb) in enumerate(fold_boundaries)
    ]
    _fold_workers = min(n_folds, MAX_FOLD_WORKERS)
    with ThreadPoolExecutor(max_workers=_fold_workers) as pool:
        results = list(pool.map(_run_swing_fold, fold_args))
    report.folds = sorted(results, key=lambda f: f.fold)

    return report


# ── Intraday walk-forward ──────────────────────────────────────────────────────

def run_intraday_walkforward(
    n_folds: int = 3,
    total_days: int = 730,
    symbols: Optional[List[str]] = None,
    meta_model=None,
    pm_abstention_vix: float = 0.0,
    pm_abstention_spy_ma_days: int = 0,
    scan_offsets: Optional[List[int]] = None,
    model_version: Optional[int] = None,
    transaction_cost_pct: float = 0.0015,
    purge_days: int = 2,
    embargo_days: Optional[int] = None,  # WF-1: post-test gap (None = same as purge_days)
    use_opportunity_score: bool = False,
    use_dispersion_gate: bool = False,
    earnings_blackout: Optional[Dict[str, set]] = None,  # Phase 2b: pre-built calendar
    macro_blocked_dates: Optional[set] = None,  # WF-5a: FOMC/NFP/CPI/GDP blocked dates
    benign_blocked_dates: Optional[set] = None,  # P1: adverse-regime dates
    use_regime_gate: bool = False,  # Phase R5: regime gates (R5-A/B/C)
    regime_map: Optional[Dict] = None,  # Phase R5: {date: label} from WF-4 regime.py
    as_of: Optional[date] = None,  # WF-C2: pin fold boundaries for reproducibility
    allow_in_sample: bool = False,  # OOS-guard bypass: label run in-sample, cannot promote
) -> WalkForwardReport:
    from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
    from app.data.intraday_cache import load_many, available_symbols as poly_syms

    report = WalkForwardReport(model_type="intraday")
    # Design note (deep-review pass 3): same single-model-across-folds convention
    # as the swing harness. See run_swing_walkforward for rationale.
    model, version = _load_model("intraday", version=model_version)
    if model is None:
        _err("No intraday model found — retrain first.")
        return report

    _warn(
        f"NOTE: Using pre-trained intraday model v{version} for all {n_folds} folds — "
        "this is a generalization test, not a true expanding-window walk-forward "
        "(no per-fold retrain). See WalkForwardReport.is_true_walkforward."
    )

    from app.utils.constants import RUSSELL_1000_TICKERS
    from app.data.universe_history import members_at as _members_at
    symbols = symbols or list(RUSSELL_1000_TICKERS)

    # WF-C2: pin fold boundaries via as_of for reproducibility.
    if as_of is not None:
        end_all = as_of
        logger.info("WF-C2: intraday walk-forward pinned to --as-of=%s", as_of)
    else:
        end_all = datetime.now().date()
        _warn(
            "WF-C2: --as-of not specified; intraday fold boundaries anchored to "
            f"datetime.now() ({end_all}). Results will drift on subsequent runs. "
            "Pass --as-of YYYY-MM-DD for reproducibility."
        )
    start_all = end_all - timedelta(days=total_days + 10)

    _subheader(f"Intraday walk-forward: {n_folds} folds | {total_days}d | "
               f"{len(symbols)} symbols | model v{version}")

    # Load 5-min data from Polygon cache
    cache_syms = set(poly_syms())
    if cache_syms:
        logger.info("Loading from Polygon cache (%d symbols available)", len(cache_syms))
        symbols_data = load_many(
            [s for s in symbols if s in cache_syms],
            start=start_all, end=end_all,
        )
    else:
        _warn("Polygon cache empty — falling back to yfinance (≤55 days)")
        import yfinance as yf
        symbols_data = {}
        for sym in symbols[:100]:  # cap at 100 for yfinance fallback
            try:
                df = yf.download(sym, period="55d", interval="5m",
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 12:
                    symbols_data[sym] = df
            except Exception:
                pass

    spy_data: Optional[pd.DataFrame] = None
    if "SPY" in symbols_data:
        spy_data = symbols_data["SPY"]

    # Phase 86: fetch SPY daily bars for market-condition features
    spy_daily_data: Optional[pd.DataFrame] = None
    try:
        import yfinance as yf
        _spy_daily = yf.download("SPY", period="3y", progress=False, auto_adjust=True)
        if isinstance(_spy_daily.columns, pd.MultiIndex):
            _spy_daily.columns = _spy_daily.columns.get_level_values(0)
        _spy_daily.columns = [c.lower() for c in _spy_daily.columns]
        if len(_spy_daily) >= 6:
            spy_daily_data = _spy_daily
            logger.info("SPY daily bars loaded: %d rows", len(spy_daily_data))
    except Exception as exc:
        logger.warning("SPY daily fetch failed (Phase 86 features will use defaults): %s", exc)

    # Download VIX daily bars for opportunity score (Phase 2a)
    if use_opportunity_score:
        try:
            import yfinance as yf
            _vix_daily = yf.download("^VIX", period="3y", progress=False, auto_adjust=True)
            if isinstance(_vix_daily.columns, pd.MultiIndex):
                _vix_daily.columns = _vix_daily.columns.get_level_values(0)
            _vix_daily.columns = [c.lower() for c in _vix_daily.columns]
            if len(_vix_daily) >= 5:
                symbols_data["^VIX"] = _vix_daily
                logger.info("VIX daily bars loaded: %d rows (for opportunity score)", len(_vix_daily))
        except Exception as exc:
            logger.warning("VIX download failed — opportunity score will use SPY-only: %s", exc)

    logger.info("Intraday data loaded: %d symbols", len(symbols_data))

    # Fold boundaries — split the test period into n_folds equal segments.
    # Exclude daily-bar overlay symbols (^VIX, SPY) which may span 3y and would
    # inflate all_days_sorted far beyond the intraday data window.
    _daily_overlay_keys = {"^VIX", "SPY"}
    all_days_sorted = sorted({
        d for sym, df in symbols_data.items()
        if sym not in _daily_overlay_keys
        for d in pd.to_datetime(df.index).date
    })
    if not all_days_sorted:
        _err("No trading days found in data.")
        return report

    # WF-1: embargo defaults to purge_days when not specified.
    _embargo_intra = embargo_days if embargo_days is not None else purge_days

    # purge_days trading-day gap between train_end and test_start prevents intraday
    # label leakage (2-day hold means labels at train boundary use test-window bars).
    # embargo_days post-test gap prevents test rows entering next fold's training set.
    segment_size = max(1, len(all_days_sorted) // (n_folds + 1))
    fold_boundaries = []
    for fi in range(n_folds):
        tr_end_idx = segment_size * (fi + 1) - 1
        te_start_idx = min(tr_end_idx + purge_days + 1, len(all_days_sorted) - 1)
        te_end_idx = min(segment_size * (fi + 1) + segment_size - 1, len(all_days_sorted) - 1)
        fold_boundaries.append((
            all_days_sorted[0],
            all_days_sorted[tr_end_idx],
            all_days_sorted[te_start_idx],
            all_days_sorted[te_end_idx],
            _embargo_intra,
        ))

    # OOS-guard: every test fold must start strictly after the model's training cutoff.
    # BUG-8 fix: pass trading_day_set so purge_days is counted in trading days, not
    # calendar days. Without this, a Friday trained_through + purge=2 gives a Sunday
    # cutoff that a Monday te_start clears with only 1 trading day of actual gap.
    from scripts.walkforward.oos_guard import assert_model_oos
    assert_model_oos(
        trained_through=getattr(model, "trained_through", None),
        fold_boundaries=[(tr, te, ts, te2) for tr, te, ts, te2, _ in fold_boundaries],
        purge_days=purge_days,
        model_label=f"intraday v{version}",
        allow_in_sample=allow_in_sample,
        trading_day_set=set(all_days_sorted),
    )
    if allow_in_sample:
        report.in_sample_override = True

    def _run_intraday_fold(args):
        fold_idx, tr_start, tr_end, te_start, te_end, emb = args
        _subheader(f"Fold {fold_idx}/{n_folds}  train:{tr_start}->{tr_end}  "
                   f"test:{te_start}->{te_end}  purge={purge_days}d  embargo={emb}d")
        t_fold = time.time()
        # Point-in-time filter: only use symbols that were in the index at fold train start
        pit_members = set(_members_at("russell1000", tr_start))
        fold_symbols_data = {s: d for s, d in symbols_data.items() if s in pit_members}
        sim = IntradayAgentSimulator(
            model=model,
            meta_model=meta_model,
            pm_abstention_vix=pm_abstention_vix,
            pm_abstention_spy_ma_days=pm_abstention_spy_ma_days,
            scan_offsets=scan_offsets,
            transaction_cost_pct=transaction_cost_pct,
            use_opportunity_score=use_opportunity_score,
            use_dispersion_gate=use_dispersion_gate,
            earnings_blackout=earnings_blackout,
            macro_blocked_dates=macro_blocked_dates,
            benign_blocked_dates=benign_blocked_dates,
            use_regime_gate=use_regime_gate,
            regime_map=regime_map,
        )
        result = sim.run(
            fold_symbols_data,
            spy_data=spy_data,
            start_date=te_start,
            end_date=te_end,
            spy_daily_data=spy_daily_data,
        )
        elapsed = time.time() - t_fold
        stop_exits = result.exit_breakdown.get("STOP", 0)
        stop_rate = stop_exits / max(result.total_trades, 1)
        # WF-1: additional metrics
        trade_rets = (getattr(result, "trade_returns", None)
                      or [t.pnl_pct for t in (getattr(result, "trades", None) or [])
                          if t.exit_reason != "OPEN"])
        pf = _compute_profit_factor(trade_rets)
        fold_yrs = _fold_years(te_start, te_end)
        calmar = _compute_calmar(result.total_return_pct, result.max_drawdown_pct, fold_yrs)
        equity = getattr(result, "equity_curve", []) or []
        kr = _compute_k_ratio(equity)
        _ok(f"Fold {fold_idx} done in {elapsed:.1f}s — {result.total_trades} trades, "
            f"Sharpe {result.sharpe_ratio:.2f}  PF={pf:.2f}  Calmar={calmar:.2f}")
        return FoldResult(
            fold=fold_idx,
            train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=result.total_trades,
            win_rate=result.win_rate,
            sharpe=result.sharpe_ratio,
            max_drawdown=result.max_drawdown_pct,
            total_return=result.total_return_pct,
            stop_exit_rate=stop_rate,
            model_version=version,
            profit_factor=pf,
            calmar_ratio=calmar,
            k_ratio=kr,
            n_obs=max(len(equity) - 1, 0),  # daily returns count for DSR
        )

    fold_args = [
        (i + 1, tr_start, tr_end, te_start, te_end, emb)
        for i, (tr_start, tr_end, te_start, te_end, emb) in enumerate(fold_boundaries)
    ]
    report.folds = [_run_intraday_fold(args) for args in fold_args]

    return report


# ── WF-3: CPCV helpers ────────────────────────────────────────────────────────

def _momentum_baseline_scorer(lookback_days: int = 60):
    """Return a factor_scorer callable that ranks symbols by trailing return (PIT-safe)."""
    import pandas as _pd

    def _scorer(day, symbols_data, vix_history=None):
        ts = _pd.Timestamp(day)
        scores = []
        for sym, df in symbols_data.items():
            if sym in ("SPY", "^VIX", "VIX"):
                continue
            if df is None or df.empty:
                continue
            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns:
                continue
            hist = df[df.index < ts][close_col]
            if len(hist) < lookback_days + 1:
                continue
            ret = float(hist.iloc[-1] / hist.iloc[-(lookback_days + 1)] - 1)
            scores.append((sym, ret))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    return _scorer


def _cpcv_swing_gate_ok(cpcv_result, args) -> bool:
    """Did the swing CPCV produce a genuine pass?

    Returns True only when the run produced a result whose gate passed. A None
    result (no model / skipped) or zero surviving paths (n_paths == 0) is a
    FAILURE for the overall exit status — this is the fix for the display bug
    where a failed/empty CPCV still printed "ALL GATES PASSED".
    """
    if cpcv_result is None:
        return False
    # Zero surviving paths (every fold skipped/failed) → not a pass.
    if not getattr(cpcv_result, "path_sharpes", None):
        return False
    try:
        # Phase-4: evaluate at the requested promotion tier (paper|capital). Under
        # GATE_MODE='significance' this is how the CAPITAL tier becomes reachable;
        # under mean_sharpe these kwargs are ignored by gate_passed().
        return bool(cpcv_result.gate_passed(
            dsr_n=args.dsr_n, paper_gate=args.paper_gate,
            tier=getattr(args, "gate_tier", "paper"),
            paper_confirmation=getattr(args, "paper_confirmation", False),
            regime_waiver_approved=getattr(args, "regime_waiver_approved", False),
        ))
    except Exception:
        return False


def _run_cpcv_swing(args, symbols, swing_ver, meta_model, earnings_cal, passed):
    """Run CPCV for swing model and print the result."""
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.strategies.swing import SwingStrategy
    from app.utils.constants import RUSSELL_1000_TICKERS
    model, version = _load_model("swing", version=swing_ver)
    if model is None:
        _warn("CPCV: no swing model found — skipping")
        return
    syms = symbols or list(RUSSELL_1000_TICKERS)

    _factor_scorer = None
    if getattr(args, "rebalance_momentum_baseline", False):
        _factor_scorer = _momentum_baseline_scorer(lookback_days=60)
        print("  CPCV: momentum baseline mode — bypassing ML model, using 60d trailing return ranker")
    elif getattr(args, "rebalance_ic_composite", False):
        from app.ml.factor_scorer import IcCompositeScorer
        _factor_scorer = IcCompositeScorer()
        print("  CPCV: IC composite mode — Phase 88 deterministic IC-weighted scorer (v219 weights)")
    elif getattr(args, "rebalance_ic_composite_v220", False):
        from app.ml.factor_scorer import IcCompositeV220Scorer
        _factor_scorer = IcCompositeV220Scorer()
        print("  CPCV: IC composite v220 mode — Phase 90 regime-conditional two-composite scorer")
    elif getattr(args, "rebalance_ic_composite_v221", False):
        from app.ml.factor_scorer import IcCompositeV221Scorer
        _factor_scorer = IcCompositeV221Scorer()
        print("  CPCV: IC composite v221 mode — Phase 91 fundamentals-downweighted scorer")
    elif getattr(args, "rebalance_ic_composite_v222", False):
        from app.ml.factor_scorer import IcCompositeV222Scorer
        _factor_scorer = IcCompositeV222Scorer()
        print("  CPCV: IC composite v222 mode — Phase 92 hybrid (v221 base + v220 breadth switch)")

    strategy = SwingStrategy(
        model=model, version=version, symbols=syms,
        atr_stop_mult=args.stop_mult, atr_target_mult=args.target_mult,
        meta_model=meta_model,
        pm_abstention_vix=args.pm_abstention_vix,
        pm_abstention_spy_ma_days=args.pm_abstention_spy_ma_days,
        pm_abstention_spy_5d=args.pm_abstention_spy_5d,
        transaction_cost_pct=args.swing_cost_bps / 10_000 / 2,
        use_opportunity_score=args.pm_opportunity_score,
        no_prefilters=args.no_prefilters,
        earnings_blackout=earnings_cal,
        feature_cache_workers=args.feature_cache_workers,
        feature_cache_executor=args.feature_cache_executor,
        feature_cache_disable=args.feature_cache_disable,
        sim_scan_interval_days=args.sim_scan_interval_days,
        rebalance_mode=getattr(args, "rebalance_mode", False),
        rebalance_days=getattr(args, "rebalance_days", 20),
        rebalance_target_n=getattr(args, "rebalance_target_n", 30),
        rebalance_sector_cap=getattr(args, "rebalance_sector_cap", 0.30),
        rebalance_add_threshold=getattr(args, "rebalance_add_threshold", 15),
        rebalance_drop_threshold=getattr(args, "rebalance_drop_threshold", 30),
        rebalance_min_adv=getattr(args, "rebalance_min_adv", 0.0),
        rebalance_regime_gate=getattr(args, "rebalance_regime_gate", False),
        rebalance_regime_spy_ma_days=getattr(args, "rebalance_regime_spy_ma_days", 200),
        rebalance_regime_vix_bull=getattr(args, "rebalance_regime_vix_bull", 20.0),
        rebalance_regime_vix_bear=getattr(args, "rebalance_regime_vix_bear", 30.0),
        rebalance_inv_vol=getattr(args, "rebalance_inv_vol", False),
        rebalance_inv_vol_lookback=getattr(args, "rebalance_inv_vol_lookback", 20),
        rebalance_inv_vol_min_mult=getattr(args, "rebalance_inv_vol_min_mult", 0.5),
        rebalance_inv_vol_max_mult=getattr(args, "rebalance_inv_vol_max_mult", 2.0),
        rebalance_spy_vol_damper=getattr(args, "rebalance_spy_vol_damper", False),
        rebalance_spy_vol_damper_scale=getattr(args, "rebalance_spy_vol_damper_scale", 0.50),
        rebalance_hard_exit_bear=getattr(args, "rebalance_hard_exit_bear", False),
        rebalance_flat_stop_pct=getattr(args, "rebalance_flat_stop_pct", 0.0),
        rebalance_factor_stability_gate=getattr(args, "rebalance_factor_stability_gate", False),
        rebalance_factor_stability_lookback=getattr(args, "rebalance_factor_stability_lookback", 63),
        rebalance_factor_stability_ic_threshold=getattr(args, "rebalance_factor_stability_ic_threshold", 0.02),
        rebalance_dispersion_gate=getattr(args, "rebalance_dispersion_gate", False),
        rebalance_dispersion_k=getattr(args, "rebalance_dispersion_k", 5),
        rebalance_dispersion_L=getattr(args, "rebalance_dispersion_L", 126),
        # RANKER v2 Spike A — dollar-neutral L/S short leg through the per-fold CPCV
        # path. Defaults (enable_shorts=False + AgentSimulator-matching grosses) keep
        # the long-only swing path byte-identical.
        enable_shorts=getattr(args, "enable_shorts", False),
        long_gross=getattr(args, "long_gross", 0.95),
        short_gross=getattr(args, "short_gross", 0.55),
        short_target_n=getattr(args, "short_target_n", 30),
        short_min_adv=getattr(args, "short_min_adv", 50_000_000.0),
        short_add_threshold=getattr(args, "short_add_threshold", 15),
        short_drop_threshold=getattr(args, "short_drop_threshold", 30),
        # RANKER v2 §3.1 re-architecture — NET-sector cap (Failure B fix) + SPY
        # beta-hedge overlay + realized net-exposure capture. Defaults OFF →
        # long-only / existing L-S CPCV runs byte-identical.
        net_sector_cap=getattr(args, "net_sector_cap", False),
        spy_beta_hedge=getattr(args, "spy_beta_hedge", False),
        spy_beta_lookback=getattr(args, "spy_beta_lookback", 60),
        spy_hedge_max_gross=getattr(args, "spy_hedge_max_gross", 0.30),
        capture_net_exposure=(True if getattr(args, "capture_net_exposure", False)
                              else None),
        net_beta_lookback=getattr(args, "net_beta_lookback", 60),
        factor_scorer=_factor_scorer,
        no_atr_stops=getattr(args, "no_atr_stops", False),
    )
    strategy.model_type = "swing"
    strategy.allow_in_sample = getattr(args, "allow_in_sample", False)

    # Phase 1: per-fold retraining (true out-of-sample). Construct the retrainer
    # from the active SWING_RETRAIN architecture so each fold fits the same model
    # type / feature set on only its own training window.
    if getattr(args, "per_fold_retrain", False):
        from scripts.walkforward.retrainers import SwingFoldRetrainer, TrainWindowCache
        from app.ml.retrain_config import SWING_RETRAIN
        _retrainer = SwingFoldRetrainer(base_config=dict(
            model_type=SWING_RETRAIN["model_type"],
            label_scheme=SWING_RETRAIN["label_scheme"],
            feature_keep_list=SWING_RETRAIN["feature_keep_list"],
            fetch_fundamentals=SWING_RETRAIN.get("fetch_fundamentals", False),
            n_workers=SWING_RETRAIN.get("n_workers", 0),
        ))
        strategy.retrainer = _retrainer
        strategy.per_fold_retrain = True
        strategy._train_cache = TrainWindowCache(_retrainer)
        _subheader("PER-FOLD RETRAIN MODE (swing): true out-of-sample — each fold "
                   "fits a fresh model on its own training window")

    from datetime import datetime, timedelta
    # Anchor fetch window to the same clock as fold boundaries (retrain_as_of),
    # so data and folds are always aligned. Previously used datetime.now() which
    # could diverge from the retrain_as_of()-anchored fold boundaries in run_cpcv,
    # silently truncating the final test fold when run near midnight or after the
    # sacred-holdout clamp activates. --as-of override is still respected.
    _cpcv_as_of_raw = getattr(args, "as_of", None)
    if _cpcv_as_of_raw:
        try:
            _cpcv_as_of = datetime.strptime(_cpcv_as_of_raw, "%Y-%m-%d")
            end_all = _cpcv_as_of
            logger.info("WF-R5: CPCV swing pinned to --as-of=%s", _cpcv_as_of.date())
        except Exception:
            from app.ml.retrain_config import retrain_as_of as _retrain_as_of_fn
            end_all = datetime.combine(_retrain_as_of_fn(), datetime.min.time())
    else:
        from app.ml.retrain_config import retrain_as_of as _retrain_as_of_fn
        end_all = datetime.combine(_retrain_as_of_fn(), datetime.min.time())
        logger.info("CPCV swing: fetch window anchored to retrain_as_of()=%s", end_all.date())
    start_all = end_all - timedelta(days=args.years * 365 + 30)
    strategy.fetch_data(start_all, end_all)
    cpcv_result = run_cpcv(
        strategy=strategy,
        purge_days=args.swing_purge_days,
        embargo_days=args.swing_embargo_days,
        n_folds=args.cpcv_k,
        n_paths=args.cpcv_paths,
        total_years=args.years,
        train_years=args.swing_train_years,
        allow_sacred_holdout=args.allow_sacred_holdout,
    )
    cpcv_result.print()
    _dump_cpcv_result_json(cpcv_result, "swing")
    # Alpha-v4 P0: optional purged sequential-WF baseline for trained models —
    # the honest, full-coverage complement to CPCV (sanity only, not a gate).
    if getattr(args, "sequential_baseline", False):
        try:
            from scripts.walkforward.sequential_baseline import (
                run_sequential_baseline, print_baseline_vs_cpcv,
            )
            wf = run_sequential_baseline(
                strategy, n_folds=args.cpcv_k, total_years=args.years,
                purge_days=args.swing_purge_days, embargo_days=args.swing_embargo_days,
                train_years=args.swing_train_years,
                allow_sacred_holdout=args.allow_sacred_holdout,
            )
            print_baseline_vs_cpcv(wf, cpcv_result)
        except Exception as _e:
            _warn(f"sequential-WF baseline skipped: {_e}")
    return cpcv_result


def _dump_cpcv_result_json(cpcv_result, model_type: str) -> None:
    """Persist a CPCV result summary (metrics + realized net-exposure) to a
    timestamped JSON under logs/, so arms are comparable and auditable after the
    run. Never raises (diagnostic side-effect only)."""
    try:
        import json as _json
        from pathlib import Path as _Path
        from datetime import datetime as _dt
        r = cpcv_result
        detail = r.gate_detail()
        _fl = lambda xs: [float(x) for x in (xs or [])]
        payload = {
            "model_type": getattr(r, "model_type", model_type),
            "n_folds": r.n_folds, "n_paths": r.n_paths,
            "n_combinations": r.n_combinations, "n_skipped": r.n_skipped,
            # Alpha-v4 P0: completeness + fold-coverage instrumentation.
            "n_overlap_bypassed": getattr(r, "n_overlap_bypassed", 0),
            "coverage": getattr(r, "coverage", None),
            "coverage_ok": getattr(r, "coverage_ok", True),
            "coverage_warnings": getattr(r, "coverage_warnings", []),
            "path_fold_members": getattr(r, "path_fold_members", []),
            "mean_sharpe": r.mean_sharpe, "std_sharpe": r.std_sharpe,
            "p5_sharpe": r.p5_sharpe, "p95_sharpe": r.p95_sharpe,
            "pct_positive": r.pct_positive, "path_sharpe_tstat": r.path_sharpe_tstat,
            "avg_deployment_pct": r.avg_deployment_pct,
            "avg_deployment_adjusted_sharpe": r.avg_deployment_adjusted_sharpe,
            "avg_profit_factor": r.avg_profit_factor, "avg_calmar": r.avg_calmar,
            "gate_passed": r.gate_passed(),
            "gate_failed": [k for k, (v, ok) in detail.items() if not ok],
            # Realized net-exposure (L/S arm) — the §3.1 validity signals.
            "net_exposure_captured": r.net_exposure_captured,
            "mean_net_beta": r.mean_net_beta, "p95_abs_net_beta": r.p95_abs_net_beta,
            "max_abs_net_beta": r.max_abs_net_beta, "net_beta_clean": r.net_beta_clean,
            "mean_net_dollar": r.mean_net_dollar, "max_abs_net_dollar": r.max_abs_net_dollar,
            "max_abs_net_sector": r.max_abs_net_sector, "mean_gross": r.mean_gross,
            "path_sharpes": _fl(r.path_sharpes),
            "path_mean_net_betas": _fl(r.path_mean_net_betas),
            "path_mean_net_dollars": _fl(r.path_mean_net_dollars),
            "path_mean_grosses": _fl(r.path_mean_grosses),
        }
        out_dir = _Path("logs"); out_dir.mkdir(exist_ok=True)
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"cpcv_{payload['model_type']}_{ts}.json"
        out.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  CPCV result JSON written: {out}")
    except Exception as _e:
        print(f"  (CPCV result JSON dump skipped: {_e})")


def _run_cpcv_intraday(args, symbols, intraday_ver, intraday_meta_model, earnings_cal, passed):
    """Run CPCV for intraday model and print the result."""
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.strategies.intraday import IntradayStrategy
    from app.utils.constants import RUSSELL_1000_TICKERS
    from datetime import date as _date, timedelta
    model, version = _load_model("intraday", version=intraday_ver)
    if model is None:
        _warn("CPCV: no intraday model found — skipping")
        return
    syms = symbols or list(RUSSELL_1000_TICKERS)
    intraday_scan_offsets = [12, 18, 24] if args.intraday_multi_scan else None
    strategy = IntradayStrategy(
        model=model, version=version, symbols=syms,
        meta_model=intraday_meta_model,
        pm_abstention_vix=args.pm_abstention_vix,
        pm_abstention_spy_ma_days=args.pm_abstention_spy_ma_days,
        scan_offsets=intraday_scan_offsets,
        transaction_cost_pct=args.intraday_cost_bps / 10_000 / 2,
        use_opportunity_score=args.pm_opportunity_score,
        use_dispersion_gate=args.dispersion_gate,
        earnings_blackout=earnings_cal,
    )
    strategy.model_type = "intraday"
    strategy.allow_in_sample = getattr(args, "allow_in_sample", False)

    # Phase 2: per-fold retraining (true out-of-sample) for intraday. Construct
    # the retrainer from the active INTRADAY_RETRAIN architecture so each fold
    # fits a fresh model on only its own training window.
    if getattr(args, "per_fold_retrain", False):
        from scripts.walkforward.retrainers import IntradayFoldRetrainer, TrainWindowCache
        from app.ml.retrain_config import INTRADAY_RETRAIN
        _retrainer = IntradayFoldRetrainer(base_config=dict(
            model_dir="app/ml/models",
            provider=INTRADAY_RETRAIN.get("provider", "alpaca"),
        ))
        strategy.retrainer = _retrainer
        strategy.per_fold_retrain = True
        strategy._train_cache = TrainWindowCache(_retrainer)
        # Feasibility guard: full Russell-1000 intraday per-fold is OOM-infeasible.
        # Force a reduced liquidity universe; warn loudly when full CPCV is asked.
        _top_n = getattr(args, "intraday_top_n", 150) or 150
        strategy.top_n_by_liquidity = _top_n
        _subheader("PER-FOLD RETRAIN MODE (intraday): true out-of-sample — each fold "
                   f"fits a fresh model on its own training window (universe capped to "
                   f"top-{_top_n} by liquidity)")
        if args.cpcv_k > 4:
            _warn(
                f"INTRADAY PER-FOLD FEASIBILITY: cpcv_k={args.cpcv_k} > 4. Full CPCV "
                f"retrains C(k,p) windows of 5-min features and is expensive/OOM-prone. "
                f"The design recommends reduced-universe WF with k=4. Proceeding, but "
                f"expect long runtime and high memory."
            )

    # Anchor fetch window to retrain_as_of() (same clock as fold boundaries in run_cpcv)
    # to prevent silent truncation of the final test fold.
    _cpcv_intraday_as_of = getattr(args, "as_of", None)
    if _cpcv_intraday_as_of:
        try:
            end_date = datetime.strptime(_cpcv_intraday_as_of, "%Y-%m-%d").date()
            logger.info("WF-R5: CPCV intraday pinned to --as-of=%s", end_date)
        except Exception:
            from app.ml.retrain_config import retrain_as_of as _retrain_as_of_fn
            end_date = _retrain_as_of_fn()
    else:
        from app.ml.retrain_config import retrain_as_of as _retrain_as_of_fn
        end_date = _retrain_as_of_fn()
        logger.info("CPCV intraday: fetch window anchored to retrain_as_of()=%s", end_date)
    start_date = end_date - timedelta(days=args.days + 10)
    strategy.fetch_data(start_date, end_date)
    cpcv_result = run_cpcv(
        strategy=strategy,
        purge_days=args.intraday_purge_days,
        embargo_days=args.intraday_embargo_days,
        n_folds=args.cpcv_k,
        n_paths=args.cpcv_paths,
        total_days=args.days,
        allow_sacred_holdout=args.allow_sacred_holdout,
    )
    cpcv_result.print()
    # Alpha-v4 P0: optional purged sequential-WF baseline (sanity only, not a gate).
    if getattr(args, "sequential_baseline", False):
        try:
            from scripts.walkforward.sequential_baseline import (
                run_sequential_baseline, print_baseline_vs_cpcv,
            )
            wf = run_sequential_baseline(
                strategy, n_folds=args.cpcv_k, total_days=args.days,
                purge_days=args.intraday_purge_days, embargo_days=args.intraday_embargo_days,
                allow_sacred_holdout=args.allow_sacred_holdout,
            )
            print_baseline_vs_cpcv(wf, cpcv_result)
        except Exception as _e:
            _warn(f"sequential-WF baseline skipped: {_e}")
    return cpcv_result


# ── Entry point ───────────────────────────────────────────────────────────────

def _cleanup_workers(signum=None, frame=None) -> None:
    """Kill any surviving multiprocessing worker processes on exit/interrupt."""
    for child in multiprocessing.active_children():
        try:
            child.terminate()
        except Exception:
            pass
    for child in multiprocessing.active_children():
        try:
            child.join(timeout=1.0)
            if child.is_alive():
                child.kill()
        except Exception:
            pass
    if signum is not None:
        raise SystemExit(130)


# Register cleanup on normal exit and on Ctrl-C / SIGTERM so WF workers never
# survive as orphaned processes when a run is interrupted.
atexit.register(_cleanup_workers)
signal.signal(signal.SIGINT, _cleanup_workers)
try:
    signal.signal(signal.SIGTERM, _cleanup_workers)
except (AttributeError, ValueError):
    pass  # SIGTERM unavailable in some Windows console contexts


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward Tier 3 validation")
    parser.add_argument("--model", choices=["swing", "intraday", "both"], default="both")
    parser.add_argument("--folds", type=int, default=3, help="Number of OOS folds")
    parser.add_argument("--years", type=int, default=5,
                        help="Total years of swing history (default: 5)")
    parser.add_argument("--days", type=int, default=730,
                        help="Total days of intraday history (default: 730)")
    parser.add_argument("--symbols", nargs="+", default=None, metavar="TICKER")
    parser.add_argument("--stop-mult", type=float, default=0.5,
                        help="ATR stop multiplier (default: 0.5)")
    parser.add_argument("--target-mult", type=float, default=1.5,
                        help="ATR target multiplier (default: 1.5)")
    parser.add_argument("--no-atr-stops", action="store_true", default=False,
                        help="Phase 4: disable ATR stops entirely. Positions hold to max_hold_bars."
                             " Overrides --stop-mult. Use with --no-pm-opportunity-score for clean signal isolation.")
    parser.add_argument("--rebalance-mode", action="store_true", default=False,
                        help="Phase RA: use top-N portfolio rebalance instead of signal entries")
    parser.add_argument("--rebalance-days", type=int, default=20,
                        help="Calendar days between rebalances (default: 20)")
    parser.add_argument("--rebalance-target-n", type=int, default=30,
                        help="Target number of positions in rebalance mode (default: 30)")
    parser.add_argument("--rebalance-sector-cap", type=float, default=0.30,
                        help="Max sector weight in rebalance portfolio (default: 0.30)")
    parser.add_argument("--rebalance-add-threshold", type=int, default=15,
                        help="Add a symbol if its rank <= this (default: 15)")
    parser.add_argument("--rebalance-drop-threshold", type=int, default=30,
                        help="Drop a symbol if its rank > this (default: 30)")
    parser.add_argument("--rebalance-min-adv", type=float, default=20_000_000.0,
                        help="Min avg daily dollar volume for liquidity filter (default: 20M)")
    parser.add_argument("--rebalance-regime-gate", action="store_true", default=False,
                        help="Phase RB.1: scale gross exposure by regime "
                             "(Bull=100%%, Neutral=70%%, Bear=30%%)")
    parser.add_argument("--rebalance-regime-vix-bull", type=float, default=20.0,
                        help="VIX threshold below which regime is BULL (default: 20)")
    parser.add_argument("--rebalance-regime-vix-bear", type=float, default=30.0,
                        help="VIX threshold at/above which regime is BEAR (default: 30)")
    parser.add_argument("--rebalance-regime-spy-ma-days", type=int, default=200,
                        help="SPY MA lookback for regime detection (default: 200)")
    parser.add_argument("--rebalance-hard-exit-bear", action="store_true", default=False,
                        help="LX6b: force-close ALL long positions when regime==BEAR at rebalance, "
                             "and skip new entries. Requires --rebalance-regime-gate.")
    parser.add_argument("--rebalance-flat-stop", type=float, default=0.0, dest="rebalance_flat_stop_pct",
                        help="LX8: per-position trailing stop pct from high-water-mark (e.g. 0.07 = 7%%). "
                             "0 = disabled (default). PIT-safe: trails from previous day's highest price.")
    parser.add_argument("--rebalance-inv-vol", action="store_true", default=False,
                        help="Phase RB.2: use inverse-volatility position sizing")
    parser.add_argument("--rebalance-inv-vol-lookback", type=int, default=20,
                        help="Lookback days for realized vol estimate (default: 20)")
    parser.add_argument("--rebalance-inv-vol-min-mult", type=float, default=0.5,
                        help="Min weight vs equal weight (default: 0.5x)")
    parser.add_argument("--rebalance-inv-vol-max-mult", type=float, default=2.0,
                        help="Max weight vs equal weight (default: 2.0x)")
    parser.add_argument("--rebalance-spy-vol-damper", action="store_true", default=False,
                        help="Phase 91: halve gross_mult when SPY 21d realized vol > 80th pct "
                             "of trailing 252d rolling vols. Defensive sizing only.")
    parser.add_argument("--rebalance-spy-vol-damper-scale", type=float, default=0.50,
                        help="Phase 91: gross_mult scale factor in high-vol regime (default: 0.50)")
    parser.add_argument("--rebalance-momentum-baseline", action="store_true", default=False,
                        help="CPCV diagnostic: replace v216 with 60d trailing-return momentum ranker "
                             "(same REBALANCE harness, same regime gate + inv-vol sizing)")
    parser.add_argument("--rebalance-ic-composite", action="store_true", default=False,
                        help="Phase 88: use IC-weighted deterministic factor composite (v219) "
                             "instead of ML model. Weights from 2026-05-24 IC audit h20 IC IR.")
    parser.add_argument("--rebalance-ic-composite-v220", action="store_true", default=False,
                        help="Phase 90: regime-conditional two-composite switch. Composite A "
                             "(momentum-tilted, breadth>60%%) vs Composite B (quality/v219). "
                             "5pp hysteresis deadband, EMA-smoothed breadth signal.")
    parser.add_argument("--rebalance-ic-composite-v221", action="store_true", default=False,
                        help="Phase 91: v219 with fundamentals down-weighted 70%%. Stateless. "
                             "Fixes quality-feature drag in rate-shock / blow-off folds.")
    parser.add_argument("--rebalance-ic-composite-v222", action="store_true", default=False,
                        help="Phase 92: hybrid — v221 base weights (fundamentals x0.30) + v220 "
                             "breadth-regime switch (bull>60%%: momentum tilt, shock<55%%: v221). "
                             "Best-of-both: shock-regime quality + bull-market momentum.")
    parser.add_argument("--rebalance-ic-composite-v224", action="store_true", default=False,
                        help="Phase v224: momentum-enhanced — v221 + mom_63d (3m momentum), "
                             "reduced contrarian weights (reversal/downtrend x0.5).")
    # Phase LX1 / B2 — long-side edge experiments
    parser.add_argument("--rebalance-lx1", action="store_true", default=False,
                        help="LX1: equal-weight 5-feature IC-validated scorer "
                             "(momentum_252d_ex1m, profit_margin, operating_margin, "
                             "price_to_52w_high, -pe_ratio). No learned weights. "
                             "Phase A1 out-of-sample IC-validated features only.")
    parser.add_argument("--rebalance-b2-baseline", action="store_true", default=False,
                        help="B2 naive baseline: all symbols score 0.0 (no selection). "
                             "Edge comes only from SPY>200d MA regime gate + inv-vol sizing. "
                             "Reference floor that all selection models must beat.")
    parser.add_argument("--rebalance-beta-neutralize", action="store_true", default=False,
                        help="LX9-A: beta-residualized LX1 scorer. Cross-sectionally regress "
                             "each feature vs trailing 252d beta to SPY, rank by residuals. "
                             "Addresses momentum crash / beta-in-the-book in VIX spikes.")
    # Phase 2 — L/S extension
    parser.add_argument("--enable-shorts", action="store_true", default=False,
                        help="Phase 2: add short sleeve to IC composite rebalance. "
                             "Shorts bottom-N from the same IC scorer. Asymmetric regime gate: "
                             "shorts scale UP in bear market, DOWN in bull.")
    parser.add_argument("--short-target-n", type=int, default=30,
                        help="Number of short positions (default: 30)")
    parser.add_argument("--short-bull-n", type=int, default=None,
                        help="Override short count in BULL regime (None = use --short-target-n)")
    parser.add_argument("--long-gross", type=float, default=0.95,
                        help="Long book gross as fraction of equity (default: 0.95 = 95%%)")
    parser.add_argument("--short-gross", type=float, default=0.55,
                        help="Short book gross as fraction of equity (default: 0.55 = 55%%)")
    parser.add_argument("--short-min-adv", type=float, default=50_000_000.0,
                        help="Minimum ADV for short candidates in $ (default: 50M; higher than long 20M)")
    parser.add_argument("--short-add-threshold", type=int, default=15,
                        help="Hysteresis: add short if rank-from-bottom <= this (default: 15)")
    parser.add_argument("--short-drop-threshold", type=int, default=30,
                        help="Hysteresis: drop short if rank-from-bottom > this (default: 30)")
    parser.add_argument("--net-sector-cap", action="store_true", default=False,
                        help="RANKER v2 §3.1 (Failure B fix): cap NET sector exposure "
                             "(long minus short per sector) instead of short COUNT per sector. "
                             "Lets the sector-concentrated R1K short tail fill where the longs "
                             "do not, so the short leg reaches its gross target. Requires "
                             "--enable-shorts.")
    parser.add_argument("--capture-net-exposure", action="store_true", default=False,
                        help="RANKER v2 §3.1: capture realized net beta / net dollar / net "
                             "sector per EOD (PURE-ADDITIVE diagnostic). Auto-on with "
                             "--enable-shorts; this flag forces it on explicitly.")
    parser.add_argument("--net-beta-lookback", type=int, default=60,
                        help="RANKER v2 §3.1: trailing days for PIT net-beta capture (default 60)")
    parser.add_argument("--spy-beta-hedge", action="store_true", default=False,
                        help="SPY beta hedge. Long-only: replace short book with a SPY short "
                             "sized to the long book's rolling 60d beta (Option A). With "
                             "--enable-shorts (RANKER v2 §3.1): OVERLAY a SPY short sized to the "
                             "RESIDUAL net beta of the whole book (longs minus single-name shorts) "
                             "to drive realized net beta -> 0. Requires --enable-shorts for the "
                             "overlay behavior.")
    parser.add_argument("--spy-beta-lookback", type=int, default=60,
                        help="Rolling days for beta computation (default: 60)")
    parser.add_argument("--spy-hedge-max-gross", type=float, default=0.30,
                        help="Max SPY short gross as fraction of equity (default: 0.30 = 30%%)")
    parser.add_argument("--spy-hedge-vix-lo", type=float, default=0.0,
                        help="v222: VIX gate low (hedge=0 below this). 0=disabled. Recommended: 14")
    parser.add_argument("--spy-hedge-vix-hi", type=float, default=0.0,
                        help="v222: VIX gate high (hedge=full above this). 0=disabled. Recommended: 22")
    parser.add_argument("--rebalance-atr-stops", action="store_true", default=False,
                        help="v222: set 1.5×ATR stop / 3.0×ATR target on rebalance long entries (fix payoff shape)")
    parser.add_argument("--rebalance-factor-stability-gate", action="store_true", default=False,
                        help="Phase 89: add cross-sectional factor stability gate (rolling realized "
                             "rank-IC filter). Multiplied on top of SPY+VIX gate. "
                             "Requires --rebalance-ic-composite or --rebalance-momentum-baseline.")
    parser.add_argument("--rebalance-factor-stability-lookback", type=int, default=63,
                        help="Phase 89: lookback days for rolling realized IC (default: 63)")
    parser.add_argument("--rebalance-factor-stability-ic-threshold", type=float, default=0.02,
                        help="Phase 89: IC threshold for gate on/off (default: 0.02)")
    parser.add_argument("--rebalance-dispersion-gate", action="store_true", default=False,
                        help="Phase 89 v2: cross-sectional return dispersion gate. "
                             "DRR=MAD(5d returns)/126d-baseline; throttles 1.5x-2.5x, floored at 10%%.")
    parser.add_argument("--rebalance-dispersion-k", type=int, default=5,
                        help="Phase 89 v2: dispersion return window in trading days (default: 5)")
    parser.add_argument("--rebalance-dispersion-L", type=int, default=126,
                        help="Phase 89 v2: dispersion baseline lookback in trading days (default: 126)")
    parser.add_argument("--meta-model-version", type=int, default=0,
                        help="Swing MetaLabelModel version to load (0 = none)")
    parser.add_argument("--intraday-meta-model-version", type=int, default=0,
                        help="Intraday MetaLabelModel version to load (0 = none)")
    parser.add_argument("--pm-abstention-vix", type=float, default=0.0,
                        help="PM abstention gate: skip entries when VIX >= this level (0 = off)")
    parser.add_argument("--pm-abstention-spy-ma-days", type=int, default=0,
                        help="PM abstention gate: skip entries when SPY < N-day SMA (0 = off)")
    parser.add_argument("--pm-abstention-spy-5d", action="store_true", default=False,
                        help="PM abstention gate: skip entries when SPY 5-day return <= 0 (Phase 55)")
    parser.add_argument("--intraday-multi-scan", action="store_true", default=False,
                        help="Phase 51: scan at offsets [12, 18, 24] instead of single scan at 12")
    parser.add_argument("--intraday-model-version", type=int, default=0,
                        help="Load a specific intraday model version (0 = active). "
                             "Use this to re-test historical/retired models.")
    parser.add_argument("--swing-model-version", type=int, default=0,
                        help="Load a specific swing model version (0 = active). "
                             "Use this to re-test historical/retired models.")
    parser.add_argument("--record-results", action="store_true", default=False,
                        help="Write tier3 Sharpe + gate result back to ModelVersion DB record")
    parser.add_argument("--as-of", type=str, default=None, metavar="YYYY-MM-DD",
                        help="WF-C2: pin fold boundary 'end' date for reproducibility. "
                             "When omitted, datetime.now() is used and a WARNING is printed.")
    parser.add_argument("--no-db-write", action="store_true", default=False,
                        help="WF-C audit: hard-block any DB writes from the WF harness. "
                             "Disables --record-results if both are passed. Walk-forward "
                             "should be purely read-only with respect to the app database.")
    parser.add_argument("--swing-cost-bps", type=float, default=5.0,
                        help="Round-trip transaction cost in bps for swing (default: 5bps)")
    parser.add_argument("--intraday-cost-bps", type=float, default=15.0,
                        help="Round-trip transaction cost in bps for intraday (default: 15bps)")
    parser.add_argument("--swing-purge-days", type=int, default=85,
                        help="Calendar days to skip between train_end and test_start for swing. "
                             "Must be >= 60d feature lookback + 20d label horizon + 5d buffer = 85d. "
                             "Old default was 15 (too short, caused train/test overlap). Default: 85")
    parser.add_argument("--swing-embargo-days", type=int, default=None,
                        help="WF-1: Calendar days to skip after test_end before next fold trains "
                             "(defaults to --swing-purge-days if not set)")
    parser.add_argument("--intraday-purge-days", type=int, default=2,
                        help="Trading days to skip between train_end and test_start for intraday "
                             "(prevents 2-day hold label leakage; default: 2)")
    parser.add_argument("--intraday-embargo-days", type=int, default=None,
                        help="WF-1: Trading days to skip after test_end before next fold trains "
                             "(defaults to --intraday-purge-days if not set)")
    parser.add_argument("--pm-opportunity-score", action="store_true", default=True,
                        help="WF-5a: apply PM continuous opportunity score gate in simulation "
                             "(score<0.35=skip, 0.35-0.65=cap at 2 candidates). On by default. "
                             "Use --no-pm-opportunity-score to disable.")
    parser.add_argument("--no-pm-opportunity-score", dest="pm_opportunity_score",
                        action="store_false",
                        help="WF-5a: disable the PM opportunity score gate.")
    parser.add_argument("--dispersion-gate", action="store_true", default=True,
                        help="WF-5a: skip intraday entries on days where cross-sectional return "
                             "dispersion < 0.5x rolling 60-day median. On by default. "
                             "Use --no-dispersion-gate to disable.")
    parser.add_argument("--no-dispersion-gate", dest="dispersion_gate",
                        action="store_false",
                        help="WF-5a: disable the dispersion gate.")
    # BUG-21: default True to match retrain_cron production gate. RSI/EMA prefilters
    # are a legacy artifact; the ML model should score the full universe.
    # Use --with-prefilters to restore legacy behavior for diagnostics.
    parser.add_argument("--no-prefilters", action="store_true", default=True,
                        help="(default ON) Bypass RSI/EMA pre-filters — ML model scores full universe. "
                             "Matches retrain_cron production gate. Use --with-prefilters to restore legacy.")
    parser.add_argument("--with-prefilters", dest="no_prefilters", action="store_false",
                        help="Legacy: re-enable RSI 40-70 and EMA20/50 pre-filters (diagnostic only).")
    parser.add_argument("--swing-train-years", type=int, default=None,
                        help="Phase 3b: rolling training window per fold — limit each fold's "
                             "training data to the N most recent years before train_end "
                             "(None = expanding window). Use 2 to exclude 2021-2022 bull regime.")
    parser.add_argument("--earnings-blackout", action="store_true", default=True,
                        help="WF-5a: skip entries within earnings blackout window "
                             "(swing: 3d before; intraday: 1d before / 3d after). On by default. "
                             "Use --no-earnings-blackout to disable.")
    parser.add_argument("--no-earnings-blackout", dest="earnings_blackout",
                        action="store_false",
                        help="WF-5a: disable the earnings blackout gate.")
    parser.add_argument("--regime-gate", action="store_true", default=False,
                        help="Phase R5: enable intraday regime gates (R5-A/B/C). Downloads SPY+VIX "
                             "history to build regime map. Off by default (requires WF-4 regime tagger).")
    parser.add_argument("--macro-gate", action="store_true", default=True,
                        help="WF-5a: block entries on FOMC/NFP/CPI/GDP days. On by default. "
                             "Use --no-macro-gate to disable.")
    parser.add_argument("--no-macro-gate", dest="macro_gate",
                        action="store_false",
                        help="WF-5a: disable the macro event gate.")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="Phase 1d: run N bootstrap iterations with ±30d fold perturbation "
                             "to quantify selection bias. 0 = disabled. 100 recommended.")
    # WF-3: CPCV flags
    parser.add_argument("--cpcv", action="store_true", default=False,
                        help="WF-3: run Combinatorial Purged Cross-Validation instead of "
                             "standard expanding walk-forward. Requires --cpcv-k and --cpcv-paths.")
    parser.add_argument("--cpcv-k", type=int, default=6,
                        help="WF-3: number of CPCV groups k (default: 6). "
                             "C(k, paths) combinations will be tested.")
    parser.add_argument("--cpcv-paths", type=int, default=2,
                        help="WF-3: number of test paths per combination (default: 2). "
                             "Larger = more combinations, slower but higher statistical power.")
    parser.add_argument("--sequential-baseline", action="store_true", default=False,
                        help="Alpha-v4 P0: also run a purged SEQUENTIAL walk-forward "
                             "(full-coverage, zero holes) alongside CPCV and print a "
                             "comparison. The honest baseline for TRAINED models; "
                             "sanity only (not a gate). Doubles runtime (per-fold "
                             "retrain runs twice), so off by default.")
    # P1: BenignGate — block test-fold entries on adverse-regime days
    parser.add_argument("--benign-gate", action="store_true", default=False,
                        help="P1: block new entries on days where PIT composite regime score < "
                             "BENIGN_REGIME_THRESHOLD (from macro_history.parquet). "
                             "Off by default. Adds ~1s to setup time.")
    parser.add_argument("--wf-max-symbols", type=int, default=0, metavar="N",
                        help="Speed-up: cap the swing WF universe to the top-N symbols by "
                             "average dollar volume. 0 = use all (default). "
                             "300 gives ~2-3x speedup with negligible Sharpe impact.")
    parser.add_argument("--feature-cache-workers", type=int, default=0,
                        help="Workers for per-fold feature pre-computation (0 = auto: "
                             "min(cpu_count, 12)). Ignored when --feature-cache-disable is set.")
    parser.add_argument("--feature-cache-executor", choices=["process", "thread"],
                        default="process",
                        help="Executor for feature cache build: 'process' (default, avoids GIL) "
                             "or 'thread' (lower overhead, useful on Windows with small universes).")
    parser.add_argument("--feature-cache-disable", action="store_true", default=False,
                        help="Disable the feature cache and fall back to per-day per-symbol "
                             "engineer_features() calls (original behavior). Use for debugging.")
    parser.add_argument("--sim-scan-interval-days", type=int, default=1,
                        help="Score symbols every N trading days instead of daily (default: 1). "
                             "N=5 gives ~5x sim speedup with minor strategy-behavior change. "
                             "Exits still run daily regardless of this setting.")
    # P3: DSR n-trials override and paper-gate mode
    parser.add_argument("--dsr-n", type=int, default=N_TRIALS_TESTED, metavar="N",
                        help=f"P3: number of model variants tried historically (for DSR correction). "
                             f"Default={N_TRIALS_TESTED}. Set to actual count (e.g. 200) for honest "
                             f"selection-bias correction. Warn if < 100.")
    parser.add_argument("--paper-gate", action="store_true", default=False,
                        help="P3: use paper-trading readiness gate instead of production gate. "
                             "Criteria: avg_sharpe > 0.50 and min_fold_sharpe > -0.40 (less strict). "
                             "DSR check still applies. Useful for deploy-to-paper decisions. "
                             "Ignored under GATE_MODE='significance' (use --gate-tier instead).")
    # Phase-4 FIX-1: significance-gate promotion tier. The CPCV gate is evaluated
    # at this tier so the CAPITAL tier is REACHABLE via an explicit promotion run
    # (default 'paper' — the conservative forward-validate tier). 'capital' requires
    # the stricter t-stat / n_folds / mean floors and (for event-sparse strategies
    # with no regime data) an explicit --regime-waiver-approved sign-off.
    parser.add_argument("--gate-tier", choices=["paper", "capital"], default="paper",
                        help="Phase-4: promotion tier for the significance gate "
                             "(GATE_MODE='significance'). 'paper' (default) = forward-"
                             "validate, no capital; 'capital' = real money, stricter "
                             "floors. No effect under GATE_MODE='mean_sharpe'.")
    parser.add_argument("--paper-confirmation", action="store_true", default=False,
                        help="Phase-4: documented live-paper confirmation for the "
                             "CAPITAL tier OR-path (CAPITAL_GATE_REQUIRE_PAPER_CONFIRMATION). "
                             "Only meaningful with --gate-tier capital.")
    parser.add_argument("--regime-waiver-approved", action="store_true", default=False,
                        help="Phase-4 FIX-2: explicit human sign-off to waive the "
                             "regime backstop on the CAPITAL tier when "
                             "worst_regime_sharpe is None due to event-sparsity. "
                             "PAPER auto-waives (flagged); CAPITAL never does without this.")
    # R5b: realism options
    parser.add_argument("--delisted-haircut", type=float, default=0.0,
                        help="R5b: when a position has no bar data at fold-end (delisted/halt), "
                             "exit at entry_price*(1-haircut) for longs (or entry_price*(1-haircut) "
                             "for shorts). Default 0.0 preserves prior behavior (P&L=0). "
                             "Recommended: 0.90 to model bankruptcy losses.")
    parser.add_argument("--per-fold-ic-weights", action="store_true", default=False,
                        help="R5b: compute Option-A per-fold IC weights from daily_ic.parquet "
                             "(filter date <= tr_end). Default off (uses static pre-2021-04-26 weights). "
                             "Falls back to static if parquet missing or has < 252 IC observations.")
    parser.add_argument("--daily-ic-parquet", type=str, default=None,
                        help="R5b: explicit path to daily_ic.parquet. If omitted, auto-detect newest "
                             "under data/diagnostics/feature_ic/.")
    # P0: sacred holdout bypass (one-shot promotion run only)
    parser.add_argument("--allow-sacred-holdout", action="store_true", default=False,
                        help="P0: bypass the SACRED_HOLDOUT_START guard. Use ONLY for the "
                             "single, final promotion-candidate evaluation. Logs a banner "
                             "warning. See app/ml/retrain_config.py.")
    # OOS-guard bypass: allows running WF/CPCV on test folds inside the model's
    # training period. Results are explicitly labeled in-sample and cannot promote
    # past gates. Use for diagnostics / baseline benchmarks only.
    parser.add_argument("--allow-in-sample", action="store_true", default=False,
                        help="Bypass the OOS guard — allow test folds inside the model's "
                             "training period. Results labeled in-sample; cannot promote "
                             "past gates. Use for diagnostics only.")
    parser.add_argument("--per-fold-retrain", action="store_true", default=False,
                        help="SWING + INTRADAY (Phase 1/2): retrain a fresh model inside "
                             "each WF/CPCV fold on only that fold's training window — true "
                             "out-of-sample. Default (off) uses the frozen-model "
                             "generalization test. For intraday this forces a reduced "
                             "liquidity universe (--intraday-top-n) for feasibility. "
                             "See docs/living/PIPELINE_ARCHITECTURE.md.")
    parser.add_argument("--intraday-top-n", type=int, default=150,
                        help="Intraday per-fold-retrain universe cap: keep top-N symbols "
                             "by 20-day median dollar volume before building the per-fold "
                             "matrix. Full Russell-1000 per-fold intraday is OOM-infeasible; "
                             "the design recommends reduced-universe WF (k=4), not full CPCV. "
                             "Default 150. Ignored in frozen intraday mode.")
    args = parser.parse_args()

    # WF-C2: parse --as-of into a date
    _as_of_date: Optional[date] = None
    if getattr(args, "as_of", None):
        try:
            _as_of_date = datetime.strptime(args.as_of, "%Y-%m-%d").date()
        except Exception:
            _err(f"--as-of must be YYYY-MM-DD (got {args.as_of!r})")
            return 2

    # WF-C audit: enforce read-only DB policy
    if getattr(args, "no_db_write", False) and getattr(args, "record_results", False):
        _warn("--no-db-write set: ignoring --record-results (WF is read-only)")
        args.record_results = False

    # P0: hard guard against using sacred holdout data in development WF runs.
    from app.ml.retrain_config import assert_no_sacred_holdout as _assert_holdout_wf
    # WF-R5: prefer --as-of for reproducibility (else fall back to today).
    _wf_end_today = _as_of_date if _as_of_date is not None else date.today()
    _assert_holdout_wf(
        _wf_end_today,
        allow_sacred_holdout=args.allow_sacred_holdout,
        context="walkforward_tier3.main",
    )

    # P3: warn if dsr_n is too low to give honest selection-bias correction
    if args.dsr_n < 100 and not args.paper_gate:
        _warn(
            f"--dsr-n={args.dsr_n} is likely understated. "
            f"Set to the actual number of model variants tried (typically 100-300) "
            f"for an honest DSR correction. Use --paper-gate to suppress this warning."
        )

    symbols = [s.upper() for s in args.symbols] if args.symbols else None
    passed = True

    meta_model = None
    if args.meta_model_version > 0:
        from app.ml.meta_model import MetaLabelModel
        try:
            meta_model = MetaLabelModel.load("app/ml/models", args.meta_model_version, model_type="swing")
            _ok(f"Swing MetaLabelModel v{args.meta_model_version} loaded")
        except Exception as e:
            _warn(f"Could not load swing MetaLabelModel v{args.meta_model_version}: {e}")

    intraday_meta_model = None
    if args.intraday_meta_model_version > 0:
        from app.ml.meta_model import MetaLabelModel
        try:
            intraday_meta_model = MetaLabelModel.load(
                "app/ml/models", args.intraday_meta_model_version, model_type="intraday"
            )
            _ok(f"Intraday MetaLabelModel v{args.intraday_meta_model_version} loaded")
        except Exception as e:
            _warn(f"Could not load intraday MetaLabelModel v{args.intraday_meta_model_version}: {e}")

    _header("MrTrader — Walk-Forward Tier 3 Validation")

    swing_ver = args.swing_model_version if args.swing_model_version > 0 else None
    intraday_ver = args.intraday_model_version if args.intraday_model_version > 0 else None

    # WF-5a: pre-fetch macro event blocked dates once for the full WF range
    macro_blocked_dates: Optional[set] = None
    if args.macro_gate:
        from datetime import datetime as _dt2
        # WF-R5: prefer --as-of for reproducibility (else fall back to today).
        _wf_end = _as_of_date if _as_of_date is not None else _dt2.now().date()
        _wf_start = _wf_end - timedelta(days=max(args.years * 365, args.days) + 30)
        try:
            from scripts.walkforward.macro_calendar import load_macro_blocked_dates
            macro_blocked_dates = load_macro_blocked_dates(_wf_start, _wf_end)
            _ok(f"Macro gate: {len(macro_blocked_dates)} blocked dates loaded "
                f"({_wf_start} → {_wf_end})")
        except Exception as _me:
            _warn(f"Macro gate calendar fetch failed: {_me} — macro gate disabled for this run")

    # P1: pre-compute benign-blocked dates from macro_history.parquet
    benign_blocked_dates: Optional[set] = None
    if getattr(args, "benign_gate", False):
        try:
            from app.ml.regime_score_pit import build_regime_score_map
            from app.ml.retrain_config import BENIGN_REGIME_THRESHOLD
            _score_map_raw = build_regime_score_map()
            # WF-R5 (FIX 2): shift regime score by one trading day to remove
            # one-bar look-ahead. compute_pit_regime_series scores day D using
            # SPY/VIX closes AT D — but the BenignGate consumes that score on
            # day D to influence trades executed at D's open. Shift forward so
            # day D's gate uses D-1's close-derived score.
            if _score_map_raw:
                import pandas as _pd_shift
                _s = _pd_shift.Series(_score_map_raw).sort_index()
                _s_shift = _s.shift(1).dropna()
                _score_map = {d: float(v) for d, v in _s_shift.items()}
            else:
                _score_map = {}
            benign_blocked_dates = {
                d for d, score in _score_map.items() if score < BENIGN_REGIME_THRESHOLD
            }
            _ok(f"BenignGate: {len(benign_blocked_dates)} adverse-regime dates "
                f"(threshold={BENIGN_REGIME_THRESHOLD})")
        except Exception as _be:
            _warn(f"BenignGate setup failed: {_be} — benign gate disabled for this run")

    # Phase 2b: pre-fetch earnings calendar once (used by both swing and intraday)
    earnings_cal: Optional[Dict[str, set]] = None
    if args.earnings_blackout:
        from datetime import datetime as _dt
        # WF-R5: prefer --as-of for reproducibility (else fall back to today).
        _end = _as_of_date if _as_of_date is not None else _dt.now().date()
        _start = _end - timedelta(days=max(args.years * 365, args.days) + 30)
        _syms = symbols or []
        if not _syms:
            # Default to SP100 + Russell1000 union if no explicit symbols
            try:
                from app.utils.constants import RUSSELL_1000_TICKERS
                _syms = list(RUSSELL_1000_TICKERS)
            except Exception:
                _syms = []
        if _syms:
            earnings_cal = _fetch_earnings_calendar(_syms, _start, _end)

    if args.model in ("swing", "both"):
        t0 = time.time()
        _swing_kwargs = dict(
            n_folds=args.folds,
            total_years=args.years,
            symbols=symbols,
            atr_stop_mult=args.stop_mult,
            atr_target_mult=args.target_mult,
            meta_model=meta_model,
            pm_abstention_vix=args.pm_abstention_vix,
            pm_abstention_spy_ma_days=args.pm_abstention_spy_ma_days,
            pm_abstention_spy_5d=args.pm_abstention_spy_5d,
            model_version=swing_ver,
            transaction_cost_pct=args.swing_cost_bps / 10_000 / 2,
            purge_days=args.swing_purge_days,
            embargo_days=args.swing_embargo_days,
            use_opportunity_score=args.pm_opportunity_score,
            no_prefilters=args.no_prefilters,
            train_years=args.swing_train_years,
            earnings_blackout=earnings_cal,
            macro_blocked_dates=macro_blocked_dates,
            benign_blocked_dates=benign_blocked_dates,
            wf_max_symbols=args.wf_max_symbols if args.wf_max_symbols > 0 else None,
            feature_cache_workers=args.feature_cache_workers,
            feature_cache_executor=args.feature_cache_executor,
            feature_cache_disable=args.feature_cache_disable,
            sim_scan_interval_days=args.sim_scan_interval_days,
            no_atr_stops=args.no_atr_stops,
            rebalance_mode=args.rebalance_mode,
            rebalance_days=args.rebalance_days,
            rebalance_target_n=args.rebalance_target_n,
            rebalance_sector_cap=args.rebalance_sector_cap,
            rebalance_add_threshold=args.rebalance_add_threshold,
            rebalance_drop_threshold=args.rebalance_drop_threshold,
            rebalance_min_adv=args.rebalance_min_adv,
            rebalance_regime_gate=args.rebalance_regime_gate,
            rebalance_regime_vix_bull=args.rebalance_regime_vix_bull,
            rebalance_regime_vix_bear=args.rebalance_regime_vix_bear,
            rebalance_regime_spy_ma_days=args.rebalance_regime_spy_ma_days,
            rebalance_inv_vol=args.rebalance_inv_vol,
            rebalance_inv_vol_lookback=args.rebalance_inv_vol_lookback,
            rebalance_inv_vol_min_mult=args.rebalance_inv_vol_min_mult,
            rebalance_inv_vol_max_mult=args.rebalance_inv_vol_max_mult,
            rebalance_spy_vol_damper=getattr(args, "rebalance_spy_vol_damper", False),
            rebalance_spy_vol_damper_scale=getattr(args, "rebalance_spy_vol_damper_scale", 0.50),
            rebalance_hard_exit_bear=getattr(args, "rebalance_hard_exit_bear", False),
        rebalance_flat_stop_pct=getattr(args, "rebalance_flat_stop_pct", 0.0),
            enable_shorts=getattr(args, "enable_shorts", False),
            short_target_n=getattr(args, "short_target_n", 30),
            short_bull_n=getattr(args, "short_bull_n", None),
            long_gross=getattr(args, "long_gross", 0.95),
            short_gross=getattr(args, "short_gross", 0.55),
            short_min_adv=getattr(args, "short_min_adv", 50_000_000.0),
            short_add_threshold=getattr(args, "short_add_threshold", 15),
            short_drop_threshold=getattr(args, "short_drop_threshold", 30),
            spy_beta_hedge=getattr(args, "spy_beta_hedge", False),
            spy_beta_lookback=getattr(args, "spy_beta_lookback", 60),
            spy_hedge_max_gross=getattr(args, "spy_hedge_max_gross", 0.30),
            spy_hedge_vix_lo=getattr(args, "spy_hedge_vix_lo", 0.0),
            spy_hedge_vix_hi=getattr(args, "spy_hedge_vix_hi", 0.0),
            rebalance_atr_stops=getattr(args, "rebalance_atr_stops", False),
            rebalance_factor_stability_gate=getattr(args, "rebalance_factor_stability_gate", False),
            rebalance_factor_stability_lookback=getattr(args, "rebalance_factor_stability_lookback", 63),
            rebalance_factor_stability_ic_threshold=getattr(args, "rebalance_factor_stability_ic_threshold", 0.02),
            rebalance_dispersion_gate=getattr(args, "rebalance_dispersion_gate", False),
            rebalance_dispersion_k=getattr(args, "rebalance_dispersion_k", 5),
            rebalance_dispersion_L=getattr(args, "rebalance_dispersion_L", 126),
            as_of=_as_of_date,
            delisted_haircut=getattr(args, "delisted_haircut", 0.0),
            per_fold_ic_weights=getattr(args, "per_fold_ic_weights", False),
            daily_ic_parquet=getattr(args, "daily_ic_parquet", None),
            allow_in_sample=getattr(args, "allow_in_sample", False),
        )
        if getattr(args, "rebalance_momentum_baseline", False):
            _swing_kwargs["scorer_instance"] = _momentum_baseline_scorer(lookback_days=60)
            print("  WF: momentum baseline mode — 60d trailing return ranker")
        elif getattr(args, "rebalance_ic_composite", False):
            from app.ml.factor_scorer import IcCompositeScorer
            _swing_kwargs["scorer_instance"] = IcCompositeScorer()
            print("  WF: IC composite mode — Phase 88 deterministic IC-weighted scorer (v219 weights)")
        elif getattr(args, "rebalance_ic_composite_v220", False):
            from app.ml.factor_scorer import IcCompositeV220Scorer
            _swing_kwargs["scorer_instance"] = IcCompositeV220Scorer()
            print("  WF: IC composite v220 mode — Phase 90 regime-conditional two-composite scorer")
        elif getattr(args, "rebalance_ic_composite_v221", False):
            from app.ml.factor_scorer import IcCompositeV221Scorer
            _swing_kwargs["scorer_instance"] = IcCompositeV221Scorer()
            print("  WF: IC composite v221 mode — Phase 91 fundamentals-downweighted scorer")
        elif getattr(args, "rebalance_ic_composite_v222", False):
            from app.ml.factor_scorer import IcCompositeV222Scorer
            _swing_kwargs["scorer_instance"] = IcCompositeV222Scorer()
            print("  WF: IC composite v222 mode — Phase 92 hybrid (v221 base + v220 breadth switch)")
        elif getattr(args, "rebalance_ic_composite_v224", False):
            from app.ml.factor_scorer import IcCompositeV224Scorer
            _swing_kwargs["scorer_instance"] = IcCompositeV224Scorer()
            print("  WF: IC composite v224 mode — momentum-enhanced (v221 + mom_63d, reduced contrarian)")
        elif getattr(args, "rebalance_lx1", False):
            from app.ml.factor_scorer import LX1EqualWeightScorer
            _swing_kwargs["scorer_instance"] = LX1EqualWeightScorer()
            print("  WF: LX1 mode — equal-weight 5-feature IC-validated scorer "
                  "(momentum_252d_ex1m, profit_margin, operating_margin, "
                  "price_to_52w_high, -pe_ratio)")
        elif getattr(args, "rebalance_beta_neutralize", False):
            from app.ml.factor_scorer import LX9ABetaNeutralScorer
            _swing_kwargs["scorer_instance"] = LX9ABetaNeutralScorer()
            print("  WF: LX9-A mode — beta-residualized LX1 scorer "
                  "(cross-sectional OLS residuals vs trailing 252d SPY beta)")
        elif getattr(args, "rebalance_b2_baseline", False):
            from app.ml.factor_scorer import B2EqualWeightUniverseScorer
            _swing_kwargs["scorer_instance"] = B2EqualWeightUniverseScorer()
            print("  WF: B2 baseline mode — no stock selection, regime gate + inv-vol only")
        # Per-fold-retrain short-circuit: the legacy run_swing_walkforward is a
        # frozen-model generalization test (is_true_walkforward=False, cannot promote)
        # that bypasses SwingStrategy/FoldEngine, so it has NO per-fold path and would
        # crash on a None trained_through. When --per-fold-retrain + --cpcv are set, skip
        # the legacy WF and run ONLY the genuine per-fold CPCV (the trustworthy number).
        if getattr(args, "per_fold_retrain", False) and args.cpcv:
            print("  Per-fold-retrain mode: skipping legacy frozen WF; running per-fold CPCV only.")
            _cpcv_res = _run_cpcv_swing(args, symbols, swing_ver, meta_model, earnings_cal, passed)
            # The per-fold CPCV is the ONLY signal in this branch — its gate result
            # IS the overall result. Previously the return value was ignored, so a
            # failed CPCV (or zero surviving paths) still printed "ALL GATES PASSED".
            if not _cpcv_swing_gate_ok(_cpcv_res, args):
                passed = False
        else:
            swing_report = run_swing_walkforward(**_swing_kwargs)
            swing_report.print(dsr_n=args.dsr_n, paper_gate=args.paper_gate)
            print(f"  Swing walk-forward elapsed: {time.time()-t0:.0f}s")
            if args.bootstrap > 0:
                _bootstrap_folds(run_swing_walkforward, n_bootstrap=args.bootstrap, **_swing_kwargs)
            if args.cpcv and args.model in ("swing", "both"):
                _cpcv_res = _run_cpcv_swing(args, symbols, swing_ver, meta_model, earnings_cal, passed)
                if not _cpcv_swing_gate_ok(_cpcv_res, args):
                    passed = False
            if not swing_report.gate_passed(dsr_n=args.dsr_n, paper_gate=args.paper_gate):
                passed = False
            # CRITICAL-1: block auto-promotion when the result is implausibly strong.
            # requires_human_review() does NOT affect gate_passed(); enforce it here at
            # the runner level so an implausible Sharpe cannot silently auto-promote.
            if swing_report.requires_human_review():
                from app.ml.retrain_config import SHARPE_IMPLAUSIBILITY_CEILING
                logger.warning(
                    "AUTO-PROMOTION BLOCKED: swing avg Sharpe %.3f > SHARPE_IMPLAUSIBILITY_CEILING %.1f. "
                    "Manual review required before promoting. See docs/living/PIPELINE_ARCHITECTURE.md §12 KL-6.",
                    swing_report.avg_sharpe, SHARPE_IMPLAUSIBILITY_CEILING,
                )
                passed = False
            if args.record_results and swing_report.folds:
                from app.ml.training import ModelTrainer
                loaded_ver = swing_report.folds[0].model_version if swing_report.folds else 0
                # C-1 fix: gate sentinel on combined decision — gate_passed AND not requires_human_review.
                # Using raw gate_passed() here would write the sentinel even when AUTO-PROMOTION BLOCKED.
                _swing_promotable = (swing_report.gate_passed(dsr_n=args.dsr_n, paper_gate=args.paper_gate)
                                     and not swing_report.requires_human_review())
                ModelTrainer.record_tier3_result(
                    version=loaded_ver,
                    avg_sharpe=swing_report.avg_sharpe,
                    fold_sharpes=[f.sharpe for f in swing_report.folds],
                    gate_passed=_swing_promotable,
                )

    if args.model in ("intraday", "both"):
        t0 = time.time()
        intraday_scan_offsets = [12, 18, 24] if args.intraday_multi_scan else None
        # Phase R5: pre-fetch regime map if --regime-gate enabled
        _regime_map: Optional[Dict] = None
        if getattr(args, "regime_gate", False):
            from datetime import datetime as _dt3
            _rg_end = _dt3.now().date()
            _rg_start = _rg_end - timedelta(days=args.days + 80)
            try:
                from scripts.walkforward.regime import load_regime_map
                _regime_map = load_regime_map(_rg_start, _rg_end)
                _ok(f"R5 regime map: {len(_regime_map)} dates tagged ({_rg_start} → {_rg_end})")
            except Exception as _re:
                _warn(f"R5 regime map fetch failed: {_re} — regime gate disabled for this run")
        _intraday_kwargs = dict(
            n_folds=args.folds,
            total_days=args.days,
            symbols=symbols,
            meta_model=intraday_meta_model,
            pm_abstention_vix=args.pm_abstention_vix,
            pm_abstention_spy_ma_days=args.pm_abstention_spy_ma_days,
            scan_offsets=intraday_scan_offsets,
            model_version=intraday_ver,
            transaction_cost_pct=args.intraday_cost_bps / 10_000 / 2,
            purge_days=args.intraday_purge_days,
            use_opportunity_score=args.pm_opportunity_score,
            use_dispersion_gate=args.dispersion_gate,
            earnings_blackout=earnings_cal,
            embargo_days=args.intraday_embargo_days,
            macro_blocked_dates=macro_blocked_dates,
            benign_blocked_dates=benign_blocked_dates,
            use_regime_gate=getattr(args, "regime_gate", False),
            regime_map=_regime_map,
            as_of=_as_of_date,
            allow_in_sample=getattr(args, "allow_in_sample", False),
        )
        # Per-fold-retrain short-circuit (mirror swing): run_intraday_walkforward
        # is a frozen-model generalization test (is_true_walkforward=False) with a
        # bespoke loop and NO per-fold path — it calls assert_model_oos on the
        # frozen model and would either run an in-sample frozen number or, with a
        # per-fold model, has no plumbing. When --per-fold-retrain + --cpcv are
        # set, skip the legacy WF and run ONLY the genuine per-fold CPCV.
        if getattr(args, "per_fold_retrain", False) and args.cpcv:
            print("  Per-fold-retrain mode (intraday): skipping legacy frozen WF; "
                  "running per-fold CPCV only.")
            _cpcv_intra_res = _run_cpcv_intraday(
                args, symbols, intraday_ver, intraday_meta_model, earnings_cal, passed)
            # The per-fold CPCV is the ONLY signal in this branch — its gate result
            # IS the intraday result (mirrors the swing per-fold short-circuit).
            if _cpcv_intra_res is None or not _cpcv_intra_res.gate_passed(
                    dsr_n=args.dsr_n, paper_gate=args.paper_gate,
                    tier=getattr(args, "gate_tier", "paper"),
                    paper_confirmation=getattr(args, "paper_confirmation", False),
                    regime_waiver_approved=getattr(args, "regime_waiver_approved", False)):
                passed = False
            print(f"  Intraday per-fold CPCV elapsed: {time.time()-t0:.0f}s")
        else:
            intraday_report = run_intraday_walkforward(**_intraday_kwargs)
            intraday_report.print(dsr_n=args.dsr_n, paper_gate=args.paper_gate)
            print(f"  Intraday walk-forward elapsed: {time.time()-t0:.0f}s")
            if args.bootstrap > 0:
                _bootstrap_folds(run_intraday_walkforward, n_bootstrap=args.bootstrap, **_intraday_kwargs)
            if args.cpcv and args.model in ("intraday", "both"):
                _run_cpcv_intraday(args, symbols, intraday_ver, intraday_meta_model, earnings_cal, passed)
            if not intraday_report.gate_passed(dsr_n=args.dsr_n, paper_gate=args.paper_gate):
                passed = False
            # CRITICAL-1: block auto-promotion when the result is implausibly strong.
            if intraday_report.requires_human_review():
                from app.ml.retrain_config import SHARPE_IMPLAUSIBILITY_CEILING
                logger.warning(
                    "AUTO-PROMOTION BLOCKED: intraday avg Sharpe %.3f > SHARPE_IMPLAUSIBILITY_CEILING %.1f. "
                    "Manual review required before promoting. See docs/living/PIPELINE_ARCHITECTURE.md §12 KL-6.",
                    intraday_report.avg_sharpe, SHARPE_IMPLAUSIBILITY_CEILING,
                )
                passed = False
            if args.record_results and intraday_report.folds:
                from app.ml.intraday_training import IntradayModelTrainer
                loaded_ver = intraday_report.folds[0].model_version if intraday_report.folds else 0
                # C-1 fix: gate sentinel on combined decision — mirrors swing fix above.
                _intraday_promotable = (intraday_report.gate_passed(dsr_n=args.dsr_n, paper_gate=args.paper_gate)
                                        and not intraday_report.requires_human_review())
                IntradayModelTrainer.record_tier3_result(
                    version=loaded_ver,
                    avg_sharpe=intraday_report.avg_sharpe,
                    fold_sharpes=[f.sharpe for f in intraday_report.folds],
                    gate_passed=_intraday_promotable,
                )

    print()
    if passed:
        _ok("ALL GATES PASSED — proceed to Phase 23 (paper trading)")
        return 0
    else:
        _warn("Gates not yet met — further model improvement needed before paper trading")
        return 1


if __name__ == "__main__":
    sys.exit(main())
