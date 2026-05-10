"""
Single source of truth for the weekly auto-retrain architecture and gate thresholds.

When you develop and validate a new model architecture (new model_type,
HPO settings, feature set, etc.), update SWING_RETRAIN or INTRADAY_RETRAIN
here and document the change in docs/ML_EXPERIMENT_LOG.md. The weekly
scheduled retrain will automatically use the updated config.

Rules:
  - Only change this after a new architecture passes the Tier 3 walk-forward gate
  - The config here should mirror the exact settings used for the current
    ACTIVE model version (so retrains produce the same architecture on fresh data)
  - hpo_trials controls Optuna hyperparameter search per weekly retrain;
    0 = use fixed params from model.py, >0 = search that many trials first
  - If the new model fails the walk-forward gate, the previous ACTIVE model
    is automatically restored — the new version is marked RETIRED
"""

# ── Retraining schedule ──────────────────────────────────────────────────────
# Day of week to run the weekly retrain (0=Monday … 6=Sunday).
# Retrain only fires on this day — server restarts on other days don't trigger it.
# Change to a different day if Wednesday conflicts with market events.
RETRAIN_WEEKDAY: int = 2  # Wednesday

# ── Swing model (daily bars, ~84 features, XGBoost) ─────────────────────────
# Current champion: v135 — XGBoost, default hyperparams, 84 features
# HPO: 20 trials per weekly retrain (~3 min extra, finds slightly better params)
SWING_RETRAIN: dict = dict(
    model_type="xgboost",
    hpo_trials=20,
    fetch_fundamentals=False,   # avoid OOM on Windows (prefetch_fundamentals)
    n_workers=8,
    walk_forward_folds=5,       # Phase 88: 5 folds (was 3) — one bad regime can't tank avg
    walk_forward_years=6,       # Phase 88: 6yr window → ~14mo per fold test
    exclude_risk_off_days=True,  # Phase R6b/88: down-weight (0.3×) not exclude RISK_OFF
    use_union_label=False,       # Phase 90 REVERTED: union label collapsed AUC to 0.50 OOS (5d features can't predict 15d outcome)
)

# ── Option C (label horizon) ─────────────────────────────────────────────────
# LABEL_HORIZON_DAYS: forward-return horizon for the swing label (default 5).
# Hypothesis C: 5d label is too noisy in trending regimes (AI rally 2023-24
# played out over 10-15 days). Larger horizon = thesis has time to develop.
#
# When != 5, training.py mutates module-level FORWARD_DAYS / STEP_DAYS /
# EMBARGO_WINDOWS at train_model entry. Set via env or by editing here.
#
# LABEL_ABS_HURDLE: optional ADDITIONAL absolute-return floor for cross-sectional
# label (in addition to the top-20% Sharpe-normalized cut). Currently swing has
# NO absolute hurdle; this is a new opt-in gate. Scales linearly with horizon
# (5d:0.003 → 10d:0.006 → 15d:0.009). Set to 0.0 to disable.
LABEL_HORIZON_DAYS: int = 5
LABEL_ABS_HURDLE_5D: float = 0.0  # 0.0 = disabled (default). 0.003 = 0.3% per 5d.

# ── Option B (regime-split training) ─────────────────────────────────────────
# REGIME_SPLIT_VIX_THRESHOLD: when > 0, train two separate models — one fit on
# rows where VIX < threshold (calm/trending) and one on VIX >= threshold
# (shock/volatile). Inference selects model based on current VIX level.
#
# Hypothesis B: fold 4 (tariff shock, high VIX) and fold 3 (AI rally, low VIX)
# need opposite mean-reversion vs momentum biases. A single model cannot serve
# both. Default 20.0 (S&P long-run median ~19). Range [15, 30] per spec.
#
# 0.0 = disabled (single-model path, current behaviour).
REGIME_SPLIT_VIX_THRESHOLD: float = 0.0

# Gate thresholds matching the manual training gates (docs/ML_EXPERIMENT_LOG.md)
SWING_GATE = dict(
    min_avg_sharpe=0.80,        # avg Sharpe across all folds must exceed this
    min_fold_sharpe=-0.30,      # no single fold may fall below this
)

# ── Intraday model (5-min bars, 50 features, XGBoost + LightGBM ensemble) ───
# Current champion: v29 — XGBoost HPO-tuned (n_est=577, depth=6, lr=0.0176),
# 56 features, SINGLE scan at bar 12 (60 min post-open), 730 days history.
# IMPORTANT: ENTRY_OFFSETS must stay [12] — multi-window (v30) failed the gate
# due to distribution mismatch (trained on 3 windows, deployed on 1).
INTRADAY_RETRAIN: dict = dict(
    days=730,
    fetch_spy=True,
    use_ranker=False,
    top_n_by_liquidity=None,    # None = full Russell 1000 universe
    exclude_risk_off_days=True,  # Phase R6/88: down-weight (0.3×) not exclude RISK_OFF
    wf_folds=5,                  # Phase 88: 5 folds (was 3)
)

INTRADAY_GATE = dict(
    min_avg_sharpe=1.00,        # recalibrated from 1.50 — best honest 365d result is +0.786 (v51 Run B)
    min_fold_sharpe=-0.30,      # no single fold below this floor
)

# ── Feature flags ─────────────────────────────────────────────────────────────
# USE_NIS_FEATURES: include NIS/macro LLM sentiment features in swing training.
#   False (default): NIS excluded — ~80% NaN creates a time-proxy (NaN = pre-May-2025
#   regime). XGBoost learns the time index, not the sentiment. AUC inflated on 2025 folds.
#   True: re-enable only after ≥2yr backfill via scripts/backfill_stock_nis_history.py
#   and scripts/backfill_macro_nis_llm.py. NIS still used at PM gate layer regardless.
#   Why: same pattern as USE_REALIZED_R_LABELS — code/infra preserved, behavior gated.
USE_NIS_FEATURES: bool = False

# ── Phase 93: FMP quarterly fundamentals ─────────────────────────────────────
# When True and data/fundamentals/fmp_fundamentals_history.parquet exists,
# training/inference loads PIT-correct quarterly fundamentals from FMP and
# OVERRIDES any value sourced from the older EDGAR annual parquet. Falls back
# to EDGAR (or zeros) when the FMP parquet is absent. Toggle False to A/B vs
# the EDGAR-only baseline without removing the parquet.
USE_FMP_FUNDAMENTALS: bool = True

# Quarters of history to fetch per symbol during full backfill.
# 40q ≈ 10y; 100q is FMP standard plan max.
FMP_QUARTERLY_LOOKBACK_QUARTERS: int = 40

# USE_REALIZED_R_LABELS: switch intraday labeling from cross-sectional top-20%
#   to absolute realized-R threshold. Failed walk-forward gate (avg Sharpe -4.514,
#   AUC ~0.55). Structurally incompatible with cs_normalize. Keep False.
USE_REALIZED_R_LABELS: bool = False

# MIN_REALIZED_R: threshold for realized-R labeling (only applies when
#   USE_REALIZED_R_LABELS=True). 0.35 was tried — gate failed.
MIN_REALIZED_R: float = 0.35


# ── P1: BenignModel — regime-filtered training ────────────────────────────────
# When BENIGN_FILTER_ENABLED=True, the training pipeline filters out any
# (symbol, window) row where the PIT composite regime score < BENIGN_REGIME_THRESHOLD.
# This prevents the model from learning patterns that only appear during bear markets
# (where our signals have no demonstrated edge).
#
# Default: False — existing retrain behaviour is unchanged until explicitly enabled.
# Enable via CLI flag: --benign-model (sets this to True for the run).
BENIGN_FILTER_ENABLED: bool = False

# Regime threshold: rows with composite_score < this are excluded from training.
# 0.5 = at least 3 of 5 components must be bullish (spy_ma50, spy_ma200, vix_term,
# breadth, credit). Historically filters ~21% of days (2018-2026).
BENIGN_REGIME_THRESHOLD: float = 0.5

# P1 feature keep-lists. When BENIGN_FILTER_ENABLED=True, the worker retains ONLY
# features in the relevant list after computing the full feature dict.
# Verified against app/ml/features.py output — names must match exactly.
BENIGN_SWING_FEATURES: tuple = (
    # Momentum (8)
    "rsi_14", "macd_histogram", "momentum_5d", "momentum_20d", "momentum_60d",
    "momentum_252d_ex1m", "price_change_pct", "consecutive_days",
    # Trend (7)
    "price_above_ema20", "price_above_ema50", "ema20_dist", "ema50_dist",
    "adx_14", "uptrend", "downtrend",
    # Volatility (5)
    "atr_norm", "volatility", "vol_percentile_52w", "vol_regime", "parkinson_vol",
    # Volume (3)
    "volume_ratio", "volume_trend", "vwap_distance_20d",
    # 52-week position (2)
    "price_to_52w_high", "price_to_52w_low",
    # Regime / macro (5)
    "vix_term_ratio", "spy_above_ma50", "spy_above_ma200",
    "breadth_rsp_spy_ratio_20d", "credit_hyg_ief_20d",
    # Sector (2)
    "sector_momentum", "sector_momentum_5d",
    # Fundamentals (3) — only used when FMP parquet present
    "pe_ratio", "profit_margin", "revenue_growth",
)  # 35 total

BENIGN_INTRADAY_FEATURES: tuple = (
    # Momentum (6)
    "rsi_14", "momentum_5d", "momentum_20d", "price_change_pct",
    "consecutive_days", "adx_14",
    # Trend (5)
    "price_above_ema20", "price_above_ema50", "ema20_dist", "uptrend", "downtrend",
    # Volatility (4)
    "atr_norm", "volatility", "vol_percentile_52w", "parkinson_vol",
    # Volume (3)
    "volume_ratio", "volume_trend", "vwap_distance_20d",
    # 52-week position (2)
    "price_to_52w_high", "near_52w_high",
    # Regime / macro (5)
    "vix_term_ratio", "spy_above_ma50", "spy_above_ma200",
    "breadth_rsp_spy_ratio_20d", "credit_hyg_ief_20d",
)  # 25 total


# ── P0: Sacred holdout enforcement ────────────────────────────────────────────
# SACRED_HOLDOUT_START defines the boundary date — data ON or AFTER this date is
# RESERVED for the single, final promotion-candidate evaluation. It must NEVER
# be touched during development, tuning, walk-forward, CPCV, or any iterative
# experiment. Doing so reintroduces selection bias and invalidates every honest
# baseline number we have.
#
# Enforcement (defense in depth — guard fires at multiple layers):
#   1. ModelTrainer.train_model() before fetching bars
#   2. ModelTrainer._build_rolling_matrix() on the assembled date spine
#   3. scripts/walkforward_tier3.py main() before kicking off any folds
#   4. scripts/walkforward/cpcv.py run_cpcv() when computing end_all
#   5. scripts/walkforward/engine.py FoldEngine.run() when computing end_all
#
# Bypass: pass `allow_sacred_holdout=True` (Python) or `--allow-sacred-holdout`
# (CLI). Bypass logs a prominent banner warning. Use ONLY for the final
# promotion candidate run, ONCE. Do not develop with this flag set.
#
# Date format is ISO (YYYY-MM-DD). Boundary is INCLUSIVE: end_date == this date
# is rejected (the holdout day itself is sacred).
SACRED_HOLDOUT_START: str = "2025-11-09"


def _parse_sacred_holdout_start():
    from datetime import date as _date
    y, m, d = SACRED_HOLDOUT_START.split("-")
    return _date(int(y), int(m), int(d))


def assert_no_sacred_holdout(end_date, *, allow_sacred_holdout: bool = False,
                             context: str = "training") -> None:
    """Hard-fail guard: reject any run whose data window reaches the sacred holdout.

    Raises RuntimeError if `end_date >= SACRED_HOLDOUT_START` and bypass not set.
    Logs a prominent banner warning if bypass IS set.

    Args:
        end_date: a `datetime.date`, `datetime.datetime`, pandas Timestamp,
                  or ISO date string representing the END of the data window.
        allow_sacred_holdout: explicit bypass for the one-shot promotion run.
        context: free-text label included in error/warning messages
                 ("training", "walk-forward", "cpcv", etc.).
    """
    import logging as _logging
    from datetime import date as _date, datetime as _datetime

    if end_date is None:
        return  # nothing to check

    # Coerce input → date
    if isinstance(end_date, str):
        end_d = _date.fromisoformat(end_date[:10])
    elif isinstance(end_date, _datetime):
        end_d = end_date.date()
    elif isinstance(end_date, _date):
        end_d = end_date
    else:
        # pandas.Timestamp and similar — has .date()
        end_d = end_date.date() if hasattr(end_date, "date") else end_date

    boundary = _parse_sacred_holdout_start()
    if end_d < boundary:
        return  # safe — entirely before holdout

    if allow_sacred_holdout:
        log = _logging.getLogger(__name__)
        banner = "=" * 72
        log.warning(banner)
        log.warning("SACRED HOLDOUT BYPASS ACTIVE — context=%s", context)
        log.warning("end_date=%s reaches/exceeds SACRED_HOLDOUT_START=%s",
                    end_d.isoformat(), boundary.isoformat())
        log.warning("This must ONLY be the final, one-shot promotion-candidate run.")
        log.warning("If this is anything else, ABORT NOW.")
        log.warning(banner)
        return

    raise RuntimeError(
        f"SACRED HOLDOUT VIOLATION ({context}): end_date={end_d.isoformat()} "
        f"reaches or exceeds SACRED_HOLDOUT_START={boundary.isoformat()}. "
        f"Data on/after this date is reserved for the single, final "
        f"promotion-candidate evaluation. To intentionally bypass for that "
        f"one-shot run, pass allow_sacred_holdout=True (or --allow-sacred-holdout). "
        f"See app/ml/retrain_config.py for details."
    )
