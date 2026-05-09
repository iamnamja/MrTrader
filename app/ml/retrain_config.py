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

# USE_REALIZED_R_LABELS: switch intraday labeling from cross-sectional top-20%
#   to absolute realized-R threshold. Failed walk-forward gate (avg Sharpe -4.514,
#   AUC ~0.55). Structurally incompatible with cs_normalize. Keep False.
USE_REALIZED_R_LABELS: bool = False

# MIN_REALIZED_R: threshold for realized-R labeling (only applies when
#   USE_REALIZED_R_LABELS=True). 0.35 was tried — gate failed.
MIN_REALIZED_R: float = 0.35
