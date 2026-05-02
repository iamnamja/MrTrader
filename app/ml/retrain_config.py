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
    walk_forward_folds=3,       # run gate check before promoting
)

# Gate thresholds matching the manual training gates (docs/ML_EXPERIMENT_LOG.md)
SWING_GATE = dict(
    min_avg_sharpe=0.80,        # avg Sharpe across all folds must exceed this
    min_fold_sharpe=-0.30,      # no single fold may fall below this
)

# ── Intraday model (5-min bars, 50 features, XGBoost + LightGBM ensemble) ───
# Current champion: v29 — XGBoost HPO-tuned (n_est=577, depth=6, lr=0.0176),
# 50 features, SINGLE scan at bar 12 (60 min post-open), 730 days history.
# IMPORTANT: ENTRY_OFFSETS must stay [12] — multi-window (v30) failed the gate
# due to distribution mismatch (trained on 3 windows, deployed on 1).
INTRADAY_RETRAIN: dict = dict(
    days=730,
    fetch_spy=True,
    use_ranker=False,
    top_n_by_liquidity=None,    # None = full Russell 1000 universe
)

INTRADAY_GATE = dict(
    min_avg_sharpe=1.50,        # intraday gate is stricter than swing (Sharpe > 1.5)
    min_fold_sharpe=-0.30,      # no single fold below this floor
)
