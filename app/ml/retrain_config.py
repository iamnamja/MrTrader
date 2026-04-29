"""
Single source of truth for the weekly auto-retrain architecture.

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
"""

# ── Swing model (daily bars, ~84 features, XGBoost) ─────────────────────────
# Current champion: v135 — XGBoost, default hyperparams, 84 features
# HPO: 20 trials per weekly retrain (~3 min extra, finds slightly better params)
SWING_RETRAIN: dict = dict(
    model_type="xgboost",
    hpo_trials=20,
    fetch_fundamentals=False,   # avoid OOM on Windows (prefetch_fundamentals)
    n_workers=8,
)

# ── Intraday model (5-min bars, 50 features, XGBoost + LightGBM ensemble) ───
# Current champion: v29 — XGBoost HPO-tuned (n_est=577, depth=6, lr=0.0176),
# 50 features, single scan at bar 12 (60 min post-open), 730 days history
INTRADAY_RETRAIN: dict = dict(
    days=730,
    fetch_spy=True,
    use_ranker=False,
    top_n_by_liquidity=None,    # None = full Russell 1000 universe
)
