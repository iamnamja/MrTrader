"""
retrainers.py — per-fold model retraining for true out-of-sample WF/CPCV.

Phase 1 (swing only): when per-fold-retrain mode is on, each fold trains a fresh
model on ONLY its own training window [tr_start, tr_end] from the already-in-memory
bars (no re-fetch, no network). The fresh model's trained_through is set to tr_end,
guaranteeing the per-fold OOS guard passes when te_start > tr_end + purge_days.

TrainWindowCache dedups training across CPCV combinations that share a training
window — C(k, p) combos reuse the same k folds, so without caching the same
(tr_start, tr_end) window would be retrained many times.

See docs/living/PIPELINE_ARCHITECTURE.md KL-10.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TrainWindowCache:
    """Caches fitted models by (tr_start, tr_end) so CPCV combos sharing a
    training window retrain only once. Lifetime = one WF/CPCV run."""

    def __init__(self, retrainer: "SwingFoldRetrainer"):
        self._retrainer = retrainer
        self._cache: Dict[Tuple, object] = {}

    def get(self, tr_start, tr_end, *fit_inputs):
        key = (tr_start, tr_end)
        if key not in self._cache:
            self._cache[key] = self._retrainer.train_for_window(*fit_inputs, tr_start, tr_end)
        return self._cache[key]


class SwingFoldRetrainer:
    """Trains a fresh swing model on a single fold's training window.

    base_config carries the swing architecture (model_type, label_scheme,
    feature_keep_list, n_workers, ...). seed_base + a deterministic per-window
    offset gives a reproducible-yet-window-distinct random_state.
    """

    def __init__(self, base_config: Optional[dict] = None, seed_base: int = 42):
        self._base_config = dict(base_config or {})
        self._seed_base = seed_base

    def _seed_for(self, tr_start, tr_end) -> int:
        """Deterministic per-window seed. Same window → same seed; different
        tr_end → (almost surely) different seed."""
        return self._seed_base + tr_end.toordinal() % 100000

    def train_for_window(self, symbols_data, spy_prices, regime_map, tr_start, tr_end):
        from app.ml.training import ModelTrainer
        from app.ml.retrain_config import assert_no_sacred_holdout

        # Defense in depth — never train through the sacred holdout.
        assert_no_sacred_holdout(tr_end, context="per-fold-retrain")

        cfg = self._base_config
        trainer = ModelTrainer(
            model_dir=cfg.get("model_dir", "app/ml/models"),
            provider=cfg.get("provider", "polygon"),
            use_feature_store=False,  # per-fold: keep hermetic, no cross-window cache
            model_type=cfg.get("model_type", "lambdarank"),
            label_scheme=cfg.get("label_scheme", "lambdarank"),
            feature_keep_list=cfg.get("feature_keep_list"),
            n_workers=cfg.get("n_workers", 0),
            hpo_trials=0,  # PER_FOLD_SWING_HPO_TRIALS — no per-fold HPO
        )
        # Per-fold training never touches the network; mark so any secondary
        # sacred-holdout guard in the matrix builder respects the bypass flag.
        trainer._allow_sacred_holdout = False

        X, y, fnames, meta = trainer.build_train_matrix_for_window(
            symbols_data, tr_start, tr_end,
            spy_prices=spy_prices, regime_score_map=regime_map,
            fetch_fundamentals=cfg.get("fetch_fundamentals", False),
        )
        if len(X) == 0:
            raise RuntimeError(
                f"per-fold-retrain: no training samples in window "
                f"[{tr_start}, {tr_end}] — cannot fit a fold model."
            )
        model = trainer.fit_in_memory(
            X, y, fnames, meta, seed=self._seed_for(tr_start, tr_end)
        )
        model.trained_through = tr_end  # EXPLICIT — the training-window upper bound
        logger.info(
            "per-fold-retrain: fit fresh swing model for window [%s, %s] "
            "(%d samples, %d features, seed=%d, trained_through=%s)",
            tr_start, tr_end, len(X), len(fnames),
            self._seed_for(tr_start, tr_end), model.trained_through,
        )
        return model
