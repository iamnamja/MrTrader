"""
ML model wrapper for portfolio stock selection.

Supports XGBoost (default), RandomForest, and "ensemble" (XGBoost + LogisticRegression
soft-vote blend). The ensemble mode typically improves AUC by 0.01-0.03 over XGBoost
alone by reducing overfitting on noisy training labels.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRanker
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


class PortfolioSelectorModel:
    """
    Binary classifier: predicts whether a stock will be a top performer (1)
    or poor performer (0) over the coming period.
    """

    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False
        self.predict_threshold: float = 0.5  # tuned by tune_threshold()
        self._feature_weights: Optional[np.ndarray] = None  # MI-score weights
        self._lr_model: Optional[LogisticRegression] = None  # second estimator for ensemble
        self._lgbm_model = None  # second estimator for lgbm_ensemble
        self._is_regression = False  # set True when training on float return labels

        if model_type == "lgbm_ensemble":
            if not _LGBM_AVAILABLE:
                raise ImportError("lightgbm not installed — pip install lightgbm")
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.6,
                min_child_weight=10,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=42,
                eval_metric="auc",
                verbosity=0,
            )
            self._lgbm_model = LGBMClassifier(
                n_estimators=500,
                num_leaves=31,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.7,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                min_child_samples=50,
                random_state=42,
                verbose=-1,
            )
        elif model_type == "lgbm":
            if not _LGBM_AVAILABLE:
                raise ImportError("lightgbm not installed — pip install lightgbm")
            self.model = LGBMClassifier(
                n_estimators=600,
                num_leaves=31,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.7,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                min_child_samples=50,
                random_state=42,
                verbose=-1,
            )
        elif model_type in ("xgboost", "ensemble"):
            self.model = XGBClassifier(
                n_estimators=400,
                max_depth=4,          # shallower = less overfitting with many features
                learning_rate=0.03,   # lower lr needs more trees but generalises better
                subsample=0.7,
                colsample_bytree=0.6,  # use 60% of features per tree — reduces feature correlation
                min_child_weight=10,  # require ≥10 samples per leaf — prevents noise splits
                gamma=0.1,            # min loss reduction to make a split
                reg_alpha=0.1,        # L1 regularisation
                reg_lambda=1.5,       # L2 regularisation
                random_state=42,
                eval_metric="auc",
                verbosity=0,
            )
            if model_type == "ensemble":
                self._lr_model = LogisticRegression(
                    C=0.1,              # strong L2 — keeps weights small on noisy features
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs",
                )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )

    # ─── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_pos_weight: Optional[float] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        sample_weight: Optional[np.ndarray] = None,
        feature_weights: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        val_groups: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the model on pre-engineered features.

        groups / val_groups are accepted for API compatibility with LambdaRankModel
        but are ignored by classifier/regressor models.
        """
        logger.info(
            "Training %s model — %d samples, %d features",
            self.model_type, X.shape[0], X.shape[1],
        )

        # Regression mode: float labels (raw returns) → swap to XGBRegressor
        self._is_regression = (
            np.issubdtype(y.dtype, np.floating)
            and not np.all(np.isin(y, [0.0, 1.0]))
        )
        if self._is_regression:
            logger.info("Regression mode detected (float labels) — using XGBRegressor")
            self.model = XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
                random_state=42, verbosity=0,
            )
            scale_pos_weight = None  # not applicable for regression

        if scale_pos_weight is not None and self.model_type in ("xgboost", "ensemble", "lgbm_ensemble"):
            self.model.set_params(scale_pos_weight=scale_pos_weight)
            logger.info("scale_pos_weight=%.2f", scale_pos_weight)

        if sample_weight is not None:
            logger.info(
                "sample_weight: min=%.3f max=%.3f mean=%.3f",
                float(sample_weight.min()), float(sample_weight.max()), float(sample_weight.mean()),
            )

        # Apply MI-score feature weights: scale columns by normalized weights
        # before StandardScaler so the model sees higher-MI features with more signal.
        if feature_weights is not None and len(feature_weights) == X.shape[1]:
            self._feature_weights = feature_weights / (feature_weights.mean() + 1e-9)
            X = X * self._feature_weights
            if X_val is not None:
                X_val = X_val * self._feature_weights
            logger.info(
                "Feature weights applied: min=%.3f max=%.3f",
                float(self._feature_weights.min()), float(self._feature_weights.max()),
            )

        X_scaled = self.scaler.fit_transform(X)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if self.model_type == "lgbm" and X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            from lightgbm import early_stopping, log_evaluation
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[early_stopping(early_stopping_rounds, verbose=False), log_evaluation(-1)],
                **fit_kwargs,
            )
            logger.info("LGBM early stopping: best iteration = %s", getattr(self.model, "best_iteration_", "n/a"))
        elif self._is_regression and X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.set_params(early_stopping_rounds=early_stopping_rounds, eval_metric="rmse")
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
                **fit_kwargs,
            )
            logger.info("Regressor early stopping: best iteration = %s", getattr(self.model, "best_iteration", "n/a"))
        elif self.model_type in ("xgboost", "ensemble", "lgbm_ensemble") and X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
                **fit_kwargs,
            )
            logger.info("Early stopping: best iteration = %s", getattr(self.model, "best_iteration", "n/a"))
        else:
            self.model.fit(X_scaled, y, **fit_kwargs)

        # Train the LR blend model for ensemble mode
        if self.model_type == "ensemble" and self._lr_model is not None:
            lr_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
            self._lr_model.fit(X_scaled, y, **lr_kwargs)
            logger.info("Ensemble LR component trained")

        # Train the LGBM blend model for lgbm_ensemble mode
        if self.model_type == "lgbm_ensemble" and self._lgbm_model is not None:
            lgbm_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
            self._lgbm_model.fit(X_scaled, y, **lgbm_kwargs)
            logger.info("LGBM ensemble component trained")

        self.feature_names = feature_names
        self.is_trained = True

        # Log top features if available
        if feature_names and hasattr(self.model, "feature_importances_"):
            pairs = sorted(
                zip(feature_names, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
            logger.info("Top 5 features: %s", pairs[:5])

        logger.info("Training complete")

    # ─── Prediction ───────────────────────────────────────────────────────────

    def predict(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (class_predictions, class_1_probabilities).
        threshold defaults to self.predict_threshold (tuned or 0.5 if not tuned).
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained or loaded yet.")

        # Re-apply feature weights if they were set during training
        if self._feature_weights is not None and len(self._feature_weights) == X.shape[1]:
            X = X * self._feature_weights

        X_scaled = self.scaler.transform(X)

        if self._is_regression:
            # Regression: use predict(), normalize scores to [0,1] for compatibility
            raw_scores = self.model.predict(X_scaled).astype(float)
            lo, hi = raw_scores.min(), raw_scores.max()
            probabilities = (raw_scores - lo) / (hi - lo + 1e-9)
            t = threshold if threshold is not None else 0.5
            predictions = (probabilities >= t).astype(int)
            return predictions, probabilities

        primary_proba = self.model.predict_proba(X_scaled)[:, 1]

        if self.model_type == "ensemble" and self._lr_model is not None:
            lr_proba = self._lr_model.predict_proba(X_scaled)[:, 1]
            probabilities = 0.70 * primary_proba + 0.30 * lr_proba
        elif self.model_type == "lgbm_ensemble" and self._lgbm_model is not None:
            lgbm_proba = self._lgbm_model.predict_proba(X_scaled)[:, 1]
            probabilities = 0.50 * primary_proba + 0.50 * lgbm_proba
        else:
            probabilities = primary_proba

        t = threshold if threshold is not None else self.predict_threshold
        predictions = (probabilities >= t).astype(int)
        return predictions, probabilities

    def tune_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """
        Find the probability threshold that maximises F1 (or precision×recall
        harmonic mean) on a held-out validation set. Stores result in
        self.predict_threshold and returns it.
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        _, probabilities = self.predict(X_val, threshold=0.5)

        # Regression mode: convert float y_val to binary (top-20% = 1)
        if self._is_regression and np.issubdtype(y_val.dtype, np.floating):
            y_eval = (y_val >= np.percentile(y_val, 80)).astype(int)
        else:
            y_eval = y_val

        best_threshold, best_score = 0.5, 0.0

        for t in np.arange(0.20, 0.65, 0.05):
            preds = (probabilities >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_eval, preds, zero_division=0)
            else:
                prec = precision_score(y_eval, preds, zero_division=0)
                rec = recall_score(y_eval, preds, zero_division=0)
                score = 2 * prec * rec / (prec + rec + 1e-9)
            if score > best_score:
                best_score, best_threshold = score, float(t)

        self.predict_threshold = best_threshold
        logger.info(
            "Threshold tuned: %.2f  (best %s=%.4f)",
            best_threshold, metric, best_score,
        )
        return best_threshold

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str, version: int, model_name: str = "model") -> str:
        """Pickle model + scaler to directory. Returns model file path."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        model_path = Path(directory) / f"{model_name}_v{version}.pkl"
        scaler_path = Path(directory) / f"{model_name}_scaler_v{version}.pkl"
        meta_path = Path(directory) / f"{model_name}_meta_v{version}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(meta_path, "wb") as f:
            pickle.dump({
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "lr_model": self._lr_model,
                "lgbm_model": self._lgbm_model,
                "predict_threshold": self.predict_threshold,
                "feature_weights": self._feature_weights,
                "is_regression": self._is_regression,
            }, f)

        logger.info("Model v%d saved to %s", version, directory)
        return str(model_path)

    def load(self, directory: str, version: int, model_name: str = "model") -> None:
        """Load pickled model + scaler from directory."""
        # Support legacy filenames (model_v{n}.pkl) and new namespaced ones
        model_path = Path(directory) / f"{model_name}_v{version}.pkl"
        if not model_path.exists():
            model_path = Path(directory) / f"model_v{version}.pkl"
        scaler_path = Path(directory) / f"{model_name}_scaler_v{version}.pkl"
        if not scaler_path.exists():
            scaler_path = Path(directory) / f"scaler_v{version}.pkl"
        meta_path = Path(directory) / f"{model_name}_meta_v{version}.pkl"
        if not meta_path.exists():
            meta_path = Path(directory) / f"meta_v{version}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.feature_names = meta.get("feature_names")
                self.model_type = meta.get("model_type", self.model_type)
                self._lr_model = meta.get("lr_model")
                self._lgbm_model = meta.get("lgbm_model")
                self.predict_threshold = meta.get("predict_threshold", 0.5)
                self._feature_weights = meta.get("feature_weights")
                self._is_regression = meta.get("is_regression", False)

        self.is_trained = True
        logger.info("Model v%d loaded from %s", version, directory)

    def feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        """Return sorted (feature, importance) pairs, or None if unavailable."""
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return None
        names = self.feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
        pairs = sorted(
            zip(names, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs


# ── Stage feature sets ────────────────────────────────────────────────────────
#
# Three-stage split tuned for a 10-day holding horizon:
#   Stage 1 (weight 0.20) — slow quality gate: balance-sheet / analyst signals
#                           that change quarterly. Veto junk, not pick winners.
#   Stage 2 (weight 0.40) — near-term catalysts: features that are genuinely
#                           forward-looking at 10 days (earnings proximity,
#                           fresh earnings surprises, sentiment, short squeeze).
#   Stage 3 (weight 0.40) — entry timing: oscillators + volume patterns that
#                           pinpoint the right day to enter within a setup.
#
# Two-stage model keeps FUNDAMENTAL_FEATURES / TECHNICAL_FEATURES for backward
# compatibility with saved v30 models.

# Stage 1 — slow quality fundamentals (veto filter, low weight)
FUNDAMENTAL_FEATURES = {
    "pe_ratio", "pb_ratio", "profit_margin", "revenue_growth", "debt_to_equity",
    "insider_score", "fmp_inst_ownership_pct", "fmp_inst_change_pct",
    "fmp_analyst_upgrades_30d", "fmp_analyst_downgrades_30d", "fmp_analyst_momentum_30d",
    "sector_momentum",
    # Polygon quarterly financials — point-in-time, slow quarterly signal
    "fcf_margin", "operating_leverage", "rd_intensity",
}

# Stage 2 — 10-day catalysts (what can actually move the stock THIS week)
CATALYST_FEATURES = {
    "earnings_proximity_days", "earnings_surprise", "earnings_surprise_1q",
    "earnings_surprise_2q_avg", "days_since_earnings", "fmp_surprise_1q",
    "fmp_surprise_2q_avg", "fmp_days_since_earnings", "fmp_consecutive_beats",
    "fmp_revenue_surprise_1q", "short_interest_pct",
    "options_put_call_ratio", "options_iv_atm", "options_iv_premium",
    "news_sentiment_3d", "news_sentiment_7d", "news_article_count_7d",
    "news_sentiment_momentum", "regime_score",
}

# Stage 3 — entry timing (oscillators + volume, pinpoint the entry day)
TIMING_FEATURES = {
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_histogram",
    "stoch_k", "williams_r_14", "cci_20", "stochrsi_k", "stochrsi_d", "stochrsi_signal",
    "bb_position", "keltner_position", "mean_reversion_zscore",
    "atr_norm", "atr_trend", "volatility", "parkinson_vol", "vol_of_vol",
    "vol_percentile_52w", "vol_regime", "vol_expansion",
    "volume_ratio", "volume_trend", "volume_surge_3d", "vpt_momentum", "cmf_20",
    "momentum_5d", "momentum_20d", "price_change_pct",
    "consolidation_position", "range_expansion", "price_acceleration",
    "price_efficiency_20d", "vol_price_confirmation",
}

# Stage 2 of the legacy TwoStageModel — kept for backward compatibility
TECHNICAL_FEATURES = {
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_histogram",
    "ema_20", "ema_50", "price_above_ema20", "price_above_ema50", "price_above_ema200",
    "dist_from_ema200", "price_change_pct", "price_to_52w_high", "price_to_52w_low",
    "near_52w_high", "volume_ratio", "volume_trend", "uptrend", "downtrend",
    "volatility", "momentum_5d", "momentum_20d", "momentum_60d", "momentum_252d_ex1m",
    "atr_norm", "bb_position", "stoch_k", "adx_14", "rs_vs_spy", "consecutive_days",
    "rs_vs_spy_5d", "rs_vs_spy_10d", "rs_vs_spy_60d", "vol_percentile_52w",
    "vol_regime", "vol_of_vol", "atr_trend", "parkinson_vol", "vpt_momentum",
    "range_expansion", "vwap_distance_20d", "momentum_20d_sector_neutral",
    "momentum_60d_sector_neutral", "mean_reversion_zscore", "up_day_ratio_20d",
    "trend_consistency_63d", "vol_price_confirmation", "price_efficiency_20d",
    "williams_r_14", "cci_20", "price_acceleration", "stochrsi_k", "stochrsi_d",
    "stochrsi_signal", "keltner_position", "cmf_20", "dema_20_dist",
    "vol_expansion", "adx_slope", "volume_surge_3d", "consolidation_position",
}


class TwoStageModel:
    """
    Two-stage pipeline: quality screen (Stage 1) then timing filter (Stage 2).

    Stage 1 (fundamental quality): trained on fundamental/sentiment features.
    Stage 2 (technical timing):    trained on price/volume/momentum features.

    A stock gets label=1 only if BOTH stages predict positive.
    Blend weight controls how much each stage contributes to the final probability.
    """

    def __init__(self, model_type: str = "xgboost", blend: float = 0.5):
        self.model_type = model_type
        self.blend = blend  # weight on Stage 1; (1-blend) on Stage 2
        self.stage1 = PortfolioSelectorModel(model_type=model_type)
        self.stage2 = PortfolioSelectorModel(model_type=model_type)
        self.feature_names: Optional[List[str]] = None
        self.stage1_idx: List[int] = []
        self.stage2_idx: List[int] = []
        self.is_trained = False
        self.predict_threshold: float = 0.35

    def _split_feature_indices(self, feature_names: List[str]):
        self.stage1_idx = [i for i, n in enumerate(feature_names) if n in FUNDAMENTAL_FEATURES]
        self.stage2_idx = [i for i, n in enumerate(feature_names) if n in TECHNICAL_FEATURES]
        # Any feature not in either set goes to both
        neither = [i for i, n in enumerate(feature_names)
                   if n not in FUNDAMENTAL_FEATURES and n not in TECHNICAL_FEATURES]
        self.stage1_idx = sorted(set(self.stage1_idx + neither))
        self.stage2_idx = sorted(set(self.stage2_idx + neither))
        logger.info(
            "Two-stage split: stage1=%d fundamental features, stage2=%d technical features",
            len(self.stage1_idx), len(self.stage2_idx),
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_pos_weight: Optional[float] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        sample_weight: Optional[np.ndarray] = None,
        feature_weights: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        val_groups: Optional[np.ndarray] = None,
    ) -> None:
        self.feature_names = feature_names
        self._split_feature_indices(feature_names or [f"f{i}" for i in range(X.shape[1])])

        s1_names = [feature_names[i] for i in self.stage1_idx] if feature_names else None
        s2_names = [feature_names[i] for i in self.stage2_idx] if feature_names else None
        s1_weights = feature_weights[self.stage1_idx] if feature_weights is not None else None
        s2_weights = feature_weights[self.stage2_idx] if feature_weights is not None else None

        X_val1 = X_val[:, self.stage1_idx] if X_val is not None else None
        X_val2 = X_val[:, self.stage2_idx] if X_val is not None else None

        logger.info("Training Stage 1 (fundamental quality)...")
        self.stage1.train(
            X[:, self.stage1_idx], y, s1_names,
            scale_pos_weight=scale_pos_weight,
            X_val=X_val1, y_val=y_val,
            early_stopping_rounds=early_stopping_rounds,
            sample_weight=sample_weight,
            feature_weights=s1_weights,
        )

        logger.info("Training Stage 2 (technical timing)...")
        self.stage2.train(
            X[:, self.stage2_idx], y, s2_names,
            scale_pos_weight=scale_pos_weight,
            X_val=X_val2, y_val=y_val,
            early_stopping_rounds=early_stopping_rounds,
            sample_weight=sample_weight,
            feature_weights=s2_weights,
        )
        self.is_trained = True

    def predict(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("TwoStageModel has not been trained yet.")
        _, p1 = self.stage1.predict(X[:, self.stage1_idx], threshold=0.5)
        _, p2 = self.stage2.predict(X[:, self.stage2_idx], threshold=0.5)
        probabilities = self.blend * p1 + (1 - self.blend) * p2
        t = threshold if threshold is not None else self.predict_threshold
        predictions = (probabilities >= t).astype(int)
        return predictions, probabilities

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray, metric: str = "f1") -> float:
        from sklearn.metrics import f1_score
        _, probabilities = self.predict(X_val, threshold=0.5)
        best_t, best_score = 0.5, 0.0
        for t in np.arange(0.20, 0.65, 0.05):
            preds = (probabilities >= t).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_score:
                best_score, best_t = score, float(t)
        self.predict_threshold = best_t
        logger.info("Two-stage threshold tuned: %.2f (F1=%.4f)", best_t, best_score)
        return best_t

    def feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        pairs1 = self.stage1.feature_importance() or []
        pairs2 = self.stage2.feature_importance() or []
        combined = {n: v * self.blend for n, v in pairs1}
        for n, v in pairs2:
            combined[n] = combined.get(n, 0.0) + v * (1 - self.blend)
        return sorted(combined.items(), key=lambda x: x[1], reverse=True) if combined else None

    def save(self, directory: str, version: int, model_name: str = "model") -> str:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f"{model_name}_v{version}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("TwoStageModel v%d saved to %s", version, directory)
        return str(path)

    def load(self, directory: str, version: int, model_name: str = "model") -> None:
        path = Path(directory) / f"{model_name}_v{version}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        logger.info("TwoStageModel v%d loaded", version)


# ── Three-stage model ─────────────────────────────────────────────────────────

class ThreeStageModel:
    """
    Three-stage pipeline tuned for a 10-day holding horizon.

    Stage 1 — Quality gate   (weight 0.20): slow balance-sheet fundamentals.
              Veto junk companies; contributes little positive signal at 10d.
    Stage 2 — Catalyst       (weight 0.40): near-term catalysts that can move
              a stock within 10 days (earnings proximity, surprise, sentiment,
              short interest, options flow).
    Stage 3 — Entry timing   (weight 0.40): oscillators + volume patterns that
              pinpoint the right day to enter within a setup.

    Features not found in any set go to all three stages (shared context).
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        w1: float = 0.20,
        w2: float = 0.40,
        w3: float = 0.40,
    ):
        assert abs(w1 + w2 + w3 - 1.0) < 1e-6, "Stage weights must sum to 1.0"
        self.model_type = model_type
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.stage1 = PortfolioSelectorModel(model_type=model_type)
        self.stage2 = PortfolioSelectorModel(model_type=model_type)
        self.stage3 = PortfolioSelectorModel(model_type=model_type)
        self.feature_names: Optional[List[str]] = None
        self.stage1_idx: List[int] = []
        self.stage2_idx: List[int] = []
        self.stage3_idx: List[int] = []
        self.is_trained = False
        self.predict_threshold: float = 0.35

    def _split_feature_indices(self, feature_names: List[str]) -> None:
        s1 = set(FUNDAMENTAL_FEATURES)
        s2 = set(CATALYST_FEATURES)
        s3 = set(TIMING_FEATURES)
        shared = [
            i for i, n in enumerate(feature_names)
            if n not in s1 and n not in s2 and n not in s3
        ]
        self.stage1_idx = sorted({i for i, n in enumerate(feature_names) if n in s1} | set(shared))
        self.stage2_idx = sorted({i for i, n in enumerate(feature_names) if n in s2} | set(shared))
        self.stage3_idx = sorted({i for i, n in enumerate(feature_names) if n in s3} | set(shared))
        logger.info(
            "Three-stage split: stage1=%d quality, stage2=%d catalyst, stage3=%d timing  (shared=%d)",
            len(self.stage1_idx) - len(shared),
            len(self.stage2_idx) - len(shared),
            len(self.stage3_idx) - len(shared),
            len(shared),
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_pos_weight: Optional[float] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        sample_weight: Optional[np.ndarray] = None,
        feature_weights: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        val_groups: Optional[np.ndarray] = None,
    ) -> None:
        self.feature_names = feature_names
        self._split_feature_indices(feature_names or [f"f{i}" for i in range(X.shape[1])])

        def _names(idx): return [feature_names[i] for i in idx] if feature_names else None
        def _fw(idx): return feature_weights[idx] if feature_weights is not None else None
        def _val(idx): return X_val[:, idx] if X_val is not None else None

        for stage, idx, label in [
            (self.stage1, self.stage1_idx, "Stage 1 (quality gate)"),
            (self.stage2, self.stage2_idx, "Stage 2 (catalyst)"),
            (self.stage3, self.stage3_idx, "Stage 3 (entry timing)"),
        ]:
            logger.info("Training %s ...", label)
            stage.train(
                X[:, idx], y, _names(idx),
                scale_pos_weight=scale_pos_weight,
                X_val=_val(idx), y_val=y_val,
                early_stopping_rounds=early_stopping_rounds,
                sample_weight=sample_weight,
                feature_weights=_fw(idx),
            )

        self.is_trained = True

    def predict(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("ThreeStageModel has not been trained yet.")
        _, p1 = self.stage1.predict(X[:, self.stage1_idx], threshold=0.5)
        _, p2 = self.stage2.predict(X[:, self.stage2_idx], threshold=0.5)
        _, p3 = self.stage3.predict(X[:, self.stage3_idx], threshold=0.5)
        probabilities = self.w1 * p1 + self.w2 * p2 + self.w3 * p3
        t = threshold if threshold is not None else self.predict_threshold
        return (probabilities >= t).astype(int), probabilities

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray, metric: str = "f1") -> float:
        from sklearn.metrics import f1_score
        _, probabilities = self.predict(X_val, threshold=0.5)
        # Regression labels: convert to binary (top-20% = 1) before F1
        if np.issubdtype(y_val.dtype, np.floating) and not np.all(np.isin(y_val, [0.0, 1.0])):
            y_eval = (y_val >= np.percentile(y_val, 80)).astype(int)
        else:
            y_eval = y_val
        best_t, best_score = 0.5, 0.0
        for t in np.arange(0.20, 0.65, 0.05):
            preds = (probabilities >= t).astype(int)
            score = f1_score(y_eval, preds, zero_division=0)
            if score > best_score:
                best_score, best_t = score, float(t)
        self.predict_threshold = best_t
        logger.info("Three-stage threshold tuned: %.2f (F1=%.4f)", best_t, best_score)
        return best_t

    def feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        combined: dict = {}
        for stage, w in [(self.stage1, self.w1), (self.stage2, self.w2), (self.stage3, self.w3)]:
            for n, v in (stage.feature_importance() or []):
                combined[n] = combined.get(n, 0.0) + v * w
        return sorted(combined.items(), key=lambda x: x[1], reverse=True) if combined else None

    def save(self, directory: str, version: int, model_name: str = "model") -> str:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f"{model_name}_v{version}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ThreeStageModel v%d saved to %s", version, directory)
        return str(path)

    def load(self, directory: str, version: int, model_name: str = "model") -> None:
        path = Path(directory) / f"{model_name}_v{version}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        logger.info("ThreeStageModel v%d loaded", version)


# ── LambdaRank model ──────────────────────────────────────────────────────────

class LambdaRankModel:
    """
    Learning-to-rank model using LightGBM LambdaRank.

    Instead of binary classification, directly optimizes ranking quality
    (NDCG) within each window group. Stocks are assigned ordinal relevance
    labels 0-4 (quintiles by 10-day return within the window).

    Predict returns a normalized ranking score [0, 1] — higher = better rank.
    """

    def __init__(self):
        if not _LGBM_AVAILABLE:
            raise ImportError("lightgbm not installed — pip install lightgbm")
        self.model_type = "lambdarank"
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False
        self.predict_threshold: float = 0.5
        self._feature_weights: Optional[np.ndarray] = None
        self.model = LGBMRanker(
            objective="lambdarank",
            n_estimators=600,
            num_leaves=31,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            lambdarank_truncation_level=4,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_pos_weight: Optional[float] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        sample_weight: Optional[np.ndarray] = None,
        feature_weights: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        val_groups: Optional[np.ndarray] = None,
    ) -> None:
        if groups is None:
            raise ValueError("LambdaRankModel requires 'groups' (array of group sizes).")

        logger.info(
            "LambdaRank training — %d samples, %d features, %d groups",
            X.shape[0], X.shape[1], len(groups),
        )

        if feature_weights is not None and len(feature_weights) == X.shape[1]:
            self._feature_weights = feature_weights / (feature_weights.mean() + 1e-9)
            X = X * self._feature_weights
            if X_val is not None:
                X_val = X_val * self._feature_weights

        X_scaled = self.scaler.fit_transform(X)

        # LGBMRanker limits each validation query to ≤10k rows; our test set is a
        # single large group so we skip early stopping and just train all estimators.
        self.model.fit(X_scaled, y, group=groups)
        logger.info("LambdaRank fit done; best_iteration=%s", getattr(self.model, "best_iteration_", "n/a"))

        self.feature_names = feature_names
        self.is_trained = True
        logger.info("LambdaRank training complete")

    def predict(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("LambdaRankModel has not been trained yet.")

        if self._feature_weights is not None and len(self._feature_weights) == X.shape[1]:
            X = X * self._feature_weights

        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.predict(X_scaled).astype(float)
        lo, hi = raw_scores.min(), raw_scores.max()
        probabilities = (raw_scores - lo) / (hi - lo + 1e-9)
        t = threshold if threshold is not None else self.predict_threshold
        predictions = (probabilities >= t).astype(int)
        return predictions, probabilities

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray, metric: str = "f1") -> float:
        from sklearn.metrics import f1_score
        _, probabilities = self.predict(X_val, threshold=0.5)
        # y_val may be raw float returns (test set) or quintile ints
        if np.issubdtype(y_val.dtype, np.floating) and not np.all(np.isin(y_val, [0.0, 1.0, 2.0, 3.0, 4.0])):
            # raw return labels: top-20% = positive
            y_bin = (y_val >= np.percentile(y_val, 80)).astype(int)
        else:
            # quintile labels 0-4: top quintile (4) = positive
            y_bin = (y_val >= 4).astype(int)
        best_t, best_score = 0.5, 0.0
        for t in np.arange(0.20, 0.65, 0.05):
            preds = (probabilities >= t).astype(int)
            score = f1_score(y_bin, preds, zero_division=0)
            if score > best_score:
                best_score, best_t = score, float(t)
        self.predict_threshold = best_t
        logger.info("LambdaRank threshold tuned: %.2f (F1=%.4f)", best_t, best_score)
        return best_t

    def feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return None
        names = self.feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
        return sorted(zip(names, self.model.feature_importances_), key=lambda x: x[1], reverse=True)

    def save(self, directory: str, version: int, model_name: str = "model") -> str:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f"{model_name}_v{version}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("LambdaRankModel v%d saved to %s", version, directory)
        return str(path)

    def load(self, directory: str, version: int, model_name: str = "model") -> None:
        path = Path(directory) / f"{model_name}_v{version}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        logger.info("LambdaRankModel v%d loaded", version)
