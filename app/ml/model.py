"""
ML model wrapper for portfolio stock selection.

Supports XGBoost (default) and RandomForest.
Handles versioned save/load via pickle.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class PortfolioSelectorModel:
    """
    Binary classifier: predicts whether a stock will be a top performer (1)
    or poor performer (0) over the coming period.
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False

        if model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=400,
                max_depth=4,          # shallower = less overfitting with many features
                learning_rate=0.03,   # lower lr needs more trees but generalises better
                subsample=0.7,
                colsample_bytree=0.6, # use 60% of features per tree — reduces feature correlation
                min_child_weight=10,  # require ≥10 samples per leaf — prevents noise splits
                gamma=0.1,            # min loss reduction to make a split
                reg_alpha=0.1,        # L1 regularisation
                reg_lambda=1.5,       # L2 regularisation
                random_state=42,
                eval_metric="auc",
                verbosity=0,
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
    ) -> None:
        """
        Fit the model on pre-engineered features.

        Args:
            X:                    Feature matrix (n_samples, n_features).
            y:                    Binary labels (1 = good performer, 0 = poor).
            feature_names:        Optional list of feature names for logging.
            scale_pos_weight:     XGBoost class-weight ratio (n_neg / n_pos).
            X_val / y_val:        Optional validation set for early stopping.
            early_stopping_rounds: Stop if AUC doesn't improve for N rounds.
            sample_weight:        Per-sample importance weights (n_samples,).
                                  Combines recency, volatility regime, outcome
                                  margin, liquidity, and sector diversity weights.
        """
        logger.info(
            "Training %s model — %d samples, %d features",
            self.model_type, X.shape[0], X.shape[1],
        )

        if scale_pos_weight is not None and self.model_type == "xgboost":
            self.model.set_params(scale_pos_weight=scale_pos_weight)
            logger.info("scale_pos_weight=%.2f", scale_pos_weight)

        if sample_weight is not None:
            logger.info(
                "sample_weight: min=%.3f max=%.3f mean=%.3f",
                float(sample_weight.min()), float(sample_weight.max()), float(sample_weight.mean()),
            )

        X_scaled = self.scaler.fit_transform(X)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if self.model_type == "xgboost" and X_val is not None and y_val is not None:
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

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (class_predictions, class_1_probabilities).

        Raises:
            RuntimeError: if model has not been trained/loaded yet.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained or loaded yet.")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities

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
            pickle.dump({"feature_names": self.feature_names}, f)

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
