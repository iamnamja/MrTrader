"""
Phase 37 — Meta-Label Model (Expected R gate)

Secondary XGBoost regressor trained on historical Tier 3 trade outcomes.
Predicts Expected R (pnl_pct) for a proposed entry given its features.
Used in AgentSimulator to skip entries where predicted E[R] < threshold.

Training data: (feature_vector, pnl_pct) pairs from Tier 3 simulation
Prediction:    float E[R]; entry skipped if below min_expected_r

Design note: this is NOT re-ranking — it's an orthogonal quality gate.
The PM model ranks stocks by likelihood of hitting target; the meta-model
asks "given PM picked this, is today specifically a good day to enter?"
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_FNAME = "swing_meta_label_v{version}.pkl"
_DEFAULT_VERSION = 1


class MetaLabelModel:
    """
    XGBoost regression model that predicts expected trade P&L (pnl_pct).

    Trained on historical Tier 3 outcomes: for each completed trade,
    features at entry → actual pnl_pct. Predicts expected return for
    a proposed entry; entries below `min_expected_r` are skipped.
    """

    def __init__(self, min_expected_r: float = 0.002):
        self.min_expected_r = min_expected_r
        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.version = 0
        self._scaler = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
    ) -> dict:
        """Train on (features, pnl_pct) pairs. Returns OOS metrics."""
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_absolute_error

        self.feature_names = list(feature_names)

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        self._scaler = StandardScaler()
        X_tr_s = self._scaler.fit_transform(X_tr)
        X_val_s = self._scaler.transform(X_val)

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            tree_method="hist",
            early_stopping_rounds=20,
            verbosity=0,
        )
        self.model.fit(
            X_tr_s, y_tr,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )

        preds = self.model.predict(X_val_s)
        r2 = r2_score(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        corr = float(np.corrcoef(y_val, preds)[0, 1]) if len(y_val) > 2 else 0.0

        self.is_trained = True
        logger.info("MetaLabel trained: R²=%.3f MAE=%.4f corr=%.3f on %d val samples",
                    r2, mae, corr, len(y_val))
        return {"r2": r2, "mae": mae, "corr": corr, "n_train": len(X_tr), "n_val": len(X_val)}

    def predict_expected_r(self, features: dict) -> float:
        """Return predicted E[R] for a single trade's feature dict."""
        if not self.is_trained or self.model is None:
            return float("inf")
        try:
            x = np.array([[features.get(f, 0.0) for f in self.feature_names]], dtype=float)
            x = np.nan_to_num(x, nan=0.0)
            if self._scaler is not None:
                x = self._scaler.transform(x)
            return float(self.model.predict(x)[0])
        except Exception:
            return float("inf")  # fail open (don't block the trade)

    def should_enter(self, features: dict) -> bool:
        """Return True if predicted E[R] >= min_expected_r."""
        return self.predict_expected_r(features) >= self.min_expected_r

    def save(self, model_dir: str, version: int) -> str:
        path = Path(model_dir) / _MODEL_FNAME.format(version=version)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.version = version
        logger.info("MetaLabelModel saved -> %s", path)
        return str(path)

    @staticmethod
    def load(model_dir: str, version: int) -> "MetaLabelModel":
        path = Path(model_dir) / _MODEL_FNAME.format(version=version)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("MetaLabelModel loaded v%d from %s", version, path)
        return obj

    @staticmethod
    def load_latest(model_dir: str) -> Optional["MetaLabelModel"]:
        """Load the highest-version MetaLabelModel pkl in model_dir, or None."""
        files = sorted(Path(model_dir).glob("swing_meta_label_v*.pkl"))
        if not files:
            return None
        with open(files[-1], "rb") as f:
            obj = pickle.load(f)
        logger.info("MetaLabelModel loaded (latest) from %s", files[-1])
        return obj


def collect_trade_features(
    trades,
    symbols_data: dict,
    model_feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) training pairs from completed Tier 3 trades.

    For each trade: re-compute features at entry date → X row.
    y = trade.pnl_pct (actual outcome).

    Returns (X, y) arrays ready for MetaLabelModel.train().
    """
    from app.ml.features import FeatureEngineer
    from datetime import date

    fe = FeatureEngineer()
    rows_X: List[List[float]] = []
    rows_y: List[float] = []

    for t in trades:
        sym = t.symbol
        df = symbols_data.get(sym)
        if df is None:
            continue

        entry_dt = t.entry_date if isinstance(t.entry_date, date) else t.entry_date.date()
        idx = df.index.date if hasattr(df.index, "date") else __import__("pandas").DatetimeIndex(df.index).date
        window = df.loc[idx <= entry_dt]
        if len(window) < 60:
            continue

        try:
            feats = fe.engineer_features(sym, window, fetch_fundamentals=False, as_of_date=entry_dt, regime_score=0.5)
            if feats is None:
                continue
            row = [feats.get(f, 0.0) for f in model_feature_names]
            rows_X.append(row)
            rows_y.append(float(t.pnl_pct))
        except Exception:
            continue

    if not rows_X:
        return np.empty((0, len(model_feature_names))), np.empty(0)

    X = np.array(rows_X, dtype=float)
    y = np.array(rows_y, dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    return X, y
