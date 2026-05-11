"""MVP logistic regime classifier (R5).

Predicts a tradeable-regime probability from 5 macro features:
  1. SPY 20d log return
  2. SPY close / SPY 200d MA - 1  (centred ratio)
  3. VIX level
  4. VIX 20d percentile (rolling 252d)
  5. HYG 20d log return

Label: 1 if (SPY > SPY_200d_MA) AND (VIX < vix_threshold), else 0.

Output: predict_proba()[:,1] used as a sizing multiplier in PM:
    size *= max(REGIME_FLOOR, regime_prob)

Train on 2015-2023, validate on 2024. Hold 2025+ as untouched OOS.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "spy_20d_logret",
    "spy_ma200_ratio",
    "vix_level",
    "vix_20d_pct_rank",
    "hyg_20d_logret",
]

REGIME_FLOOR = 0.25  # minimum sizing multiplier even in adverse regime


def build_regime_features(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    hyg_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build 5-feature regime panel from daily OHLCV DataFrames.

    All DataFrames must have a DatetimeIndex and a 'Close' column.
    Returns a DataFrame indexed by date with columns = FEATURE_NAMES.
    No lookahead: features at index t use only data up to and including t.
    """
    spy_close = spy_df["Close"].copy()
    vix_close = vix_df["Close"].copy()
    hyg_close = hyg_df["Close"].copy()

    # 1. SPY 20d log return
    spy_20d = np.log(spy_close / spy_close.shift(20))

    # 2. SPY / SPY_200d_MA - 1
    spy_ma200 = spy_close.rolling(200, min_periods=150).mean()
    spy_ma200_ratio = (spy_close / spy_ma200) - 1.0

    # 3. VIX level
    vix_level = vix_close

    # 4. VIX 20d percentile (rolling 252d rank)
    vix_pct_rank = vix_close.rolling(252, min_periods=63).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]),
        raw=False,
    )

    # 5. HYG 20d log return
    hyg_20d = np.log(hyg_close / hyg_close.shift(20))

    features = pd.DataFrame({
        "spy_20d_logret": spy_20d,
        "spy_ma200_ratio": spy_ma200_ratio,
        "vix_level": vix_level,
        "vix_20d_pct_rank": vix_pct_rank,
        "hyg_20d_logret": hyg_20d,
    })
    features.index = pd.to_datetime(features.index).normalize()
    return features.dropna()


def build_regime_labels(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    vix_threshold: float = 25.0,
) -> pd.Series:
    """Build binary regime label: 1 = tradeable, 0 = adverse.

    Label at day t: 1 if SPY_close > SPY_200d_MA AND VIX < vix_threshold.
    No lookahead — MA and VIX are point-in-time as of day t.
    """
    spy_close = spy_df["Close"].copy()
    vix_close = vix_df["Close"].copy()

    spy_ma200 = spy_close.rolling(200, min_periods=150).mean()
    spy_above_ma = spy_close > spy_ma200
    vix_below_thresh = vix_close < vix_threshold

    label = (spy_above_ma & vix_below_thresh).astype(int)
    label.index = pd.to_datetime(label.index).normalize()
    label.name = "regime_label"
    return label


class RegimeClassifier:
    """Logistic regression regime classifier.

    Inputs:  5 macro features (FEATURE_NAMES)
    Outputs: predict_proba()[:,1] — probability of tradeable regime
    """

    def __init__(self, vix_threshold: float = 25.0):
        self.vix_threshold = vix_threshold
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self._is_fitted = False
        self._train_start: Optional[str] = None
        self._train_end: Optional[str] = None
        self._val_auc: Optional[float] = None
        self._val_brier: Optional[float] = None
        self._label_mean: Optional[float] = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        train_end: str = "2023-12-31",
    ) -> "RegimeClassifier":
        """Fit on rows up to train_end (inclusive)."""
        idx = features.index.normalize()
        mask = idx <= pd.Timestamp(train_end)
        X_train = features.loc[mask][FEATURE_NAMES].values
        y_train = labels.reindex(features.index[mask]).values

        valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train, y_train = X_train[valid], y_train[valid]

        self._scaler.fit(X_train)
        self._model.fit(self._scaler.transform(X_train), y_train)
        self._is_fitted = True
        self._train_start = str(features.index[mask][0].date())
        self._train_end = train_end
        self._label_mean = float(y_train.mean())

        logger.info(
            "RegimeClassifier fitted on %d samples (%s to %s). Label mean=%.2f",
            len(X_train), self._train_start, train_end, self._label_mean,
        )
        return self

    def predict_proba_series(self, features: pd.DataFrame) -> pd.Series:
        """Return regime probability for each row in features."""
        if not self._is_fitted:
            raise RuntimeError("RegimeClassifier not fitted — call fit() first.")
        X = features[FEATURE_NAMES].values
        valid_mask = ~np.isnan(X).any(axis=1)
        probs = np.full(len(X), 0.5)  # default: neutral for NaN rows
        if valid_mask.any():
            probs[valid_mask] = self._model.predict_proba(
                self._scaler.transform(X[valid_mask])
            )[:, 1]
        return pd.Series(probs, index=features.index, name="regime_prob")

    def predict_proba_date(self, row: pd.Series) -> float:
        """Return regime probability for a single feature row."""
        if not self._is_fitted:
            raise RuntimeError("RegimeClassifier not fitted — call fit() first.")
        X = np.array([[row[f] for f in FEATURE_NAMES]])
        if np.isnan(X).any():
            return 0.5
        return float(self._model.predict_proba(self._scaler.transform(X))[0, 1])

    def sizing_weight(self, regime_prob: float) -> float:
        """Convert regime probability to a PM sizing multiplier ∈ [REGIME_FLOOR, 1.0]."""
        return max(REGIME_FLOOR, min(1.0, float(regime_prob)))

    def save(self, path: str | Path) -> None:
        """Pickle model to path.pkl and write metadata to path_meta.json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        meta = {
            "feature_names": FEATURE_NAMES,
            "vix_threshold": self.vix_threshold,
            "train_start": self._train_start,
            "train_end": self._train_end,
            "label_mean": self._label_mean,
            "val_auc": self._val_auc,
            "val_brier": self._val_brier,
        }
        meta_path = path.parent / (path.stem + "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("RegimeClassifier saved to %s (meta: %s)", path, meta_path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeClassifier":
        """Load a pickled RegimeClassifier."""
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(f"Loaded object is {type(obj)}, expected RegimeClassifier")
        logger.info("RegimeClassifier loaded from %s", path)
        return obj
