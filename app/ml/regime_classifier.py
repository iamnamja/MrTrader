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


class RegimeRuleScorer:
    """Rule-based regime scorer (regime_v4).

    Replaces the broken regime_v3 ML model (Phase A4: 100% NEUTRAL output).
    Uses three interpretable signals:
        1. SPY > 200d MA              (trend filter)
        2. VIX < vix_cap              (volatility filter, default 25)
        3. Breadth > breadth_thresh   (% R1K above 50d MA, if available)

    Composite score ∈ [0, 1]:
        score = (w1 * spy_signal + w2 * vix_signal + w3 * breadth_signal)
    where w1=0.50, w2=0.35, w3=0.15 (breadth is noisier / less available).

    Regime thresholds (matches Phase A3 B2 SPY MA200 gate):
        score >= 0.50  → BULL   (full sizing)
        score >= 0.35  → NEUTRAL (half sizing)
        score <  0.35  → RISK_OFF (floor sizing)

    Interface: same as RegimeClassifier — predict_proba_date() and
    predict_proba_series() return a probability in [0, 1] that is used
    directly as a sizing multiplier in the PM.

    Validation (2025-02 → 2025-05, tariff shock):
        Expected: ≥60% of days score < 0.50 (NEUTRAL/RISK_OFF).
        A4 showed regime_v3 returned NEUTRAL 100% of days → useless.
    """

    VERSION = "v4_rule"

    def __init__(
        self,
        vix_cap: float = 25.0,
        breadth_thresh: float = 0.40,
        spy_weight: float = 0.50,
        vix_weight: float = 0.35,
        breadth_weight: float = 0.15,
    ):
        self.vix_cap = vix_cap
        self.breadth_thresh = breadth_thresh
        self.spy_weight = spy_weight
        self.vix_weight = vix_weight
        self.breadth_weight = breadth_weight
        self._is_fitted = True  # rule-based, always "fitted"

    def score_row(
        self,
        spy_above_ma200: bool | float,
        vix: float,
        breadth: float | None = None,
    ) -> float:
        """Return composite regime score ∈ [0, 1] for a single day.

        Args:
            spy_above_ma200: True/1 if SPY close > SPY 200d MA
            vix: VIX close level
            breadth: optional fraction of R1K symbols above their 50d MA
        """
        spy_sig = 1.0 if spy_above_ma200 else 0.0
        vix_sig = max(0.0, min(1.0, (self.vix_cap - vix) / self.vix_cap))
        if breadth is not None:
            breadth_sig = 1.0 if breadth >= self.breadth_thresh else breadth / self.breadth_thresh
            total_w = self.spy_weight + self.vix_weight + self.breadth_weight
            score = (
                self.spy_weight * spy_sig
                + self.vix_weight * vix_sig
                + self.breadth_weight * breadth_sig
            ) / total_w
        else:
            total_w = self.spy_weight + self.vix_weight
            score = (self.spy_weight * spy_sig + self.vix_weight * vix_sig) / total_w
        return float(np.clip(score, 0.0, 1.0))

    def label(self, score: float) -> str:
        """Map score to regime label string."""
        if score >= 0.50:
            return "BULL"
        elif score >= 0.35:
            return "NEUTRAL"
        else:
            return "RISK_OFF"

    def sizing_weight(self, score: float) -> float:
        """Convert score to PM sizing multiplier ∈ [REGIME_FLOOR, 1.0]."""
        return max(REGIME_FLOOR, min(1.0, score))

    def predict_proba_series(self, macro_df: pd.DataFrame) -> pd.Series:
        """Compute daily regime score from a macro_history DataFrame.

        Expected columns (any subset is fine — missing ones are ignored):
            spy_close, spy_ma200 OR spy_above_ma200
            vix (VIX close)
            breadth_pct_above_50ma (optional)
        Returns: pd.Series of scores indexed by date.
        """
        results = []
        for date, row in macro_df.iterrows():
            # SPY signal
            if "spy_above_ma200" in macro_df.columns:
                spy_sig = bool(row["spy_above_ma200"])
            elif "spy_close" in macro_df.columns and "spy_ma200" in macro_df.columns:
                spy_sig = float(row["spy_close"]) > float(row["spy_ma200"])
            elif "spy_ma200_ratio" in macro_df.columns:
                spy_sig = float(row["spy_ma200_ratio"]) > 0
            else:
                spy_sig = True  # no data → conservative neutral assumption
            # VIX signal
            vix = float(row.get("vix", 20.0) or 20.0)
            # Breadth signal
            breadth = None
            if "breadth_pct_above_50ma" in macro_df.columns:
                b = row.get("breadth_pct_above_50ma")
                if b is not None and not np.isnan(float(b)):
                    breadth = float(b)
            results.append((date, self.score_row(spy_sig, vix, breadth)))
        return pd.Series(dict(results), name="regime_score")

    def predict_proba_date(self, row: pd.Series) -> float:
        """Single-row interface matching RegimeClassifier.predict_proba_date."""
        spy_sig = bool(row.get("spy_above_ma200", row.get("spy_ma200_ratio", 0) > 0))
        vix = float(row.get("vix", 20.0) or 20.0)
        breadth = None
        if "breadth_pct_above_50ma" in row.index:
            b = row.get("breadth_pct_above_50ma")
            if b is not None and not np.isnan(float(b)):
                breadth = float(b)
        return self.score_row(spy_sig, vix, breadth)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        meta = {
            "type": "RegimeRuleScorer",
            "version": self.VERSION,
            "vix_cap": self.vix_cap,
            "breadth_thresh": self.breadth_thresh,
            "spy_weight": self.spy_weight,
            "vix_weight": self.vix_weight,
            "breadth_weight": self.breadth_weight,
        }
        meta_path = path.parent / (path.stem + "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("RegimeRuleScorer v4 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeRuleScorer":
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(f"Loaded object is {type(obj)}, expected RegimeRuleScorer")
        logger.info("RegimeRuleScorer loaded from %s", path)
        return obj
