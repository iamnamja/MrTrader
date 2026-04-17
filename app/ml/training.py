"""
Model training pipeline — swing (daily) model.

Key improvements over v1:
  - Rolling quarterly windows: ~12 windows x 82 symbols = ~900 samples
  - Time-based train/test split (train on older periods, test on recent)
    prevents data leakage and gives honest out-of-sample metrics
  - Uses DataProvider abstraction — swap yfinance for any future source
    by passing provider="polygon" etc.
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import settings
from app.database.models import ModelVersion
from app.database.session import get_session
from app.ml.features import FeatureEngineer
from app.ml.model import PortfolioSelectorModel
from app.utils.constants import SP_100_TICKERS, SECTOR_MAP

logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"

# Rolling window config
WINDOW_DAYS = 63        # ~1 quarter of trading days
FORWARD_DAYS = 63       # predict return over next quarter
STEP_DAYS = 63          # step between windows (non-overlapping quarters)
TEST_FRACTION = 0.25    # most recent 25% of windows = test set


class ModelTrainer:
    """
    Orchestrates data fetching, rolling-window labelling,
    feature engineering, and XGBoost training for the swing model.
    """

    def __init__(self, model_dir: str = MODEL_DIR, provider: str = "yfinance"):
        self.model_dir = model_dir
        self.feature_engineer = FeatureEngineer()
        self.model = PortfolioSelectorModel(model_type="xgboost")
        self._provider_name = provider

    @property
    def _provider(self):
        from app.data import get_provider
        return get_provider(self._provider_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def train_model(
        self,
        symbols: Optional[List[str]] = None,
        years: Optional[int] = None,
        fetch_fundamentals: bool = True,
    ) -> int:
        """
        Full pipeline: fetch -> rolling windows -> features -> train -> save.
        Returns version number of the saved model.
        """
        symbols = symbols or SP_100_TICKERS
        years = years or settings.historical_data_years

        logger.info(
            "Starting swing training — %d symbols, %d years, provider=%s",
            len(symbols), years, self._provider_name,
        )

        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365 * years + FORWARD_DAYS + 30)

        symbols_data = self._fetch_data(symbols, start_dt, end_dt)
        if not symbols_data:
            raise RuntimeError("No historical data fetched.")

        X_train, y_train, X_test, y_test, feature_names = self._build_rolling_matrix(
            symbols_data, fetch_fundamentals=fetch_fundamentals
        )
        if len(X_train) == 0:
            raise RuntimeError("No valid training samples after rolling windows.")

        logger.info(
            "Train: %d samples | Test: %d samples | Features: %d",
            len(X_train), len(X_test), len(feature_names),
        )

        self.model.train(X_train, y_train, feature_names)

        # Evaluate on held-out test set
        metrics = self._evaluate(X_test, y_test)
        logger.info("Out-of-sample metrics: %s", metrics)

        version = self._next_version("swing")
        saved_path = self.model.save(self.model_dir, version, model_name="swing")
        self._record_version(version, len(X_train), len(X_test), saved_path, years, metrics)

        return version

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_data(
        self, symbols: List[str], start: date, end: date
    ) -> Dict[str, pd.DataFrame]:
        logger.info("Fetching daily bars %s -> %s", start, end)
        data = self._provider.get_daily_bars_bulk(symbols, start, end)
        logger.info("Got data for %d / %d symbols", len(data), len(symbols))
        return data

    # ── Rolling window labelling ──────────────────────────────────────────────

    def _build_rolling_matrix(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        fetch_fundamentals: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        For each non-overlapping WINDOW_DAYS window across all symbols:
          - features: computed from bars in [window_start, window_end]
          - label:    top/bottom 30% of forward return over next FORWARD_DAYS

        Returns (X_train, y_train, X_test, y_test, feature_names).
        Test set = most recent TEST_FRACTION of windows (time-based split).
        """
        # Build sorted list of window start dates from the earliest common date
        all_dates = sorted(set.intersection(
            *[set(df.index.date) for df in symbols_data.values()]
        ))
        if len(all_dates) < WINDOW_DAYS + FORWARD_DAYS:
            logger.warning("Not enough common dates for rolling windows")
            return np.array([]), np.array([]), np.array([]), np.array([]), []

        # Window start indices (step by STEP_DAYS)
        window_starts = list(range(0, len(all_dates) - WINDOW_DAYS - FORWARD_DAYS, STEP_DAYS))
        if not window_starts:
            return np.array([]), np.array([]), np.array([]), np.array([]), []

        # Time-based split: last TEST_FRACTION windows -> test
        split_idx = max(1, int(len(window_starts) * (1 - TEST_FRACTION)))
        train_window_starts = window_starts[:split_idx]
        test_window_starts = window_starts[split_idx:]

        # Regime score once per run (same macro context)
        regime_score = self._get_regime_score()

        # Pre-fetch fundamentals once per symbol to warm the cache.
        # The rolling loop calls engineer_features ~20x per symbol; without this
        # each call would hit the yfinance API independently.
        if fetch_fundamentals:
            from app.ml.fundamental_fetcher import prefetch_fundamentals
            logger.info("Pre-fetching fundamentals for %d symbols...", len(symbols_data))
            prefetch_fundamentals(list(symbols_data.keys()))

        X_train, y_train = self._windows_to_matrix(
            symbols_data, all_dates, train_window_starts,
            regime_score, fetch_fundamentals
        )
        X_test, y_test = self._windows_to_matrix(
            symbols_data, all_dates, test_window_starts,
            regime_score, fetch_fundamentals
        )

        feature_names = self._last_feature_names
        return X_train, y_train, X_test, y_test, feature_names

    def _windows_to_matrix(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        all_dates: list,
        window_starts: list,
        regime_score: Optional[float],
        fetch_fundamentals: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_rows, y_vals = [], []
        self._last_feature_names: List[str] = []

        for w_start_idx in window_starts:
            w_end_idx = w_start_idx + WINDOW_DAYS
            fwd_end_idx = min(w_end_idx + FORWARD_DAYS, len(all_dates) - 1)

            w_start_date = all_dates[w_start_idx]
            w_end_date = all_dates[w_end_idx]
            fwd_end_date = all_dates[fwd_end_idx]

            # Forward returns for each symbol in this window
            fwd_returns: Dict[str, float] = {}
            for symbol, df in symbols_data.items():
                try:
                    idx = df.index.date
                    entry = df.loc[idx == w_end_date, "close"]
                    exit_ = df.loc[idx == fwd_end_date, "close"]
                    if len(entry) and len(exit_):
                        fwd_returns[symbol] = float(
                            (exit_.iloc[0] - entry.iloc[0]) / entry.iloc[0]
                        )
                except Exception:
                    pass

            if len(fwd_returns) < 6:
                continue

            # Label: top 30% = 1, bottom 30% = 0, middle = skip
            sorted_ret = sorted(fwd_returns.items(), key=lambda x: x[1])
            n = len(sorted_ret)
            lo = int(n * 0.30)
            hi = int(n * 0.70)
            labels = {}
            for i, (sym, _) in enumerate(sorted_ret):
                if i < lo:
                    labels[sym] = 0
                elif i >= hi:
                    labels[sym] = 1
                # middle 40% skipped

            # Features for each labeled symbol
            for symbol, label in labels.items():
                df = symbols_data[symbol]
                try:
                    idx = df.index.date
                    window_df = df.loc[(idx >= w_start_date) & (idx <= w_end_date)]
                except Exception:
                    continue

                if len(window_df) < FeatureEngineer.MIN_BARS:
                    continue

                sector = SECTOR_MAP.get(symbol)
                features = self.feature_engineer.engineer_features(
                    symbol, window_df,
                    sector=sector,
                    regime_score=regime_score,
                    fetch_fundamentals=fetch_fundamentals,
                )
                if features is None:
                    continue

                if not self._last_feature_names:
                    self._last_feature_names = list(features.keys())

                X_rows.append(list(features.values()))
                y_vals.append(label)

        return np.array(X_rows), np.array(y_vals)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        if len(X_test) == 0:
            return {}
        try:
            from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
            preds, proba = self.model.predict(X_test)
            return {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
                "auc": round(float(roc_auc_score(y_test, proba)), 4),
                "n_test": len(y_test),
            }
        except Exception as exc:
            logger.warning("Evaluation failed: %s", exc)
            return {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_regime_score(self) -> Optional[float]:
        try:
            from app.strategy.regime_detector import RegimeDetector
            det = RegimeDetector().get_regime_detail()
            return float(det.get("composite_score", 0.5))
        except Exception:
            return 0.5

    # ── Legacy compatibility: kept so existing code calling _create_labels works
    def _create_labels(
        self, symbols_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Optional[int]]:
        """Single-window labels (kept for CLI dry-run compatibility)."""
        returns = []
        for symbol, df in symbols_data.items():
            close = df["close"]
            if len(close) >= 2:
                ret = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
                returns.append((symbol, float(ret)))
        returns.sort(key=lambda x: x[1])
        n = len(returns)
        lo, hi = int(n * 0.30), int(n * 0.70)
        labels: Dict[str, Optional[int]] = {}
        for i, (sym, _) in enumerate(returns):
            labels[sym] = 0 if i < lo else (1 if i >= hi else None)
        return labels

    def _build_feature_matrix(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        labels: Dict[str, Optional[int]],
        fetch_fundamentals: bool = True,
    ):
        """Single-window feature matrix (kept for CLI dry-run compatibility)."""
        regime_score = self._get_regime_score()
        X_rows, y_vals = [], []
        feature_names: Optional[List[str]] = None
        for symbol, df in symbols_data.items():
            label = labels.get(symbol)
            if label is None:
                continue
            sector = SECTOR_MAP.get(symbol)
            features = self.feature_engineer.engineer_features(
                symbol, df, sector=sector,
                regime_score=regime_score,
                fetch_fundamentals=fetch_fundamentals,
            )
            if features is None:
                continue
            if feature_names is None:
                feature_names = list(features.keys())
            X_rows.append(list(features.values()))
            y_vals.append(label)
        return np.array(X_rows), np.array(y_vals), feature_names or []

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _next_version(self, model_name: str = "swing") -> int:
        db = get_session()
        try:
            latest = (
                db.query(ModelVersion)
                .filter_by(model_name=model_name)
                .order_by(ModelVersion.version.desc())
                .first()
            )
            return (latest.version + 1) if latest else 1
        finally:
            db.close()

    def _record_version(
        self, version: int, n_train: int, n_test: int,
        model_path: str, years: int, metrics: Dict
    ) -> None:
        db = get_session()
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365 * years)
            db.add(ModelVersion(
                model_name="swing",
                version=version,
                training_date=datetime.utcnow(),
                data_range_start=start_dt.strftime("%Y-%m-%d"),
                data_range_end=end_dt.strftime("%Y-%m-%d"),
                performance={**metrics, "n_train": n_train, "n_test": n_test},
                status="ACTIVE",
                model_path=model_path,
            ))
            db.commit()
            logger.info("Swing model v%d saved to DB", version)
        except Exception as exc:
            db.rollback()
            logger.error("Failed to save model version: %s", exc)
        finally:
            db.close()
