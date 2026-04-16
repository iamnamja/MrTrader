"""
Daily model training pipeline for the portfolio selection model.

Uses yfinance for historical OHLCV data (free, no API key required).
Labels: top 30% return stocks = 1, bottom 30% = 0, middle 40% = skipped.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import settings
from app.database.models import ModelVersion
from app.database.session import get_session
from app.ml.features import FeatureEngineer
from app.ml.model import PortfolioSelectorModel
from app.utils.constants import SP_100_TICKERS

logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"


class ModelTrainer:
    """Orchestrates data fetching, labelling, feature engineering, and model training."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.feature_engineer = FeatureEngineer()
        self.model = PortfolioSelectorModel(model_type="xgboost")

    # ─── Public API ───────────────────────────────────────────────────────────

    def train_model(
        self,
        symbols: Optional[List[str]] = None,
        years: Optional[int] = None,
    ) -> int:
        """
        Full pipeline: fetch → label → engineer → train → save → log.

        Returns:
            Version number of the newly trained model.
        """
        symbols = symbols or SP_100_TICKERS
        years = years or settings.historical_data_years

        logger.info("Starting training pipeline — %d symbols, %d years", len(symbols), years)

        symbols_data = self._fetch_historical_data(symbols, years)
        if not symbols_data:
            raise RuntimeError("No historical data fetched — check network / yfinance.")

        labels = self._create_labels(symbols_data)

        X, y, feature_names = self._build_feature_matrix(symbols_data, labels)
        if len(X) == 0:
            raise RuntimeError("No valid samples after feature engineering.")

        self.model.train(X, y, feature_names)

        version = self._next_version()
        saved_path = self.model.save(self.model_dir, version)
        self._record_version(version, len(X), saved_path, years)

        logger.info("Training complete — model v%d saved", version)
        return version

    # ─── Data Fetching ────────────────────────────────────────────────────────

    def _fetch_historical_data(
        self, symbols: List[str], years: int
    ) -> Dict[str, pd.DataFrame]:
        """Download daily OHLCV bars from yfinance for each symbol."""
        end = datetime.now()
        start = end - timedelta(days=365 * years)
        data: Dict[str, pd.DataFrame] = {}

        logger.info("Downloading data from %s to %s", start.date(), end.date())

        for symbol in symbols:
            try:
                df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
                # yfinance may return MultiIndex columns when auto_adjust=True
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if not df.empty and "close" in df.columns:
                    data[symbol] = df
                    logger.debug("Downloaded %d bars for %s", len(df), symbol)
            except Exception as e:
                logger.warning("Could not download %s: %s", symbol, e)

        logger.info("Downloaded data for %d / %d symbols", len(data), len(symbols))
        return data

    # ─── Labelling ────────────────────────────────────────────────────────────

    def _create_labels(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Optional[int]]:
        """
        Label stocks by total return over the period:
          - Top 30%  → 1  (good performer)
          - Bottom 30% → 0  (poor performer)
          - Middle 40% → None (skipped)
        """
        returns: List[tuple] = []
        for symbol, df in symbols_data.items():
            close = df["close"]
            if len(close) >= 2:
                total_return = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
                returns.append((symbol, float(total_return)))

        returns.sort(key=lambda x: x[1])
        n = len(returns)
        low_threshold = int(n * 0.30)
        high_threshold = int(n * 0.70)

        labels: Dict[str, Optional[int]] = {}
        for i, (symbol, _) in enumerate(returns):
            if i < low_threshold:
                labels[symbol] = 0
            elif i >= high_threshold:
                labels[symbol] = 1
            else:
                labels[symbol] = None

        good = sum(1 for v in labels.values() if v == 1)
        bad = sum(1 for v in labels.values() if v == 0)
        logger.info("Labels: %d good performers, %d poor performers, %d skipped", good, bad, n - good - bad)
        return labels

    # ─── Feature Matrix ───────────────────────────────────────────────────────

    def _build_feature_matrix(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        labels: Dict[str, Optional[int]],
    ):
        """Build numpy arrays X, y and feature_names from labelled symbol data."""
        X_rows, y_vals = [], []
        feature_names: Optional[List[str]] = None

        for symbol, df in symbols_data.items():
            label = labels.get(symbol)
            if label is None:
                continue

            features = self.feature_engineer.engineer_features(symbol, df)
            if features is None:
                continue

            if feature_names is None:
                feature_names = list(features.keys())

            X_rows.append(list(features.values()))
            y_vals.append(label)

        logger.info("Feature matrix: %d samples × %d features", len(X_rows), len(feature_names or []))
        return np.array(X_rows), np.array(y_vals), feature_names or []

    # ─── DB helpers ───────────────────────────────────────────────────────────

    def _next_version(self) -> int:
        db = get_session()
        try:
            latest = (
                db.query(ModelVersion)
                .filter_by(model_name="portfolio_selector")
                .order_by(ModelVersion.version.desc())
                .first()
            )
            return (latest.version + 1) if latest else 1
        finally:
            db.close()

    def _record_version(self, version: int, n_samples: int, model_path: str, years: int) -> None:
        db = get_session()
        try:
            end = datetime.now()
            start = end - timedelta(days=365 * years)
            record = ModelVersion(
                model_name="portfolio_selector",
                version=version,
                training_date=datetime.utcnow(),
                data_range_start=start.strftime("%Y-%m-%d"),
                data_range_end=end.strftime("%Y-%m-%d"),
                performance={"n_samples": n_samples},
                status="ACTIVE",
                model_path=model_path,
            )
            db.add(record)
            db.commit()
            logger.info("Model v%d metadata saved to DB", version)
        except Exception as e:
            db.rollback()
            logger.error("Failed to save model version to DB: %s", e)
        finally:
            db.close()
