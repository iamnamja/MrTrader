"""
Intraday model training pipeline.

Label: did price move >= TARGET_PCT in the right direction within HOLD_BARS bars?
Window: each trading day is one window per stock.
Train/test split: last TEST_FRACTION of days = test (time-based, no leakage).

Data source: yfinance 5-min for training history, Alpaca for live.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.data import get_provider
from app.database.models import ModelVersion
from app.database.session import get_session
from app.ml.intraday_features import compute_intraday_features, MIN_BARS
from app.ml.model import PortfolioSelectorModel

logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"

# Training config
TARGET_PCT = 0.005       # 0.5% move = winning intraday trade
HOLD_BARS = 24           # 2 hours of 5-min bars to achieve the target
TEST_FRACTION = 0.20     # most recent 20% of days = test
MIN_DAYS = 20            # minimum trading days needed for a symbol


class IntradayModelTrainer:
    """
    Orchestrates 5-min bar fetching, per-day labelling,
    feature engineering, and XGBoost training for the intraday model.
    """

    def __init__(self, model_dir: str = MODEL_DIR, provider: str = "yfinance"):
        self.model_dir = model_dir
        self.model = PortfolioSelectorModel(model_type="xgboost")
        self._provider_name = provider

    @property
    def _provider(self):
        return get_provider(self._provider_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def train_model(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 60,
        fetch_spy: bool = True,
    ) -> int:
        """
        Full pipeline: fetch 5-min bars -> per-day labelling -> features -> train -> save.
        Returns version number of the saved model.
        """
        from app.utils.constants import SP_100_TICKERS
        symbols = symbols or SP_100_TICKERS[:30]  # default: first 30, 5-min data is large

        logger.info(
            "Starting intraday training — %d symbols, %d days, provider=%s",
            len(symbols), days, self._provider_name,
        )

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days + 5)  # buffer for weekends

        symbols_data = self._fetch_data(symbols, start_dt, end_dt)
        spy_data = None
        if fetch_spy:
            spy_data = self._provider.get_intraday_bars("SPY", start_dt, end_dt, interval_minutes=5)

        if not symbols_data:
            raise RuntimeError("No 5-min data fetched.")

        X_train, y_train, X_test, y_test, feature_names = self._build_daily_matrix(
            symbols_data, spy_data
        )

        if len(X_train) == 0:
            raise RuntimeError("No valid training samples after per-day labelling.")

        logger.info(
            "Train: %d samples | Test: %d samples | Features: %d",
            len(X_train), len(X_test), len(feature_names),
        )

        self.model.train(X_train, y_train, feature_names)

        metrics = self._evaluate(X_test, y_test)
        logger.info("Intraday OOS metrics: %s", metrics)

        version = self._next_version("intraday")
        saved_path = self.model.save(self.model_dir, version, model_name="intraday")
        self._record_version(version, len(X_train), len(X_test), saved_path, days, metrics)

        return version

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_data(
        self, symbols: List[str], start: datetime, end: datetime
    ) -> Dict[str, pd.DataFrame]:
        logger.info("Fetching 5-min bars %s -> %s for %d symbols", start, end, len(symbols))
        data = self._provider.get_intraday_bars_bulk(symbols, start, end, interval_minutes=5)
        logger.info("Got 5-min data for %d / %d symbols", len(data), len(symbols))
        return data

    # ── Per-day labelling + feature matrix ───────────────────────────────────

    def _build_daily_matrix(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        spy_data: Optional[pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        For each (symbol, trading_day):
          - features: computed from bars up to HOLD_BARS before session end
          - label:    did price hit +TARGET_PCT within the next HOLD_BARS bars?

        Time-based split: last TEST_FRACTION of trading days -> test.
        """
        # Collect all trading days across symbols
        all_days: set = set()
        for df in symbols_data.values():
            if df is not None and len(df) > 0:
                idx = pd.DatetimeIndex(df.index)
                for d in idx.normalize().unique():
                    all_days.add(d.date())

        sorted_days = sorted(all_days)
        if len(sorted_days) < MIN_DAYS:
            logger.warning("Not enough trading days (%d) for intraday training", len(sorted_days))
            return np.array([]), np.array([]), np.array([]), np.array([]), []

        # Time-based split
        split_idx = max(1, int(len(sorted_days) * (1 - TEST_FRACTION)))
        train_days = sorted_days[:split_idx]
        test_days = sorted_days[split_idx:]

        X_train, y_train = self._days_to_matrix(symbols_data, spy_data, train_days)
        X_test, y_test = self._days_to_matrix(symbols_data, spy_data, test_days)

        return X_train, y_train, X_test, y_test, self._last_feature_names

    def _days_to_matrix(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        spy_data: Optional[pd.DataFrame],
        trading_days: List[date],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_rows, y_vals = [], []
        self._last_feature_names: List[str] = []

        for sym, df in symbols_data.items():
            if df is None or len(df) == 0:
                continue

            df_idx = pd.DatetimeIndex(df.index)

            for day in trading_days:
                # Bars for this symbol on this day
                day_mask = df_idx.normalize().date == day
                day_bars = df.loc[day_mask]

                if len(day_bars) < MIN_BARS + HOLD_BARS:
                    continue

                # Feature window: bars up to (session - HOLD_BARS)
                feat_bars = day_bars.iloc[:-HOLD_BARS]
                future_bars = day_bars.iloc[-HOLD_BARS:]

                if len(feat_bars) < MIN_BARS:
                    continue

                # Label: price rises >= TARGET_PCT within HOLD_BARS
                entry_price = float(feat_bars["close"].iloc[-1])
                future_highs = future_bars["high"].values.astype(float)
                label = int(any(h >= entry_price * (1 + TARGET_PCT) for h in future_highs))

                # SPY bars for same day
                spy_day_bars = None
                if spy_data is not None and len(spy_data) > 0:
                    spy_idx = pd.DatetimeIndex(spy_data.index)
                    spy_mask = spy_idx.normalize().date == day
                    spy_day_bars = spy_data.loc[spy_mask] if spy_mask.any() else None

                # Prior day close / high / low for gap and S/R features
                prior_close, prior_day_high, prior_day_low = self._prior_day_ohlc(df, day)

                feats = compute_intraday_features(
                    feat_bars, spy_day_bars, prior_close,
                    prior_day_high=prior_day_high,
                    prior_day_low=prior_day_low,
                )
                if feats is None:
                    continue

                if not self._last_feature_names:
                    self._last_feature_names = list(feats.keys())

                X_rows.append(list(feats.values()))
                y_vals.append(label)

        return np.array(X_rows), np.array(y_vals)

    def _prior_day_ohlc(
        self, df: pd.DataFrame, day: date
    ) -> tuple:
        """Return (prior_close, prior_day_high, prior_day_low) for the day before `day`."""
        idx = pd.DatetimeIndex(df.index)
        prior_mask = idx.normalize().date < day
        if not prior_mask.any():
            return None, None, None
        prior_bars = df.loc[prior_mask]
        last_day = prior_bars.index.normalize().date[-1]
        last_day_bars = prior_bars.loc[prior_bars.index.normalize().date == last_day]
        if len(last_day_bars) == 0:
            return None, None, None
        return (
            float(last_day_bars["close"].iloc[-1]),
            float(last_day_bars["high"].max()),
            float(last_day_bars["low"].min()),
        )

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
            logger.warning("Intraday evaluation failed: %s", exc)
            return {}

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _next_version(self, model_name: str = "intraday") -> int:
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
        model_path: str, days: int, metrics: Dict
    ) -> None:
        db = get_session()
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)
            db.add(ModelVersion(
                model_name="intraday",
                version=version,
                training_date=datetime.utcnow(),
                data_range_start=start_dt.strftime("%Y-%m-%d"),
                data_range_end=end_dt.strftime("%Y-%m-%d"),
                performance={**metrics, "n_train": n_train, "n_test": n_test},
                status="ACTIVE",
                model_path=model_path,
            ))
            db.commit()
            logger.info("Intraday model v%d saved to DB", version)
        except Exception as exc:
            db.rollback()
            logger.error("Failed to save intraday model version: %s", exc)
        finally:
            db.close()
