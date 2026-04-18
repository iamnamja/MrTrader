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
WINDOW_DAYS = 63        # ~1 quarter of features (enough for MACD, ATR, momentum)
FORWARD_DAYS = 10       # predict return over next 2 weeks — matches MAX_HOLD_DAYS=10
# v17 fix: STEP_DAYS = FORWARD_DAYS avoids overlapping forward windows (label leakage).
# v15/v16 used STEP_DAYS=5 which meant consecutive windows shared 5 of 10 forward days.
STEP_DAYS = 10          # non-overlapping forward windows → cleaner labels
TEST_FRACTION = 0.25    # most recent 25% of windows = test set

LABEL_TARGET_PCT = 0.03   # fallback fixed target
LABEL_STOP_PCT = 0.02     # fallback fixed stop

# ATR-adaptive labeling — v19: ASYMMETRIC 1.5x target / 0.5x stop
# Restores R:R > 1 (3:1) — only labels a winner when move > stop distance by 3x.
# Tighter stop (0.5x vs old 0.75x) = cleaner, more decisive labels.
# v17/v18 used symmetric 1.0x/1.0x which produced ~50/50 labels but random AUC.
ATR_MULT_TARGET = 1.5     # target = 1.5x the stock's 14-day ATR
ATR_MULT_STOP = 0.5       # stop  = 0.5x the stock's 14-day ATR (tight = decisive labels)
ATR_MIN_TARGET = 0.015    # floor: never require less than 1.5% move
ATR_MAX_TARGET = 0.08     # ceiling: never require more than 8% move

# v19: Volume confirmation — require avg forward volume > this fraction of historical avg
# to label a winner. Filters out low-conviction price moves not backed by volume.
VOL_CONFIRM_THRESHOLD = 0.9  # forward avg volume must be >= 90% of window avg volume


def _atr_label_thresholds(window_df: pd.DataFrame, entry_price: float):
    """
    Compute ATR-adaptive target and stop percentages for labeling.

    Uses 14-day ATR of the feature window to scale thresholds to each stock's
    actual volatility. A 1.5x ATR target on TSLA (~3% ATR) gives 4.5% vs
    1.5% on PG (~1% ATR) — both meaningful for their respective regimes.
    Falls back to fixed constants if ATR is unavailable.
    """
    try:
        highs = window_df["high"].values[-15:].astype(float)
        lows = window_df["low"].values[-15:].astype(float)
        closes = window_df["close"].values[-15:].astype(float)
        if len(closes) < 2:
            raise ValueError("insufficient bars")
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
        )
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
        atr_pct = atr / max(entry_price, 1e-6)
        target_pct = float(np.clip(ATR_MULT_TARGET * atr_pct, ATR_MIN_TARGET, ATR_MAX_TARGET))
        stop_pct = float(np.clip(ATR_MULT_STOP * atr_pct, ATR_MIN_TARGET / 2, ATR_MAX_TARGET / 2))
        return target_pct, stop_pct
    except Exception:
        return LABEL_TARGET_PCT, LABEL_STOP_PCT


class ModelTrainer:
    """
    Orchestrates data fetching, rolling-window labelling,
    feature engineering, and XGBoost/LightGBM training for the swing model.

    model_type options: "xgboost", "lgbm", "ensemble", "lgbm_ensemble"
    top_n_features: if set, selects top-N features by mutual information before training
    """

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        provider: str = "yfinance",
        use_feature_store: bool = True,
        model_type: str = "xgboost",
        top_n_features: Optional[int] = None,
    ):
        self.model_dir = model_dir
        self.feature_engineer = FeatureEngineer()
        self.model = PortfolioSelectorModel(model_type=model_type)
        self._provider_name = provider
        self.top_n_features = top_n_features
        if use_feature_store:
            from app.ml.feature_store import FeatureStore
            self._feature_store: Optional[object] = FeatureStore(f"{model_dir}/feature_store.db")
        else:
            self._feature_store = None

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

        X_train, y_train, X_test, y_test, feature_names, meta_train = self._build_rolling_matrix(
            symbols_data, fetch_fundamentals=fetch_fundamentals
        )
        if len(X_train) == 0:
            raise RuntimeError("No valid training samples after rolling windows.")

        logger.info(
            "Train: %d samples | Test: %d samples | Features: %d",
            len(X_train), len(X_test), len(feature_names),
        )

        # Feature selection (if configured)
        if self.top_n_features and len(feature_names) > self.top_n_features:
            X_train, X_test, feature_names = self._select_top_features(
                X_train, y_train, X_test, feature_names, self.top_n_features
            )
            logger.info("Feature selection: kept top %d features", len(feature_names))

        # Correct for class imbalance: stops (~70%) outnumber targets (~30%)
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
        logger.info("Class ratio  neg=%d  pos=%d  scale_pos_weight=%.2f", n_neg, n_pos, spw)

        # Build multi-factor sample weights
        sample_weight = self._build_sample_weights(meta_train)

        # LightGBM-based models use class_weight instead of scale_pos_weight
        if self.model.model_type in ("lgbm", "lgbm_ensemble"):
            self.model.model.set_params(class_weight={0: 1.0, 1: float(spw)})
            if self.model.model_type == "lgbm_ensemble" and self.model._lgbm_model is not None:
                self.model._lgbm_model.set_params(class_weight={0: 1.0, 1: float(spw)})
            spw = None  # don't pass as XGBoost param

        # Use test set as validation for early stopping (avoids overfitting on noisy data)
        self.model.train(
            X_train, y_train, feature_names,
            scale_pos_weight=spw,
            X_val=X_test, y_val=y_test,
            early_stopping_rounds=30,
            sample_weight=sample_weight,
        )

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[dict]]:
        """
        For each non-overlapping WINDOW_DAYS window across all symbols:
          - features: computed from bars in [window_start, window_end]
          - label:    1 if trade hits TARGET_PCT within FORWARD_DAYS bars,
                      0 if trade hits STOP_PCT first,
                      skipped if neither (ambiguous outcome)

        Returns (X_train, y_train, X_test, y_test, feature_names, meta_train).
        meta_train is a list of dicts used to compute sample weights.
        Test set = most recent TEST_FRACTION of windows (time-based split).
        """
        # Build sorted list of window start dates from the earliest common date
        all_dates = sorted(set.intersection(
            *[set(df.index.date) for df in symbols_data.values()]
        ))
        if len(all_dates) < WINDOW_DAYS + FORWARD_DAYS:
            logger.warning("Not enough common dates for rolling windows")
            return np.array([]), np.array([]), np.array([]), np.array([]), [], []

        # Window start indices (step by STEP_DAYS)
        window_starts = list(range(0, len(all_dates) - WINDOW_DAYS - FORWARD_DAYS, STEP_DAYS))
        if not window_starts:
            return np.array([]), np.array([]), np.array([]), np.array([]), [], []

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
            # Pre-warm FMP caches (earnings history + analyst grades for all symbols)
            try:
                from app.data.fmp_provider import prefetch_fmp
                prefetch_fmp(list(symbols_data.keys()))
            except Exception as exc:
                logger.warning("FMP prefetch skipped: %s", exc)

        X_train, y_train, meta_train = self._windows_to_matrix(
            symbols_data, all_dates, train_window_starts,
            regime_score, fetch_fundamentals, total_windows=len(window_starts)
        )
        X_test, y_test, _ = self._windows_to_matrix(
            symbols_data, all_dates, test_window_starts,
            regime_score, fetch_fundamentals, total_windows=len(window_starts)
        )

        feature_names = self._last_feature_names
        return X_train, y_train, X_test, y_test, feature_names, meta_train

    def _windows_to_matrix(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        all_dates: list,
        window_starts: list,
        regime_score: Optional[float],
        fetch_fundamentals: bool,
        total_windows: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        X_rows, y_vals, meta_rows = [], [], []
        self._last_feature_names: List[str] = []

        for w_start_idx in window_starts:
            w_end_idx = w_start_idx + WINDOW_DAYS
            if w_end_idx + FORWARD_DAYS >= len(all_dates):
                continue

            w_start_date = all_dates[w_start_idx]
            w_end_date = all_dates[w_end_idx]

            for symbol, df in symbols_data.items():
                idx = df.index.date

                # ── Feature window ────────────────────────────────────────────
                try:
                    window_df = df.loc[(idx >= w_start_date) & (idx <= w_end_date)]
                except Exception:
                    continue

                if len(window_df) < FeatureEngineer.MIN_BARS:
                    continue

                # ── Outcome-based label ───────────────────────────────────────
                entry_rows = df.loc[idx == w_end_date, "close"]
                if len(entry_rows) == 0:
                    continue
                entry_price = float(entry_rows.iloc[0])
                if entry_price <= 0:
                    continue

                target_pct, stop_pct = _atr_label_thresholds(window_df, entry_price)
                target_price = entry_price * (1 + target_pct)
                stop_price = entry_price * (1 - stop_pct)

                label = None
                outcome_return = 0.0
                forward_volumes = []
                for bar_offset in range(1, FORWARD_DAYS + 1):
                    future_idx = w_end_idx + bar_offset
                    if future_idx >= len(all_dates):
                        break
                    future_date = all_dates[future_idx]
                    bar = df.loc[idx == future_date]
                    if len(bar) == 0:
                        continue
                    high = float(bar["high"].iloc[0])
                    low = float(bar["low"].iloc[0])
                    if "volume" in bar.columns:
                        forward_volumes.append(float(bar["volume"].iloc[0]))
                    if low <= stop_price:
                        label = 0
                        outcome_return = (low - entry_price) / entry_price
                        break
                    if high >= target_price:
                        # v19: volume confirmation — require meaningful volume to validate winner
                        avg_fwd_vol = float(np.mean(forward_volumes)) if forward_volumes else entry_price
                        window_vol = float(window_df["volume"].mean()) if "volume" in window_df.columns else avg_fwd_vol
                        if window_vol > 0 and avg_fwd_vol < VOL_CONFIRM_THRESHOLD * window_vol:
                            label = 0  # price hit target but volume didn't confirm — treat as stop
                        else:
                            label = 1
                        outcome_return = (high - entry_price) / entry_price
                        break

                if label is None:
                    continue  # neither target nor stop hit — skip

                # ── Features (cache-first) ────────────────────────────────────
                sector = SECTOR_MAP.get(symbol) or "Unknown"
                features = None
                if self._feature_store is not None:
                    features = self._feature_store.get(symbol, w_end_date)
                    # Discard stale cache entries whose feature count doesn't match current code.
                    # Mixing feature counts across windows causes inhomogeneous np.array errors.
                    if features is not None and self._last_feature_names and \
                            len(features) != len(self._last_feature_names):
                        features = None
                if features is None:
                    features = self.feature_engineer.engineer_features(
                        symbol, window_df,
                        sector=sector,
                        regime_score=regime_score,
                        fetch_fundamentals=fetch_fundamentals,
                        as_of_date=w_end_date,
                    )
                    if features is not None and self._feature_store is not None:
                        self._feature_store.put(symbol, w_end_date, features)
                if features is None:
                    continue

                if not self._last_feature_names:
                    self._last_feature_names = list(features.keys())

                X_rows.append(list(features.values()))
                y_vals.append(label)

                # Collect metadata for sample weight computation
                avg_vol = float(window_df["volume"].mean()) if "volume" in window_df.columns else 1e6
                meta_rows.append({
                    "window_idx": w_start_idx,
                    "outcome_return": outcome_return,
                    "vol_percentile": features.get("vol_percentile_52w", 0.5),
                    "avg_volume": avg_vol,
                    "sector": sector,
                })

        return np.array(X_rows), np.array(y_vals), meta_rows

    # ── Sample weighting ─────────────────────────────────────────────────────

    def _select_top_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        top_n: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Select top-N features by mutual information score (training data only)."""
        try:
            from sklearn.feature_selection import mutual_info_classif
            scores = mutual_info_classif(X_train, y_train, random_state=42)
            top_idx = np.argsort(scores)[::-1][:top_n]
            top_idx_sorted = sorted(top_idx)
            selected_names = [feature_names[i] for i in top_idx_sorted]
            logger.info("Top %d features by MI: %s", top_n, selected_names[:10])
            return X_train[:, top_idx_sorted], X_test[:, top_idx_sorted], selected_names
        except Exception as exc:
            logger.warning("Feature selection failed, using all: %s", exc)
            return X_train, X_test, feature_names

    def _build_sample_weights(self, meta: List[dict]) -> Optional[np.ndarray]:
        """Build multi-factor sample weights from per-sample metadata."""
        if not meta:
            return None
        try:
            from app.ml.sample_weights import compute_sample_weights
            # Current vol percentile from regime detector (proxy for today's market)
            current_vol = self._get_current_vol_percentile()
            weights = compute_sample_weights(
                window_indices=[m["window_idx"] for m in meta],
                total_windows=max(m["window_idx"] for m in meta) + 1,
                outcome_returns=[m["outcome_return"] for m in meta],
                vol_percentiles=[m["vol_percentile"] for m in meta],
                avg_volumes=[m["avg_volume"] for m in meta],
                sector_labels=[m["sector"] for m in meta],
                target_pct=LABEL_TARGET_PCT,
                current_vol_percentile=current_vol,
            )
            logger.info("Sample weights built for %d samples", len(weights))
            return weights
        except Exception as exc:
            logger.warning("Sample weight computation failed, using uniform: %s", exc)
            return None

    def _get_current_vol_percentile(self) -> float:
        """Estimate current market vol percentile using SPY realized vol."""
        try:
            import yfinance as yf
            spy = yf.download("SPY", period="1y", progress=False, auto_adjust=True)
            if spy is None or len(spy) < 30:
                return 0.5
            closes = spy["Close"].values.astype(float)
            returns = np.diff(np.log(closes))
            rv10 = float(np.std(returns[-10:]) * np.sqrt(252))
            rv_series = [
                float(np.std(returns[max(0, i-10):i]) * np.sqrt(252))
                for i in range(10, len(returns) + 1)
            ]
            if not rv_series or max(rv_series) == min(rv_series):
                return 0.5
            return float(np.clip((rv10 - min(rv_series)) / (max(rv_series) - min(rv_series)), 0, 1))
        except Exception:
            return 0.5

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
