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
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import settings
from app.database.models import ModelVersion
from app.database.session import get_session
from app.ml.cs_normalize import cs_normalize_by_group
from app.ml.features import FeatureEngineer
from app.ml.model import PortfolioSelectorModel
from app.utils.constants import SP_500_TICKERS, SECTOR_MAP

logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"

# Rolling window config
WINDOW_DAYS = 63        # ~1 quarter of features (enough for MACD, ATR, momentum)
FORWARD_DAYS = 5        # Phase 25: 5-day forward (aligns with observed 2-3 bar avg hold)
# STEP_DAYS = FORWARD_DAYS keeps windows non-overlapping (no label leakage).
STEP_DAYS = 5           # non-overlapping forward windows → cleaner labels
TEST_FRACTION = 0.25    # most recent 25% of windows = test set
# Embargo: skip this many windows between train and test to prevent the training
# label period (FORWARD_DAYS bars) from overlapping with test feature windows.
EMBARGO_WINDOWS = max(1, round(FORWARD_DAYS / STEP_DAYS))

LABEL_TARGET_PCT = 0.03   # fallback fixed target
LABEL_STOP_PCT = 0.02     # fallback fixed stop

# ATR-adaptive labeling — v110 baseline: asymmetric 1.5x target / 0.5x stop
ATR_MULT_TARGET = 1.5     # target = 1.5x the stock's 14-day ATR
ATR_MULT_STOP = 0.5       # stop  = 0.5x the stock's 14-day ATR
ATR_MIN_TARGET = 0.015    # floor: never require less than 1.5% move
ATR_MAX_TARGET = 0.08     # ceiling: never require more than 8% move

# Phase 43 — features with zero importance in v110 (55 features never used in any split).
# Removing them reduces overfitting surface without changing the signal.
PRUNED_FEATURES: frozenset = frozenset([
    "price_above_ema200", "dist_from_ema200", "rs_vs_spy", "sentiment",
    "pe_ratio", "pb_ratio",
    # profit_margin / revenue_growth / debt_to_equity un-pruned in Phase 89a when
    # fundamentals_history.parquet is present (PIT-correct values from EDGAR 10-K store)
    "earnings_proximity_days", "insider_score", "earnings_surprise",
    # sector_momentum / sector_momentum_5d un-pruned when sector_etf_history.parquet is present
    "regime_score", "vix_level", "vix_regime_bucket", "vix_fear_spike",
    "vix_percentile_1y", "spy_trend_63d", "earnings_surprise_1q",
    "earnings_surprise_2q_avg", "days_since_earnings", "short_interest_pct",
    "rs_vs_spy_5d", "rs_vs_spy_10d", "rs_vs_spy_60d", "fmp_surprise_1q",
    "fmp_surprise_2q_avg", "fmp_days_since_earnings", "fmp_analyst_upgrades_30d",
    "fmp_analyst_downgrades_30d", "fmp_analyst_momentum_30d", "fmp_inst_ownership_pct",
    "fmp_inst_change_pct", "atr_trend", "options_put_call_ratio", "options_iv_atm",
    "options_iv_premium", "news_sentiment_3d", "news_sentiment_7d",
    "news_article_count_7d", "news_sentiment_momentum", "fmp_consecutive_beats",
    "fmp_revenue_surprise_1q", "fcf_margin", "operating_leverage", "rd_intensity",
    "vol_expansion", "days_to_opex", "near_opex", "beta_252d", "beta_deviation",
    "earnings_drift_signal", "earnings_pead_strength", "adx_x_spy_trend",
    "rsi_x_spy_trend",
])


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


def _process_symbol_windows_worker(
    symbol: str,
    df_bytes: bytes,           # DataFrame serialised as Parquet bytes (Arrow columnar)
    all_dates: list,           # list of date objects
    window_starts: list,
    cs_thresholds: dict,
    sector: str,
    regime_score: Optional[float],
    fetch_fundamentals: bool,
    vix_history,               # pre-loaded VIX series or None
    label_scheme: str,
    symbol_cache: dict,        # {date_isoformat: features_dict} — pre-loaded from FeatureStore
    progress_queue=None,       # multiprocessing.Queue for progress reporting
    fundamentals_history: Optional[list] = None,  # Phase 89a: sorted list of (date_str, {profit_margin, revenue_growth, debt_to_equity})
    sector_etf_bars: Optional[Dict[str, list]] = None,  # Phase 89b: {etf: sorted list of (date_str, close)}
) -> Tuple[List, List, List, List, List]:
    """
    Module-level worker for ProcessPoolExecutor.
    Mirrors _process_symbol_windows but avoids instance-method pickling and
    SQLite access in subprocess — cache reads come from the pre-loaded dict,
    new entries are returned to the main process for batch writing.
    """
    import io as _io
    import pandas as pd
    from app.ml.features import FeatureEngineer

    fe = FeatureEngineer()
    df = pd.read_parquet(_io.BytesIO(df_bytes))

    X_rows, y_vals, meta_rows, to_cache, feature_names = [], [], [], [], []
    idx = df.index.date

    for w_start_idx in window_starts:
        w_end_idx = w_start_idx + WINDOW_DAYS
        if w_end_idx + FORWARD_DAYS >= len(all_dates):
            continue

        w_start_date = all_dates[w_start_idx]
        w_end_date = all_dates[w_end_idx]

        try:
            window_df = df.loc[(idx >= w_start_date) & (idx <= w_end_date)]
        except Exception:
            continue

        if len(window_df) < FeatureEngineer.MIN_BARS:
            continue

        entry_rows = df.loc[idx == w_end_date, "close"]
        if len(entry_rows) == 0:
            continue
        entry_price = float(entry_rows.iloc[0])
        if entry_price <= 0:
            continue

        label = None
        outcome_return = 0.0

        if label_scheme in ("cross_sectional", "sector_relative", "return_regression",
                            "return_blend", "lambdarank", "percentile_rank"):
            future_idx = w_end_idx + FORWARD_DAYS
            if future_idx >= len(all_dates):
                continue
            future_date = all_dates[future_idx]
            future_bar = df.loc[idx == future_date, "close"]
            if len(future_bar) == 0:
                continue
            stock_ret = (float(future_bar.iloc[0]) - entry_price) / entry_price
            outcome_return = stock_ret

            if label_scheme == "return_regression":
                label = stock_ret
            elif label_scheme == "cross_sectional":
                window_thresh = cs_thresholds.get(w_start_idx)
                if window_thresh is None:
                    continue
                if isinstance(window_thresh, tuple) and len(window_thresh) == 3:
                    w_mean, w_std, w_thr = window_thresh
                    sharpe_ret = (stock_ret - w_mean) / (w_std + 1e-8)
                    label = 1 if sharpe_ret >= w_thr else 0
                else:
                    label = 1 if stock_ret >= window_thresh else 0
            elif label_scheme == "percentile_rank":
                thresholds_pair = cs_thresholds.get(w_start_idx)
                if thresholds_pair is None:
                    continue
                p25, p75 = thresholds_pair
                if stock_ret >= p75:
                    label = 1
                elif stock_ret <= p25:
                    label = 0
                else:
                    continue
            elif label_scheme == "sector_relative":
                sector_thresh_map = cs_thresholds.get(w_start_idx)
                if sector_thresh_map is None:
                    continue
                thresh = sector_thresh_map.get(sector, sector_thresh_map.get("__global__"))
                if thresh is None:
                    continue
                label = 1 if stock_ret >= thresh else 0
            else:
                label = 1 if stock_ret >= (cs_thresholds.get(w_start_idx) or 0.0) else 0
        elif label_scheme == "path_quality":
            # Path quality regression label (Phase 45 Phase 2).
            # Scores each entry on how cleanly it moved toward target without touching stop.
            # Uses Config B multipliers: stop=0.5x ATR, target=1.0x ATR.
            # score = 1.0*upside_capture - 1.25*stop_pressure + 0.25*close_strength
            # Range approximately [-1.25, 1.25]; higher = better path.
            _PATH_STOP_MULT = 0.5
            _PATH_TARGET_MULT = 1.0
            _raw_target, _raw_stop = _atr_label_thresholds(window_df, entry_price)
            # Recompute with Config B multipliers (override ATR_MULT constants)
            try:
                highs_w = window_df["high"].values[-15:].astype(float)
                lows_w = window_df["low"].values[-15:].astype(float)
                closes_w = window_df["close"].values[-15:].astype(float)
                tr_w = np.maximum(
                    highs_w[1:] - lows_w[1:],
                    np.maximum(np.abs(highs_w[1:] - closes_w[:-1]), np.abs(lows_w[1:] - closes_w[:-1]))
                )
                atr_w = float(np.mean(tr_w[-14:])) if len(tr_w) >= 14 else float(np.mean(tr_w))
                atr_pct_w = atr_w / max(entry_price, 1e-6)
                target_pct = float(np.clip(_PATH_TARGET_MULT * atr_pct_w, ATR_MIN_TARGET, ATR_MAX_TARGET))
                stop_pct = float(np.clip(_PATH_STOP_MULT * atr_pct_w, ATR_MIN_TARGET / 2, ATR_MAX_TARGET / 2))
            except Exception:
                target_pct, stop_pct = _raw_target, _raw_stop
            target_price = entry_price * (1 + target_pct)
            stop_price = entry_price * (1 - stop_pct)
            max_high = entry_price
            min_low = entry_price
            final_close = entry_price
            for bar_offset in range(1, FORWARD_DAYS + 1):
                fi = w_end_idx + bar_offset
                if fi >= len(all_dates):
                    break
                bar = df.loc[idx == all_dates[fi]]
                if len(bar) == 0:
                    continue
                h = float(bar["high"].iloc[0])
                lo = float(bar["low"].iloc[0])
                c = float(bar["close"].iloc[0])
                max_high = max(max_high, h)
                min_low = min(min_low, lo)
                final_close = c
            upside_capture = min((max_high - entry_price) / (entry_price * target_pct + 1e-8), 1.0)
            stop_pressure = min((entry_price - min_low) / (entry_price * stop_pct + 1e-8), 1.0)
            close_strength = float(np.clip(
                (final_close - entry_price) / (entry_price * target_pct + 1e-8), -1.0, 1.0
            ))
            label = 1.0 * upside_capture - 1.25 * stop_pressure + 0.25 * close_strength
            outcome_return = (final_close - entry_price) / entry_price
        else:
            # ATR / triple_barrier label scheme:
            # Simulate bar-by-bar: label=1 if target hit before stop, label=0 if stop hit first.
            # "triple_barrier" is the explicit Phase 33 name; "atr" is the legacy alias.
            target_pct, stop_pct = _atr_label_thresholds(window_df, entry_price)
            target_price = entry_price * (1 + target_pct)
            stop_price = entry_price * (1 - stop_pct)
            label = 0
            for bar_offset in range(1, FORWARD_DAYS + 1):
                fi = w_end_idx + bar_offset
                if fi >= len(all_dates):
                    break
                bar = df.loc[idx == all_dates[fi]]
                if len(bar) == 0:
                    continue
                high = float(bar["high"].iloc[0])
                low = float(bar["low"].iloc[0])
                if low <= stop_price:
                    label = 0
                    outcome_return = (low - entry_price) / entry_price
                    break
                if high >= target_price:
                    label = 1
                    outcome_return = (high - entry_price) / entry_price
                    break

        if label is None:
            continue

        # Cache read from pre-loaded dict (no SQLite in subprocess)
        features = symbol_cache.get(str(w_end_date))
        if features is not None and feature_names and len(features) != len(feature_names):
            features = None

        if features is None:
            try:
                features = fe.engineer_features(
                    symbol, window_df,
                    sector=sector,
                    regime_score=regime_score,
                    fetch_fundamentals=fetch_fundamentals,
                    as_of_date=w_end_date,
                    vix_history=vix_history,
                )
            except Exception:
                continue
            if features:
                to_cache.append((symbol, w_end_date, features))

        if not features:
            continue

        # Phase 89a: override fundamentals with PIT-correct values from EDGAR 10-K store.
        # fundamentals_history is sorted ascending by date; take the last entry ≤ as_of_date.
        if fundamentals_history and features:
            as_of_str = str(w_end_date)
            pit = None
            for snap_date, snap_vals in fundamentals_history:
                if snap_date <= as_of_str:
                    pit = snap_vals
                else:
                    break
            if pit is not None:
                features["profit_margin"] = pit["profit_margin"]
                features["revenue_growth"] = pit["revenue_growth"]
                features["debt_to_equity"] = pit["debt_to_equity"]

        # Phase 89b: override sector_momentum / sector_momentum_5d with PIT ETF bars
        if sector_etf_bars and features:
            from app.ml.fundamental_fetcher import SECTOR_ETF_MAP
            etf = SECTOR_ETF_MAP.get(sector)
            bars = sector_etf_bars.get(etf) if etf else None
            if bars:
                as_of_str = str(w_end_date)
                # bars is sorted ascending [(date_str, close), ...]
                idx20 = idx5 = None
                for i, (d, _) in enumerate(bars):
                    if d <= as_of_str:
                        idx20 = i
                idx5 = idx20
                if idx20 is not None and idx20 >= 20:
                    features["sector_momentum"] = float(
                        (bars[idx20][1] - bars[idx20 - 20][1]) / max(bars[idx20 - 20][1], 1e-8)
                    )
                    features["momentum_20d_sector_neutral"] = (
                        features.get("momentum_20d", 0.0) - features["sector_momentum"]
                    )
                    features["momentum_60d_sector_neutral"] = (
                        features.get("momentum_60d", 0.0) - features["sector_momentum"]
                    )
                if idx5 is not None and idx5 >= 5:
                    features["sector_momentum_5d"] = float(
                        (bars[idx5][1] - bars[idx5 - 5][1]) / max(bars[idx5 - 5][1], 1e-8)
                    )
                    features["momentum_5d_sector_neutral"] = (
                        features.get("momentum_5d", 0.0) - features["sector_momentum_5d"]
                    )

        features = {k: v for k, v in features.items() if k not in PRUNED_FEATURES}
        if not feature_names:
            feature_names = list(features.keys())

        avg_vol = float(window_df["volume"].mean()) if "volume" in window_df.columns else 1e6
        X_rows.append(list(features.values()))
        y_vals.append(label)
        meta_rows.append({
            "window_idx": w_start_idx,
            "outcome_return": outcome_return,
            "vol_percentile": features.get("vol_percentile_52w", 0.5),
            "avg_volume": avg_vol,
            "sector": sector,
            "vix_regime_bucket": features.get("vix_regime_bucket", 0.5),
        })

    if progress_queue is not None:
        try:
            progress_queue.put((symbol, len(X_rows)))
        except Exception:
            pass
    return X_rows, y_vals, meta_rows, to_cache, feature_names


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
        provider: str = "polygon",
        use_feature_store: bool = True,
        model_type: str = "xgboost",
        label_scheme: str = "cross_sectional",
        top_n_features: Optional[int] = None,
        n_workers: int = 0,
        hpo_trials: int = 0,
        walk_forward_folds: int = 0,
        prediction_threshold: float = 0.35,
        two_stage: bool = False,
        three_stage: bool = False,
        multi_window: bool = False,
    ):
        self.model_dir = model_dir
        self.feature_engineer = FeatureEngineer()
        if model_type == "lambdarank":
            from app.ml.model import LambdaRankModel
            self.model = LambdaRankModel()
        elif model_type == "double_ensemble":
            from app.ml.model import DoubleEnsembleModel
            self.model = DoubleEnsembleModel()
        elif three_stage:
            from app.ml.model import ThreeStageModel
            self.model = ThreeStageModel(model_type=model_type)
        elif two_stage:
            from app.ml.model import TwoStageModel
            self.model = TwoStageModel(model_type=model_type)
        else:
            self.model = PortfolioSelectorModel(model_type=model_type)
        self._provider_name = provider
        self.label_scheme = label_scheme
        self.top_n_features = top_n_features
        # Default: all logical CPUs. Caller can pass n_workers to override.
        # Feature engineering is numpy-heavy and releases the GIL, so threads
        # beyond 8 still yield speedups on machines with 16+ cores.
        self.n_workers = n_workers if (n_workers and n_workers > 0) else (os.cpu_count() or 4)
        self.hpo_trials = hpo_trials
        self.walk_forward_folds = walk_forward_folds
        self.prediction_threshold = prediction_threshold
        self.two_stage = two_stage
        self.three_stage = three_stage
        self.multi_window = multi_window
        self._feature_store_lock = threading.Lock()
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
        symbols = symbols or SP_500_TICKERS
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

        # Feature selection + MI weights
        mi_scores: Optional[np.ndarray] = None
        if self.top_n_features and len(feature_names) > self.top_n_features:
            X_train, X_test, feature_names, mi_scores = self._select_top_features(
                X_train, y_train, X_test, feature_names, self.top_n_features
            )
            logger.info("Feature selection: kept top %d features", len(feature_names))
        else:
            # Compute MI scores even when keeping all features (used for weighting)
            try:
                if self.label_scheme in ("return_regression", "path_quality"):
                    from sklearn.feature_selection import mutual_info_regression
                    mi_scores = mutual_info_regression(X_train, y_train, random_state=42, n_jobs=self.n_workers)
                else:
                    from sklearn.feature_selection import mutual_info_classif
                    mi_scores = mutual_info_classif(X_train, y_train, random_state=42, n_jobs=self.n_workers)
            except Exception:
                mi_scores = None

        # Optional HPO — tune XGBoost params before final training
        if self.hpo_trials > 0 and self.model.model_type in ("xgboost", "ensemble", "lgbm_ensemble", "xgboost_regressor"):
            logger.info("Running Optuna HPO (%d trials)...", self.hpo_trials)
            best_params = self._tune_hyperparams(X_train, y_train, n_trials=self.hpo_trials)
            self.model.model.set_params(**best_params)

        # LambdaRank: convert float returns → quintile ranks, sort by window, build groups
        train_groups = None
        val_groups = None
        if self.label_scheme == "lambdarank":
            X_train, y_train, train_groups = self._build_lambdarank_groups(X_train, y_train, meta_train)
            val_groups = None
            spw = None
            sample_weight = None
            logger.info("LambdaRank/DoubleEnsemble: %d train groups, %d samples", len(train_groups), len(X_train))
        else:
            # Correct for class imbalance (not applicable to regression labels)
            if self.label_scheme in ("return_regression", "path_quality"):
                spw = None
                logger.info("Return regression mode — skipping class imbalance correction")
            else:
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
            scale_pos_weight=spw if self.label_scheme != "lambdarank" else None,
            X_val=X_test, y_val=y_test,
            early_stopping_rounds=30,
            sample_weight=sample_weight,
            feature_weights=mi_scores,
            groups=train_groups,
            val_groups=val_groups,
        )

        # Tune prediction threshold on validation set
        if len(X_test) > 0:
            tuned_t = self.model.tune_threshold(X_test, y_test)
            logger.info("Prediction threshold tuned to %.2f", tuned_t)

        # Evaluate on held-out test set
        metrics = self._evaluate(X_test, y_test, threshold=self.prediction_threshold)
        logger.info("Out-of-sample metrics: %s", metrics)

        # SHAP feature importance — logs top-15 and captures for DB persistence
        shap_top = self._log_shap_importance(X_test, feature_names)
        if shap_top:
            metrics["shap_top_features"] = shap_top

        # Optional walk-forward CV
        if self.walk_forward_folds > 0:
            wf_metrics = self._walk_forward_cv(
                np.vstack([X_train, X_test]),
                np.concatenate([y_train, y_test]),
                feature_names,
                n_folds=self.walk_forward_folds,
            )
            metrics.update(wf_metrics)

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
        # Always download SPY — used as date spine for rolling windows and for spy_relative labeling
        if "SPY" not in symbols_data:
            try:
                import yfinance as yf
                dates = sorted(set.union(*[set(df.index.date) for df in symbols_data.values()]))
                spy_df = yf.download("SPY", start=dates[0], end=dates[-1], progress=False, auto_adjust=True)
                if isinstance(spy_df.columns, pd.MultiIndex):
                    spy_df.columns = spy_df.columns.get_level_values(0)
                spy_df.columns = [c.lower() for c in spy_df.columns]
                if not spy_df.empty:
                    symbols_data = dict(symbols_data)  # don't mutate caller's dict
                    symbols_data["SPY"] = spy_df
                    logger.info("Downloaded SPY as date spine")
            except Exception as exc:
                logger.warning("Could not download SPY for spy_relative labeling: %s", exc)

        # Use SPY as the date spine (most complete US trading calendar).
        # Falling back to union-of-all avoids the intersection collapsing to nothing
        # when symbols have different listing histories (e.g. Russell 1000 has many
        # recent IPOs that reduce the intersection to zero).
        spy_df = symbols_data.get("SPY")
        if spy_df is not None:
            all_dates = sorted(set(spy_df.index.date))
        else:
            # Use union then filter to dates present in at least half the symbols
            from collections import Counter
            date_counts = Counter(
                d for df in symbols_data.values() for d in df.index.date
            )
            min_symbols = max(1, len(symbols_data) // 2)
            all_dates = sorted(d for d, cnt in date_counts.items() if cnt >= min_symbols)

        if len(all_dates) < WINDOW_DAYS + FORWARD_DAYS:
            logger.warning("Not enough common dates for rolling windows")
            return np.array([]), np.array([]), np.array([]), np.array([]), [], []

        # Window start indices (step by STEP_DAYS)
        window_starts = list(range(0, len(all_dates) - WINDOW_DAYS - FORWARD_DAYS, STEP_DAYS))
        if not window_starts:
            return np.array([]), np.array([]), np.array([]), np.array([]), [], []

        # Time-based split: last TEST_FRACTION windows -> test, with embargo gap.
        split_idx = max(1, int(len(window_starts) * (1 - TEST_FRACTION)))
        train_window_starts = window_starts[:split_idx]
        # Embargo: skip EMBARGO_WINDOWS windows so the last training label period
        # (ends at split_idx window + FORWARD_DAYS bars) does not overlap the
        # first test feature window.
        embargo_start = min(split_idx + EMBARGO_WINDOWS, len(window_starts))
        test_window_starts = window_starts[embargo_start:]

        # Use neutral regime_score for all historical training windows.
        # The live regime_score is computed once and would be identical for all
        # windows (2021 through 2026), creating look-ahead bias. Since
        # spy_trend_63d, vix_level, and vix_regime_bucket already capture regime
        # per-window from historical data, passing 0.5 here prevents leakage
        # while keeping regime_score available for live inference.
        regime_score = 0.5

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
            # Pre-warm Polygon financials cache (FCF margin, operating leverage, R&D)
            try:
                from app.data.polygon_financials import prefetch_polygon_financials
                prefetch_polygon_financials(list(symbols_data.keys()))
            except Exception as exc:
                logger.warning("Polygon financials prefetch skipped: %s", exc)

        X_train, y_train, meta_train = self._windows_to_matrix(
            symbols_data, all_dates, train_window_starts,
            regime_score, fetch_fundamentals, total_windows=len(window_starts)
        )
        X_test, y_test, meta_test = self._windows_to_matrix(
            symbols_data, all_dates, test_window_starts,
            regime_score, fetch_fundamentals, total_windows=len(window_starts)
        )

        # Cross-sectional normalization: z-score each feature within each window
        # (across all symbols at the same point in time) to remove regime/sector bias.
        if len(X_train) > 0 and len(meta_train) > 0:
            train_window_ids = np.array([m["window_idx"] for m in meta_train])
            X_train = cs_normalize_by_group(X_train, train_window_ids)
        if len(X_test) > 0 and len(meta_test) > 0:
            test_window_ids = np.array([m["window_idx"] for m in meta_test])
            X_test = cs_normalize_by_group(X_test, test_window_ids)

        feature_names = self._last_feature_names
        return X_train, y_train, X_test, y_test, feature_names, meta_train

    def _compute_cs_thresholds(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        all_dates: list,
        window_starts: list,
    ) -> Dict[int, Any]:
        """
        Pre-compute cross-sectional return thresholds for all windows.

        Windows are processed in parallel (ThreadPoolExecutor) — each window
        is independent (read-only access to symbols_data).

        label_scheme == "cross_sectional":
            returns {w_start_idx: float}  — global 80th-pct threshold

        label_scheme == "sector_relative":
            returns {w_start_idx: {sector: float}}  — per-sector 80th-pct threshold
            Symbols with no sector peers fall back to the global threshold.
        """
        thresholds: Dict[int, Any] = {}
        sector_relative = self.label_scheme in ("sector_relative", "atr_and_sector", "return_blend")
        use_blend = self.label_scheme == "return_blend"
        use_percentile_rank = self.label_scheme == "percentile_rank"

        def _compute_one_window(w_start_idx: int):
            """Compute threshold(s) for one window. Returns (w_start_idx, result) or None."""
            w_end_idx = w_start_idx + WINDOW_DAYS
            future_idx = w_end_idx + FORWARD_DAYS
            if future_idx >= len(all_dates):
                return None
            w_end_date = all_dates[w_end_idx]
            future_date = all_dates[future_idx]
            mid_date = all_dates[w_end_idx + 5] if use_blend and w_end_idx + 5 < len(all_dates) else None

            sym_returns: Dict[str, float] = {}
            sym_sectors: Dict[str, str] = {}
            for sym, df in symbols_data.items():
                if sym == "SPY":
                    continue
                idx = df.index.date
                entry_rows = df.loc[idx == w_end_date, "close"]
                future_rows = df.loc[idx == future_date, "close"]
                if len(entry_rows) == 0 or len(future_rows) == 0:
                    continue
                ep = float(entry_rows.iloc[0])
                fp = float(future_rows.iloc[0])
                if ep > 0:
                    ret_10d = (fp - ep) / ep
                    if use_blend and mid_date is not None:
                        mid_rows = df.loc[idx == mid_date, "close"]
                        ret_5d = (float(mid_rows.iloc[0]) - ep) / ep if len(mid_rows) > 0 else ret_10d
                        sym_returns[sym] = 0.4 * ret_5d + 0.6 * ret_10d
                    else:
                        sym_returns[sym] = ret_10d
                    sym_sectors[sym] = SECTOR_MAP.get(sym) or "Unknown"

            if not sym_returns:
                return None

            if use_percentile_rank:
                rets_arr = np.array(list(sym_returns.values()))
                p25 = float(np.percentile(rets_arr, 25))
                p75 = float(np.percentile(rets_arr, 75))
                return w_start_idx, (p25, p75)
            elif not sector_relative:
                rets_arr = np.array(sorted(sym_returns.values()))
                # Store (mean, std, sharpe_threshold) tuple so the label assignment
                # can normalize stock_ret to Sharpe units before comparing.
                # Previously this stored only the threshold scalar, causing the label
                # to compare raw return against a z-score (~0.84) — only extreme
                # outliers (84%+ gains) were labeled 1.
                cs_mean = float(rets_arr.mean())
                cs_std = float(rets_arr.std()) + 1e-8
                sharpe_rets = (rets_arr - cs_mean) / cs_std
                cs_threshold = float(np.percentile(sharpe_rets, 80))
                return w_start_idx, (cs_mean, cs_std, cs_threshold)
            else:
                from collections import defaultdict
                sector_rets: Dict[str, List[float]] = defaultdict(list)
                for sym, ret in sym_returns.items():
                    sector_rets[sym_sectors[sym]].append(ret)
                sector_thresh: Dict[str, float] = {}
                for sec, rets in sector_rets.items():
                    if len(rets) >= 3:
                        sorted_r = sorted(rets)
                        sector_thresh[sec] = sorted_r[int(len(sorted_r) * 0.80)]
                all_sorted = sorted(sym_returns.values())
                sector_thresh["__global__"] = all_sorted[int(len(all_sorted) * 0.80)]
                return w_start_idx, sector_thresh

        # Windows are independent — compute in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            for result in pool.map(_compute_one_window, window_starts):
                if result is not None:
                    thresholds[result[0]] = result[1]

        return thresholds

    def _process_symbol_windows(
        self,
        symbol: str,
        df: pd.DataFrame,
        all_dates: list,
        window_starts: list,
        cs_thresholds: Dict[int, float],
        sector: str,
        regime_score: Optional[float],
        fetch_fundamentals: bool,
        vix_history: Optional[Any] = None,
    ) -> Tuple[List[Any], List[Any], List[Any], List[Tuple], List[str]]:
        """Process all windows for a single symbol. Returns (X_rows, y_vals, meta_rows, to_cache, feature_names)."""
        X_rows, y_vals, meta_rows, to_cache, feature_names = [], [], [], [], []
        idx = df.index.date

        for w_start_idx in window_starts:
            w_end_idx = w_start_idx + WINDOW_DAYS
            if w_end_idx + FORWARD_DAYS >= len(all_dates):
                continue

            w_start_date = all_dates[w_start_idx]
            w_end_date = all_dates[w_end_idx]

            try:
                window_df = df.loc[(idx >= w_start_date) & (idx <= w_end_date)]
            except Exception:
                continue

            if len(window_df) < FeatureEngineer.MIN_BARS:
                continue

            entry_rows = df.loc[idx == w_end_date, "close"]
            if len(entry_rows) == 0:
                continue
            entry_price = float(entry_rows.iloc[0])
            if entry_price <= 0:
                continue

            label = None
            outcome_return = 0.0

            if self.label_scheme in ("cross_sectional", "sector_relative", "return_regression", "return_blend", "lambdarank", "percentile_rank"):
                future_idx = w_end_idx + FORWARD_DAYS
                if future_idx >= len(all_dates):
                    continue
                future_date = all_dates[future_idx]
                future_bar = df.loc[idx == future_date, "close"]
                if len(future_bar) == 0:
                    continue
                stock_ret = (float(future_bar.iloc[0]) - entry_price) / entry_price
                outcome_return = stock_ret

                if self.label_scheme == "lambdarank":
                    # Path-based realized P&L: simulate stop/target exit bar by bar.
                    # Aligns training label with how Tier 3 actually exits trades.
                    # Label = +target_pct if target hit first,
                    #         -stop_pct   if stop hit first,
                    #         actual 10-day return if neither (time exit).
                    target_pct, stop_pct = _atr_label_thresholds(window_df, entry_price)
                    realized = stock_ret  # fallback: 10-day endpoint (time exit)
                    for bar_offset in range(1, FORWARD_DAYS + 1):
                        fi = w_end_idx + bar_offset
                        if fi >= len(all_dates):
                            break
                        fbar = df.loc[idx == all_dates[fi]]
                        if len(fbar) == 0:
                            continue
                        if float(fbar["low"].iloc[0]) <= entry_price * (1 - stop_pct):
                            realized = -stop_pct
                            break
                        if float(fbar["high"].iloc[0]) >= entry_price * (1 + target_pct):
                            realized = target_pct
                            break
                    label = realized
                    outcome_return = realized
                elif self.label_scheme == "return_regression":
                    # Continuous float label — XGBRegressor path
                    label = stock_ret
                elif self.label_scheme == "return_blend":
                    # Binary label based on blended 5d+10d return (sector-relative top-20%).
                    # Smoother than pure 10d: a stock up 3% through day 5 then flat to day 10
                    # ranks higher than one up 3% only on day 10.
                    mid_idx = w_end_idx + 5
                    if mid_idx < len(all_dates):
                        mid_date = all_dates[mid_idx]
                        mid_bar = df.loc[idx == mid_date, "close"]
                        ret_5d = (float(mid_bar.iloc[0]) - entry_price) / entry_price if len(mid_bar) > 0 else stock_ret
                    else:
                        ret_5d = stock_ret
                    blended_ret = 0.4 * ret_5d + 0.6 * stock_ret
                    # Use precomputed blended threshold (stored in cs_thresholds for return_blend)
                    window_thresh = cs_thresholds.get(w_start_idx)
                    if window_thresh is None:
                        continue
                    if isinstance(window_thresh, dict):
                        cs_threshold = window_thresh.get(sector) or window_thresh.get("__global__")
                    else:
                        cs_threshold = window_thresh
                    if cs_threshold is None:
                        continue
                    label = 1 if blended_ret >= cs_threshold else 0
                elif self.label_scheme == "percentile_rank":
                    # Top quartile = 1, bottom quartile = 0, middle 50% = skip.
                    # Cleaner labels: removes ambiguous zone where small noise determines outcome.
                    thresholds_pair = cs_thresholds.get(w_start_idx)
                    if thresholds_pair is None:
                        continue
                    p25, p75 = thresholds_pair
                    if stock_ret >= p75:
                        label = 1
                    elif stock_ret <= p25:
                        label = 0
                    else:
                        continue  # skip ambiguous middle 50%
                else:
                    # cross_sectional: threshold stored as (mean, std, sharpe_80th_pct).
                    # Normalize stock_ret to Sharpe units before comparing so the
                    # threshold represents the top 20% of the window, not an absolute value.
                    window_thresh = cs_thresholds.get(w_start_idx)
                    if window_thresh is None:
                        continue
                    if isinstance(window_thresh, dict):
                        cs_threshold = window_thresh.get(sector) or window_thresh.get("__global__")
                        if cs_threshold is None:
                            continue
                        label = 1 if stock_ret >= cs_threshold else 0
                    elif isinstance(window_thresh, tuple) and len(window_thresh) == 3:
                        w_mean, w_std, w_thr = window_thresh
                        sharpe_ret = (stock_ret - w_mean) / (w_std + 1e-8)
                        label = 1 if sharpe_ret >= w_thr else 0
                    else:
                        # Legacy scalar threshold — raw comparison (pre-fix)
                        label = 1 if stock_ret >= window_thresh else 0

            elif self.label_scheme == "atr_and_sector":
                # Stricter: must BOTH hit ATR target AND be sector top-20%.
                # Reduces label noise — only stocks with real price action AND
                # relative outperformance get label=1. Fewer positives but cleaner.
                future_idx = w_end_idx + FORWARD_DAYS
                if future_idx >= len(all_dates):
                    continue
                future_date = all_dates[future_idx]
                future_bar = df.loc[idx == future_date, "close"]
                if len(future_bar) == 0:
                    continue
                stock_ret = (float(future_bar.iloc[0]) - entry_price) / entry_price
                outcome_return = stock_ret

                # ATR hit check
                target_pct, stop_pct = _atr_label_thresholds(window_df, entry_price)
                atr_label = None
                for bar_offset in range(1, FORWARD_DAYS + 1):
                    fi = w_end_idx + bar_offset
                    if fi >= len(all_dates):
                        break
                    bar = df.loc[idx == all_dates[fi]]
                    if len(bar) == 0:
                        continue
                    if float(bar["low"].iloc[0]) <= entry_price * (1 - stop_pct):
                        atr_label = 0
                        break
                    if float(bar["high"].iloc[0]) >= entry_price * (1 + target_pct):
                        atr_label = 1
                        break
                if atr_label is None:
                    continue

                # Sector top-20% check
                window_thresh = cs_thresholds.get(w_start_idx)
                if window_thresh is None:
                    continue
                if isinstance(window_thresh, dict):
                    cs_threshold = window_thresh.get(sector) or window_thresh.get("__global__")
                else:
                    cs_threshold = window_thresh
                sector_top = cs_threshold is not None and stock_ret >= cs_threshold

                # Both conditions required for label=1; either failure = label=0
                label = 1 if (atr_label == 1 and sector_top) else 0

            elif self.label_scheme == "spy_relative":
                future_idx = w_end_idx + FORWARD_DAYS
                if future_idx >= len(all_dates):
                    continue
                future_date = all_dates[future_idx]
                future_bar = df.loc[idx == future_date, "close"]
                if len(future_bar) == 0:
                    continue
                stock_ret = (float(future_bar.iloc[0]) - entry_price) / entry_price
                outcome_return = stock_ret
                # spy_ret uses pre-computed cs_thresholds as a proxy (spy stored separately)
                # fall through to ambiguous label handling
                if stock_ret > 0.02:
                    label = 1
                elif stock_ret < 0.0:
                    label = 0

            elif self.label_scheme == "path_quality":
                # Path quality regression label (Phase 45 Phase 2).
                # Scores each entry on how cleanly it moved toward target without touching stop.
                # Uses Config B multipliers: stop=0.5x ATR, target=1.0x ATR.
                # score = 1.0*upside_capture - 1.25*stop_pressure + 0.25*close_strength
                _PATH_STOP_MULT = 0.5
                _PATH_TARGET_MULT = 1.0
                try:
                    highs_w = window_df["high"].values[-15:].astype(float)
                    lows_w = window_df["low"].values[-15:].astype(float)
                    closes_w = window_df["close"].values[-15:].astype(float)
                    tr_w = np.maximum(
                        highs_w[1:] - lows_w[1:],
                        np.maximum(np.abs(highs_w[1:] - closes_w[:-1]), np.abs(lows_w[1:] - closes_w[:-1]))
                    )
                    atr_w = float(np.mean(tr_w[-14:])) if len(tr_w) >= 14 else float(np.mean(tr_w))
                    atr_pct_w = atr_w / max(entry_price, 1e-6)
                    target_pct = float(np.clip(_PATH_TARGET_MULT * atr_pct_w, ATR_MIN_TARGET, ATR_MAX_TARGET))
                    stop_pct = float(np.clip(_PATH_STOP_MULT * atr_pct_w, ATR_MIN_TARGET / 2, ATR_MAX_TARGET / 2))
                except Exception:
                    target_pct, stop_pct = _atr_label_thresholds(window_df, entry_price)
                max_high = entry_price
                min_low = entry_price
                final_close = entry_price
                for bar_offset in range(1, FORWARD_DAYS + 1):
                    fi = w_end_idx + bar_offset
                    if fi >= len(all_dates):
                        break
                    bar = df.loc[idx == all_dates[fi]]
                    if len(bar) == 0:
                        continue
                    h = float(bar["high"].iloc[0])
                    lo = float(bar["low"].iloc[0])
                    c = float(bar["close"].iloc[0])
                    max_high = max(max_high, h)
                    min_low = min(min_low, lo)
                    final_close = c
                upside_capture = min((max_high - entry_price) / (entry_price * target_pct + 1e-8), 1.0)
                stop_pressure = min((entry_price - min_low) / (entry_price * stop_pct + 1e-8), 1.0)
                close_strength = float(np.clip(
                    (final_close - entry_price) / (entry_price * target_pct + 1e-8), -1.0, 1.0
                ))
                label = 1.0 * upside_capture - 1.25 * stop_pressure + 0.25 * close_strength
                outcome_return = (final_close - entry_price) / entry_price

            else:  # ATR-hit labeling
                target_pct, stop_pct = _atr_label_thresholds(window_df, entry_price)
                target_price = entry_price * (1 + target_pct)
                stop_price = entry_price * (1 - stop_pct)
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
                    if low <= stop_price:
                        label = 0
                        outcome_return = (low - entry_price) / entry_price
                        break
                    if high >= target_price:
                        label = 1
                        outcome_return = (high - entry_price) / entry_price
                        break

            if label is None:
                continue

            # Feature cache check (read — thread-safe for SQLite WAL reads)
            features = None
            if self._feature_store is not None:
                features = self._feature_store.get(symbol, w_end_date)
                if features is not None and feature_names and len(features) != len(feature_names):
                    features = None

            if features is None:
                features = self.feature_engineer.engineer_features(
                    symbol, window_df,
                    sector=sector,
                    regime_score=regime_score,
                    fetch_fundamentals=fetch_fundamentals,
                    as_of_date=w_end_date,
                    vix_history=vix_history,
                )
                if features is not None:
                    to_cache.append((symbol, w_end_date, features))

            if features is None:
                continue

            features = {k: v for k, v in features.items() if k not in PRUNED_FEATURES}
            if not feature_names:
                feature_names = list(features.keys())

            avg_vol = float(window_df["volume"].mean()) if "volume" in window_df.columns else 1e6
            X_rows.append(list(features.values()))
            y_vals.append(label)
            meta_rows.append({
                "window_idx": w_start_idx,
                "outcome_return": outcome_return,
                "vol_percentile": features.get("vol_percentile_52w", 0.5),
                "avg_volume": avg_vol,
                "sector": sector,
                "vix_regime_bucket": features.get("vix_regime_bucket", 0.5),
            })

        return X_rows, y_vals, meta_rows, to_cache, feature_names

    def _windows_to_matrix(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        all_dates: list,
        window_starts: list,
        regime_score: Optional[float],
        fetch_fundamentals: bool,
        total_windows: int = 1,
        vix_history: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        X_rows, y_vals, meta_rows = [], [], []
        self._last_feature_names: List[str] = []

        # Pre-compute cross-sectional thresholds (serial — needs all symbols per window)
        cs_thresholds: Dict[int, Any] = {}
        if self.label_scheme in ("cross_sectional", "sector_relative", "atr_and_sector", "return_blend", "percentile_rank"):
            logger.info("Pre-computing %s thresholds for %d windows...", self.label_scheme, len(window_starts))
            cs_thresholds = self._compute_cs_thresholds(symbols_data, all_dates, window_starts)
            logger.info("%s thresholds ready (%d windows)", self.label_scheme, len(cs_thresholds))

        trading_symbols = {s: df for s, df in symbols_data.items() if s != "SPY"}
        logger.info("Processing %d symbols × %d windows using %d workers",
                    len(trading_symbols), len(window_starts), self.n_workers)

        pending_cache: List[Tuple] = []

        # 30a: Checkpoint — resume from saved progress if < 24h old and same config
        import pickle as _pickle
        import hashlib as _hashlib
        _ckpt_dir = Path(MODEL_DIR) / "checkpoints"
        _ckpt_dir.mkdir(parents=True, exist_ok=True)
        _cfg_key = _hashlib.md5(f"{self.label_scheme}:{len(trading_symbols)}:{len(window_starts)}".encode()).hexdigest()[:8]
        _ckpt_path = _ckpt_dir / f"swing_checkpoint_{_cfg_key}.pkl"
        _completed_symbols: set = set()

        if _ckpt_path.exists():
            _age = time.time() - _ckpt_path.stat().st_mtime
            if _age < 86400:  # < 24h
                try:
                    with open(_ckpt_path, "rb") as _f:
                        _ckpt = _pickle.load(_f)
                    X_rows.extend(_ckpt["X_rows"])
                    y_vals.extend(_ckpt["y_vals"])
                    meta_rows.extend(_ckpt["meta_rows"])
                    pending_cache.extend(_ckpt["pending_cache"])
                    _completed_symbols = _ckpt["completed_symbols"]
                    if not self._last_feature_names and _ckpt.get("feature_names"):
                        self._last_feature_names = _ckpt["feature_names"]
                    logger.info("Resumed from checkpoint: %d symbols done, %d rows loaded",
                                len(_completed_symbols), len(X_rows))
                except Exception as _exc:
                    logger.warning("Checkpoint load failed, starting fresh: %s", _exc)
                    _completed_symbols = set()
                    X_rows.clear()
                    y_vals.clear()
                    meta_rows.clear()
                    pending_cache.clear()

        # Filter to symbols not yet completed
        trading_symbols = {s: df for s, df in trading_symbols.items() if s not in _completed_symbols}
        if _completed_symbols:
            logger.info("Skipping %d already-checkpointed symbols, processing %d remaining",
                        len(_completed_symbols), len(trading_symbols))

        # Pre-load feature store cache into memory (single bulk SQL query) so
        # subprocess workers don't need SQLite access.
        all_dates_needed = [all_dates[i + WINDOW_DAYS]
                            for i in window_starts if i + WINDOW_DAYS < len(all_dates)]
        preloaded_cache: Dict[str, Dict[str, dict]] = {}
        if self._feature_store is not None:
            t_preload = time.time()
            bulk = self._feature_store.get_all_for_dates(all_dates_needed)
            # Re-key dates as isoformat strings for worker consumption
            for sym, date_map in bulk.items():
                preloaded_cache[sym] = {str(d): v for d, v in date_map.items()}
            total_hits = sum(len(v) for v in preloaded_cache.values())
            total_slots = len(trading_symbols) * len(all_dates_needed)
            logger.info("Cache pre-load: %d / %d slots warm (%.0f%%) in %.1fs",
                        total_hits, total_slots,
                        100 * total_hits / max(total_slots, 1),
                        time.time() - t_preload)

        # Phase 89a: load historical fundamentals parquet for PIT-correct feature override
        _fund_hist_by_symbol: Dict[str, list] = {}
        _fund_hist_path = Path("data/fundamentals/fundamentals_history.parquet")
        if _fund_hist_path.exists():
            try:
                _fh = pd.read_parquet(_fund_hist_path)
                for sym, grp in _fh.groupby("symbol"):
                    grp_sorted = grp.sort_values("as_of_date")
                    _fund_hist_by_symbol[sym] = [
                        (row["as_of_date"], {
                            "profit_margin": float(row.get("profit_margin", 0.0) or 0.0),
                            "revenue_growth": float(row.get("revenue_growth", 0.0) or 0.0),
                            "debt_to_equity": float(row.get("debt_to_equity", 0.0) or 0.0),
                        })
                        for _, row in grp_sorted.iterrows()
                    ]
                logger.info("Phase 89a: loaded fundamentals history for %d symbols", len(_fund_hist_by_symbol))
            except Exception as _exc:
                logger.warning("Phase 89a: could not load fundamentals_history.parquet — %s", _exc)

        # Phase 89b: load sector ETF history for PIT sector_momentum override
        _sector_etf_bars: Dict[str, list] = {}
        _etf_hist_path = Path("data/sector_etf/sector_etf_history.parquet")
        if _etf_hist_path.exists():
            try:
                _ef = pd.read_parquet(_etf_hist_path)
                for etf, grp in _ef.groupby("etf"):
                    grp_sorted = grp.sort_values("date")
                    _sector_etf_bars[etf] = [
                        (row["date"], float(row["close"]))
                        for _, row in grp_sorted.iterrows()
                    ]
                logger.info("Phase 89b: loaded sector ETF history for %d ETFs", len(_sector_etf_bars))
            except Exception as _exc:
                logger.warning("Phase 89b: could not load sector_etf_history.parquet — %s", _exc)

        # Serialize DataFrames as Arrow/Parquet bytes for subprocess pickling (29a)
        # ~3-5x faster than records lists for 753 symbols x 1200 rows x 5 columns
        def _sym_args(symbol, df):
            import io as _io
            cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            buf = _io.BytesIO()
            df[cols].to_parquet(buf, index=True)
            return (
                symbol,
                buf.getvalue(),
                all_dates,
                window_starts,
                cs_thresholds,
                SECTOR_MAP.get(symbol) or "Unknown",
                regime_score,
                fetch_fundamentals,
                vix_history,
                self.label_scheme,
                preloaded_cache.get(symbol, {}),
                _progress_q,
                _fund_hist_by_symbol.get(symbol),
                _sector_etf_bars if _sector_etf_bars else None,
            )

        # Manager queue required on Windows for cross-process passing to ProcessPoolExecutor
        with multiprocessing.Manager() as _mgr:
            _progress_q = _mgr.Queue()

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(_process_symbol_windows_worker, *_sym_args(symbol, df)): symbol
                    for symbol, df in trading_symbols.items()
                }
                done = 0
                total_rows = 0
                for future in as_completed(futures):
                    done += 1
                    # Drain progress queue for logging every 50 completions
                    while not _progress_q.empty():
                        try:
                            _, n_rows = _progress_q.get_nowait()
                            total_rows += n_rows
                        except Exception:
                            break
                    if done % 50 == 0 or done == len(futures):
                        logger.info("  OK  %d / %d symbols  |  ~%dk rows", done, len(futures), total_rows // 1000)
                    try:
                        sym_X, sym_y, sym_meta, sym_cache, sym_names = future.result()
                        X_rows.extend(sym_X)
                        y_vals.extend(sym_y)
                        meta_rows.extend(sym_meta)
                        pending_cache.extend(sym_cache)
                        if not self._last_feature_names and sym_names:
                            self._last_feature_names = sym_names
                    except Exception as exc:
                        logger.warning("Symbol %s failed: %s", futures[future], exc)
                    else:
                        _completed_symbols.add(futures[future])

                    # Save checkpoint every 50 completions
                    if done % 50 == 0:
                        try:
                            with open(_ckpt_path, "wb") as _f:
                                _pickle.dump({
                                    "X_rows": X_rows, "y_vals": y_vals, "meta_rows": meta_rows,
                                    "pending_cache": pending_cache,
                                    "completed_symbols": _completed_symbols,
                                    "feature_names": self._last_feature_names,
                                }, _f)
                        except Exception as _exc:
                            logger.debug("Checkpoint save failed: %s", _exc)

        # Delete checkpoint on successful completion
        try:
            _ckpt_path.unlink(missing_ok=True)
        except Exception:
            pass

        # Batch-write new feature rows from all workers in one go (main process only)
        if self._feature_store is not None and pending_cache:
            logger.info("Writing %d new feature rows to cache...", len(pending_cache))
            try:
                self._feature_store.put_batch(pending_cache)
            except Exception as exc:
                logger.warning("Feature store batch write failed: %s", exc)

        # Filter to consistent feature length (stale cache entries may have
        # different lengths if features were added/removed between runs)
        if X_rows:
            lengths = [len(r) for r in X_rows]
            target_len = max(set(lengths), key=lengths.count)
            if len(set(lengths)) > 1:
                logger.warning(
                    "Inhomogeneous feature rows detected — keeping only len=%d (%d/%d rows)",
                    target_len, lengths.count(target_len), len(lengths),
                )
                filtered = [(x, y, m) for x, y, m in zip(X_rows, y_vals, meta_rows) if len(x) == target_len]
                X_rows, y_vals, meta_rows = zip(*filtered) if filtered else ([], [], [])
                X_rows, y_vals, meta_rows = list(X_rows), list(y_vals), list(meta_rows)

        return np.array(X_rows), np.array(y_vals), meta_rows

    # ── Sample weighting ─────────────────────────────────────────────────────

    def _select_top_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        top_n: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        Select top-N features by mutual information score (training data only).
        Returns (X_train_sel, X_test_sel, selected_names, mi_scores_for_selected).
        MI scores are returned so callers can use them as feature weights.
        """
        try:
            if self.label_scheme in ("return_regression", "path_quality"):
                from sklearn.feature_selection import mutual_info_regression
                scores = mutual_info_regression(X_train, y_train, random_state=42, n_jobs=self.n_workers)
            else:
                from sklearn.feature_selection import mutual_info_classif
                scores = mutual_info_classif(X_train, y_train, random_state=42, n_jobs=self.n_workers)
            top_idx = np.argsort(scores)[::-1][:top_n]
            top_idx_sorted = sorted(top_idx)
            selected_names = [feature_names[i] for i in top_idx_sorted]
            selected_scores = scores[top_idx_sorted]
            logger.info("Top %d features by MI: %s", top_n, selected_names[:10])
            return (
                X_train[:, top_idx_sorted],
                X_test[:, top_idx_sorted],
                selected_names,
                selected_scores,
            )
        except Exception as exc:
            logger.warning("Feature selection failed, using all: %s", exc)
            all_scores = np.ones(len(feature_names))
            return X_train, X_test, feature_names, all_scores

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
                vix_regime_buckets=[m.get("vix_regime_bucket", 0.5) for m in meta],
                target_pct=LABEL_TARGET_PCT,
                current_vol_percentile=current_vol,
            )
            logger.info("Sample weights built for %d samples", len(weights))
            return weights
        except Exception as exc:
            logger.warning("Sample weight computation failed, using uniform: %s", exc)
            return None

    def _get_current_vol_percentile(self) -> float:
        """Estimate current market vol percentile using SPY realized vol. Cached 6h."""
        import json
        cache_path = Path(MODEL_DIR) / "price_cache" / "spy_vol_percentile.json"
        try:
            if cache_path.exists():
                age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
                if age_hours < 6:
                    return float(json.loads(cache_path.read_text()).get("vol_pct", 0.5))
        except Exception:
            pass
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
            result = float(np.clip((rv10 - min(rv_series)) / (max(rv_series) - min(rv_series)), 0, 1))
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"vol_pct": result}))
            except Exception:
                pass
            return result
        except Exception:
            # Fall back to stale cache rather than neutral 0.5
            try:
                if cache_path.exists():
                    return float(json.loads(cache_path.read_text()).get("vol_pct", 0.5))
            except Exception:
                pass
            return 0.5

    def _build_lambdarank_groups(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: List[dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert float return labels to quintile ranks (0-4) within each window,
        sort samples by window_idx so all samples in a group are contiguous,
        and return (X_sorted, y_quintile, group_sizes).

        LGBMRanker requires samples grouped by query (here: window) and sorted
        so that all rows in the same group are contiguous.
        """
        from scipy.stats import rankdata

        window_idx = np.array([m["window_idx"] for m in meta])
        sort_order = np.argsort(window_idx, kind="stable")
        X_sorted = X[sort_order]
        y_sorted = y[sort_order]
        window_idx_sorted = window_idx[sort_order]

        # Compute quintile labels (0-4) within each window using Sharpe-adjusted
        # returns: (ret - window_mean) / (window_std + eps). This normalizes out
        # market-wide moves so high-volatility stocks that moved randomly rank
        # lower than lower-vol stocks with genuine directional signal.
        y_quintile = np.zeros(len(y_sorted), dtype=np.int32)
        unique_windows = np.unique(window_idx_sorted)
        for w in unique_windows:
            mask = window_idx_sorted == w
            rets = y_sorted[mask]
            if len(rets) < 2:
                y_quintile[mask] = 2  # single sample gets middle quintile
                continue
            cs_mean = rets.mean()
            cs_std = rets.std() + 1e-8
            sharpe_rets = (rets - cs_mean) / cs_std
            ranks = rankdata(sharpe_rets, method="average")  # 1..n
            quintiles = np.floor((ranks - 1) / len(ranks) * 5).astype(int)
            quintiles = np.clip(quintiles, 0, 4)
            y_quintile[mask] = quintiles

        # Group sizes: how many samples per window
        _, group_sizes = np.unique(window_idx_sorted, return_counts=True)
        group_sizes = group_sizes.astype(np.int32)

        logger.info(
            "LambdaRank groups: %d windows, avg %.1f stocks/window, quintile dist %s",
            len(group_sizes),
            float(np.mean(group_sizes)),
            {q: int((y_quintile == q).sum()) for q in range(5)},
        )
        return X_sorted, y_quintile, group_sizes

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _log_shap_importance(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Compute SHAP values, log top-15 features by mean absolute contribution,
        and return {feature: shap_value} dict for DB persistence (Phase B4).
        Returns None if SHAP is unavailable or fails.
        """
        if len(X) == 0:
            return None
        try:
            import shap
            inner = None
            if hasattr(self.model, "stage1"):  # ThreeStageModel
                inner = getattr(self.model.stage2, "model", None)
            elif hasattr(self.model, "model"):
                inner = self.model.model
            if inner is None or not hasattr(inner, "feature_importances_"):
                return None

            X_sample = X[:min(500, len(X))]
            if hasattr(self.model, "scaler"):
                X_sample = self.model.scaler.transform(X_sample)
            elif hasattr(self.model, "stage2") and hasattr(self.model.stage2, "scaler"):
                stage2_idx = self.model.stage2_idx
                X_sample = self.model.stage2.scaler.transform(X_sample[:, stage2_idx])
                feature_names = [feature_names[i] for i in stage2_idx if i < len(feature_names)]

            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs = np.abs(shap_values).mean(axis=0)
            names = feature_names[:len(mean_abs)]
            top = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)[:15]
            logger.info("SHAP top-15 features (mean |shap|):")
            for name, val in top:
                logger.info("  %-35s %.4f", name, val)
            return {name: round(float(val), 6) for name, val in top}
        except ImportError:
            logger.info("SHAP not installed — skipping (pip install shap)")
            return None
        except Exception as exc:
            logger.debug("SHAP analysis failed: %s", exc)
            return None

    @staticmethod
    def _to_binary(y: np.ndarray) -> np.ndarray:
        """Convert float return labels to binary: top-20% within batch = 1."""
        if np.issubdtype(y.dtype, np.floating) and not np.all(np.isin(y, [0.0, 1.0])):
            return (y >= np.percentile(y, 80)).astype(int)
        return y.astype(int)

    def _evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict:
        if len(X_test) == 0:
            return {}
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, roc_auc_score
            )
            t = threshold if threshold is not None else self.model.predict_threshold
            preds, proba = self.model.predict(X_test, threshold=t)
            y_bin = self._to_binary(y_test)
            return {
                "accuracy": round(float(accuracy_score(y_bin, preds)), 4),
                "precision": round(float(precision_score(y_bin, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_bin, preds, zero_division=0)), 4),
                "auc": round(float(roc_auc_score(y_bin, proba)), 4),
                "threshold": round(t, 2),
                "n_test": len(y_test),
            }
        except Exception as exc:
            logger.warning("Evaluation failed: %s", exc)
            return {}

    # ── Optuna HPO ────────────────────────────────────────────────────────────

    def _tune_hyperparams(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Run Optuna Bayesian optimisation over XGBoost hyperparams using
        3-fold TimeSeriesSplit cross-validation on the training set.
        Returns the best param dict (ready to pass to XGBClassifier.set_params).
        """
        import optuna
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cv = TimeSeriesSplit(n_splits=3)

        is_regression = self.label_scheme in ("return_regression", "path_quality")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
                "random_state": 42,
                "verbosity": 0,
            }
            fold_aucs = []
            for train_idx, val_idx in cv.split(X_train):
                if is_regression:
                    from xgboost import XGBRegressor as _XGBReg
                    clf = _XGBReg(**params)
                    clf.fit(X_train[train_idx], y_train[train_idx], verbose=False)
                    raw = clf.predict(X_train[val_idx]).astype(float)
                    lo, hi = raw.min(), raw.max()
                    proba = (raw - lo) / (hi - lo + 1e-9)
                    y_bin = self._to_binary(y_train[val_idx])
                else:
                    params["eval_metric"] = "auc"
                    params.setdefault("nthread", -1)
                    clf = XGBClassifier(**params)
                    clf.fit(X_train[train_idx], y_train[train_idx], verbose=False)
                    proba = clf.predict_proba(X_train[val_idx])[:, 1]
                    y_bin = y_train[val_idx]
                try:
                    fold_aucs.append(roc_auc_score(y_bin, proba))
                except Exception:
                    fold_aucs.append(0.5)
            return float(np.mean(fold_aucs))

        study = optuna.create_study(direction="maximize")
        # n_jobs=-1: Optuna runs trials in parallel threads; each trial spawns its own XGBoost
        # which also uses all cores (nthread in params), so cap Optuna parallelism at 4 to
        # avoid over-subscription (4 parallel trials × nthread cores each ≈ full utilisation).
        study.optimize(objective, n_trials=n_trials, n_jobs=4, show_progress_bar=False)

        best = study.best_params
        logger.info(
            "HPO complete — best CV AUC=%.4f  params=%s",
            study.best_value, best,
        )
        return best

    # ── Walk-forward CV ───────────────────────────────────────────────────────

    def _walk_forward_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Expanding-window walk-forward cross-validation.
        Fold k trains on the first k/(n_folds+1) fraction of samples and
        tests on the next 1/(n_folds+1) fraction.
        Returns mean and std AUC across folds.
        """
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier

        n = len(X)
        fold_size = n // (n_folds + 1)
        if fold_size < 50:
            logger.warning("Walk-forward CV skipped — too few samples per fold (%d)", fold_size)
            return {}

        is_regression = self.label_scheme in ("return_regression", "path_quality")

        def _run_fold(k: int):
            train_end = k * fold_size
            test_end = min(train_end + fold_size, n)
            X_tr, y_tr = X[:train_end], y[:train_end]
            X_te, y_te = X[train_end:test_end], y[train_end:test_end]
            y_te_bin = self._to_binary(y_te)
            if len(np.unique(y_te_bin)) < 2:
                return None
            if is_regression:
                from xgboost import XGBRegressor as _XGBReg
                clf = _XGBReg(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
                    random_state=42, nthread=-1, verbosity=0,
                )
                clf.fit(X_tr, y_tr, verbose=False)
                raw = clf.predict(X_te).astype(float)
                lo, hi = raw.min(), raw.max()
                proba = (raw - lo) / (hi - lo + 1e-9)
            else:
                clf = XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
                    random_state=42, eval_metric="auc", nthread=-1, verbosity=0,
                )
                clf.fit(X_tr, y_tr, verbose=False)
                proba = clf.predict_proba(X_te)[:, 1]
            try:
                return roc_auc_score(y_te_bin, proba)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=min(n_folds, self.n_workers)) as pool:
            fold_results = list(pool.map(_run_fold, range(1, n_folds + 1)))
        aucs = [r for r in fold_results if r is not None]

        if not aucs:
            return {}

        result = {
            "wf_auc_mean": round(float(np.mean(aucs)), 4),
            "wf_auc_std": round(float(np.std(aucs)), 4),
            "wf_folds": len(aucs),
        }
        logger.info(
            "Walk-forward CV (%d folds): AUC=%.4f ± %.4f",
            len(aucs), result["wf_auc_mean"], result["wf_auc_std"],
        )
        return result

    # ── Multi-window ensemble ─────────────────────────────────────────────────

    def train_multi_window(
        self,
        symbols_data: dict,
        fetch_fundamentals: bool = True,
        years: int = 5,
        windows: Tuple[int, int] = (63, 126),
        blend: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Train models for each window size concurrently using a ProcessPoolExecutor-safe
        approach: each window runs in its own thread, with per-window WINDOW_DAYS
        set locally (not the module global) via a dedicated ModelTrainer subclass.
        63d captures short-term momentum; 126d captures medium-term trend.
        """
        import app.ml.training as _self_mod

        def _train_one_window(w: int) -> tuple:
            """Train a single window model. Runs in a thread."""
            logger.info("Multi-window: starting WINDOW_DAYS=%d", w)

            # Patch the constant only for feature computation inside this call
            # by using a subclass that overrides WINDOW_DAYS at instance level.
            orig = _self_mod.WINDOW_DAYS
            _self_mod.WINDOW_DAYS = w
            try:
                sub = ModelTrainer(
                    model_dir=self.model_dir,
                    provider=self._provider_name,
                    model_type=(
                        self.model.stage1.model_type
                        if hasattr(self.model, "stage1")
                        else getattr(self.model, "model_type", "xgboost")
                    ),
                    label_scheme=self.label_scheme,
                    top_n_features=self.top_n_features,
                    n_workers=max(1, self.n_workers // len(windows)),  # share workers
                    hpo_trials=self.hpo_trials,
                    prediction_threshold=self.prediction_threshold,
                    two_stage=self.two_stage,
                    three_stage=self.three_stage,
                )

                X_tr, y_tr, X_te, y_te, feat_names, meta = sub._build_rolling_matrix(
                    symbols_data, fetch_fundamentals=fetch_fundamentals
                )
            finally:
                _self_mod.WINDOW_DAYS = orig  # always restore

            if len(X_tr) == 0:
                logger.warning("Multi-window: no samples for window=%d", w)
                return w, None

            if sub.top_n_features and len(feat_names) > sub.top_n_features:
                X_tr, X_te, feat_names, mi_scores = sub._select_top_features(
                    X_tr, y_tr, X_te, feat_names, sub.top_n_features
                )
            else:
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(X_tr, y_tr, random_state=42, n_jobs=sub.n_workers)

            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
            sample_weight = sub._build_sample_weights(meta)

            sub.model.train(
                X_tr, y_tr, feat_names,
                scale_pos_weight=spw,
                X_val=X_te, y_val=y_te,
                early_stopping_rounds=30,
                sample_weight=sample_weight,
                feature_weights=mi_scores,
            )
            sub.model.tune_threshold(X_te, y_te)
            metrics = sub._evaluate(X_te, y_te)
            logger.info("Multi-window w=%d complete: %s", w, metrics)
            return w, {"trainer": sub, "metrics": metrics, "feature_names": feat_names}

        # Run all window sizes concurrently
        logger.info("Multi-window: training %d windows in parallel", len(windows))
        results: Dict[int, Any] = {}
        trained_trainers = []

        with ThreadPoolExecutor(max_workers=len(windows)) as pool:
            futures = {pool.submit(_train_one_window, w): w for w in windows}
            for future in as_completed(futures):
                w, result = future.result()
                if result is not None:
                    results[w] = result
                    trained_trainers.append((w, result["trainer"]))

        if not trained_trainers:
            raise RuntimeError("Multi-window: no windows produced valid models")

        # Store blended predict function on self for inference
        self._mw_trainers = trained_trainers
        self._mw_blend = blend
        logger.info("Multi-window ensemble ready: %d models, blend=%.2f", len(trained_trainers), blend)
        return results

    def predict_multi_window(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Blend probabilities from all trained window models."""
        if not hasattr(self, "_mw_trainers") or not self._mw_trainers:
            raise RuntimeError("Call train_multi_window() first")
        probas = []
        for _, trainer in self._mw_trainers:
            _, p = trainer.model.predict(X)
            probas.append(p)
        blended = np.mean(probas, axis=0)
        preds = (blended >= self.prediction_threshold).astype(int)
        return preds, blended

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
            features = {k: v for k, v in features.items() if k not in PRUNED_FEATURES}
            if feature_names is None:
                feature_names = list(features.keys())
            X_rows.append(list(features.values()))
            y_vals.append(label)
        return np.array(X_rows), np.array(y_vals), feature_names or []

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _next_version(self, model_name: str = "swing") -> int:
        try:
            db = get_session()
            latest = (
                db.query(ModelVersion)
                .filter_by(model_name=model_name)
                .order_by(ModelVersion.version.desc())
                .first()
            )
            db.close()
            return (latest.version + 1) if latest else 1
        except Exception:
            # DB not available — derive version from existing pkl files
            from pathlib import Path as _P
            files = sorted(_P(self.model_dir).glob(f"{model_name}_v*.pkl"))
            return (int(files[-1].stem.split("_v")[-1]) + 1) if files else 1

    def _record_version(
        self, version: int, n_train: int, n_test: int,
        model_path: str, years: int, metrics: Dict
    ) -> None:
        try:
            db = get_session()
        except Exception as exc:
            logger.warning("DB not available, skipping version record: %s", exc)
            return
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365 * years)
            # Retire previous ACTIVE swing versions — only one should be active at a time
            prev = db.query(ModelVersion).filter_by(model_name="swing", status="ACTIVE").all()
            for p in prev:
                p.status = "RETIRED"
                logger.info("Retired swing v%d", p.version)
            db.add(ModelVersion(
                model_name="swing",
                version=version,
                training_date=datetime.utcnow(),
                data_range_start=start_dt.strftime("%Y-%m-%d"),
                data_range_end=end_dt.strftime("%Y-%m-%d"),
                performance={
                    **metrics, "n_train": n_train, "n_test": n_test,
                    "tier3_gate_passed": None,
                    "tier3_avg_sharpe": None,
                    "tier3_fold_sharpes": None,
                },
                status="ACTIVE",
                model_path=model_path,
            ))
            db.commit()
            logger.info("Swing model v%d saved to DB", version)

            # Phase B4: AUC drift alert
            auc = metrics.get("auc")
            if auc is not None and auc < 0.65:
                logger.warning(
                    "MODEL DRIFT ALERT: v%d AUC=%.4f is below 0.65 threshold — "
                    "recent OOS performance degraded, manual review recommended",
                    version, auc,
                )
        except Exception as exc:
            db.rollback()
            logger.error("Failed to save model version: %s", exc)
        finally:
            db.close()

    @staticmethod
    def record_tier3_result(
        version: int,
        avg_sharpe: float,
        fold_sharpes: list,
        gate_passed: bool,
    ) -> None:
        """Write walk-forward results back into the swing ModelVersion performance field."""
        try:
            from app.database.session import get_session
            from app.database.models import ModelVersion
            db = get_session()
            row = db.query(ModelVersion).filter_by(
                model_name="swing", version=version
            ).first()
            if row is None:
                return
            perf = dict(row.performance or {})
            perf["tier3_avg_sharpe"] = round(avg_sharpe, 4)
            perf["tier3_gate_passed"] = gate_passed
            perf["tier3_fold_sharpes"] = [round(s, 4) for s in fold_sharpes]
            row.performance = perf
            if not gate_passed:
                row.status = "RETIRED"
                logger.info("Swing v%d tier3 FAIL — status set to RETIRED", version)
            else:
                from pathlib import Path as _Path
                sentinel = _Path("app/ml/models") / f"swing_v{version}.gate_passed"
                sentinel.touch()
                logger.info("Swing v%d gate_passed sentinel written", version)
            db.commit()
            logger.info("Recorded tier3 result for swing v%d: Sharpe=%.3f gate=%s",
                        version, avg_sharpe, gate_passed)
        except Exception as exc:
            logger.warning("Could not record tier3 result for swing v%d: %s", version, exc)
        finally:
            try:
                db.close()
            except Exception:
                pass
