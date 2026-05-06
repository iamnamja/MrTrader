"""
Intraday model training pipeline — optimized for 1000 symbols × 2 years of 5-min bars.

Label: outcome-based — price hits +TARGET_PCT before -STOP_PCT within HOLD_BARS bars.
Window: each trading day is one window per stock.
Train/test split: last TEST_FRACTION of days = test (time-based, no leakage).

Data sources (in priority order):
  1. Polygon.io Parquet cache at data/intraday/ (populated by fetch_intraday_history.py)
     — 2 years of history, recommended for training
  2. Alpaca 5-min bars with Parquet disk cache at data/cache/5min/ (≈60 days max)
     — fallback when Polygon cache is missing

Speed optimizations:
  - Parquet cache per symbol; only re-fetches stale/missing symbols on rerun
  - Chunked parallel API calls (FETCH_CHUNK_SIZE symbols/call, FETCH_WORKERS threads)
  - Parallel feature computation (FEATURE_WORKERS threads, one task per symbol)
  - XGBoost trained with nthread=-1 (all CPU cores)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import multiprocessing
import numpy as np
import pandas as pd

from app.data import get_provider
from app.database.models import ModelVersion
from app.database.session import get_session
from app.ml.cs_normalize import cs_normalize_by_group
from app.ml.intraday_features import compute_intraday_features, MIN_BARS, FEATURE_NAMES as _INTRADAY_FEATURE_NAMES
from app.ml.model import PortfolioSelectorModel

logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"
CACHE_DIR = Path("data/cache/5min")
META_FILE = CACHE_DIR / "_meta.json"

# Training config
TARGET_PCT = 0.005       # 0.5% intraday profit target (fallback)
STOP_PCT = 0.003         # 0.3% intraday stop loss (fallback)
HOLD_BARS = 24           # 2 hours of 5-min bars to achieve the target
TEST_FRACTION = 0.20     # most recent 20% of days = test
MIN_DAYS = 20            # minimum trading days needed per symbol

# ATR-adaptive labeling (Phase 87: realized-R outcome labels)
ATR_MULT_TARGET = 0.8    # Phase 47-3: compressed from 1.2 → closer target for 2h window
ATR_MULT_STOP = 0.4      # Phase 47-3: compressed from 0.6 → tighter stop, maintains ~2:1 R:R
ATR_MIN_TARGET = 0.003   # floor: never require less than 0.3%
MIN_ABSOLUTE_MOVE = 0.003  # Phase 87A: 0.30% minimum to cover commissions + spread
MIN_REALIZED_R = 0.40      # Phase 88: loosened from 0.5 → 0.40 (experiment log recommendation)

ENTRY_OFFSETS = [12]  # single scan at bar 12 (~60 min post-open); v30 proved multi-window hurts
ATR_MAX_TARGET = 0.025   # ceiling: never require more than 2.5%

# Phase 87: frozen HPO params (set from thorough 100-trial search; reuse on every retrain)
# None = run HPO to find initial params; once found, set these to freeze.
FROZEN_HPO_PARAMS: Optional[dict] = None  # type: ignore[assignment]

# Phase 87: ensemble seeds (3 XGBoost models blended by average probability)
ENSEMBLE_SEEDS = [42, 123, 777]

# Parallelism config
FETCH_CHUNK_SIZE = 100   # symbols per Alpaca API call
FETCH_WORKERS = 4        # parallel API calls (stay well under 180 req/min)
FEATURE_WORKERS = min(24, multiprocessing.cpu_count())  # threads; numpy/pandas release GIL


class IntradayModelTrainer:
    """
    Orchestrates 5-min bar fetching, per-day labelling,
    feature engineering, and XGBoost training for the intraday model.

    Key design:
      - Parquet cache per symbol avoids re-fetching on reruns
      - Symbols are fetched in parallel chunks via ThreadPoolExecutor
      - Feature rows are computed per-symbol in parallel
    """

    def __init__(self, model_dir: str = MODEL_DIR, provider: str = "alpaca"):
        self.model_dir = model_dir
        self.model = PortfolioSelectorModel(model_type="xgboost")
        self._provider_name = provider
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def _provider(self):
        return get_provider(self._provider_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def train_model(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 730,
        fetch_spy: bool = True,
        force_refresh: bool = False,
        use_ranker: bool = False,
        top_n_by_liquidity: Optional[int] = None,
    ) -> int:
        """
        Full pipeline: fetch/cache 5-min bars → label → features → train → save.

        Args:
            symbols:       List of tickers. Defaults to RUSSELL_1000_TICKERS.
            days:          Calendar days of history (730 ≈ 2 years).
            fetch_spy:     Include SPY relative-strength features.
            force_refresh: Ignore Parquet cache and re-fetch everything.
            use_ranker:    If True, use XGBRanker (rank:pairwise) with path_quality scores
                           as targets, grouped by trading day. Aligns training objective with
                           PM's top-N daily selection task.
            top_n_by_liquidity: If set, filter symbol universe to top-N by 20-day median
                           dollar volume before training. Improves signal quality by removing
                           illiquid/delisted names.

        Returns:
            Version number of the saved model.
        """
        from app.utils.constants import RUSSELL_1000_TICKERS
        symbols = symbols or RUSSELL_1000_TICKERS

        end_dt = datetime.now(tz=timezone.utc)
        start_dt = end_dt - timedelta(days=days + 10)  # buffer for weekends/holidays

        self._force_refresh = force_refresh
        logger.info(
            "Intraday training | symbols=%d  days=%d  provider=%s  cache=%s",
            len(symbols), days, self._provider_name, CACHE_DIR,
        )

        # ── 1. Fetch / refresh data ───────────────────────────────────────────
        t0 = datetime.now()
        symbols_data = self._fetch_data(symbols, start_dt, end_dt)
        spy_data = self._fetch_spy(start_dt, end_dt, force_refresh) if fetch_spy else None
        daily_data = self._fetch_daily_all(symbols, start_dt, end_dt)
        spy_daily_data = self._fetch_daily_all(["SPY"], start_dt, end_dt).get("SPY")
        logger.info("Data fetch complete in %.1fs — %d/%d symbols",
                    (datetime.now() - t0).total_seconds(), len(symbols_data), len(symbols))

        if not symbols_data:
            raise RuntimeError("No 5-min data fetched.")

        # ── 1b. Liquidity filter: keep top-N symbols by 20-day median dollar volume ──
        if top_n_by_liquidity is not None and daily_data:
            dv_scores: Dict[str, float] = {}
            for sym, df in daily_data.items():
                if df is None or len(df) < 5:
                    continue
                df_tail = df.tail(20)
                dv = (df_tail["close"] * df_tail["volume"]).median()
                dv_scores[sym] = float(dv) if pd.notna(dv) else 0.0
            ranked = sorted(dv_scores, key=lambda s: dv_scores[s], reverse=True)
            keep_set = set(ranked[:top_n_by_liquidity])
            before = len(symbols_data)
            symbols_data = {s: v for s, v in symbols_data.items() if s in keep_set}
            daily_data = {s: v for s, v in daily_data.items() if s in keep_set}
            logger.info(
                "Liquidity filter: kept %d/%d symbols (top-%d by 20d median dollar volume)",
                len(symbols_data), before, top_n_by_liquidity,
            )

        # ── 2. Build feature matrix (parallel per symbol) ─────────────────────
        t1 = datetime.now()
        X_train, y_train, X_test, y_test, feature_names, raw_train = self._build_daily_matrix(
            symbols_data, spy_data, daily_data, spy_daily_data
        )
        logger.info("Feature matrix built in %.1fs — %d train / %d test rows, %d features",
                    (datetime.now() - t1).total_seconds(),
                    len(X_train), len(X_test), len(feature_names))

        if len(X_train) == 0:
            raise RuntimeError("No valid training samples after labelling.")

        # ── 3. Train ──────────────────────────────────────────────────────────
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
        logger.info("Class balance: pos=%d  neg=%d  scale_pos_weight=%.2f", n_pos, n_neg, spw)

        # Sample weights: exponential recency decay (half-life = 180 days)
        sample_weight = None
        if len(raw_train) > 0:
            day_ords = raw_train[:, 0]
            max_ord = day_ords.max()
            half_life = 180.0
            sample_weight = np.exp((day_ords - max_ord) * np.log(2) / half_life).astype(np.float32)
            sample_weight /= sample_weight.mean()  # normalise to mean=1
            logger.info("Sample weights: min=%.3f  max=%.3f", sample_weight.min(), sample_weight.max())

        if use_ranker:
            # XGBRanker path: use raw path_quality scores as targets, grouped by day.
            # Sort rows by day_ordinal so XGBRanker sees contiguous groups.
            self.model = PortfolioSelectorModel(model_type="xgboost_ranker")
            raw_scores = raw_train[:, 1].astype(np.float32)  # path_quality scores
            sort_idx = np.argsort(raw_train[:, 0], kind="stable")
            X_train_r = X_train[sort_idx]
            y_ranker = raw_scores[sort_idx]
            # Compute groups: count of samples per unique day (in sorted order)
            sorted_day_ords = raw_train[sort_idx, 0]
            unique_days, group_counts = np.unique(sorted_day_ords, return_counts=True)
            # XGBRanker expects one weight per *group* (day), not per sample.
            # Use mean recency weight for each day.
            if sample_weight is not None:
                sw_sorted = sample_weight[sort_idx]
                day_weights = np.array([
                    float(sw_sorted[sorted_day_ords == d].mean()) for d in unique_days
                ], dtype=np.float32)
            else:
                day_weights = None
            logger.info("XGBRanker: %d groups (days), score range [%.3f, %.3f]",
                        len(group_counts), float(y_ranker.min()), float(y_ranker.max()))
            self.model.train(X_train_r, y_ranker, feature_names,
                             sample_weight=day_weights,
                             groups=group_counts.astype(np.int32))
        else:
            # Phase 87: frozen HPO params + 3-seed ensemble.
            # If FROZEN_HPO_PARAMS is set, skip HPO and use them directly.
            # If None, run thorough 100-trial HPO once to find good params.
            from xgboost import XGBClassifier
            hpo_params = FROZEN_HPO_PARAMS
            if hpo_params is None:
                try:
                    import optuna
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    from sklearn.model_selection import StratifiedKFold

                    def _objective(trial):
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                            "max_depth": trial.suggest_int("max_depth", 3, 7),
                            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                            "min_child_weight": trial.suggest_int("min_child_weight", 3, 40),
                            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                            "scale_pos_weight": spw,
                            "nthread": -1, "verbosity": 0, "eval_metric": "auc",
                            "random_state": 42,
                        }
                        cv = StratifiedKFold(n_splits=3, shuffle=False)
                        aucs = []
                        for tr_idx, va_idx in cv.split(X_train, y_train):
                            Xtr, Xva = X_train[tr_idx], X_train[va_idx]
                            ytr, yva = y_train[tr_idx], y_train[va_idx]
                            sw_tr = sample_weight[tr_idx] if sample_weight is not None else None
                            clf = XGBClassifier(**params)
                            clf.fit(Xtr, ytr, sample_weight=sw_tr)
                            from sklearn.metrics import roc_auc_score
                            aucs.append(roc_auc_score(yva, clf.predict_proba(Xva)[:, 1]))
                        return float(np.mean(aucs))

                    study = optuna.create_study(direction="maximize")
                    study.optimize(_objective, n_trials=100, show_progress_bar=False)
                    hpo_params = study.best_params
                    logger.info("HPO best params (freeze these in FROZEN_HPO_PARAMS): %s  AUC=%.4f",
                                hpo_params, study.best_value)
                except Exception as exc:
                    logger.warning("HPO failed, using defaults: %s", exc)
                    hpo_params = {}
            else:
                logger.info("Using frozen HPO params: %s", hpo_params)

            # Phase 87: 3-seed XGBoost ensemble — train on seeds 42, 123, 777 and blend.
            ensemble_models = []
            for seed in ENSEMBLE_SEEDS:
                seed_params = {**(hpo_params or {}), "random_state": seed,
                               "scale_pos_weight": spw, "nthread": -1, "verbosity": 0,
                               "eval_metric": "auc"}
                clf = XGBClassifier(**seed_params)
                clf.fit(X_train, y_train, sample_weight=sample_weight)
                ensemble_models.append(clf)
                logger.info("Trained XGBoost seed=%d", seed)

            # Store ensemble on model object for inference blending
            self.model.ensemble_models = ensemble_models
            # Primary model (seed=42) for backward-compat save/load
            self.model.model.set_params(**(hpo_params or {}),
                                        random_state=ENSEMBLE_SEEDS[0],
                                        scale_pos_weight=spw)
            self.model.train(X_train, y_train, feature_names, scale_pos_weight=spw,
                             sample_weight=sample_weight)

        # Phase 87: blend 3-seed XGBoost ensemble probabilities for evaluation
        ensemble_proba_test = None
        if not use_ranker and hasattr(self.model, "ensemble_models") and len(self.model.ensemble_models) > 1:
            try:
                seed_probas = [m.predict_proba(X_test)[:, 1] for m in self.model.ensemble_models]
                ensemble_proba_test = np.mean(seed_probas, axis=0)
                logger.info("3-seed XGBoost ensemble blended for evaluation")
            except Exception as exc:
                logger.warning("Ensemble blending failed: %s", exc)

        # LightGBM ensemble: train LGBM alongside XGBoost, soft-vote probabilities
        lgbm_proba_test = None
        try:
            from lightgbm import LGBMClassifier
            lgbm = LGBMClassifier(
                n_estimators=400, learning_rate=0.03, max_depth=6,
                subsample=0.73, colsample_bytree=0.58, min_child_samples=24,
                class_weight={0: 1.0, 1: float(spw)},
                n_jobs=24, random_state=42, verbose=-1,
            )
            lgbm.fit(X_train, y_train, sample_weight=sample_weight)
            lgbm_proba_test = lgbm.predict_proba(X_test)[:, 1]
            logger.info("LightGBM trained — ensemble enabled")
        except Exception as exc:
            logger.warning("LightGBM training failed: %s", exc)

        # Final blended proba: average XGBoost ensemble + LGBM if both available
        if ensemble_proba_test is not None and lgbm_proba_test is not None:
            final_proba_test = (ensemble_proba_test + lgbm_proba_test) / 2.0
        elif ensemble_proba_test is not None:
            final_proba_test = ensemble_proba_test
        else:
            final_proba_test = lgbm_proba_test

        metrics = self._evaluate(X_test, y_test, lgbm_proba=final_proba_test)
        logger.info("OOS metrics: %s", metrics)

        # ── 4. Save ───────────────────────────────────────────────────────────
        version = self._next_version("intraday")
        saved_path = self.model.save(self.model_dir, version, model_name="intraday")
        training_config = {
            "days": days,
            "n_features": len(feature_names),
            "feature_names": list(feature_names),
            "entry_offsets": ENTRY_OFFSETS,
            "lgbm_ensemble": lgbm_proba_test is not None,
            "xgb_3seed_ensemble": ensemble_proba_test is not None,
            "frozen_hpo": FROZEN_HPO_PARAMS is not None,
            "label_scheme": "cross_sectional_top20pct_abs_hurdle_0.30pct",
            "use_ranker": use_ranker,
            "top_n_by_liquidity": top_n_by_liquidity,
        }
        self._record_version(version, len(X_train), len(X_test), saved_path, days, metrics,
                             training_config=training_config)

        logger.info("Total training time: %.1fs", (datetime.now() - t0).total_seconds())
        return version

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, symbol: str) -> Path:
        return CACHE_DIR / f"{symbol}.parquet"

    def _load_meta(self) -> dict:
        if META_FILE.exists():
            try:
                return json.loads(META_FILE.read_text())
            except Exception:
                pass
        return {}

    def _save_meta(self, meta: dict) -> None:
        META_FILE.write_text(json.dumps(meta, indent=2))

    def _cache_is_fresh(self, symbol: str, start: datetime, meta: dict) -> bool:
        """Return True if the cached Parquet covers [start, today] and was fetched today."""
        info = meta.get(symbol)
        if not info:
            return False
        path = self._cache_path(symbol)
        if not path.exists():
            return False
        fetched = datetime.fromisoformat(info.get("fetched_at", "2000-01-01"))
        cached_start = datetime.fromisoformat(info.get("start", "2000-01-01"))
        today = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        return (
            (today - fetched).total_seconds() < 86400 and   # fetched within 24h
            cached_start <= start.replace(tzinfo=None)       # covers requested range
        )

    def _read_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_parquet(self._cache_path(symbol))
        except Exception:
            return None

    def _write_cache(self, symbol: str, df: pd.DataFrame, start: datetime, meta: dict) -> None:
        try:
            df.to_parquet(self._cache_path(symbol))
            meta[symbol] = {
                "fetched_at": datetime.utcnow().isoformat(),
                "start": start.replace(tzinfo=None).isoformat(),
            }
        except Exception as exc:
            logger.debug("Cache write failed for %s: %s", symbol, exc)

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_all(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        force_refresh: bool,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch 5-min bars for all symbols with Parquet caching.
        Symbols already in fresh cache are loaded from disk; the rest are
        fetched from Alpaca in parallel chunks of FETCH_CHUNK_SIZE.
        """
        meta = self._load_meta()
        result: Dict[str, pd.DataFrame] = {}
        to_fetch: List[str] = []

        for sym in symbols:
            if not force_refresh and self._cache_is_fresh(sym, start, meta):
                df = self._read_cache(sym)
                if df is not None and len(df) > 0:
                    result[sym] = df
                    continue
            to_fetch.append(sym)

        logger.info("Cache hit: %d symbols | fetching: %d symbols", len(result), len(to_fetch))

        if to_fetch:
            fetched = self._fetch_parallel(to_fetch, start, end)
            for sym, df in fetched.items():
                result[sym] = df
                self._write_cache(sym, df, start, meta)
            self._save_meta(meta)

        return result

    def _fetch_parallel(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Split symbols into chunks, fetch each chunk in parallel."""
        chunks = [symbols[i:i + FETCH_CHUNK_SIZE]
                  for i in range(0, len(symbols), FETCH_CHUNK_SIZE)]
        result: Dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=FETCH_WORKERS, thread_name_prefix="fetch") as pool:
            futures = {
                pool.submit(self._fetch_chunk, chunk, start, end): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    data = future.result()
                    result.update(data)
                    logger.info("Fetched chunk: %d/%d symbols so far",
                                len(result), len(symbols))
                except Exception as exc:
                    logger.warning("Chunk fetch failed (%s…): %s", chunk[0], exc)

        return result

    def _fetch_chunk(
        self, symbols: List[str], start: datetime, end: datetime
    ) -> Dict[str, pd.DataFrame]:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from app.config import settings

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed="iex",   # free-tier uses IEX; paid plans can use "sip"
        )
        resp = data_client.get_stock_bars(req)

        result = {}
        for sym in symbols:
            try:
                bars = resp[sym]
                if not bars:
                    continue
                records = [{"open": b.open, "high": b.high, "low": b.low,
                            "close": b.close, "volume": b.volume}
                           for b in bars]
                timestamps = [b.timestamp for b in bars]
                df = pd.DataFrame(records,
                                  index=pd.DatetimeIndex(timestamps, name="timestamp"))
                result[sym] = df.astype(float)
            except (KeyError, TypeError):
                pass
        return result

    def _fetch_spy(
        self, start: datetime, end: datetime, force_refresh: bool
    ) -> Optional[pd.DataFrame]:
        meta = self._load_meta()
        if not force_refresh and self._cache_is_fresh("SPY", start, meta):
            df = self._read_cache("SPY")
            if df is not None:
                return df
        try:
            chunk = self._fetch_chunk(["SPY"], start, end)
            df = chunk.get("SPY")
            if df is not None:
                self._write_cache("SPY", df, start, meta)
                self._save_meta(meta)
            return df
        except Exception as exc:
            logger.warning("SPY fetch failed: %s", exc)
            return None

    def _fetch_daily(self, symbols, start, end):
        return self._fetch_daily_all(symbols, start, end)

    # Public-facing names used by tests and train_model
    def _fetch_data(self, symbols, start, end):
        # Prefer Polygon Parquet cache when available (2yr history)
        from app.data.intraday_cache import load_many, available_symbols
        polygon_syms = set(available_symbols())
        if polygon_syms:
            start_date = start.date() if hasattr(start, "date") else start
            end_date = end.date() if hasattr(end, "date") else end
            polygon_hit = load_many(
                [s for s in symbols if s in polygon_syms],
                start=start_date, end=end_date,
            )
            missing = [s for s in symbols if s not in polygon_hit]
            if missing:
                alpaca_hit = self._fetch_all(missing, start, end,
                                             force_refresh=self._force_refresh)
                polygon_hit.update(alpaca_hit)
            logger.info(
                "Data load: %d from Polygon cache, %d from Alpaca",
                len([s for s in symbols if s in polygon_syms and s in polygon_hit]),
                len([s for s in symbols if s not in polygon_syms]),
            )
            return polygon_hit
        return self._fetch_all(symbols, start, end, force_refresh=self._force_refresh)

    def _build_daily_matrix(self, symbols_data, spy_data, daily_data=None, spy_daily_data=None):
        X_tr, y_tr, X_te, y_te, fnames, raw_tr = self._build_matrix_parallel(
            symbols_data, spy_data, daily_data or {}, spy_daily_data
        )
        return X_tr, y_tr, X_te, y_te, fnames, raw_tr

    def _fetch_daily_all(
        self, symbols: List[str], start: datetime, end: datetime
    ) -> Dict[str, pd.DataFrame]:
        try:
            daily_start = start - timedelta(days=365)  # extra year for vol percentile
            data = self._provider.get_daily_bars_bulk(symbols, daily_start, end)
            logger.info("Daily bars: %d/%d symbols", len(data), len(symbols))
            return data
        except Exception as exc:
            logger.warning("Daily bar fetch failed: %s", exc)
            return {}

    # ── Feature matrix (parallel) ─────────────────────────────────────────────

    def _build_matrix_parallel(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        spy_data: Optional[pd.DataFrame],
        daily_data: Dict[str, pd.DataFrame],
        spy_daily_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        # Collect all trading days across all symbols
        all_days: set = set()
        for df in symbols_data.values():
            if df is not None and len(df) > 0:
                idx = pd.DatetimeIndex(df.index)
                for d in idx.normalize().unique():
                    all_days.add(d.date())

        sorted_days = sorted(all_days)
        if len(sorted_days) < MIN_DAYS:
            logger.warning("Only %d trading days — need at least %d", len(sorted_days), MIN_DAYS)
            return np.array([]), np.array([]), np.array([]), np.array([]), [], []

        split_idx = max(1, int(len(sorted_days) * (1 - TEST_FRACTION)))
        train_days = set(sorted_days[:split_idx])
        # Embargo: skip 1 day so the last training day's 2h label period
        # does not bleed into the first test day's features via daily bar history.
        embargo_start = min(split_idx + 1, len(sorted_days))
        test_days = set(sorted_days[embargo_start:])

        # Precompute SPY day-slices once to avoid per-symbol work
        spy_by_day = _index_by_day(spy_data) if spy_data is not None else {}

        X_train_parts, raw_train_parts = [], []
        X_test_parts, raw_test_parts = [], []
        feature_names: List[str] = []

        syms = list(symbols_data.keys())
        tasks = [
            (sym, symbols_data[sym], spy_by_day, daily_data.get(sym), train_days, test_days, spy_daily_data)
            for sym in syms
        ]

        with ThreadPoolExecutor(max_workers=FEATURE_WORKERS,
                                thread_name_prefix="features") as pool:
            futures = {pool.submit(_symbol_to_rows, t): t[0] for t in tasks}
            done = 0
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    rows_train, raw_train, rows_test, raw_test, fnames = future.result()
                    if rows_train:
                        X_train_parts.append(np.array(rows_train, dtype=np.float32))
                        raw_train_parts.append(np.array(raw_train, dtype=np.float64))
                    if rows_test:
                        X_test_parts.append(np.array(rows_test, dtype=np.float32))
                        raw_test_parts.append(np.array(raw_test, dtype=np.float64))
                    if fnames and not feature_names:
                        feature_names.extend(fnames)
                except Exception as exc:
                    logger.debug("Feature computation failed for %s: %s", sym, exc)
                done += 1
                if done % 100 == 0:
                    logger.info("Features: %d/%d symbols processed", done, len(syms))

        # ── Phase 89: restore cross-sectional top-20% labels ────────────────────
        # raw_parts carry [day_ordinal, raw_2h_return]. Per-day top-20% → label=1.
        # cs_normalize applied to features (not labels) for cross-sectional alignment.
        # Phase 1 (label fix): minimum absolute 2h return to qualify as a positive label.
        # Prevents labeling "least bad" stocks as winners on down days.
        # 0.30% = roughly 1/3 of the 0.8×ATR target — a stock must actually move up.
        CS_ABSOLUTE_HURDLE = 0.0030

        def _cross_sectional_labels(X_parts, raw_parts):
            """Rank raw 2h returns within each day; top 20% AND above hurdle get label=1."""
            if not X_parts:
                return np.array([]), np.array([])
            X = np.vstack(X_parts)
            raws = np.concatenate(raw_parts)   # shape (N, 2): [day_ordinal, raw_return]
            days_ord = raws[:, 0]
            raw_returns = raws[:, 1]
            labels = np.zeros(len(raw_returns), dtype=np.int8)
            for day_val in np.unique(days_ord):
                mask = days_ord == day_val
                day_rets = raw_returns[mask]
                if mask.sum() < 2:
                    continue
                threshold = np.percentile(day_rets, 80)  # top 20%
                # Must be top-20% AND have a minimum positive absolute return
                labels[mask] = (
                    (day_rets >= threshold) & (day_rets >= CS_ABSOLUTE_HURDLE)
                ).astype(np.int8)
            # Cross-sectional normalization: z-score each feature within each day
            X = cs_normalize_by_group(X, days_ord)
            return X, labels

        X_train, y_train = _cross_sectional_labels(X_train_parts, raw_train_parts)
        X_test, y_test = _cross_sectional_labels(X_test_parts, raw_test_parts)

        # Keep raw_train for sample-weight computation in caller
        raw_train = np.concatenate(raw_train_parts) if raw_train_parts else np.array([])
        return X_train, y_train, X_test, y_test, feature_names, raw_train

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                  lgbm_proba: Optional[np.ndarray] = None) -> Dict:
        if len(X_test) == 0:
            return {}
        try:
            from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
            _, xgb_proba = self.model.predict(X_test)
            if lgbm_proba is not None:
                proba = (xgb_proba + lgbm_proba) / 2.0
                logger.info("Using XGBoost+LightGBM ensemble probabilities")
            else:
                proba = xgb_proba
            preds = (proba >= 0.5).astype(int)
            return {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
                "auc": round(float(roc_auc_score(y_test, proba)), 4),
                "n_test": len(y_test),
            }
        except Exception as exc:
            logger.warning("Evaluation failed: %s", exc)
            return {}

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _next_version(self, model_name: str = "intraday") -> int:
        try:
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
        except Exception:
            from pathlib import Path as _P
            files = sorted(_P(self.model_dir).glob(f"{model_name}_v*.pkl"))
            return (int(files[-1].stem.split("_v")[-1]) + 1) if files else 1

    @staticmethod
    def record_tier3_result(
        version: int,
        avg_sharpe: float,
        fold_sharpes: list,
        gate_passed: bool,
    ) -> None:
        """Write walk-forward results back into the ModelVersion performance field."""
        try:
            db = get_session()
            row = db.query(ModelVersion).filter_by(
                model_name="intraday", version=version
            ).first()
            if row is None:
                return
            perf = dict(row.performance or {})
            perf["tier3_sharpe"] = round(avg_sharpe, 4)
            perf["tier3_gate_passed"] = gate_passed
            perf["tier3_fold_sharpes"] = [round(s, 4) for s in fold_sharpes]
            row.performance = perf
            if not gate_passed:
                row.status = "RETIRED"
                logger.info("Intraday v%d tier3 FAIL — status set to RETIRED", version)
            else:
                # Write gate_passed sentinel so the pkl fallback loader can find this version
                from pathlib import Path as _Path
                model_dir = _Path("app/ml/models")
                sentinel = model_dir / f"intraday_v{version}.gate_passed"
                sentinel.touch()
                logger.info("Intraday v%d gate_passed sentinel written", version)
            db.commit()
            logger.info("Recorded tier3 result for intraday v%d: Sharpe=%.3f gate=%s",
                        version, avg_sharpe, gate_passed)
        except Exception as exc:
            logger.warning("Could not record tier3 result: %s", exc)
        finally:
            try:
                db.close()
            except Exception:
                pass

    def _record_version(
        self, version: int, n_train: int, n_test: int,
        model_path: str, days: int, metrics: Dict,
        training_config: Optional[Dict] = None,
    ) -> None:
        try:
            db = get_session()
        except Exception as exc:
            logger.warning("DB not available, skipping version record: %s", exc)
            return
        try:
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=days)
            # Retire previous ACTIVE versions — only one version is active at a time
            prev = db.query(ModelVersion).filter_by(
                model_name="intraday", status="ACTIVE"
            ).all()
            for p in prev:
                p.status = "RETIRED"
                logger.info("Retired intraday v%d", p.version)
            performance = {
                **metrics,
                "n_train": n_train,
                "n_test": n_test,
                # Tier 3 walk-forward results — populated later by record_tier3_result()
                "tier3_sharpe": None,
                "tier3_gate_passed": None,
                "tier3_fold_sharpes": None,
            }
            if training_config:
                performance["training_config"] = training_config
            db.add(ModelVersion(
                model_name="intraday",
                version=version,
                training_date=datetime.utcnow(),
                data_range_start=start_dt.strftime("%Y-%m-%d"),
                data_range_end=end_dt.strftime("%Y-%m-%d"),
                performance=performance,
                status="ACTIVE",
                model_path=model_path,
            ))
            db.commit()
            logger.info("Intraday model v%d recorded in DB", version)
        except Exception as exc:
            db.rollback()
            logger.error("Failed to record model version: %s", exc)
        finally:
            db.close()


# ── Module-level helpers (used in ThreadPoolExecutor — must be picklable) ─────

def _index_by_day(df: pd.DataFrame) -> Dict[date, pd.DataFrame]:
    """Pre-slice a DataFrame into {date: day_bars} for O(1) lookup."""
    if df is None or len(df) == 0:
        return {}
    idx = pd.DatetimeIndex(df.index)
    result = {}
    for d in idx.normalize().unique():
        mask = idx.normalize() == d
        result[d.date()] = df.loc[mask]
    return result


def _symbol_to_rows(
    args: tuple,
) -> Tuple[List, List, List, List, List[str]]:
    """
    Compute feature rows + outcome labels for one symbol across all days.
    Returns (train_rows, train_labels, test_rows, test_labels, feature_names).

    Accepts a single tuple so ProcessPoolExecutor can pickle it:
        (sym, df, spy_by_day, daily_df, train_days, test_days)
    """
    sym, df, spy_by_day, daily_df, train_days, test_days, spy_daily_df = args if len(args) == 7 else (*args, None)

    if df is None or len(df) == 0:
        return [], [], [], [], [], []

    # Pre-group 5-min bars by date once — avoids O(n_bars × n_days) repeated masking
    df_idx = pd.DatetimeIndex(df.index)
    date_arr = np.array(df_idx.normalize().date)
    day_groups: Dict[date, pd.DataFrame] = {}
    for d in np.unique(date_arr):
        day_groups[d] = df.iloc[date_arr == d]

    # Pre-group daily bars by date for O(1) slicing
    daily_date_arr = None
    if daily_df is not None and len(daily_df) > 0:
        d_idx = pd.DatetimeIndex(daily_df.index)
        daily_date_arr = np.array(d_idx.normalize().date)

    # SPY daily bars date array for point-in-time slicing (Phase 86b)
    spy_daily_date_arr = None
    if spy_daily_df is not None and len(spy_daily_df) > 0:
        s_idx = pd.DatetimeIndex(spy_daily_df.index)
        spy_daily_date_arr = np.array(s_idx.normalize().date)

    train_rows, train_raw = [], []
    test_rows, test_raw = [], []
    feature_names: List[str] = []

    all_days = sorted(train_days | test_days)

    # Phase 50: multi-offset entry — sample at 3 points in the session so the model
    # learns how edge varies by time-of-day (open / mid / afternoon segments).
    # Each offset is the number of 5-min bars to use as features; remaining bars = hold window.
    ENTRY_OFFSETS = [12]  # single scan; multi-window (v30) caused distribution mismatch at inference

    for i, day in enumerate(all_days):
        day_bars = day_groups.get(day)
        if day_bars is None or len(day_bars) < max(ENTRY_OFFSETS) + HOLD_BARS:
            continue

        for entry_offset in ENTRY_OFFSETS:
            if len(day_bars) < entry_offset + HOLD_BARS:
                continue
            feat_bars = day_bars.iloc[:entry_offset]
            future_bars = day_bars.iloc[entry_offset:entry_offset + HOLD_BARS]

            if len(feat_bars) < MIN_BARS:
                continue

            # Prior-day OHLC — use pre-grouped dict, look back one step
            prior_close = prior_high = prior_low = None
            prior_days = [d for d in all_days[:i] if d in day_groups]
            if prior_days:
                prev = day_groups[prior_days[-1]]
                prior_close = float(prev["close"].iloc[-1])
                prior_high = float(prev["high"].max())
                prior_low = float(prev["low"].min())

            # Phase 87: realized-R outcome label (replaces path_quality cross-sectional ranking).
            # Label = 1 if the trade achieves ≥0.5R gain AND ≥0.30% absolute move.
            # Days with no qualifying setups get zero positive labels — model learns to abstain.
            if prior_high is not None and prior_low is not None and prior_close is not None:
                prior_range = prior_high - prior_low
                stop_dist_abs = ATR_MULT_STOP * prior_range
                target_dist_abs = ATR_MULT_TARGET * prior_range
            else:
                stop_dist_abs = STOP_PCT * 100.0   # fallback in absolute $ terms ~ $0.30
                target_dist_abs = TARGET_PCT * 100.0

            entry = float(feat_bars["close"].iloc[-1])
            stop_dist_use = max(stop_dist_abs, entry * 0.002)    # floor 0.2%
            target_dist_use = max(target_dist_abs, entry * 0.003)  # floor 0.3%

            max_high = entry
            min_low = entry
            realized_exit = entry
            for _, fbar in future_bars.iterrows():
                h = float(fbar["high"])
                lo = float(fbar["low"])
                c = float(fbar["close"])
                if lo <= entry - stop_dist_use:
                    realized_exit = entry - stop_dist_use
                    break
                if h >= entry + target_dist_use:
                    realized_exit = entry + target_dist_use
                    break
                max_high = max(max_high, h)
                min_low = min(min_low, lo)
                realized_exit = c

            # Phase 89: raw 2h return — cross-sectional top-20% ranking applied in caller
            best_return = (realized_exit - entry) / max(entry, 1e-8)

            # Daily bars up to this day (O(1) slice via precomputed date array)
            daily_as_of = None
            if daily_df is not None and daily_date_arr is not None:
                daily_as_of = daily_df.iloc[daily_date_arr < day]

            spy_daily_as_of = None
            if spy_daily_df is not None and spy_daily_date_arr is not None:
                spy_daily_as_of = spy_daily_df.iloc[spy_daily_date_arr < day]

            feats = compute_intraday_features(
                feat_bars,
                spy_by_day.get(day),
                prior_close,
                prior_day_high=prior_high,
                prior_day_low=prior_low,
                daily_bars=daily_as_of,
                spy_daily_bars=spy_daily_as_of,
                symbol=sym,
                as_of_date=day,
            )
            if feats is None:
                continue

            # Filter feats to FEATURE_NAMES (authoritative list) — preserves canonical order
            # and excludes features added by compute_intraday_features() that are not in the list
            # (e.g. market-wide regime proxies zeroed by cs_normalize, NIS when disabled).
            feats = {k: feats[k] for k in _INTRADAY_FEATURE_NAMES if k in feats}

            if not feature_names:
                feature_names = list(feats.keys())

            row = list(feats.values())
            # raw = [day_ordinal, best_return] for cross-sectional ranking in caller
            day_ord = float(day.toordinal())
            if day in train_days:
                train_rows.append(row)
                train_raw.append([day_ord, best_return])
            else:
                test_rows.append(row)
                test_raw.append([day_ord, best_return])

    return train_rows, train_raw, test_rows, test_raw, feature_names
