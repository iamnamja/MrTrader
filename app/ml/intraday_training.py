"""
Intraday model training pipeline — optimized for 1000 symbols × 2 years of 5-min bars.

Label: outcome-based — price hits +TARGET_PCT before -STOP_PCT within HOLD_BARS bars.
Window: each trading day is one window per stock.
Train/test split: last TEST_FRACTION of days = test (time-based, no leakage).

Data source: Alpaca 5-min bars with Parquet disk cache (incremental refresh).

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
from app.ml.intraday_features import compute_intraday_features, MIN_BARS
from app.ml.model import PortfolioSelectorModel

logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"
CACHE_DIR = Path("data/cache/5min")
META_FILE = CACHE_DIR / "_meta.json"

# Training config
TARGET_PCT = 0.005       # 0.5% intraday profit target
STOP_PCT = 0.003         # 0.3% intraday stop loss
HOLD_BARS = 24           # 2 hours of 5-min bars to achieve the target
TEST_FRACTION = 0.20     # most recent 20% of days = test
MIN_DAYS = 20            # minimum trading days needed per symbol

# Parallelism config
FETCH_CHUNK_SIZE = 100   # symbols per Alpaca API call
FETCH_WORKERS = 4        # parallel API calls (stay well under 180 req/min)
FEATURE_WORKERS = min(16, multiprocessing.cpu_count())  # threads; numpy/pandas release GIL


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
    ) -> int:
        """
        Full pipeline: fetch/cache 5-min bars → label → features → train → save.

        Args:
            symbols:       List of tickers. Defaults to RUSSELL_1000_TICKERS.
            days:          Calendar days of history (730 ≈ 2 years).
            fetch_spy:     Include SPY relative-strength features.
            force_refresh: Ignore Parquet cache and re-fetch everything.

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
        logger.info("Data fetch complete in %.1fs — %d/%d symbols",
                    (datetime.now() - t0).total_seconds(), len(symbols_data), len(symbols))

        if not symbols_data:
            raise RuntimeError("No 5-min data fetched.")

        # ── 2. Build feature matrix (parallel per symbol) ─────────────────────
        t1 = datetime.now()
        X_train, y_train, X_test, y_test, feature_names = self._build_daily_matrix(
            symbols_data, spy_data, daily_data
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

        self.model.train(X_train, y_train, feature_names, scale_pos_weight=spw)

        metrics = self._evaluate(X_test, y_test)
        logger.info("OOS metrics: %s", metrics)

        # ── 4. Save ───────────────────────────────────────────────────────────
        version = self._next_version("intraday")
        saved_path = self.model.save(self.model_dir, version, model_name="intraday")
        self._record_version(version, len(X_train), len(X_test), saved_path, days, metrics)

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
        return self._fetch_all(symbols, start, end, force_refresh=self._force_refresh)

    def _build_daily_matrix(self, symbols_data, spy_data, daily_data=None):
        return self._build_matrix_parallel(symbols_data, spy_data, daily_data or {})

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
            return np.array([]), np.array([]), np.array([]), np.array([]), []

        split_idx = max(1, int(len(sorted_days) * (1 - TEST_FRACTION)))
        train_days = set(sorted_days[:split_idx])
        test_days = set(sorted_days[split_idx:])

        # Precompute SPY day-slices once to avoid per-symbol work
        spy_by_day = _index_by_day(spy_data) if spy_data is not None else {}

        X_train_parts, y_train_parts = [], []
        X_test_parts, y_test_parts = [], []
        feature_names: List[str] = []

        syms = list(symbols_data.keys())
        # Pack args as tuples for ProcessPoolExecutor (must be picklable)
        tasks = [
            (sym, symbols_data[sym], spy_by_day, daily_data.get(sym), train_days, test_days)
            for sym in syms
        ]

        with ThreadPoolExecutor(max_workers=FEATURE_WORKERS,
                                thread_name_prefix="features") as pool:
            futures = {pool.submit(_symbol_to_rows, t): t[0] for t in tasks}
            done = 0
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    rows_train, labels_train, rows_test, labels_test, fnames = future.result()
                    if rows_train:
                        X_train_parts.append(np.array(rows_train, dtype=np.float32))
                        y_train_parts.append(np.array(labels_train, dtype=np.int8))
                    if rows_test:
                        X_test_parts.append(np.array(rows_test, dtype=np.float32))
                        y_test_parts.append(np.array(labels_test, dtype=np.int8))
                    if fnames and not feature_names:
                        feature_names.extend(fnames)
                except Exception as exc:
                    logger.debug("Feature computation failed for %s: %s", sym, exc)
                done += 1
                if done % 100 == 0:
                    logger.info("Features: %d/%d symbols processed", done, len(syms))

        X_train = np.vstack(X_train_parts) if X_train_parts else np.array([])
        y_train = np.concatenate(y_train_parts) if y_train_parts else np.array([])
        X_test = np.vstack(X_test_parts) if X_test_parts else np.array([])
        y_test = np.concatenate(y_test_parts) if y_test_parts else np.array([])

        return X_train, y_train, X_test, y_test, feature_names

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
        model_path: str, days: int, metrics: Dict,
    ) -> None:
        db = get_session()
        try:
            end_dt = datetime.utcnow()
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
    sym, df, spy_by_day, daily_df, train_days, test_days = args

    if df is None or len(df) == 0:
        return [], [], [], [], []

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

    train_rows, train_labels = [], []
    test_rows, test_labels = [], []
    feature_names: List[str] = []

    all_days = sorted(train_days | test_days)

    for i, day in enumerate(all_days):
        day_bars = day_groups.get(day)
        if day_bars is None or len(day_bars) < MIN_BARS + HOLD_BARS:
            continue

        feat_bars = day_bars.iloc[:-HOLD_BARS]
        future_bars = day_bars.iloc[-HOLD_BARS:]

        if len(feat_bars) < MIN_BARS:
            continue

        # Outcome label
        entry = float(feat_bars["close"].iloc[-1])
        target_price = entry * (1 + TARGET_PCT)
        stop_price = entry * (1 - STOP_PCT)
        label = 0
        for _, bar in future_bars.iterrows():
            if float(bar["high"]) >= target_price:
                label = 1
                break
            if float(bar["low"]) <= stop_price:
                break

        # Prior-day OHLC — use pre-grouped dict, look back one step
        prior_close = prior_high = prior_low = None
        prior_days = [d for d in all_days[:i] if d in day_groups]
        if prior_days:
            prev = day_groups[prior_days[-1]]
            prior_close = float(prev["close"].iloc[-1])
            prior_high = float(prev["high"].max())
            prior_low = float(prev["low"].min())

        # Daily bars up to this day (O(1) slice via precomputed date array)
        daily_as_of = None
        if daily_df is not None and daily_date_arr is not None:
            daily_as_of = daily_df.iloc[daily_date_arr < day]

        feats = compute_intraday_features(
            feat_bars,
            spy_by_day.get(day),
            prior_close,
            prior_day_high=prior_high,
            prior_day_low=prior_low,
            daily_bars=daily_as_of,
        )
        if feats is None:
            continue

        if not feature_names:
            feature_names = list(feats.keys())

        row = list(feats.values())
        if day in train_days:
            train_rows.append(row)
            train_labels.append(label)
        else:
            test_rows.append(row)
            test_labels.append(label)

    return train_rows, train_labels, test_rows, test_labels, feature_names


