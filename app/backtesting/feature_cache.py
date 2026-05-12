"""
Pre-computed feature cache for walk-forward backtesting.

Eliminates the bottleneck of calling FeatureEngineer.engineer_features()
for every (symbol, day) pair during AgentSimulator's inner loop. Instead,
features are computed once per fold in parallel (ProcessPoolExecutor),
stored as per-symbol float32 arrays, and looked up in O(1) during simulation.

Expected speedup: ~25-30x over the baseline sequential approach.

Memory budget: ~750 syms × 1260 days × 80 features × 4 bytes ≈ 300 MB per fold.
Cache is built and discarded per-fold (never held for all 3 folds simultaneously).

Usage in run_fold:
    cache = build_feature_cache(
        symbols_data=fold_symbols_data,
        trading_days=test_trading_days,
        feature_names=model.feature_names,
        regime_score_history=regime_scores,
        vix_history=vix_series,
        workers=12,
    )
    sim = AgentSimulator(..., feature_cache=cache)
    result = sim.run(fold_symbols_data, ...)
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Synthetic symbols that are never PM-scored — skip during cache build.
_SKIP_SYMBOLS = frozenset({"^VIX", "VIX", "SPY"})

# Module-level singleton so each worker process constructs FeatureEngineer once.
_WORKER_FE = None


def _init_worker() -> None:
    global _WORKER_FE
    from app.ml.features import FeatureEngineer
    _WORKER_FE = FeatureEngineer()


def _build_symbol_rows(
    sym: str,
    df_records: list,           # df.to_dict("records") — picklable
    df_index: list,             # list of date objects (df.index.date)
    trading_days: List[date],
    feature_names: List[str],
    regime_score_history: Dict[date, float],
    vix_index: Optional[list],  # dates for vix
    vix_values: Optional[list],  # close values for vix
    macro_index: Optional[list] = None,   # dates for macro_history
    macro_records: Optional[list] = None,  # macro_history.to_dict("records")
) -> Tuple[str, List[date], List[list]]:
    """Worker function: compute raw features for one symbol across all trading days.

    Returns (sym, valid_dates, feature_rows) where valid_dates and feature_rows
    are aligned (one entry per day with sufficient history).
    """
    global _WORKER_FE
    if _WORKER_FE is None:
        from app.ml.features import FeatureEngineer
        _WORKER_FE = FeatureEngineer()

    # Reconstruct DataFrame from records (picklable form)
    df = pd.DataFrame.from_records(df_records, index=pd.DatetimeIndex(
        [pd.Timestamp(d) for d in df_index]
    ))

    # Reconstruct vix_history Series if provided
    vix_history = None
    if vix_index is not None and vix_values is not None:
        vix_history = pd.Series(
            vix_values,
            index=pd.DatetimeIndex([pd.Timestamp(d) for d in vix_index]),
        )

    # Reconstruct macro_history DataFrame if provided
    macro_history = None
    if macro_index is not None and macro_records is not None:
        macro_history = pd.DataFrame.from_records(macro_records)
        macro_history.insert(0, "date", macro_index)

    valid_dates: List[date] = []
    feature_rows: List[list] = []

    for day in trading_days:
        # bars strictly before today (exclude_today=True equivalent)
        mask = df.index.date < day
        bars = df.loc[mask]
        if len(bars) < 60:
            continue
        try:
            regime_score = regime_score_history.get(day, 0.5)
            feats = _WORKER_FE.engineer_features(
                sym, bars,
                fetch_fundamentals=False,
                as_of_date=day,
                regime_score=regime_score,
                vix_history=vix_history,
                macro_history=macro_history,
            )
            if feats is None:
                continue
            row = [float(feats.get(f, 0.0)) for f in feature_names]
            valid_dates.append(day)
            feature_rows.append(row)
        except Exception:
            continue

    return sym, valid_dates, feature_rows


@dataclass
class FeatureCache:
    """Per-fold pre-computed raw feature cache.

    Attributes:
        feature_names: canonical column order (matches model.feature_names).
        matrix:        sym → (T_sym, F) float32 array of raw features.
        dates:         sym → (T_sym,) array of date objects (sorted ascending).
        date_index:    sym → {date: row_idx} for O(1) lookup.
    """
    feature_names: List[str]
    matrix: Dict[str, np.ndarray] = field(default_factory=dict)
    dates: Dict[str, np.ndarray] = field(default_factory=dict)
    date_index: Dict[str, Dict[date, int]] = field(default_factory=dict)

    def get_row(self, sym: str, day: date) -> Optional[np.ndarray]:
        """Return raw feature vector for (sym, day), or None if not cached."""
        idx = self.date_index.get(sym, {}).get(day)
        if idx is None:
            return None
        return self.matrix[sym][idx]

    def get_features(self, sym: str, day: date) -> Optional[Dict[str, float]]:
        """Return raw feature dict for (sym, day), or None if not cached."""
        row = self.get_row(sym, day)
        if row is None:
            return None
        return dict(zip(self.feature_names, row.tolist()))

    def symbols_with(self, day: date) -> List[str]:
        """Return all symbols that have a cached feature vector for day."""
        return [s for s, idx_map in self.date_index.items() if day in idx_map]

    @property
    def n_symbols(self) -> int:
        return len(self.matrix)

    @property
    def memory_mb(self) -> float:
        return sum(m.nbytes for m in self.matrix.values()) / 1_048_576


def build_feature_cache(
    symbols_data: Dict[str, pd.DataFrame],
    trading_days: List[date],
    feature_names: List[str],
    regime_score_history: Optional[Dict[date, float]] = None,
    vix_history: Optional[pd.Series] = None,
    macro_history: Optional[pd.DataFrame] = None,
    workers: int = 0,
    executor: str = "process",
    skip_symbols: Iterable[str] = _SKIP_SYMBOLS,
    log_progress_every: int = 50,
) -> FeatureCache:
    """Pre-compute raw feature vectors for all (symbol, day) pairs.

    Args:
        symbols_data:          fold symbol → OHLCV DataFrame (PIT-filtered).
        trading_days:          test-fold trading days to compute features for.
        feature_names:         model feature column names (canonical order).
        regime_score_history:  {date: float} PIT regime scores (optional).
        vix_history:           VIX close Series (optional).
        macro_history:         macro_history DataFrame (VIX, SPY, RSP, HYG, IEF, VIX3M) for regime features.
        workers:               parallel workers (0 = auto: cpu_count - 1, max 12).
        executor:              "process" (recommended) or "thread".
        skip_symbols:          synthetic symbols to exclude.
        log_progress_every:    log a progress line every N symbols.

    Returns:
        FeatureCache ready for injection into AgentSimulator.
    """
    skip = frozenset(skip_symbols)
    syms = [s for s in symbols_data if s not in skip]
    n_syms = len(syms)
    n_days = len(trading_days)
    F = len(feature_names)

    if workers <= 0:
        workers = max(2, min(os.cpu_count() or 4, 12))

    regime_scores = regime_score_history or {}

    # Prepare vix as plain lists for pickling
    vix_idx = vix_values = None
    if vix_history is not None:
        vix_idx = [ts.date() if hasattr(ts, "date") else ts for ts in vix_history.index]
        vix_values = vix_history.tolist()

    # Prepare macro_history as plain lists for pickling; auto-load from disk if not provided
    macro_idx = macro_recs = None
    _mh = macro_history
    if _mh is None:
        try:
            from app.data.macro_history import load_macro_history
            _mh = load_macro_history()
        except Exception as _exc:
            logger.debug("macro_history not available for feature cache: %s", _exc)
    if _mh is not None and len(_mh) > 0:
        macro_idx = list(_mh["date"].astype(str))
        macro_recs = _mh.drop(columns=["date"]).to_dict("records")

    logger.info(
        "Building feature cache: %d symbols × %d days × %d features | "
        "%d %s workers",
        n_syms, n_days, F, workers, executor,
    )

    cache = FeatureCache(feature_names=feature_names)
    completed = 0

    def _submit_args(sym: str):
        df = symbols_data[sym]
        return (
            sym,
            df.to_dict("records"),
            list(df.index.date),
            trading_days,
            feature_names,
            regime_scores,
            vix_idx,
            vix_values,
            macro_idx,
            macro_recs,
        )

    PoolClass = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    pool_kwargs = {"max_workers": workers}
    if executor == "process":
        pool_kwargs["initializer"] = _init_worker

    with PoolClass(**pool_kwargs) as pool:
        futures = {pool.submit(_build_symbol_rows, *_submit_args(sym)): sym for sym in syms}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                _, valid_dates, feature_rows = future.result()
            except Exception as exc:
                logger.warning("Feature cache: %s failed — %s", sym, exc)
                completed += 1
                continue

            if feature_rows:
                mat = np.array(feature_rows, dtype=np.float32)
                cache.matrix[sym] = mat
                cache.dates[sym] = np.array(valid_dates)
                cache.date_index[sym] = {d: i for i, d in enumerate(valid_dates)}

            completed += 1
            if completed % log_progress_every == 0 or completed == n_syms:
                logger.info(
                    "Feature cache: %d / %d symbols done (%.0f MB so far)",
                    completed, n_syms, cache.memory_mb,
                )

    logger.info(
        "Feature cache built: %d symbols, %.0f MB, %d trading days covered",
        cache.n_symbols, cache.memory_mb, n_days,
    )

    # Bug 3 fix: if a worker crashes (e.g. OOM on Windows loky), ProcessPoolExecutor
    # invalidates ALL pending futures with BrokenProcessPool. Each future is caught
    # individually above, so the build "succeeds" with an empty cache — and the
    # simulator then produces 0 trades for the whole fold (Sharpe=0). Detect this
    # mass-failure case and raise, so the caller falls back to live-compute.
    if n_syms > 0 and cache.n_symbols < max(1, n_syms // 10):
        raise RuntimeError(
            f"Feature cache build failed: only {cache.n_symbols}/{n_syms} symbols "
            f"populated (process pool likely crashed). Caller should fall back "
            f"to live-compute or rerun with fewer --feature-cache-workers."
        )
    return cache
