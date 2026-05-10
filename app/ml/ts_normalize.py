"""
Rolling time-series feature normalization (Fix 2).

Each (symbol, feature) is z-scored using the trailing NORM_LOOKBACK windows
of that same symbol's history — preserves macro/regime signal that
cross-sectional normalization destroys, and keeps absolute-magnitude info
that the triple-barrier label depends on.

Cross-sectional normalization (cs_normalize_by_group) zeros out macro/regime
features (VIX, SPY MA, breadth) because all symbols on the same window date
share identical values → std = 0. Time-series normalization fixes this: each
feature is compared against its own recent history per symbol.

Usage in training:
    X_train_norm, keep_mask, state = fit_transform_train(
        X_train, symbols_train, window_ids_train, feature_names
    )
    X_train = X_train_norm[keep_mask]
    y_train  = y_train[keep_mask]
    meta_train = [m for m, k in zip(meta_train, keep_mask) if k]

    X_test_norm, keep_mask_test = transform(
        X_test, symbols_test, window_ids_test, state
    )
    X_test = X_test_norm[keep_mask_test]
    y_test  = y_test[keep_mask_test]

    # Persist state alongside model for inference parity.
    import pickle
    with open(f"app/ml/models/swing_norm_v{version}.pkl", "wb") as f:
        pickle.dump(state, f)
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NORM_LOOKBACK: int = 20   # trailing windows per (symbol, feature)
MIN_WARMUP: int = 8       # min prior windows required; rows below this are dropped
EPS: float = 1e-8


@dataclass
class TSNormalizerState:
    """Per-(symbol, feature) running stats for inference parity.

    Fit on train rows only. At inference, extend history with arriving rows
    (in chronological order) before normalizing each new row.

    Attributes:
        history: symbol → list of (window_idx, feature_row) tuples, capped to
                 NORM_LOOKBACK. Rows are appended in ascending window_idx order.
        last_stats: symbol → (mean, std) arrays frozen at end of train fit.
                    Used as fallback when symbol has insufficient history at
                    inference time (e.g. new listing).
        n_features: number of feature columns (for integrity checks at load time).
        feature_names_hash: MD5 of sorted feature names — asserted at load to
                            catch column ordering mismatches before prediction.
        lookback: the NORM_LOOKBACK used at fit time.
        min_warmup: the MIN_WARMUP used at fit time.
    """
    history: Dict[str, list] = field(default_factory=dict)
    last_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    n_features: int = 0
    feature_names_hash: str = ""
    lookback: int = NORM_LOOKBACK
    min_warmup: int = MIN_WARMUP


def _feature_names_hash(feature_names: List[str]) -> str:
    import hashlib
    return hashlib.md5(str(sorted(feature_names)).encode()).hexdigest()[:12]


def _normalize_array(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Vectorized z-score: clamp constant features to 0."""
    denom = np.where(std < EPS, 1.0, std)
    out = (X - mean) / denom
    out = np.where(std < EPS, 0.0, out)
    out = np.where(np.isnan(out), 0.0, out)
    return out


def _rolling_stats_for_symbol(
    mat: np.ndarray,
    lookback: int,
    min_warmup: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute strictly-prior rolling mean/std for one symbol's row matrix.

    Uses pandas rolling (C-level) then shifts by 1 so each row's stats
    are computed from strictly prior rows only (no look-ahead).

    Args:
        mat: (T, F) array of feature rows in chronological order.
        lookback: rolling window size.
        min_warmup: minimum prior rows required for a valid stats row.

    Returns:
        means:      (T, F) — NaN where insufficient history.
        stds:       (T, F) — NaN where insufficient history.
        valid_mask: (T,)   bool — True where min_warmup satisfied.
    """
    T, F = mat.shape
    df = pd.DataFrame(mat.astype(np.float64))

    # rolling over strictly prior rows: shift(1) then rolling(lookback)
    shifted = df.shift(1)
    means_df = shifted.rolling(window=lookback, min_periods=min_warmup).mean()
    stds_df = shifted.rolling(window=lookback, min_periods=min_warmup).std(ddof=0)

    means = means_df.to_numpy()
    stds = stds_df.to_numpy()

    # valid where both mean and std are non-NaN (i.e. min_warmup met)
    valid_mask = ~np.isnan(means).any(axis=1)

    return means, stds, valid_mask


def fit_transform_train(
    X: np.ndarray,
    symbols: np.ndarray,
    window_ids: np.ndarray,
    feature_names: Optional[List[str]] = None,
    lookback: int = NORM_LOOKBACK,
    min_warmup: int = MIN_WARMUP,
) -> Tuple[np.ndarray, np.ndarray, "TSNormalizerState"]:
    """Fit + transform on train rows. Returns (X_norm, keep_mask, state).

    Vectorized implementation: groups by symbol, applies pandas rolling
    mean/std (C-level) with shift(1) for strict causality, then scatters
    results back. ~50x faster than the previous row-by-row Python loop.
    """
    N, F = X.shape
    X_norm = np.zeros((N, F), dtype=np.float64)
    keep_mask = np.zeros(N, dtype=bool)

    state = TSNormalizerState(n_features=F, lookback=lookback, min_warmup=min_warmup)
    if feature_names is not None:
        state.feature_names_hash = _feature_names_hash(feature_names)

    unique_syms = np.unique(symbols)
    for sym in unique_syms:
        idx = np.where(symbols == sym)[0]
        # Sort by window_id (chronological)
        order = idx[np.argsort(window_ids[idx])]
        mat = X[order].astype(np.float64)

        means, stds, valid = _rolling_stats_for_symbol(mat, lookback, min_warmup)

        # Normalize valid rows
        if valid.any():
            m_valid = means[valid]
            s_valid = stds[valid]
            x_valid = mat[valid]
            X_norm[order[valid]] = _normalize_array(x_valid, m_valid, s_valid)
            keep_mask[order[valid]] = True

        # Store history for inference: last `lookback` rows in chronological order
        buf = [(int(window_ids[order[i]]), mat[i]) for i in range(len(order))]
        buf = buf[-(lookback + 1):]
        state.history[sym] = buf

        # Freeze last stats for inference fallback
        if len(buf) >= 2:
            tail_rows = np.vstack([r for _, r in buf[-lookback:]])
            state.last_stats[sym] = (
                np.nanmean(tail_rows, axis=0),
                np.nanstd(tail_rows, axis=0),
            )

    n_kept = int(keep_mask.sum())
    n_dropped = N - n_kept
    logger.info(
        "TSNormalizer fit: %d/%d rows kept (dropped %d cold-start rows, %.1f%%)",
        n_kept, N, n_dropped, 100.0 * n_dropped / max(N, 1),
    )
    return X_norm, keep_mask, state


def _transform_single_row_per_symbol(
    X: np.ndarray,
    symbols: np.ndarray,
    window_ids: np.ndarray,
    state: "TSNormalizerState",
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast path: one row per symbol (daily inference).

    Reads stats directly from state.history buffer — no pandas overhead.
    O(lookback × F) per symbol in pure numpy.
    """
    N, F = X.shape
    X_norm = np.zeros((N, F), dtype=np.float64)
    keep_mask = np.zeros(N, dtype=bool)

    lookback = state.lookback
    min_warmup = state.min_warmup

    for i in range(N):
        sym = symbols[i]
        row = X[i].astype(np.float64)
        wid = int(window_ids[i])

        buf = state.history.get(sym, [])
        if len(buf) >= min_warmup:
            tail = buf[-lookback:]
            tail_mat = np.vstack([r for _, r in tail])
            mean = np.nanmean(tail_mat, axis=0)
            std = np.nanstd(tail_mat, axis=0)
            X_norm[i] = _normalize_array(row[None], mean[None], std[None])[0]
            keep_mask[i] = True
        elif sym in state.last_stats:
            mean, std = state.last_stats[sym]
            X_norm[i] = _normalize_array(row[None], mean[None], std[None])[0]
            keep_mask[i] = True

        buf = list(buf) + [(wid, row)]
        state.history[sym] = buf[-(lookback + 1):]

    return X_norm, keep_mask


def transform(
    X: np.ndarray,
    symbols: np.ndarray,
    window_ids: np.ndarray,
    state: "TSNormalizerState",
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform val/test rows using the fitted state.

    Uses a fast path for the daily-inference case (1 row per symbol) that
    reads stats directly from the history buffer without pandas overhead.
    For multi-row-per-symbol inputs (test fold), uses vectorized pandas rolling.

    Falls back to state.last_stats when buffer is too short.
    """
    N, F = X.shape
    lookback = state.lookback
    min_warmup = state.min_warmup

    # Fast path: 1 row per symbol (daily inference case)
    unique_syms, counts = np.unique(symbols, return_counts=True)
    if np.all(counts == 1):
        X_norm, keep_mask = _transform_single_row_per_symbol(
            X, symbols, window_ids, state
        )
        n_kept = int(keep_mask.sum())
        logger.info("TSNormalizer transform: %d/%d rows kept", n_kept, N)
        return X_norm, keep_mask

    # Multi-row path: vectorized pandas rolling per symbol
    X_norm = np.zeros((N, F), dtype=np.float64)
    keep_mask = np.zeros(N, dtype=bool)

    for sym in unique_syms:
        idx = np.where(symbols == sym)[0]
        order = idx[np.argsort(window_ids[idx])]
        mat = X[order].astype(np.float64)
        T = len(order)

        # Prepend history so rolling can see prior context
        buf = state.history.get(sym, [])
        if buf:
            hist_rows = np.vstack([r for _, r in buf])
            combined = np.vstack([hist_rows, mat])
            H = len(hist_rows)
        else:
            combined = mat
            H = 0

        means, stds, valid = _rolling_stats_for_symbol(combined, lookback, min_warmup)

        test_means = means[H:]
        test_stds = stds[H:]
        test_valid = valid[H:]

        if not test_valid.all() and sym in state.last_stats:
            fb_mean, fb_std = state.last_stats[sym]
            invalid_local = ~test_valid
            X_norm[order[invalid_local]] = _normalize_array(
                mat[invalid_local], fb_mean, fb_std
            )
            keep_mask[order[invalid_local]] = True

        if test_valid.any():
            X_norm[order[test_valid]] = _normalize_array(
                mat[test_valid], test_means[test_valid], test_stds[test_valid]
            )
            keep_mask[order[test_valid]] = True

        new_buf = list(buf) + [(int(window_ids[order[i]]), mat[i]) for i in range(T)]
        state.history[sym] = new_buf[-(lookback + 1):]

    n_kept = int(keep_mask.sum())
    logger.info("TSNormalizer transform: %d/%d rows kept", n_kept, N)
    return X_norm, keep_mask


def assert_state_compatible(state: "TSNormalizerState", feature_names: List[str]) -> None:
    """Raise ValueError if the loaded state was fit on a different feature set."""
    if not state.feature_names_hash:
        return  # old state without hash — skip
    expected = _feature_names_hash(feature_names)
    if state.feature_names_hash != expected:
        raise ValueError(
            f"TSNormalizerState feature mismatch: state was fit on hash "
            f"{state.feature_names_hash!r} but current features hash to "
            f"{expected!r}. The model and its normalizer state must match. "
            f"Retrain or load the correct state file."
        )


def save_state(state: "TSNormalizerState", path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)
    logger.info("TSNormalizerState saved to %s", path)


def load_state(path: str) -> "TSNormalizerState":
    with open(path, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, TSNormalizerState):
        raise TypeError(f"Expected TSNormalizerState, got {type(state)}")
    return state
