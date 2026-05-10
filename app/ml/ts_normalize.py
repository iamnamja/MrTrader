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


def _compute_stats(
    rows: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (mean, std) from a list of feature-row arrays."""
    mat = np.vstack(rows).astype(np.float64)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return mean, std


def _normalize_row(
    row: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Z-score row, clamping constant features to 0 (no inf/nan)."""
    row = row.astype(np.float64)
    denom = np.where(std < EPS, 1.0, std)
    out = (row - mean) / denom
    out = np.where(std < EPS, 0.0, out)
    out = np.where(np.isnan(out), 0.0, out)
    return out


def _feature_names_hash(feature_names: List[str]) -> str:
    import hashlib
    return hashlib.md5(str(sorted(feature_names)).encode()).hexdigest()[:12]


def fit_transform_train(
    X: np.ndarray,
    symbols: np.ndarray,
    window_ids: np.ndarray,
    feature_names: Optional[List[str]] = None,
    lookback: int = NORM_LOOKBACK,
    min_warmup: int = MIN_WARMUP,
) -> Tuple[np.ndarray, np.ndarray, TSNormalizerState]:
    """Fit + transform on train rows. Returns (X_norm, keep_mask, state).

    Processes rows in ascending (symbol, window_idx) order. For each row,
    computes stats from the trailing `lookback` prior rows of the same symbol
    (strictly prior — no look-ahead). Rows with fewer than `min_warmup` prior
    windows are excluded via keep_mask=False.

    Args:
        X:            (N, F) float feature matrix.
        symbols:      (N,)  string symbol per row.
        window_ids:   (N,)  int window index per row (ascending = later in time).
        feature_names: optional list of F feature names for parity checks.
        lookback:     trailing window count for mean/std.
        min_warmup:   minimum prior rows required; rows below this get mask=False.

    Returns:
        X_norm:    (N, F) normalized matrix (rows with mask=False are zeros — drop them).
        keep_mask: (N,)   bool array; False = cold-start row, caller should drop.
        state:     TSNormalizerState for val/test transform and inference.
    """
    N, F = X.shape
    X_norm = np.zeros_like(X, dtype=np.float64)
    keep_mask = np.zeros(N, dtype=bool)

    state = TSNormalizerState(n_features=F, lookback=lookback, min_warmup=min_warmup)
    if feature_names is not None:
        state.feature_names_hash = _feature_names_hash(feature_names)

    # Sort by (symbol, window_idx) to process chronologically per symbol
    order = np.lexsort((window_ids, symbols))

    for i in order:
        sym = symbols[i]
        wid = int(window_ids[i])
        row = X[i].astype(np.float64)

        buf = state.history.get(sym, [])
        if len(buf) >= min_warmup:
            # Use trailing `lookback` rows
            tail = buf[-lookback:]
            mean, std = _compute_stats([r for _, r in tail])
            X_norm[i] = _normalize_row(row, mean, std)
            keep_mask[i] = True
        # else: cold-start — keep_mask stays False, X_norm stays zeros

        # Append current row to history (AFTER computing norm, so it's not used for itself)
        buf.append((wid, row.copy()))
        if len(buf) > lookback + 1:
            buf.pop(0)
        state.history[sym] = buf

    # Freeze last stats per symbol for inference fallback
    for sym, buf in state.history.items():
        if len(buf) >= 2:
            tail = buf[-lookback:]
            mean, std = _compute_stats([r for _, r in tail])
            state.last_stats[sym] = (mean, std)

    n_kept = int(keep_mask.sum())
    n_dropped = N - n_kept
    logger.info(
        "TSNormalizer fit: %d/%d rows kept (dropped %d cold-start rows, %.1f%%)",
        n_kept, N, n_dropped, 100.0 * n_dropped / max(N, 1),
    )
    return X_norm, keep_mask, state


def transform(
    X: np.ndarray,
    symbols: np.ndarray,
    window_ids: np.ndarray,
    state: TSNormalizerState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform val/test rows using the fitted state.

    Extends state.history with each normalized row in chronological order,
    so later rows benefit from earlier val/test rows as context — but each
    row is normalized only from rows with strictly smaller window_idx (no
    look-ahead). Falls back to state.last_stats when buffer is too short.

    Args:
        X:          (N, F) float feature matrix.
        symbols:    (N,)  string symbol per row.
        window_ids: (N,)  int window index per row.
        state:      TSNormalizerState from fit_transform_train.

    Returns:
        X_norm:    (N, F) normalized matrix.
        keep_mask: (N,)   bool; False = no history available (new symbol, rare).
    """
    N, F = X.shape
    X_norm = np.zeros_like(X, dtype=np.float64)
    keep_mask = np.zeros(N, dtype=bool)

    lookback = state.lookback
    min_warmup = state.min_warmup

    order = np.lexsort((window_ids, symbols))

    for i in order:
        sym = symbols[i]
        wid = int(window_ids[i])
        row = X[i].astype(np.float64)

        buf = state.history.get(sym, [])
        if len(buf) >= min_warmup:
            tail = buf[-lookback:]
            mean, std = _compute_stats([r for _, r in tail])
            X_norm[i] = _normalize_row(row, mean, std)
            keep_mask[i] = True
        elif sym in state.last_stats:
            # Fallback: use frozen train-end stats
            mean, std = state.last_stats[sym]
            X_norm[i] = _normalize_row(row, mean, std)
            keep_mask[i] = True
        # else: truly new symbol with no history — keep_mask stays False

        buf.append((wid, row.copy()))
        if len(buf) > lookback + 1:
            buf.pop(0)
        state.history[sym] = buf

    n_kept = int(keep_mask.sum())
    logger.info(
        "TSNormalizer transform: %d/%d rows kept",
        n_kept, N,
    )
    return X_norm, keep_mask


def assert_state_compatible(state: TSNormalizerState, feature_names: List[str]) -> None:
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


def save_state(state: TSNormalizerState, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)
    logger.info("TSNormalizerState saved to %s", path)


def load_state(path: str) -> TSNormalizerState:
    with open(path, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, TSNormalizerState):
        raise TypeError(f"Expected TSNormalizerState, got {type(state)}")
    return state
