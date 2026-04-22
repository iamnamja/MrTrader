"""
Cross-sectional feature normalization utility.

Z-scores each feature column across all symbols at the same point in time.
Applied within each time window during training and across the full candidate
batch during inference — ensures model sees relative features, not absolute ones.
"""

import numpy as np


def cs_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score X column-wise across rows (cross-sectional, one point in time).

    With a single row (single-stock inference), returns X unchanged.
    """
    if len(X) < 2:
        return X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + eps)


def cs_normalize_by_group(
    X: np.ndarray, group_ids: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Apply cs_normalize independently within each group (e.g. window or trading day).

    X:         (N, F) feature matrix
    group_ids: (N,)   integer or float group identifier per row
    Returns:   (N, F) normalized matrix; groups with < 2 rows are left unchanged.
    """
    result = X.copy().astype(np.float64)
    for g in np.unique(group_ids):
        mask = group_ids == g
        if mask.sum() < 2:
            continue
        result[mask] = cs_normalize(X[mask], eps=eps)
    return result
