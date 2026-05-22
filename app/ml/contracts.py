"""
contracts.py — Shared data contract for feature vectors across all prediction paths.

A FeatureVector is the unit of data that moves from feature construction
into normalization and then into model.predict. Having a single typed
container forces all four paths (training, WF-live-compute, WF-cached,
live-PM) to agree on field names and ordering.

Usage:
    from app.ml.contracts import FeatureVector

    fv = FeatureVector.from_dict(symbol, asof_date, feature_names, feat_dict)
    # fv.values: np.ndarray, shape (n_features,)
    # fv.schema_hash: str (SHA-256 of pipe-joined feature_names)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Dict, Sequence, Tuple

import numpy as np


def schema_hash(feature_names: Sequence[str]) -> str:
    """SHA-256 of pipe-joined feature names. Detects ordering or name changes."""
    return hashlib.sha256("|".join(feature_names).encode()).hexdigest()


@dataclass(frozen=True)
class FeatureVector:
    """Immutable feature vector for a single (symbol, asof_date) observation.

    Attributes:
        symbol: Ticker, e.g. "AAPL".
        asof_date: The as-of date (point-in-time — no future data used).
        feature_names: Ordered tuple of feature names.
        values: np.ndarray of shape (n_features,), dtype float32.
        schema_hash: SHA-256 of pipe-joined feature_names.
    """

    symbol: str
    asof_date: date
    feature_names: Tuple[str, ...]
    values: np.ndarray
    schema_hash: str

    def __post_init__(self) -> None:
        expected_hash = hashlib.sha256("|".join(self.feature_names).encode()).hexdigest()
        if self.schema_hash != expected_hash:
            raise ValueError(
                f"FeatureVector({self.symbol}): schema_hash mismatch — "
                f"names imply {expected_hash[:8]}… but got {self.schema_hash[:8]}…"
            )
        if len(self.values) != len(self.feature_names):
            raise ValueError(
                f"FeatureVector({self.symbol}): {len(self.values)} values "
                f"but {len(self.feature_names)} feature names"
            )

    @classmethod
    def from_dict(
        cls,
        symbol: str,
        asof_date: date,
        feature_names: Sequence[str],
        feat_dict: Dict[str, float],
        fill_missing: float = 0.0,
    ) -> "FeatureVector":
        """Build a FeatureVector from a symbol's feature dict.

        Missing keys are filled with `fill_missing` (default 0.0).
        NaN and Inf values are replaced with `fill_missing`.
        """
        names_tuple = tuple(feature_names)
        values = np.array(
            [feat_dict.get(f, fill_missing) for f in names_tuple], dtype=np.float32
        )
        values = np.nan_to_num(values, nan=fill_missing, posinf=fill_missing, neginf=fill_missing)
        h = schema_hash(names_tuple)
        return cls(
            symbol=symbol,
            asof_date=asof_date,
            feature_names=names_tuple,
            values=values,
            schema_hash=h,
        )

    @classmethod
    def from_row(
        cls,
        symbol: str,
        asof_date: date,
        feature_names: Sequence[str],
        row: np.ndarray,
    ) -> "FeatureVector":
        """Build a FeatureVector from a pre-ordered numpy row (e.g., from FeatureCache)."""
        names_tuple = tuple(feature_names)
        values = np.array(row, dtype=np.float32)
        if len(values) != len(names_tuple):
            raise ValueError(
                f"row length {len(values)} != feature_names length {len(names_tuple)}"
            )
        h = schema_hash(names_tuple)
        return cls(
            symbol=symbol,
            asof_date=asof_date,
            feature_names=names_tuple,
            values=values,
            schema_hash=h,
        )

    def reorder(self, target_names: Sequence[str], fill_missing: float = 0.0) -> "FeatureVector":
        """Return a new FeatureVector reordered to match target_names.

        Features not in this vector are filled with `fill_missing`.
        This is the canonical way to align cache-order with model-order.
        """
        name_to_val = dict(zip(self.feature_names, self.values.tolist()))
        new_vals = np.array(
            [name_to_val.get(f, fill_missing) for f in target_names], dtype=np.float32
        )
        target_tuple = tuple(target_names)
        return FeatureVector(
            symbol=self.symbol,
            asof_date=self.asof_date,
            feature_names=target_tuple,
            values=new_vals,
            schema_hash=schema_hash(target_tuple),
        )

    def to_row(self) -> np.ndarray:
        """Return values as a 1-D float32 array."""
        return self.values.copy()

    def to_matrix_row(self) -> np.ndarray:
        """Return values as a 2-D row (1, n_features) for model.predict."""
        return self.values.reshape(1, -1)
