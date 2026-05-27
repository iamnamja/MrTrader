"""IC utility helpers for walk-forward Option-A per-fold IC weights.

R5b (2026-05-27): expose `compute_fold_ic_weights` and `find_latest_daily_ic_parquet`
so that the walk-forward harness can recalibrate IC composite weights using only
training-period data (date <= tr_end), eliminating the static pre-2021-04-26
weight assumption while remaining PIT-safe.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum number of (date, feature) IC rows after filtering to trust the calibration.
# 252 trading days * a handful of features → at least 252 observations across features.
MIN_IC_OBSERVATIONS = 252


def find_latest_daily_ic_parquet(root: str = "data/diagnostics/feature_ic") -> Optional[str]:
    """Return path to the newest daily_ic.parquet, or None if none exists."""
    p = Path(root)
    if not p.exists():
        return None
    candidates = sorted(p.glob("*/daily_ic.parquet"))
    if not candidates:
        return None
    return str(candidates[-1])


def compute_fold_ic_weights(
    daily_ic_parquet_path: str,
    tr_end: date,
    horizon: int = 20,
) -> Optional[Dict[str, float]]:
    """Compute per-feature IC-IR weights using only data with date <= tr_end.

    Steps:
      1. Load the daily IC parquet (cols: date, feature, horizon, ic, n_symbols).
      2. Filter to date <= tr_end and (if available) the requested horizon.
      3. Aggregate per feature: ic_ir = mean(ic) / std(ic).
      4. Clip negative ic_ir to 0 (do not bet on inverted features).
      5. Normalise so weights sum to 1.0.

    Returns None if the parquet is missing/empty or there are fewer than
    MIN_IC_OBSERVATIONS rows after filtering — caller should fall back to
    static weights with a warning.
    """
    try:
        df = pd.read_parquet(daily_ic_parquet_path)
    except Exception as exc:
        logger.warning("compute_fold_ic_weights: failed to load %s (%s)", daily_ic_parquet_path, exc)
        return None
    if df is None or df.empty:
        return None
    # Normalise date column
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= tr_end]
    if "horizon" in df.columns:
        # Prefer the requested horizon if present; else fall back to whatever exists.
        if (df["horizon"] == horizon).any():
            df = df[df["horizon"] == horizon]
    if len(df) < MIN_IC_OBSERVATIONS:
        logger.warning(
            "compute_fold_ic_weights: only %d IC rows for tr_end<=%s (need >= %d) — fallback to static",
            len(df), tr_end, MIN_IC_OBSERVATIONS,
        )
        return None
    grouped = df.groupby("feature")["ic"].agg(["mean", "std"]).reset_index()
    # Guard against zero std (single observation per feature)
    grouped["ic_ir"] = np.where(
        grouped["std"].fillna(0.0) > 0,
        grouped["mean"] / grouped["std"],
        0.0,
    )
    grouped["ic_ir"] = grouped["ic_ir"].clip(lower=0.0)
    total = float(grouped["ic_ir"].sum())
    if total <= 0:
        logger.warning("compute_fold_ic_weights: all features have non-positive IC-IR — fallback to static")
        return None
    weights = {row["feature"]: float(row["ic_ir"] / total) for _, row in grouped.iterrows()}
    return weights
