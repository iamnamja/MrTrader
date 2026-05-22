"""
schema_log.py — Lightweight feature-schema logging for training/WF/live alignment.

Phase 0.3-lite: surface divergences between the three prediction paths BEFORE
building abstractions. Each path emits JSON Lines records at three checkpoints:
  - "features": after feature construction
  - "normalize": after normalization
  - "predict": after model.predict

Records are written to logs/schema_YYYYMMDD.jsonl (one file per day).

Usage:
    from app.ml.schema_log import schema_hash, log_features, log_normalize, log_predict

    h = schema_hash(feature_names)
    log_features("wf", run_id, asof, feature_names, X, sym_list)
    log_normalize("wf", run_id, asof, normalizer_name, universe_size, X_norm)
    log_predict("wf", run_id, asof, model_version, h, scores, sym_list)
"""

import hashlib
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_logger = logging.getLogger("ml.schema_log")
_LOG_DIR = Path("logs")


def _jsonl_path() -> Path:
    _LOG_DIR.mkdir(exist_ok=True)
    return _LOG_DIR / f"schema_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"


def _write(record: dict) -> None:
    try:
        with open(_jsonl_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        _logger.debug("schema_log write failed (non-fatal): %s", exc)


def schema_hash(feature_names: Sequence[str]) -> str:
    """SHA-256 of pipe-joined feature names. Detects ordering or name changes."""
    return hashlib.sha256("|".join(feature_names).encode()).hexdigest()


def log_features(
    path: str,           # "train" | "wf" | "live"
    run_id: str,
    asof: date,
    feature_names: Sequence[str],
    X: np.ndarray,       # shape (N, F)
    symbols: Optional[List[str]] = None,
) -> str:
    """Log feature-stage record. Returns schema_hash for downstream use."""
    h = schema_hash(feature_names)
    n_rows, n_features = X.shape if X.ndim == 2 else (1, X.shape[0])
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())

    # 3-value fingerprint of first symbol's features
    sample_first3: list = []
    if X.ndim == 2 and X.shape[0] > 0:
        sample_first3 = [round(float(v), 6) for v in X[0, :3]]
    elif X.ndim == 1:
        sample_first3 = [round(float(v), 6) for v in X[:3]]

    sample_symbol = symbols[0] if symbols else None

    _write({
        "ts": datetime.utcnow().isoformat() + "Z",
        "path": path,
        "stage": "features",
        "run_id": run_id,
        "asof": str(asof),
        "n_rows": n_rows,
        "n_features": n_features,
        "schema_hash": h,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "sample_symbol": sample_symbol,
        "sample_first3": sample_first3,
        "feature_names_first5": list(feature_names[:5]),
    })
    return h


def log_normalize(
    path: str,
    run_id: str,
    asof: date,
    normalizer_name: str,   # "cs_normalize" | "ts_normalize" | "none"
    universe_size: int,
    X_norm: np.ndarray,
) -> None:
    """Log normalize-stage record."""
    mean_sample: list = []
    std_sample: list = []
    if X_norm.ndim == 2 and X_norm.shape[0] > 0:
        mean_sample = [round(float(v), 6) for v in np.nanmean(X_norm[:, :3], axis=0)]
        std_sample = [round(float(v), 6) for v in np.nanstd(X_norm[:, :3], axis=0)]

    _write({
        "ts": datetime.utcnow().isoformat() + "Z",
        "path": path,
        "stage": "normalize",
        "run_id": run_id,
        "asof": str(asof),
        "normalizer": normalizer_name,
        "universe_size": universe_size,
        "n_rows_out": X_norm.shape[0] if X_norm.ndim == 2 else 1,
        "mean_first3": mean_sample,
        "std_first3": std_sample,
    })


def log_predict(
    path: str,
    run_id: str,
    asof: date,
    model_version: str,
    feature_schema_hash: str,
    raw_scores: np.ndarray,
    symbols: Optional[List[str]] = None,
) -> None:
    """Log predict-stage record with score distribution and top-5."""
    top5: list = []
    if symbols and len(symbols) == len(raw_scores):
        ranked = sorted(zip(symbols, raw_scores.tolist()), key=lambda x: x[1], reverse=True)
        top5 = [[s, round(float(sc), 6)] for s, sc in ranked[:5]]

    _write({
        "ts": datetime.utcnow().isoformat() + "Z",
        "path": path,
        "stage": "predict",
        "run_id": run_id,
        "asof": str(asof),
        "model_version": model_version,
        "model_schema_hash": feature_schema_hash,
        "n_scores": len(raw_scores),
        "score_min": round(float(raw_scores.min()), 6) if len(raw_scores) else None,
        "score_max": round(float(raw_scores.max()), 6) if len(raw_scores) else None,
        "score_mean": round(float(raw_scores.mean()), 6) if len(raw_scores) else None,
        "top5": top5,
    })
