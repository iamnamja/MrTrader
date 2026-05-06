"""Phase R2 — Regime model singleton.

Loads the latest regime_model_v*.pkl on startup, exposes score() for PM scans.
Falls back to legacy opportunity score if model not loaded.
"""
from __future__ import annotations

import logging
import pickle
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"

RISK_OFF_THRESHOLD = 0.35
RISK_ON_THRESHOLD = 0.65

_CACHE_TTL_SECONDS = 300  # 5-minute cache


def _label_from_score(score: float) -> str:
    if score < RISK_OFF_THRESHOLD:
        return "RISK_OFF"
    if score >= RISK_ON_THRESHOLD:
        return "RISK_ON"
    return "NEUTRAL"


class RegimeModel:
    """Singleton regime scorer. Call RegimeModel.instance() to get the shared instance."""

    _instance: Optional["RegimeModel"] = None

    @classmethod
    def instance(cls) -> "RegimeModel":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    def __init__(self) -> None:
        self._xgb_model = None
        self._iso_model = None
        self._feature_names: list[str] = []
        self._version: Optional[str] = None
        self._cache_score: Optional[float] = None
        self._cache_label: Optional[str] = None
        self._cache_ts: float = 0.0
        self._cache_date: Optional[date] = None

    def load(self, path: Optional[Path] = None) -> bool:
        """Load model from disk. If path is None, picks latest regime_model_v*.pkl."""
        if path is None:
            candidates = sorted(MODEL_DIR.glob("regime_model_v*.pkl"))
            if not candidates:
                logger.warning("No regime model found in %s — will use legacy fallback", MODEL_DIR)
                return False
            path = candidates[-1]

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            self._xgb_model = payload["xgb_model"]
            self._iso_model = payload["iso_model"]
            self._feature_names = payload["feature_names"]
            self._version = payload.get("version", "unknown")
            logger.info("Regime model loaded: v%s from %s", self._version, path.name)
            return True
        except Exception as exc:
            logger.error("Failed to load regime model from %s: %s", path, exc)
            return False

    @property
    def loaded(self) -> bool:
        return self._xgb_model is not None

    def score(
        self,
        as_of_date: Optional[date] = None,
        trigger: str = "manual",
        _spy_df=None,
        _vix_df=None,
    ) -> dict:
        """Score regime for as_of_date. Writes to regime_snapshots. Returns dict."""
        if not self.loaded:
            return self._legacy_fallback(as_of_date, trigger)

        if as_of_date is None:
            as_of_date = date.today()

        # Check cache (same date, within TTL)
        now_ts = time.monotonic()
        if (
            self._cache_date == as_of_date
            and self._cache_score is not None
            and (now_ts - self._cache_ts) < _CACHE_TTL_SECONDS
        ):
            return {
                "regime_score": self._cache_score,
                "regime_label": self._cache_label,
                "version": f"regime_v{self._version}",
                "trigger": trigger,
                "cached": True,
            }

        try:
            from app.ml.regime_features import RegimeFeatureBuilder
            builder = RegimeFeatureBuilder()
            feats = builder.build(as_of_date, _spy_df=_spy_df, _vix_df=_vix_df)
        except Exception as exc:
            logger.error("RegimeFeatureBuilder.build failed: %s — using legacy fallback", exc)
            return self._legacy_fallback(as_of_date, trigger)

        if feats is None:
            logger.warning("RegimeFeatureBuilder returned None for %s — using legacy fallback", as_of_date)
            return self._legacy_fallback(as_of_date, trigger)

        import numpy as np
        X = np.array([[feats.get(f, 0.0) for f in self._feature_names]])
        raw = self._xgb_model.predict_proba(X)[0, 1]
        proba = float(self._iso_model.predict([raw])[0])
        label = _label_from_score(proba)

        # Update cache
        self._cache_score = proba
        self._cache_label = label
        self._cache_ts = now_ts
        self._cache_date = as_of_date

        # Write to regime_snapshots
        self._persist_snapshot(as_of_date, trigger, proba, label, feats)

        return {
            "regime_score": round(proba, 4),
            "regime_label": label,
            "version": f"regime_v{self._version}",
            "trigger": trigger,
            "cached": False,
            "features": feats,
        }

    def _persist_snapshot(
        self,
        snapshot_date: date,
        trigger: str,
        score: float,
        label: str,
        feats: dict,
    ) -> None:
        try:
            from app.database.session import get_session
            from app.database.models import RegimeSnapshot

            with get_session() as session:
                row = RegimeSnapshot(
                    snapshot_time=datetime.now(timezone.utc),
                    snapshot_date=snapshot_date,
                    snapshot_trigger=trigger,
                    regime_score=score,
                    regime_label=label,
                    risk_off_threshold=RISK_OFF_THRESHOLD,
                    risk_on_threshold=RISK_ON_THRESHOLD,
                    model_version=f"regime_v{self._version}",
                    **{k: feats.get(k) for k in self._feature_names if hasattr(RegimeSnapshot, k)},
                )
                session.add(row)
                session.commit()
        except Exception as exc:
            logger.error("Failed to persist regime snapshot: %s", exc)

    def _legacy_fallback(self, as_of_date: Optional[date], trigger: str) -> dict:
        logger.warning("Regime model not loaded — returning NEUTRAL (legacy fallback)")
        return {
            "regime_score": 0.5,
            "regime_label": "NEUTRAL",
            "version": "legacy_fallback",
            "trigger": trigger,
            "cached": False,
        }

    def invalidate_cache(self) -> None:
        self._cache_score = None
        self._cache_label = None
        self._cache_ts = 0.0
        self._cache_date = None
