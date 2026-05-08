"""Phase R7 — Regime V2 model singleton.

V2: XGBoost multi:softprob + temperature scaling.
Score = 0.5*P(CAUTION) + 1.0*P(RISK_ON)  →  continuous [0,1].
Label = argmax(probs) → RISK_OFF | RISK_CAUTION | RISK_ON.
Backwards-compatible: score() returns the same dict shape as V1.
"""
from __future__ import annotations

import logging
import pickle
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"

# Score thresholds for label derivation when only score is available (legacy path)
RISK_OFF_THRESHOLD = 0.30
RISK_ON_THRESHOLD = 0.60

_CACHE_TTL_SECONDS = 300


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
        self._temperature: float = 1.0        # V2: temperature scaling
        self._iso_model = None                # V1 compat only
        self._model_version: int = 1          # 1 or 2
        self._feature_names: list = []
        self._version: Optional[str] = None
        self._cache_score: Optional[float] = None
        self._cache_label: Optional[str] = None
        self._cache_probs: Optional[list] = None
        self._cache_ts: float = 0.0
        self._cache_date: Optional[date] = None

    def load(self, path: Optional[Path] = None) -> bool:
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
            self._feature_names = payload["feature_names"]
            self._version = str(payload.get("version", "unknown"))
            self._model_version = int(payload.get("model_version", 1))

            if self._model_version >= 2:
                self._temperature = float(payload.get("temperature", 1.0))
                self._iso_model = None
                logger.info("Regime model V2 loaded: v%s T=%.3f from %s",
                            self._version, self._temperature, path.name)
            else:
                self._iso_model = payload.get("iso_model")
                logger.info("Regime model V1 loaded: v%s from %s", self._version, path.name)
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
        _prefetched: Optional[dict] = None,
    ) -> dict:
        """Score regime for as_of_date. Writes to regime_snapshots. Returns dict."""
        if not self.loaded:
            return self._legacy_fallback(as_of_date, trigger)

        if as_of_date is None:
            as_of_date = date.today()

        now_ts = time.monotonic()
        if (
            self._cache_date == as_of_date
            and self._cache_score is not None
            and (now_ts - self._cache_ts) < _CACHE_TTL_SECONDS
        ):
            result = {
                "regime_score": self._cache_score,
                "regime_label": self._cache_label,
                "version": f"regime_v{self._version}",
                "trigger": trigger,
                "cached": True,
            }
            if self._cache_probs is not None:
                result["prob_risk_off"] = self._cache_probs[0]
                result["prob_risk_caution"] = self._cache_probs[1]
                result["prob_risk_on"] = self._cache_probs[2]
            return result

        try:
            from app.ml.regime_features import RegimeFeatureBuilder
            builder = RegimeFeatureBuilder()
            feats = builder.build(
                as_of_date,
                _spy_df=_spy_df,
                _vix_df=_vix_df,
                _prefetched=_prefetched,
            )
        except Exception as exc:
            logger.error("RegimeFeatureBuilder.build failed: %s — using legacy fallback", exc)
            return self._legacy_fallback(as_of_date, trigger)

        if feats is None:
            return self._legacy_fallback(as_of_date, trigger)

        import numpy as np
        X = np.array([[feats.get(f, 0.0) for f in self._feature_names]])

        if self._model_version >= 2:
            probs, score, label = self._score_v2(X)
        else:
            probs, score, label = self._score_v1(X)

        self._cache_score = score
        self._cache_label = label
        self._cache_probs = probs
        self._cache_ts = now_ts
        self._cache_date = as_of_date

        self._persist_snapshot(as_of_date, trigger, score, label, feats, probs)

        return {
            "regime_score": round(score, 4),
            "regime_label": label,
            "prob_risk_off": round(probs[0], 4),
            "prob_risk_caution": round(probs[1], 4),
            "prob_risk_on": round(probs[2], 4),
            "version": f"regime_v{self._version}",
            "trigger": trigger,
            "cached": False,
            "features": feats,
        }

    def _score_v2(self, X: np.ndarray) -> tuple:
        """V2: temperature-scaled multiclass. Returns (probs_list, score, label)."""
        from scipy.special import softmax as _softmax
        import xgboost as xgb_lib
        raw_logits = self._xgb_model.get_booster().predict(
            xgb_lib.DMatrix(X), output_margin=True
        )  # shape (1, 3)
        scaled = raw_logits / self._temperature
        probs = _softmax(scaled, axis=1)[0]  # (3,)
        score = float(0.5 * probs[1] + 1.0 * probs[2])
        label_idx = int(np.argmax(probs))
        label = ["RISK_OFF", "RISK_CAUTION", "RISK_ON"][label_idx]
        return [float(probs[0]), float(probs[1]), float(probs[2])], score, label

    def _score_v1(self, X: np.ndarray) -> tuple:
        """V1: isotonic-calibrated binary. Returns (probs_list, score, label)."""
        raw = self._xgb_model.predict_proba(X)[0, 1]
        score = float(self._iso_model.predict([raw])[0]) if self._iso_model else float(raw)
        label = self._label_from_score_v1(score)
        # No class probabilities in V1 — synthesize approximate values
        p_on = score
        p_off = 1.0 - score
        return [p_off, 0.0, p_on], score, label

    @staticmethod
    def _label_from_score_v1(score: float) -> str:
        if score < 0.35:
            return "RISK_OFF"
        if score >= 0.65:
            return "RISK_ON"
        return "RISK_CAUTION"

    def _persist_snapshot(
        self,
        snapshot_date: date,
        trigger: str,
        score: float,
        label: str,
        feats: dict,
        probs: list,
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
                    model_version=int(self._version) if self._version and self._version.isdigit() else None,
                    prob_risk_off=probs[0],
                    prob_risk_caution=probs[1],
                    prob_risk_on=probs[2],
                    **{k: (None if isinstance(feats.get(k), float) and feats.get(k) != feats.get(k)
                           else feats.get(k))
                       for k in self._feature_names if hasattr(RegimeSnapshot, k)},
                )
                session.add(row)
                session.commit()
        except Exception as exc:
            logger.error("Failed to persist regime snapshot: %s", exc)

    def _legacy_fallback(self, as_of_date: Optional[date], trigger: str) -> dict:
        logger.warning("Regime model not loaded — returning UNKNOWN (legacy fallback)")
        return {
            "regime_score": 0.5,
            "regime_label": "UNKNOWN",
            "prob_risk_off": None,
            "prob_risk_caution": None,
            "prob_risk_on": None,
            "version": "legacy_fallback",
            "trigger": trigger,
            "cached": False,
        }

    def invalidate_cache(self) -> None:
        self._cache_score = None
        self._cache_label = None
        self._cache_probs = None
        self._cache_ts = 0.0
        self._cache_date = None
