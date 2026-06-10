"""Phase R7 — Regime V2 model trainer.

Labels:  V2 rule-based 3-class (RISK_OFF=0, RISK_CAUTION=1, RISK_ON=2)
Model:   XGBoost multi:softprob + temperature scaling (no isotonic)
Score:   E[class/2] = 0.5*P(CAUTION) + 1.0*P(RISK_ON)  →  naturally in [0,1]
Folds:   3 expanding walk-forward folds from 2018-01-01
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from xgboost import XGBClassifier

from app.ml.retrain_config import MAX_WORKERS
from app.ml.regime_features import (
    REGIME_FEATURE_NAMES,
    RegimeFeatureBuilder,
    label_regime_day,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def regime_gate(payload: dict) -> tuple[bool, list[str]]:
    """Evaluate the regime-model promotion gate from a saved pickle payload.

    SINGLE SOURCE OF TRUTH for the gate — used by both ``scripts/train_regime_model.py``
    and ``PortfolioManager._retrain_regime`` so the thresholds can never drift apart
    (the prior bug was a 2-class Brier threshold 0.22 mis-applied to the 3-class model).

    Gate: macro_F1 min across folds >= REGIME_GATE_MACRO_F1_MIN
          AND log_loss mean (3-class cross-entropy; random baseline = log(3) ≈ 1.099)
          < REGIME_GATE_LOG_LOSS_MAX.

    Reads with safe defaults so a missing/garbage payload FAILS the gate rather than
    raising. Returns (passed, failure_reasons).
    """
    from app.ml.retrain_config import (
        REGIME_GATE_MACRO_F1_MIN,
        REGIME_GATE_LOG_LOSS_MAX,
    )
    f1_min = payload.get("wf_auc_min", 0.0)            # macro_F1 min across folds
    log_loss = payload.get("wf_log_loss_mean", 99.0)   # 3-class CE mean across folds
    failures: list[str] = []
    if not isinstance(f1_min, (int, float)) or f1_min < REGIME_GATE_MACRO_F1_MIN:
        failures.append(f"macro_F1_min {f1_min} < {REGIME_GATE_MACRO_F1_MIN}")
    if not isinstance(log_loss, (int, float)) or log_loss >= REGIME_GATE_LOG_LOSS_MAX:
        failures.append(f"log_loss {log_loss} >= {REGIME_GATE_LOG_LOSS_MAX}")
    return (not failures, failures)


# Thresholds on the continuous score for display/legacy label derivation
RISK_OFF_SCORE_THRESHOLD = 0.30   # score < 0.30 → RISK_OFF
RISK_ON_SCORE_THRESHOLD = 0.60    # score >= 0.60 → RISK_ON

# Walk-forward folds — expanding window, start 2018 to capture multiple regimes
_FOLDS = [
    (date(2018, 1, 1), date(2023, 12, 31), date(2024, 12, 31)),
    (date(2018, 1, 1), date(2024, 12, 31), date(2025, 9, 30)),
    (date(2018, 1, 1), date(2025, 9, 30), date(2026, 4, 30)),
]


def score_from_probs(probs: np.ndarray) -> np.ndarray:
    """Convert (N,3) class probability array → (N,) continuous score in [0,1].

    Score = 0*P(RISK_OFF) + 0.5*P(RISK_CAUTION) + 1.0*P(RISK_ON)
    This is naturally smooth (softmax output), no isotonic calibration needed.
    """
    return 0.5 * probs[:, 1] + 1.0 * probs[:, 2]


def label_from_score(score: float) -> str:
    """Derive regime label from continuous score. Prefer argmax where probs available."""
    if score < RISK_OFF_SCORE_THRESHOLD:
        return "RISK_OFF"
    if score >= RISK_ON_SCORE_THRESHOLD:
        return "RISK_ON"
    return "RISK_CAUTION"


class RegimeModelTrainer:
    def __init__(self) -> None:
        self._builder = RegimeFeatureBuilder()

    def load_dataset(self, start: date, end: date) -> pd.DataFrame:
        """Load regime_snapshots for date range and attach V2 rule labels."""
        from app.database.session import get_session, init_db
        from app.database.models import RegimeSnapshot

        init_db()
        rows = []
        with get_session() as session:
            snaps = (
                session.query(RegimeSnapshot)
                .filter(
                    RegimeSnapshot.snapshot_date >= start,
                    RegimeSnapshot.snapshot_date <= end,
                    RegimeSnapshot.snapshot_trigger == "backfill",
                )
                .order_by(RegimeSnapshot.snapshot_date)
                .all()
            )
            for s in snaps:
                row = {f: getattr(s, f, None) for f in REGIME_FEATURE_NAMES}
                row["snapshot_date"] = s.snapshot_date
                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"No backfill snapshots found for {start} → {end}")

        # Attach 3-class rule label — uses V2 multi-factor function
        df["label"] = df.apply(label_regime_day, axis=1)

        # Fill structurally-null features with neutral defaults
        df["nis_risk_numeric"] = df["nis_risk_numeric"].fillna(0.5).infer_objects(copy=False)
        df["nis_sizing_factor"] = df["nis_sizing_factor"].fillna(1.0).infer_objects(copy=False)

        # Drop rows with missing core market features (VIX, SPY)
        core = [f for f in REGIME_FEATURE_NAMES
                if f not in ("nis_risk_numeric", "nis_sizing_factor",
                             "breadth_pct_ma50",  # legacy col — not in V2 features
                             "vix_term_ratio", "breadth_rsp_spy_ratio_20d",
                             "credit_hyg_ief_5d", "credit_hyg_ief_20d",
                             "sector_dispersion_20d", "sector_leader_lag_20d",
                             "vix_5d_change", "spy_50d_return",
                             "spy_above_ma50", "spy_above_ma200")]
        df = df.dropna(subset=core)

        label_counts = df["label"].value_counts().to_dict()
        n = len(df)
        logger.info(
            "Dataset: %d rows | RISK_OFF=%d (%.0f%%) CAUTION=%d (%.0f%%) RISK_ON=%d (%.0f%%)",
            n,
            label_counts.get(0, 0), 100 * label_counts.get(0, 0) / max(n, 1),
            label_counts.get(1, 0), 100 * label_counts.get(1, 0) / max(n, 1),
            label_counts.get(2, 0), 100 * label_counts.get(2, 0) / max(n, 1),
        )
        return df

    def _build_xgb(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=MAX_WORKERS,
        )

    def _fit_with_temperature(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train XGBoost multiclass + temperature scaling. Returns (xgb, temperature)."""
        n = len(X)
        n_train = int(n * 0.8)
        # Time-sorted split to avoid leakage (data arrives sorted by date)
        X_tr, X_cal = X[:n_train], X[n_train:]
        y_tr, y_cal = y[:n_train], y[n_train:]

        xgb = self._build_xgb()
        xgb.fit(X_tr, y_tr)

        # Temperature scaling: find T that minimises NLL on calibration set
        raw_logits = xgb.get_booster().predict(
            __import__("xgboost").DMatrix(X_cal), output_margin=True
        )  # shape (N, 3)

        def nll(log_t):
            T = np.exp(log_t)  # constrain T > 0
            scaled = raw_logits / T
            probs = softmax(scaled, axis=1)
            probs = np.clip(probs, 1e-8, 1 - 1e-8)
            return -np.mean(np.log(probs[np.arange(len(y_cal)), y_cal]))

        result = minimize_scalar(nll, bounds=(-2, 3), method="bounded")
        temperature = float(np.exp(result.x))
        logger.info("Temperature scaling: T=%.4f (NLL=%.4f)", temperature, result.fun)

        return xgb, temperature

    def _predict_probs(self, xgb, temperature: float, X: np.ndarray) -> np.ndarray:
        """Returns (N, 3) calibrated class probabilities."""
        import xgboost as xgb_lib
        raw_logits = xgb.get_booster().predict(
            xgb_lib.DMatrix(X), output_margin=True
        )
        scaled = raw_logits / temperature
        return softmax(scaled, axis=1)

    def walk_forward(self, full_df: pd.DataFrame) -> list:
        """Run expanding-window walk-forward, return per-fold metrics."""
        from sklearn.metrics import log_loss, f1_score, confusion_matrix

        results = []
        for fold_idx, (train_start, train_end, test_end) in enumerate(_FOLDS, 1):
            train_df = full_df[
                (full_df["snapshot_date"] >= train_start)
                & (full_df["snapshot_date"] <= train_end)
            ]
            test_df = full_df[
                (full_df["snapshot_date"] > train_end)
                & (full_df["snapshot_date"] <= test_end)
            ]

            if len(train_df) < 100 or len(test_df) < 20:
                logger.warning("Fold %d: insufficient data (train=%d, test=%d)",
                               fold_idx, len(train_df), len(test_df))
                results.append({
                    "fold": fold_idx, "log_loss": None, "macro_f1": None,
                    "n_train": len(train_df), "n_test": len(test_df),
                })
                continue

            X_train = train_df[REGIME_FEATURE_NAMES].values.astype(float)
            y_train = train_df["label"].values.astype(int)
            X_test = test_df[REGIME_FEATURE_NAMES].values.astype(float)
            y_test = test_df["label"].values.astype(int)

            xgb, temp = self._fit_with_temperature(X_train, y_train)
            probs = self._predict_probs(xgb, temp, X_test)
            scores = score_from_probs(probs)
            pred_labels = np.argmax(probs, axis=1)

            ll = float(log_loss(y_test, probs, labels=[0, 1, 2]))
            f1 = float(f1_score(y_test, pred_labels, average="macro", zero_division=0))
            cm = confusion_matrix(y_test, pred_labels, labels=[0, 1, 2])

            # Critical confusion: RISK_OFF predicted as RISK_ON or vice versa
            off_on_confusion = (
                int(cm[0, 2]) + int(cm[2, 0])
            ) / max(len(y_test), 1)

            pred_distribution = {
                "RISK_OFF": int((pred_labels == 0).sum()),
                "RISK_CAUTION": int((pred_labels == 1).sum()),
                "RISK_ON": int((pred_labels == 2).sum()),
            }
            score_mean = float(scores.mean())
            score_std = float(scores.std())

            logger.info(
                "Fold %d [train→%s / test→%s]: log_loss=%.4f  macro_F1=%.3f  "
                "off↔on_confusion=%.1f%%  score_mean=%.3f±%.3f  T=%.3f",
                fold_idx, train_end, test_end, ll, f1,
                100 * off_on_confusion, score_mean, score_std, temp,
            )
            results.append({
                "fold": fold_idx,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_end": test_end.isoformat(),
                "log_loss": round(ll, 4),
                "macro_f1": round(f1, 4),
                "off_on_confusion_pct": round(100 * off_on_confusion, 2),
                "score_mean": round(score_mean, 4),
                "score_std": round(score_std, 4),
                "temperature": round(temp, 4),
                "pred_distribution": pred_distribution,
                "n_train": len(train_df),
                "n_test": len(test_df),
            })
        return results

    def train_final(self, full_df: pd.DataFrame) -> tuple:
        """Train on all data. Returns (xgb_model, temperature)."""
        X = full_df[REGIME_FEATURE_NAMES].values.astype(float)
        y = full_df["label"].values.astype(int)
        return self._fit_with_temperature(X, y)

    def train(
        self,
        start: date = date(2018, 1, 1),
        end: Optional[date] = None,
        version: Optional[int] = None,
    ) -> Path:
        """Full training pipeline. Returns path to saved model."""
        if end is None:
            end = date.today()

        df = self.load_dataset(start, end)
        fold_results = self.walk_forward(df)

        lls = [r["log_loss"] for r in fold_results if r["log_loss"] is not None]
        f1s = [r["macro_f1"] for r in fold_results if r["macro_f1"] is not None]
        ll_mean = sum(lls) / len(lls) if lls else 99.0
        f1_mean = sum(f1s) / len(f1s) if f1s else 0.0

        logger.info(
            "Walk-forward summary: log_loss mean=%.4f  macro_F1 mean=%.3f",
            ll_mean, f1_mean,
        )

        if version is None:
            existing = sorted(MODEL_DIR.glob("regime_model_v*.pkl"))
            version = len(existing) + 1

        model_path = MODEL_DIR / f"regime_model_v{version}.pkl"
        xgb_model, temperature = self.train_final(df)

        f1_min = min(
            (r["macro_f1"] for r in fold_results if r.get("macro_f1") is not None),
            default=0.0,
        )
        with open(model_path, "wb") as f:
            pickle.dump({
                "xgb_model": xgb_model,
                "temperature": temperature,
                "feature_names": REGIME_FEATURE_NAMES,
                "version": version,
                "model_version": 2,          # V2 marker — checked by regime_model.py
                "trained_at": datetime.utcnow().isoformat(),
                "train_start": start.isoformat(),
                "train_end": end.isoformat(),
                "wf_results": fold_results,
                "wf_log_loss_mean": ll_mean,
                "wf_macro_f1_mean": f1_mean,
                # Gate inputs IN THE PICKLE so the CLI and the PM weekly retrain evaluate
                # the gate without a DB query. `wf_auc_min` holds macro_F1 MIN across folds
                # (legacy key name, repurposed for the 3-class model); `brier_score` holds
                # log_loss mean (repurposed — it is 3-class cross-entropy, not a Brier
                # score). Consumed by regime_gate(). PRIOR BUG: train_regime_model.py read
                # these keys but they were only written to the DB row, never the pickle.
                "wf_auc_min": f1_min,
                "wf_auc_mean": f1_mean,
                "brier_score": ll_mean,
            }, f)

        logger.info("Regime model v%d saved → %s", version, model_path)
        self._write_model_version(version, start, end, fold_results, ll_mean, f1_mean, model_path)
        return model_path

    def _write_model_version(
        self,
        version: int,
        train_start: date,
        train_end: date,
        fold_results: list,
        ll_mean: float,
        f1_mean: float,
        model_path: Path,
    ) -> None:
        from app.database.session import get_session, init_db
        from app.database.models import RegimeModelVersion

        init_db()
        with get_session() as session:
            existing = session.query(RegimeModelVersion).filter_by(version=version).first()
            if existing:
                session.delete(existing)
                session.flush()

            row = RegimeModelVersion(
                version=version,
                trained_at=datetime.utcnow(),
                train_start=train_start,
                train_end=train_end,
                feature_names_json=json.dumps(REGIME_FEATURE_NAMES),
                wf_auc_mean=f1_mean,        # repurposed: store macro F1 here
                wf_auc_min=min(
                    (r["macro_f1"] for r in fold_results if r["macro_f1"] is not None),
                    default=0.0,
                ),
                brier_score=ll_mean,        # repurposed: store log_loss here
                notes=json.dumps(fold_results),
                model_path=str(model_path),
            )
            session.add(row)
            session.commit()
            logger.info("RegimeModelVersion v%d written to DB", version)
