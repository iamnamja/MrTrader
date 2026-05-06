"""Phase R2 — Regime model trainer.

Labels: V1 rule-based (spy_1d_return > 0 AND vix_level < 20 AND spy_ma20_dist > 0)
Model:  XGBoost + CalibratedClassifierCV(isotonic)
Walk-forward: 3 expanding folds
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from app.ml.regime_features import REGIME_FEATURE_NAMES, RegimeFeatureBuilder

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

RISK_OFF_THRESHOLD = 0.35
RISK_ON_THRESHOLD = 0.65

# Walk-forward fold boundaries (train_end, test_end) — all inclusive
_FOLDS = [
    (date(2023, 1, 1), date(2024, 12, 31), date(2025, 6, 30)),
    (date(2023, 1, 1), date(2025, 6, 30), date(2025, 12, 31)),
    (date(2023, 1, 1), date(2025, 12, 31), date(2026, 4, 30)),
]


def label_regime_day(vix_level: float, spy_1d_return: float, spy_ma20_dist: float) -> int:
    """V1 rule-based label. 1 = favorable trading environment, 0 = hostile."""
    return int(
        spy_1d_return > 0.0
        and vix_level < 20.0
        and spy_ma20_dist > 0.0
    )


class RegimeModelTrainer:
    def __init__(self) -> None:
        self._builder = RegimeFeatureBuilder()

    def load_dataset(self, start: date, end: date) -> pd.DataFrame:
        """Load regime_snapshots rows for date range, return feature+label DataFrame."""
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
                row["vix_level_raw"] = s.vix_level
                row["spy_1d_return_raw"] = s.spy_1d_return
                row["spy_ma20_dist_raw"] = s.spy_ma20_dist
                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"No backfill snapshots found for {start} → {end}")

        df["label"] = df.apply(
            lambda r: label_regime_day(
                r["vix_level_raw"], r["spy_1d_return_raw"], r["spy_ma20_dist_raw"]
            ),
            axis=1,
        )
        df = df.drop(columns=["vix_level_raw", "spy_1d_return_raw", "spy_ma20_dist_raw"])

        # Fill features that are structurally null in backfill data with neutral defaults.
        # NIS data is only live; breadth is Phase R2 future.
        df["nis_risk_numeric"] = df["nis_risk_numeric"].fillna(0.5).infer_objects(copy=False)
        df["nis_sizing_factor"] = df["nis_sizing_factor"].fillna(1.0).infer_objects(copy=False)
        df["breadth_pct_ma50"] = df["breadth_pct_ma50"].fillna(0.5).infer_objects(copy=False)

        # Drop rows where core market features (VIX, SPY) are missing
        core_features = [f for f in REGIME_FEATURE_NAMES
                         if f not in ("nis_risk_numeric", "nis_sizing_factor", "breadth_pct_ma50")]
        df = df.dropna(subset=core_features)
        logger.info("Dataset: %d rows, positive rate=%.1f%%", len(df), df["label"].mean() * 100)
        return df

    def _build_xgb(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )

    def _fit_calibrated(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train XGBoost + isotonic calibration. Returns (xgb_model, iso_model)."""
        # Use 80/20 split for calibration to avoid data leakage
        n = len(X)
        n_train = int(n * 0.8)
        idx = np.random.RandomState(42).permutation(n)
        X_tr, X_cal = X[idx[:n_train]], X[idx[n_train:]]
        y_tr, y_cal = y[idx[:n_train]], y[idx[n_train:]]

        xgb = self._build_xgb()
        xgb.fit(X_tr, y_tr)
        raw_proba = xgb.predict_proba(X_cal)[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_proba, y_cal)

        return xgb, iso

    def _predict_proba(self, xgb, iso, X: np.ndarray) -> np.ndarray:
        raw = xgb.predict_proba(X)[:, 1]
        return iso.predict(raw)

    def walk_forward(self, full_df: pd.DataFrame) -> list[dict]:
        """Run 3 expanding folds, return per-fold metrics."""
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

            if len(train_df) < 50 or len(test_df) < 10:
                logger.warning("Fold %d: insufficient data (train=%d, test=%d)", fold_idx, len(train_df), len(test_df))
                results.append({"fold": fold_idx, "auc": None, "brier": None, "n_train": len(train_df), "n_test": len(test_df)})
                continue

            X_train = train_df[REGIME_FEATURE_NAMES].values
            y_train = train_df["label"].values
            X_test = test_df[REGIME_FEATURE_NAMES].values
            y_test = test_df["label"].values

            xgb, iso = self._fit_calibrated(X_train, y_train)
            proba = self._predict_proba(xgb, iso, X_test)
            auc = roc_auc_score(y_test, proba)
            brier = brier_score_loss(y_test, proba)

            logger.info(
                "Fold %d [%s→%s / test→%s]: AUC=%.4f  Brier=%.4f  (n_train=%d, n_test=%d)",
                fold_idx, train_start, train_end, test_end, auc, brier, len(train_df), len(test_df),
            )
            results.append({
                "fold": fold_idx,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_end": test_end.isoformat(),
                "auc": round(auc, 4),
                "brier": round(brier, 4),
                "n_train": len(train_df),
                "n_test": len(test_df),
            })
        return results

    def train_final(self, full_df: pd.DataFrame) -> tuple:
        """Train on all data for production use. Returns (xgb_model, iso_model)."""
        X = full_df[REGIME_FEATURE_NAMES].values
        y = full_df["label"].values
        return self._fit_calibrated(X, y)

    def train(
        self,
        start: date = date(2023, 1, 1),
        end: Optional[date] = None,
        version: Optional[int] = None,
    ) -> Path:
        """Full training pipeline. Returns path to saved model."""
        if end is None:
            end = date.today()

        df = self.load_dataset(start, end)
        fold_results = self.walk_forward(df)

        aucs = [r["auc"] for r in fold_results if r["auc"] is not None]
        briers = [r["brier"] for r in fold_results if r["brier"] is not None]

        auc_min = min(aucs) if aucs else 0.0
        auc_mean = sum(aucs) / len(aucs) if aucs else 0.0
        brier_mean = sum(briers) / len(briers) if briers else 1.0

        logger.info("Walk-forward summary: AUC min=%.4f mean=%.4f  Brier mean=%.4f", auc_min, auc_mean, brier_mean)

        if version is None:
            # Auto-increment
            existing = sorted(MODEL_DIR.glob("regime_model_v*.pkl"))
            version = len(existing) + 1

        model_path = MODEL_DIR / f"regime_model_v{version}.pkl"
        xgb_model, iso_model = self.train_final(df)

        with open(model_path, "wb") as f:
            pickle.dump({
                "xgb_model": xgb_model,
                "iso_model": iso_model,
                "feature_names": REGIME_FEATURE_NAMES,
                "version": version,
                "trained_at": datetime.utcnow().isoformat(),
                "train_start": start.isoformat(),
                "train_end": end.isoformat(),
                "wf_results": fold_results,
                "wf_auc_mean": auc_mean,
                "wf_auc_min": auc_min,
                "brier_score": brier_mean,
            }, f)

        logger.info("Model saved → %s", model_path)
        self._write_model_version(version, start, end, fold_results, auc_mean, auc_min, brier_mean, model_path)

        return model_path

    def _write_model_version(
        self,
        version: int,
        train_start: date,
        train_end: date,
        fold_results: list[dict],
        auc_mean: float,
        auc_min: float,
        brier_score: float,
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
                wf_auc_mean=auc_mean,
                wf_auc_min=auc_min,
                brier_score=brier_score,
                notes=json.dumps(fold_results),
                model_path=str(model_path),
            )
            session.add(row)
            session.commit()
            logger.info("RegimeModelVersion v%d written to DB", version)
