"""Phase R7 — Regime V2 model trainer.

Labels:  V2 rule-based 3-class (RISK_OFF=0, RISK_CAUTION=1, RISK_ON=2)
Model:   XGBoost multi:softprob + temperature scaling (no isotonic)
Score:   E[class/2] = 0.5*P(CAUTION) + 1.0*P(RISK_ON)  →  naturally in [0,1]
Folds:   3 FIXED expanding walk-forward folds from 2018-01-01 (frozen for
         version-over-version comparability) + 1 ROLLING fold over the last
         ROLLING_TEST_WINDOW_DAYS of the dataset. The fixed folds decide
         comparability; the ROLLING fold is what makes the gate able to fail
         at all — without it the walk-forward re-scores the same three
         windows forever. See ROLLING_TEST_WINDOW_DAYS.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from xgboost import XGBClassifier

from app.ml.retrain_config import MAX_WORKERS
from app.ml.regime_features import (
    CORE_FEATURE_NAMES,
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

    Gate, on the FIXED folds:
          macro_F1 min >= REGIME_GATE_MACRO_F1_MIN
          AND log_loss mean (3-class cross-entropy; random baseline = log(3) ≈ 1.099)
              < REGIME_GATE_LOG_LOSS_MAX.
    Gate, on the ROLLING fold:
          rolling_log_loss present AND < REGIME_GATE_LOG_LOSS_MAX.

    THE ROLLING TERM IS THE ONLY ONE THAT CAN REACT TO RECENT DATA. The fixed folds
    re-score three frozen windows, so on their own they return the same numbers forever
    — which is exactly how versions v35..v42 all passed with byte-identical metrics
    while the training set sat frozen for four months (DECISIONS 2026-09-06).

    A MISSING rolling term is a FAILURE, not a pass. If the rolling fold could not be
    built or scored — short dataset, thin window, stale snapshots — then the gate has no
    evidence about the present, and "no evidence" must not read as "no problem". Failing
    is the fail-safe direction: `_retrain_regime` deletes the new pickle and keeps the
    prior passing model, so the cost of a false failure is staying on last week's
    weights, while the cost of a false pass is promoting an unvalidated model into live
    position sizing.

    Rolling is gated on LOG LOSS rather than macro_F1 because the rolling window is one
    season of market and may legitimately contain no RISK_OFF day at all; macro_F1 would
    then swing on which classes happen to appear, whereas log_loss is computed against a
    fixed labels=[0,1,2] and stays well defined.

    Reads with safe defaults so a missing/garbage payload FAILS the gate rather than
    raising. Returns (passed, failure_reasons).
    """
    from app.ml.retrain_config import (
        REGIME_GATE_MACRO_F1_MIN,
        REGIME_GATE_LOG_LOSS_MAX,
    )
    f1_min = payload.get("wf_auc_min", 0.0)            # macro_F1 min across FIXED folds
    log_loss = payload.get("wf_log_loss_mean", 99.0)   # 3-class CE mean across FIXED folds
    rolling_ll = payload.get("rolling_log_loss")       # None => could not be evaluated
    failures: list[str] = []

    # DATA STALENESS IS A GATE FAILURE, NOT A LOG LINE.
    #
    # The rolling fold is derived from the DATA's end, not the calendar. That is correct
    # for fold construction, but it means a dataset that stops advancing yields a rolling
    # fold that also stops advancing — identical window, identical rolling_log_loss,
    # forever. The un-failable gate would quietly return, one level down.
    #
    # `requested_end` is the only calendar-anchored value in the payload (it comes from
    # date.today() at train time), so comparing it against `train_end` — the data's real
    # last day — is what detects the freeze. This is precisely the condition that caused
    # the original defect: the backfill stopped on 2026-05-07 and nothing noticed for four
    # months, because the only symptom was a metric that never moved.
    # REQUIRED, not opt-in. Reading these only "if present" would make the staleness term
    # depend on payload contents — the opposite of the missing-evidence-is-a-FAILURE rule
    # applied to rolling_log_loss below, and an open door for exactly the payload shape
    # (v35..v42) that had no way to express staleness at all.
    train_end = payload.get("train_end")
    requested_end = payload.get("requested_end")
    if not train_end or not requested_end:
        failures.append(
            "train_end/requested_end missing — cannot tell whether the training data is "
            "current; refusing to promote"
        )
    else:
        try:
            lag = (date.fromisoformat(str(requested_end)[:10])
                   - date.fromisoformat(str(train_end)[:10])).days
            if lag > MAX_DATA_LAG_DAYS:
                failures.append(
                    f"regime snapshots are {lag} days stale (data ends {train_end}, "
                    f"asked through {requested_end}) > {MAX_DATA_LAG_DAYS} — the model is "
                    f"being retrained on data it has already seen"
                )
        except (TypeError, ValueError):
            failures.append(f"unparseable train_end/requested_end "
                            f"({train_end!r}/{requested_end!r})")
    if not isinstance(f1_min, (int, float)) or f1_min < REGIME_GATE_MACRO_F1_MIN:
        failures.append(f"macro_F1_min {f1_min} < {REGIME_GATE_MACRO_F1_MIN}")
    if not isinstance(log_loss, (int, float)) or log_loss >= REGIME_GATE_LOG_LOSS_MAX:
        failures.append(f"log_loss {log_loss} >= {REGIME_GATE_LOG_LOSS_MAX}")
    if not isinstance(rolling_ll, (int, float)) or rolling_ll != rolling_ll:
        failures.append(
            "rolling fold not evaluated (rolling_log_loss missing) — the gate has no "
            "evidence about recent data; extend the regime snapshots"
        )
    elif rolling_ll >= REGIME_GATE_LOG_LOSS_MAX:
        failures.append(f"rolling_log_loss {rolling_ll} >= {REGIME_GATE_LOG_LOSS_MAX}")
    return (not failures, failures)


# How far the regime snapshots may lag the retrain date before the gate refuses to promote.
# 10 days spans a holiday week plus a weekend; beyond that the dataset is not merely late,
# it has stopped. See the staleness block in regime_gate.
MAX_DATA_LAG_DAYS = 10

# Thresholds on the continuous score for display/legacy label derivation
RISK_OFF_SCORE_THRESHOLD = 0.30   # score < 0.30 → RISK_OFF
RISK_ON_SCORE_THRESHOLD = 0.60    # score >= 0.60 → RISK_ON

# Walk-forward folds — expanding window, start 2018 to capture multiple regimes.
#
# These three are FROZEN ON PURPOSE: holding them identical across versions is what makes
# the version-over-version metric series comparable, and the aggregate gate inputs are
# computed from THESE ONLY. They are NOT the whole gate — see ROLLING_TEST_WINDOW_DAYS.
_FIXED_FOLDS = [
    (date(2018, 1, 1), date(2023, 12, 31), date(2024, 12, 31)),
    (date(2018, 1, 1), date(2024, 12, 31), date(2025, 9, 30)),
    (date(2018, 1, 1), date(2025, 9, 30), date(2026, 4, 30)),
]

# The rolling fold tests the LAST `ROLLING_TEST_WINDOW_DAYS` of the dataset, training on
# everything before it.
#
# WHY THIS EXISTS (2026-09-06). With only the three fixed folds, the walk-forward re-scored
# the SAME three windows on every weekly retrain, so it returned the same numbers forever:
# versions v35 through v42 all recorded wf_auc_mean=0.9563, wf_auc_min=0.9062,
# brier=0.0569, byte-identical down to fold-1's temperature=1.2444. The promotion gate was
# being applied to a constant. It could not fail, and therefore could not detect a
# regression in the model that carries the live book's position sizing.
#
# WHY IT ROLLS BOTH ENDS. A first cut pinned the rolling fold's TRAIN end at 2026-04-30 and
# let only its test end advance. That reproduces the original defect on a slower clock: the
# train cutoff ages while the test window grows without bound, so by 2027 it would be
# scoring a year-stale fit against a year of unseen regime and failing for a reason that has
# nothing to do with the model under test. Both ends are derived from the data.
#
# 120 calendar days ≈ 82 trading days — long enough to be more than noise, short enough to
# be about the present.
ROLLING_TEST_WINDOW_DAYS = 120

# The rolling fold needs a real training history behind it; refuse to build one that would
# train on less than the first fixed fold does.
_MIN_ROLLING_TRAIN_END = _FIXED_FOLDS[0][1]

# Below this the rolling test window is too thin to certify anything, so the fold is
# dropped — and because a missing rolling term FAILS the gate, a sparse window blocks
# promotion instead of quietly certifying on a fragment.
#
# A 120-calendar-day window should hold ~82 trading days. The first cut used 20, which was
# both a no-op (it equalled the generic per-fold minimum, so the branch never bound) and
# ~4x too low: a rolling fold could pass on a quarter of its intended sample with nothing
# recording that it had. 60 leaves room for holidays and a short data hiccup while still
# refusing a window with a real gap in it — which doubles as the only check that would
# notice a HOLE in the recent snapshots, since max(snapshot_date) is blind to those.
_MIN_ROLLING_TEST_ROWS = 60


def build_folds(data_end: Optional[date]) -> list:
    """The three fixed folds, plus a rolling fold testing the last ROLLING_TEST_WINDOW_DAYS.

    `data_end` is the LAST DATE ACTUALLY IN THE DATASET, never `date.today()` — the two
    diverged by four months without anyone noticing (see `train`), which is the whole
    reason this takes a parameter instead of reading the clock.
    """
    folds = list(_FIXED_FOLDS)
    if data_end is None:
        return folds
    roll_train_end = data_end - timedelta(days=ROLLING_TEST_WINDOW_DAYS)
    if roll_train_end > _MIN_ROLLING_TRAIN_END:
        folds.append((date(2018, 1, 1), roll_train_end, data_end))
    return folds


# Back-compat alias: the fixed three. Prefer build_folds() — this name no longer describes
# the folds the gate actually runs on.
_FOLDS = _FIXED_FOLDS


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

        # Drop rows with missing core market features (VIX, SPY). CORE_FEATURE_NAMES is
        # shared with the backfill's write-guard so the two cannot drift.
        df = df.dropna(subset=list(CORE_FEATURE_NAMES))
        if df.empty:
            # Re-checked AFTER the dropna: the earlier `df.empty` guard fires only when the
            # QUERY returned nothing. Newly reachable now that the staleness guard NULLs
            # vix_level rather than carrying a stale value forward — without this, train()
            # dies in `max(df["snapshot_date"])` with a bare "max() arg is an empty
            # sequence" that names neither the cause nor the subsystem.
            raise ValueError(
                f"All {len(rows)} regime snapshots for {start} → {end} were dropped for "
                f"missing core features. The upstream market-data feed is failing; check "
                f"VIX/SPY availability before retraining."
            )

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

        data_end = max(full_df["snapshot_date"]) if len(full_df) else None
        folds = build_folds(data_end)

        # The rolling fold is the only part of this gate that can react to a regression.
        # If it is missing the gate has silently reverted to re-scoring three fixed windows,
        # which is the exact failure this was built to end — so say so at WARNING, loudly,
        # rather than letting it pass as a normal 3-fold run.
        if len(folds) == len(_FIXED_FOLDS):
            logger.warning(
                "NO ROLLING FOLD: dataset ends %s — too short to carve a %d-day test window "
                "with a real training history behind it. The gate is evaluating %d FIXED "
                "windows only and cannot detect recent degradation. Extend the regime "
                "snapshots before trusting this result.",
                data_end, ROLLING_TEST_WINDOW_DAYS, len(folds),
            )
        else:
            logger.info(
                "Folds: %d fixed + 1 rolling (%s → %s)",
                len(_FIXED_FOLDS), folds[-1][1], data_end,
            )

        results = []
        for fold_idx, (train_start, train_end, test_end) in enumerate(folds, 1):
            is_rolling = fold_idx > len(_FIXED_FOLDS)
            train_df = full_df[
                (full_df["snapshot_date"] >= train_start)
                & (full_df["snapshot_date"] <= train_end)
            ]
            test_df = full_df[
                (full_df["snapshot_date"] > train_end)
                & (full_df["snapshot_date"] <= test_end)
            ]

            min_test = _MIN_ROLLING_TEST_ROWS if is_rolling else 20
            if len(train_df) < 100 or len(test_df) < min_test:
                # A skipped fold contributes nothing to the mean/min, so a skipped ROLLING
                # fold silently restores the frozen-gate behaviour. Escalate that case.
                logger.warning(
                    "%sFold %d: insufficient data (train=%d, test=%d) — excluded from the gate",
                    "ROLLING " if is_rolling else "", fold_idx, len(train_df), len(test_df),
                )
                results.append({
                    "fold": fold_idx, "log_loss": None, "macro_f1": None,
                    "n_train": len(train_df), "n_test": len(test_df),
                    "rolling": is_rolling,
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
                "rolling": is_rolling,
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

        # The REQUESTED end and the end of the data that actually came back are different
        # things, and recording the request as if it were the data is how four months of
        # staleness stayed invisible: every weekly row claimed train_end=<today> while the
        # dataset had been frozen at 2026-05-07 (the backfill stopped; the live snapshots
        # carry a different snapshot_trigger and are filtered out by load_dataset). The
        # pickle and the DB row now record what was TRAINED ON.
        data_end = max(df["snapshot_date"])
        if (end - data_end).days > 7:
            logger.warning(
                "REGIME DATA IS STALE: requested through %s but the dataset ends %s "
                "(%d days behind). The model is being retrained on data it has already "
                "seen. Extend the regime snapshots (scripts/backfill_regime_snapshots.py).",
                end, data_end, (end - data_end).days,
            )

        fold_results = self.walk_forward(df)

        # The aggregates are computed from the FIXED folds ONLY, and the rolling fold is
        # reported on its own. Two independent reasons, both learned the hard way:
        #
        #  - COMPARABILITY. Averaging a fourth fold into wf_auc_mean/brier_score would shift
        #    those numbers for a purely mechanical reason (4 terms instead of 3), breaking
        #    comparison against the v35..v42 rows — the exact property the fixed folds exist
        #    to preserve.
        #  - MACRO-F1 IS UNSTABLE ON A SHORT WINDOW. The rolling window is one season of
        #    market, and may legitimately contain no RISK_OFF day at all (measured
        #    2026-05-07→09-04: classes [1, 2] only). `f1_score(average="macro")` then
        #    averages over whichever classes happen to appear, so a single spurious
        #    RISK_OFF prediction drags it by ~1/3 for a reason that is about the window,
        #    not the model. Folding that into `wf_auc_min` — a MIN — would let a structural
        #    artifact block promotion.
        #
        # The rolling fold is gated on LOG LOSS instead, which is computed against a fixed
        # labels=[0,1,2] and is therefore well defined however the window's classes fall.
        fixed_results = [r for r in fold_results if not r.get("rolling")]
        rolling_result = next((r for r in fold_results if r.get("rolling")), None)

        lls = [r["log_loss"] for r in fixed_results if r["log_loss"] is not None]
        f1s = [r["macro_f1"] for r in fixed_results if r["macro_f1"] is not None]
        ll_mean = sum(lls) / len(lls) if lls else 99.0
        f1_mean = sum(f1s) / len(f1s) if f1s else 0.0

        rolling_ll = (rolling_result or {}).get("log_loss")
        rolling_f1 = (rolling_result or {}).get("macro_f1")
        rolling_n = (rolling_result or {}).get("n_test")

        logger.info(
            "Walk-forward summary (fixed folds): log_loss mean=%.4f  macro_F1 mean=%.3f  "
            "| rolling: log_loss=%s macro_F1=%s n_test=%s",
            ll_mean, f1_mean, rolling_ll, rolling_f1, rolling_n,
        )

        if version is None:
            # max+1, not len+1: counting collides (and silently overwrites) the moment any
            # version is deleted or archived. See app.ml.model_versioning.
            from app.ml.model_versioning import next_version
            version = next_version(MODEL_DIR, "regime_model")

        model_path = MODEL_DIR / f"regime_model_v{version}.pkl"
        xgb_model, temperature = self.train_final(df)

        # FIXED folds only — see the aggregation note above. A structurally class-poor
        # rolling window must not become the binding MIN.
        f1_min = min(
            (r["macro_f1"] for r in fixed_results if r.get("macro_f1") is not None),
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
                "train_end": data_end.isoformat(),   # data actually trained on, NOT the requested end
                "requested_end": end.isoformat(),
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
                # ROLLING-fold gate inputs, kept SEPARATE from the aggregates above so the
                # fixed-fold series stays comparable across versions. `rolling_log_loss`
                # is the only gate term that can react to recent data; None means the
                # rolling fold could not be built or scored, which regime_gate treats as a
                # FAILURE — see the fail-safe note there.
                "rolling_log_loss": rolling_ll,
                "rolling_macro_f1": rolling_f1,
                "rolling_n_test": rolling_n,
                "rolling_test_end": data_end.isoformat(),
            }, f)

        logger.info("Regime model v%d saved → %s", version, model_path)
        self._write_model_version(version, start, data_end, fold_results, ll_mean,
                                  f1_mean, f1_min, model_path)
        return model_path

    def _write_model_version(
        self,
        version: int,
        train_start: date,
        train_end: date,
        fold_results: list,
        ll_mean: float,
        f1_mean: float,
        f1_min: float,
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
                # Passed in, NOT recomputed over `fold_results` — recomputing here would
                # silently re-include the rolling fold and write a DIFFERENT wf_auc_min
                # into the registry than the pickle carries, contaminating the very
                # version-over-version series the fixed folds exist to preserve. The
                # registry and the pickle must agree by construction.
                wf_auc_min=f1_min,
                brier_score=ll_mean,        # repurposed: store log_loss here
                notes=json.dumps(fold_results),
                model_path=str(model_path),
            )
            session.add(row)
            session.commit()
            logger.info("RegimeModelVersion v%d written to DB", version)
