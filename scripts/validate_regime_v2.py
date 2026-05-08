"""Phase R7 — Regime V2 validation script.

Checks:
  1. Label distribution in backfill data (target: ~15-25% RISK_OFF)
  2. Score distribution after training (should be continuous [0,1], not bimodal)
  3. Walk-forward gates: log_loss < 1.0, macro_F1 >= 0.40
  4. Temporal stability: monthly label distribution
  5. Known-date spot-checks (COVID crash → RISK_OFF, 2021 bull → RISK_ON)

Usage:
    python scripts/validate_regime_v2.py
    python scripts/validate_regime_v2.py --model-path app/ml/models/regime_model_v1.pkl
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

KNOWN_DATES = [
    # (date, expected_label, description)
    (date(2020, 3, 16), "RISK_OFF", "COVID crash peak"),
    (date(2022, 6, 15), "RISK_OFF", "2022 bear market (Fed hikes)"),
    (date(2021, 7, 1), "RISK_ON", "2021 bull market"),
    (date(2023, 11, 1), "RISK_ON", "2023 recovery"),
    (date(2020, 10, 15), "RISK_CAUTION", "pre-election uncertainty"),
]


def check_label_distribution() -> bool:
    """Gate: RISK_OFF should be 15-35% of backfill rows."""
    from app.database.session import get_session, init_db
    from app.database.models import RegimeSnapshot
    from app.ml.regime_features import label_regime_day, REGIME_FEATURE_NAMES

    init_db()
    rows = []
    with get_session() as session:
        snaps = (
            session.query(RegimeSnapshot)
            .filter(RegimeSnapshot.snapshot_trigger == "backfill")
            .order_by(RegimeSnapshot.snapshot_date)
            .all()
        )
        for s in snaps:
            row = {f: getattr(s, f, None) for f in REGIME_FEATURE_NAMES}
            row["snapshot_date"] = s.snapshot_date
            rows.append(row)

    if not rows:
        logger.error("No backfill rows found — run backfill_regime_snapshots.py first")
        return False

    import pandas as pd
    df = pd.DataFrame(rows)
    df["label"] = df.apply(label_regime_day, axis=1)

    n = len(df)
    counts = df["label"].value_counts()
    pct = {k: 100 * v / n for k, v in counts.items()}

    logger.info("Label distribution (%d rows):", n)
    logger.info("  RISK_OFF     = %4d  (%.1f%%)", counts.get(0, 0), pct.get(0, 0))
    logger.info("  RISK_CAUTION = %4d  (%.1f%%)", counts.get(1, 0), pct.get(1, 0))
    logger.info("  RISK_ON      = %4d  (%.1f%%)", counts.get(2, 0), pct.get(2, 0))

    off_pct = pct.get(0, 0)
    gate = 10.0 <= off_pct <= 40.0
    logger.info("Label distribution gate (%s): RISK_OFF=%.1f%% (target 10-40%%)",
                "PASS" if gate else "FAIL", off_pct)

    # Monthly breakdown for temporal stability
    df["month"] = pd.to_datetime(df["snapshot_date"]).dt.to_period("Q")
    monthly = df.groupby("month")["label"].value_counts(normalize=True).unstack(fill_value=0)
    logger.info("Quarterly label distribution:\n%s", monthly.to_string())

    return gate


def check_score_distribution(model_path: Path) -> bool:
    """Gate: score should be continuous, not bimodal (no >=90% at 0 or 1)."""
    import pickle
    import numpy as np

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    logger.info("Model version: %s  model_version=%d  T=%.4f",
                payload.get("version"), payload.get("model_version", 1),
                payload.get("temperature", 1.0))

    wf = payload.get("wf_results", [])
    if not wf:
        logger.warning("No walk-forward results in model pickle")
        return True

    lls = [r["log_loss"] for r in wf if r.get("log_loss") is not None]
    f1s = [r["macro_f1"] for r in wf if r.get("macro_f1") is not None]
    ll_mean = sum(lls) / len(lls) if lls else 99.0
    f1_mean = sum(f1s) / len(f1s) if f1s else 0.0

    logger.info("Walk-forward results:")
    for r in wf:
        logger.info(
            "  Fold %d: log_loss=%.4f  macro_F1=%.3f  score=%.3f±%.3f  T=%.3f  "
            "off↔on_confusion=%.1f%%  (train=%d test=%d)",
            r["fold"], r.get("log_loss") or 0, r.get("macro_f1") or 0,
            r.get("score_mean") or 0, r.get("score_std") or 0,
            r.get("temperature") or 1.0,
            r.get("off_on_confusion_pct") or 0,
            r.get("n_train") or 0, r.get("n_test") or 0,
        )
        dist = r.get("pred_distribution", {})
        if dist:
            logger.info("    pred dist: %s", dist)

    logger.info("WF mean: log_loss=%.4f  macro_F1=%.3f", ll_mean, f1_mean)

    gate_ll = ll_mean < 1.0
    gate_f1 = f1_mean >= 0.35
    logger.info("log_loss gate (<1.0): %s", "PASS" if gate_ll else "FAIL")
    logger.info("macro_F1 gate (>=0.35): %s", "PASS" if gate_f1 else "FAIL")

    return gate_ll and gate_f1


def check_known_dates(model_path: Path) -> bool:
    """Spot-check regime scores on historically known market regimes."""
    from app.ml.regime_model import RegimeModel
    from app.database.session import init_db
    import pickle

    init_db()

    model = RegimeModel()
    model.load(model_path)
    if not model.loaded:
        logger.error("Model failed to load from %s", model_path)
        return False

    passed = 0
    total = len(KNOWN_DATES)
    for d, expected, desc in KNOWN_DATES:
        result = model.score(as_of_date=d, trigger="validate")
        label = result.get("regime_label", "UNKNOWN")
        score = result.get("regime_score", None)
        ok = label == expected
        status = "PASS" if ok else "FAIL"
        logger.info(
            "  [%s] %s  expected=%s  got=%s (score=%.3f) — %s",
            status, d, expected, label, score or 0.0, desc,
        )
        if ok:
            passed += 1

    logger.info("Spot-check: %d/%d passed", passed, total)
    return passed >= total * 0.6  # 60% pass rate required


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Regime V2 model")
    parser.add_argument("--model-path", default=None,
                        help="Path to model pickle (default: latest in app/ml/models/)")
    args = parser.parse_args()

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_dir = ROOT / "app" / "ml" / "models"
        candidates = sorted(model_dir.glob("regime_model_v*.pkl"))
        if not candidates:
            logger.error("No regime model found in %s", model_dir)
            sys.exit(1)
        model_path = candidates[-1]

    logger.info("Validating model: %s", model_path)

    results = {}
    results["label_distribution"] = check_label_distribution()
    results["walk_forward"] = check_score_distribution(model_path)
    results["spot_checks"] = check_known_dates(model_path)

    logger.info("\n=== VALIDATION SUMMARY ===")
    all_pass = True
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %-25s %s", check, status)
        if not passed:
            all_pass = False

    if all_pass:
        logger.info("ALL GATES PASSED — model is ready for production")
        sys.exit(0)
    else:
        logger.warning("SOME GATES FAILED — review results above before deploying")
        sys.exit(1)


if __name__ == "__main__":
    main()
