#!/usr/bin/env python
"""CLI: train regime V1 model.

Usage:
    python scripts/train_regime_model.py
    python scripts/train_regime_model.py --start 2023-01-01 --end 2025-12-31
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train regime classification model")
    parser.add_argument("--start", default="2023-01-01", help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Training end date (YYYY-MM-DD), default=today")
    parser.add_argument("--version", type=int, default=None, help="Model version number (auto-increments if omitted)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    logger.info("Training regime model: %s → %s", start, end)

    from app.ml.regime_training import RegimeModelTrainer

    trainer = RegimeModelTrainer()
    model_path = trainer.train(start=start, end=end, version=args.version)

    # Validate gate
    import pickle
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    auc_min = payload["wf_auc_min"]
    brier = payload["brier_score"]

    logger.info("=" * 60)
    logger.info("Walk-forward AUC min : %.4f  (gate: >= 0.60)", auc_min)
    logger.info("Brier score (mean)   : %.4f  (gate: < 0.22)", brier)
    logger.info("Model path           : %s", model_path)

    gate_pass = auc_min >= 0.60 and brier < 0.22
    if gate_pass:
        logger.info("GATE: PASSED ✓")
    else:
        failures = []
        if auc_min < 0.60:
            failures.append(f"AUC min {auc_min:.4f} < 0.60")
        if brier >= 0.22:
            failures.append(f"Brier {brier:.4f} >= 0.22")
        logger.error("GATE: FAILED — %s", ", ".join(failures))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
