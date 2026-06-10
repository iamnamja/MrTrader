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

    # Gate via the shared evaluator (single source of truth — see regime_gate()).
    # PRIOR BUG: this read payload["wf_auc_min"]/["brier_score"] directly, but those keys
    # were only written to the DB row, NOT the pickle → KeyError; and the old cutoff was a
    # 2-class Brier value wrongly applied to the 3-class cross-entropy log-loss.
    from app.ml.regime_training import regime_gate
    from app.ml.retrain_config import REGIME_GATE_MACRO_F1_MIN, REGIME_GATE_LOG_LOSS_MAX

    f1_min = payload.get("wf_auc_min", 0.0)               # macro_F1 min across folds
    log_loss_mean = payload.get("wf_log_loss_mean", 99.0)  # 3-class CE mean across folds
    gate_pass, failures = regime_gate(payload)

    logger.info("=" * 60)
    logger.info("Walk-forward macro_F1 min : %.4f  (gate: >= %.2f)", f1_min, REGIME_GATE_MACRO_F1_MIN)
    logger.info("Log-loss mean (3-class CE): %.4f  (gate: < %.2f, random baseline=1.099)",
                log_loss_mean, REGIME_GATE_LOG_LOSS_MAX)
    logger.info("Model path                : %s", model_path)

    if gate_pass:
        logger.info("GATE: PASSED ✓")
    else:
        logger.error("GATE: FAILED — %s", ", ".join(failures))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
