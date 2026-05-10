"""
Swing Universe Expansion — SP-100 → SP-500 retrain script.

Run once after the feature/sp500-universe-expansion branch is deployed:
    python scripts/expand_universe_retrain.py

Steps:
  1. Clear the feature store (stale SP-100 entries would mix with R1K data)
  2. Retrain the swing model on RUSSELL_1000_TICKERS with 5 years of history
  3. Print the new model version number
"""
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    from app.ml.feature_store import FeatureStore
    from app.ml.training import ModelTrainer
    from app.utils.constants import RUSSELL_1000_TICKERS

    # Step 1: clear stale feature cache
    store = FeatureStore()
    before = store.count()
    store.clear()
    logger.info("Feature store cleared: %d rows removed", before)

    # Step 2: retrain on Russell 1000 universe
    logger.info("Starting R1K swing model retrain (%d symbols)...", len(RUSSELL_1000_TICKERS))
    trainer = ModelTrainer()
    version = trainer.train_model(symbols=RUSSELL_1000_TICKERS, years=5)
    logger.info("Retrain complete — model version %s", version)
    return version


if __name__ == "__main__":
    main()
