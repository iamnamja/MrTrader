"""
Portfolio Manager Agent — daily ML-driven instrument selection.

Cycle:
  1. At market open (09:30 ET, weekdays): run ML model on S&P 100
  2. Select top 10 stocks by predicted probability
  3. Send trade proposals to Risk Manager via Redis queue `trade_proposals`
  4. At 17:00 ET: retrain model with latest data
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from app.agents.base import BaseAgent
from app.ml.features import FeatureEngineer
from app.ml.model import PortfolioSelectorModel
from app.ml.training import ModelTrainer
from app.utils.constants import MARKET_CLOSE_HOUR, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, SP_100_TICKERS

logger = logging.getLogger(__name__)

TRADE_PROPOSALS_QUEUE = "trade_proposals"
TOP_N_STOCKS = 10
MIN_CONFIDENCE = 0.55          # minimum model probability to propose a trade
POSITION_RISK_PCT = 0.02       # risk 2% of account per trade for sizing


class PortfolioManager(BaseAgent):
    """
    Runs on a 60-second heartbeat. At market open it selects instruments
    via the ML model; at 17:00 it retrains the model.
    """

    def __init__(self):
        super().__init__("portfolio_manager")
        self.feature_engineer = FeatureEngineer()
        self.model = PortfolioSelectorModel(model_type="xgboost")
        self.trainer = ModelTrainer()
        self._selected_today: bool = False
        self._retrained_today: bool = False
        self._last_date: Optional[str] = None

    # ─── Lazy connectors ─────────────────────────────────────────────────────

    @property
    def _alpaca(self):
        from app.integrations import get_alpaca_client
        return get_alpaca_client()

    # ─── Main Loop ────────────────────────────────────────────────────────────

    async def run(self):
        self.logger.info("Portfolio Manager started")
        self.status = "running"
        self._try_load_model()

        while self.status == "running":
            try:
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")

                # Reset daily flags at midnight
                if today != self._last_date:
                    self._selected_today = False
                    self._retrained_today = False
                    self._last_date = today

                is_weekday = now.weekday() < 5

                # Market open: select instruments
                if (
                    is_weekday
                    and now.hour == MARKET_OPEN_HOUR
                    and now.minute == MARKET_OPEN_MINUTE
                    and not self._selected_today
                ):
                    await self.select_instruments()
                    self._selected_today = True

                # 17:00: retrain model
                if (
                    is_weekday
                    and now.hour == 17
                    and now.minute == 0
                    and not self._retrained_today
                ):
                    await self._retrain()
                    self._retrained_today = True

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                self.logger.info("Portfolio Manager cancelled — shutting down")
                self.status = "stopped"
                break
            except Exception as e:
                self.logger.error("Error in portfolio manager loop: %s", e, exc_info=True)
                await self.log_decision("PORTFOLIO_MANAGER_ERROR", reasoning={"error": str(e)})
                await asyncio.sleep(10)

    # ─── Instrument Selection ─────────────────────────────────────────────────

    async def select_instruments(self):
        """Run ML model over S&P 100, pick top stocks, send proposals."""
        self.logger.info("Selecting instruments for today...")

        if not self.model.is_trained:
            self.logger.warning("No trained model available — skipping selection")
            await self.log_decision(
                "SELECTION_SKIPPED", reasoning={"reason": "model not trained"}
            )
            return

        features_by_symbol: Dict[str, Dict[str, float]] = {}

        for symbol in SP_100_TICKERS:
            try:
                bars = self._alpaca.get_bars(symbol, timeframe="1D", limit=300)
                if bars.empty:
                    continue
                feats = self.feature_engineer.engineer_features(symbol, bars)
                if feats is not None:
                    features_by_symbol[symbol] = feats
            except Exception as e:
                self.logger.debug("Skipping %s: %s", symbol, e)

        if not features_by_symbol:
            self.logger.warning("Could not build features for any symbol")
            return

        symbols = list(features_by_symbol.keys())
        X = np.array([list(features_by_symbol[s].values()) for s in symbols])

        try:
            _, probabilities = self.model.predict(X)
        except Exception as e:
            self.logger.error("Model prediction failed: %s", e)
            return

        # Pick top N with confidence above threshold
        ranked = sorted(zip(symbols, probabilities), key=lambda x: x[1], reverse=True)
        selected = [(sym, prob) for sym, prob in ranked if prob >= MIN_CONFIDENCE][:TOP_N_STOCKS]

        self.logger.info("Selected %d instruments: %s", len(selected), [s for s, _ in selected])

        proposals = await self._build_proposals(selected)
        for proposal in proposals:
            self.send_message(TRADE_PROPOSALS_QUEUE, proposal)
            self.logger.info("Proposal sent: %s @ $%.2f (confidence=%.2f)",
                             proposal["symbol"], proposal["entry_price"], proposal["confidence"])

        await self.log_decision(
            "INSTRUMENTS_SELECTED",
            reasoning={
                "selected": [{"symbol": s, "confidence": round(float(p), 4)} for s, p in selected],
                "total_evaluated": len(symbols),
            },
        )

    # ─── Proposal Building ────────────────────────────────────────────────────

    async def _build_proposals(self, selected: List[tuple]) -> List[Dict[str, Any]]:
        """Build trade proposal dicts for the Risk Manager."""
        try:
            account = self._alpaca.get_account()
            account_value = account["portfolio_value"]
        except Exception:
            account_value = 20_000.0  # fallback

        proposals = []
        for symbol, confidence in selected:
            price = self._alpaca.get_latest_price(symbol)
            if price is None or price <= 0:
                continue

            quantity = self._calculate_quantity(price, account_value)
            proposals.append({
                "symbol": symbol,
                "direction": "BUY",
                "quantity": quantity,
                "entry_price": price,
                "confidence": float(confidence),
                "stop_loss": round(price * 0.98, 2),
                "profit_target": round(price * 1.05, 2),
                "source_agent": "portfolio_manager",
            })

        return proposals

    def _calculate_quantity(self, price: float, account_value: float) -> int:
        """Size position at POSITION_RISK_PCT of account value."""
        risk_dollars = account_value * POSITION_RISK_PCT
        qty = int(risk_dollars / price)
        return max(qty, 1)

    # ─── Retraining ───────────────────────────────────────────────────────────

    async def _retrain(self):
        """Run training pipeline in a thread so it doesn't block the event loop."""
        self.logger.info("Starting scheduled model retraining...")
        try:
            loop = asyncio.get_event_loop()
            version = await loop.run_in_executor(None, self.trainer.train_model)
            # Reload the freshly trained model
            self._try_load_model()
            self.logger.info("Model retrained → v%d now active", version)
            await self.log_decision("MODEL_RETRAINED", reasoning={"version": version})
        except Exception as e:
            self.logger.error("Retraining failed: %s", e, exc_info=True)
            await self.log_decision("RETRAINING_FAILED", reasoning={"error": str(e)})

    # ─── Model Loading ────────────────────────────────────────────────────────

    def _try_load_model(self) -> bool:
        """Attempt to load the latest model from DB. Returns True on success."""
        from app.database.models import ModelVersion
        from app.database.session import get_session

        db = get_session()
        try:
            latest = (
                db.query(ModelVersion)
                .filter_by(model_name="portfolio_selector", status="ACTIVE")
                .order_by(ModelVersion.version.desc())
                .first()
            )
            if latest and latest.model_path:
                import os
                directory = os.path.dirname(latest.model_path)
                self.model.load(directory, latest.version)
                self.logger.info("Loaded model v%d", latest.version)
                return True
            else:
                self.logger.info("No trained model found in DB — will train on first run")
                return False
        except Exception as e:
            self.logger.warning("Could not load model: %s", e)
            return False
        finally:
            db.close()


# Module-level singleton (lazy — no connections at import time)
portfolio_manager = PortfolioManager()
