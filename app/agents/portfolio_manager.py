"""
Portfolio Manager Agent — daily ML-driven instrument selection.

Cycle:
  1. At 09:30 ET: run swing model (daily bars) → proposals tagged trade_type="swing"
  2. At 09:45 ET: run intraday model (5-min bars) → proposals tagged trade_type="intraday"
  3. Send all proposals to Risk Manager via Redis queue `trade_proposals`
  4. At 17:00 ET: retrain swing model with latest daily data
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
from app.utils.constants import MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, SP_100_TICKERS

logger = logging.getLogger(__name__)

TRADE_PROPOSALS_QUEUE = "trade_proposals"
TOP_N_STOCKS = 10
TOP_N_INTRADAY = 5             # fewer intraday picks per session
MIN_CONFIDENCE = 0.55          # minimum model probability to propose a trade
POSITION_RISK_PCT = 0.02       # risk 2% of account per trade for sizing
INTRADAY_SCAN_MINUTE = 45      # 09:45 ET for intraday scan


class PortfolioManager(BaseAgent):
    """
    Runs on a 60-second heartbeat.
    09:30: swing model selection (daily bars)
    09:45: intraday model selection (5-min bars)
    17:00: retrain swing model
    """

    def __init__(self):
        super().__init__("portfolio_manager")
        self.feature_engineer = FeatureEngineer()
        self.model = PortfolioSelectorModel(model_type="xgboost")           # swing
        self.intraday_model = PortfolioSelectorModel(model_type="xgboost")  # intraday
        self.trainer = ModelTrainer()
        self._selected_today: bool = False
        self._selected_intraday_today: bool = False
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
                    self._selected_intraday_today = False
                    self._retrained_today = False
                    self._last_date = today

                is_weekday = now.weekday() < 5

                # 09:30: swing model selection
                if (
                    is_weekday
                    and now.hour == MARKET_OPEN_HOUR
                    and now.minute == MARKET_OPEN_MINUTE
                    and not self._selected_today
                ):
                    await self.select_instruments()
                    self._selected_today = True

                # 09:45: intraday model selection
                if (
                    is_weekday
                    and now.hour == MARKET_OPEN_HOUR
                    and now.minute == INTRADAY_SCAN_MINUTE
                    and not self._selected_intraday_today
                ):
                    await self.select_intraday_instruments()
                    self._selected_intraday_today = True

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

    # ─── Ticker Universe ──────────────────────────────────────────────────────

    def _get_universe(self) -> List[str]:
        """Return active tickers from DB watchlist; fall back to SP_100_TICKERS."""
        try:
            from app.database.session import get_session
            from app.database.models import WatchlistTicker
            db = get_session()
            try:
                tickers = [
                    r.symbol for r in
                    db.query(WatchlistTicker).filter(WatchlistTicker.active == 1).all()
                ]
                if tickers:
                    return tickers
            finally:
                db.close()
        except Exception as exc:
            self.logger.debug("Watchlist DB unavailable, using SP_100: %s", exc)
        return list(SP_100_TICKERS)

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

        for symbol in self._get_universe():
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
        # Align features to model's training set (handles post-training feature additions)
        model_feature_names = getattr(self.model, "feature_names", None)
        if model_feature_names:
            X = np.array([
                [features_by_symbol[s].get(f, 0.0) for f in model_feature_names]
                for s in symbols
            ])
        else:
            X = np.array([list(features_by_symbol[s].values()) for s in symbols])
        X = np.nan_to_num(X)

        try:
            _, probabilities = self.model.predict(X)
        except Exception as e:
            self.logger.error("Model prediction failed: %s", e)
            return

        # Pick top N with confidence above threshold (read live config from DB)
        min_conf, top_n = MIN_CONFIDENCE, TOP_N_STOCKS
        try:
            from app.database.session import get_session
            from app.database.agent_config import get_agent_config
            _db = get_session()
            try:
                min_conf = get_agent_config(_db, "pm.min_confidence")
                top_n = get_agent_config(_db, "pm.top_n_stocks")
            finally:
                _db.close()
        except Exception:
            pass
        # Apply FMP price-target consensus boost.
        # Stocks trading below analyst consensus get a small upward nudge to their
        # model score — this signal is forward-looking at 10 days (mean reversion
        # toward consensus) but can't be used in training (no historical timeseries).
        # Cap boost at +0.05 so it never overrides a weak model score.
        target_upside = self._fetch_target_upside(symbols)
        boosted_probs = []
        for sym, prob in zip(symbols, probabilities):
            upside = target_upside.get(sym, 0.0)
            # Only boost if upside > 10% (stock meaningfully below consensus)
            boost = min(0.05, max(0.0, upside * 0.15)) if upside > 0.10 else 0.0
            boosted_probs.append(float(prob) + boost)

        ranked = sorted(zip(symbols, boosted_probs), key=lambda x: x[1], reverse=True)
        selected = [(sym, prob) for sym, prob in ranked if prob >= min_conf][:top_n]

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
                "price_target_boost_applied": sum(1 for s in [sym for sym, _ in selected]
                                                  if target_upside.get(s, 0.0) > 0.10),
            },
        )

    # ─── Intraday Selection ───────────────────────────────────────────────────

    async def select_intraday_instruments(self):
        """Run intraday model on 5-min bars, send intraday proposals."""
        self.logger.info("Selecting intraday instruments (09:45 scan)...")

        if not self.intraday_model.is_trained:
            self.logger.warning("No intraday model available — skipping intraday scan")
            return

        features_by_symbol: Dict[str, Dict[str, float]] = {}

        for symbol in self._get_universe()[:50]:  # limit to 50 for speed
            try:
                from app.ml.intraday_features import compute_intraday_features, MIN_BARS
                bars = self._alpaca.get_bars(symbol, timeframe="5Min", limit=78)
                if bars is None or bars.empty or len(bars) < MIN_BARS:
                    continue
                # Fetch prior-day daily bar for S/R and gap features
                daily = self._alpaca.get_bars(symbol, timeframe="1Day", limit=2)
                prior_close = prior_high = prior_low = None
                if daily is not None and len(daily) >= 2:
                    prev = daily.iloc[-2]
                    prior_close = float(prev["close"])
                    prior_high = float(prev["high"])
                    prior_low = float(prev["low"])
                feats = compute_intraday_features(
                    bars, prior_close=prior_close,
                    prior_day_high=prior_high, prior_day_low=prior_low,
                )
                if feats is not None:
                    features_by_symbol[symbol] = feats
            except Exception as exc:
                self.logger.debug("Intraday feature skip %s: %s", symbol, exc)

        if not features_by_symbol:
            self.logger.warning("No intraday features computed")
            return

        symbols = list(features_by_symbol.keys())
        X = np.array([list(features_by_symbol[s].values()) for s in symbols])

        try:
            _, probabilities = self.intraday_model.predict(X)
        except Exception as exc:
            self.logger.error("Intraday model prediction failed: %s", exc)
            return

        min_conf = MIN_CONFIDENCE
        ranked = sorted(zip(symbols, probabilities), key=lambda x: x[1], reverse=True)
        selected = [(sym, prob) for sym, prob in ranked if prob >= min_conf][:TOP_N_INTRADAY]

        if not selected:
            self.logger.info("No intraday candidates above confidence threshold")
            return

        self.logger.info("Intraday selected: %s", [s for s, _ in selected])

        try:
            account = self._alpaca.get_account()
            account_value = account["portfolio_value"]
        except Exception:
            account_value = 20_000.0

        for symbol, confidence in selected:
            price = self._alpaca.get_latest_price(symbol)
            if price is None or price <= 0:
                continue
            quantity = self._calculate_quantity(price, account_value)
            proposal: Dict[str, Any] = {
                "symbol": symbol,
                "direction": "BUY",
                "quantity": quantity,
                "entry_price": price,
                "confidence": float(confidence),
                "stop_loss": round(price * 0.995, 2),      # tight 0.5% stop
                "profit_target": round(price * 1.01, 2),  # 1% target
                "source_agent": "portfolio_manager",
                "trade_type": "intraday",
            }
            self.send_message(TRADE_PROPOSALS_QUEUE, proposal)
            self.logger.info(
                "Intraday proposal: %s @ $%.2f (confidence=%.2f)",
                symbol, price, confidence,
            )

        await self.log_decision(
            "INTRADAY_INSTRUMENTS_SELECTED",
            reasoning={"selected": [{"symbol": s, "confidence": round(float(p), 4)}
                                    for s, p in selected]},
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
            proposal: Dict[str, Any] = {
                "symbol": symbol,
                "direction": "BUY",
                "quantity": quantity,
                "entry_price": price,
                "confidence": float(confidence),
                "stop_loss": round(price * 0.98, 2),
                "profit_target": round(price * 1.05, 2),
                "source_agent": "portfolio_manager",
                "trade_type": "swing",
            }
            # Optional AI signal review (non-blocking)
            try:
                from app.ai.claude_client import review_pm_signal
                from app.strategy.regime_detector import regime_detector
                regime = await asyncio.to_thread(regime_detector.get_regime)
                ai_summary = review_pm_signal(
                    symbol=symbol,
                    signal_type="ML_SELECTION",
                    confidence=float(confidence),
                    reasoning={"price": price, "stop": proposal["stop_loss"],
                               "target": proposal["profit_target"]},
                    regime=regime,
                )
                if ai_summary:
                    proposal["ai_review"] = ai_summary
            except Exception:
                pass
            proposals.append(proposal)

        return proposals

    def _fetch_target_upside(self, symbols: List[str]) -> Dict[str, float]:
        """
        Return FMP price-target upside ratio for each symbol.
        upside = (targetConsensus - current_price) / current_price
        Positive = stock trading below consensus, negative = above.
        Returns {} on any failure — boost is optional, never blocks selection.
        """
        result: Dict[str, float] = {}
        try:
            import requests
            from app.config import settings
            key = settings.fmp_api_key
            if not key:
                return result
            # Fetch current prices via Alpaca snapshot
            prices: Dict[str, float] = {}
            for sym in symbols:
                try:
                    bars = self._alpaca.get_bars(sym, timeframe="1D", limit=1)
                    if not bars.empty:
                        prices[sym] = float(bars["close"].iloc[-1])
                except Exception:
                    pass
            # Fetch consensus targets (one request per symbol — cached 24h by FMP)
            base = "https://financialmodelingprep.com/stable"
            for sym in symbols:
                price = prices.get(sym)
                if not price or price <= 0:
                    continue
                try:
                    r = requests.get(
                        f"{base}/price-target-consensus",
                        params={"symbol": sym, "apikey": key},
                        timeout=5,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        if data:
                            target = data[0].get("targetConsensus") or data[0].get("targetMedian")
                            if target and float(target) > 0:
                                result[sym] = (float(target) - price) / price
                except Exception:
                    pass
        except Exception as exc:
            self.logger.debug("Price target upside fetch failed: %s", exc)
        return result

    def _calculate_quantity(self, price: float, account_value: float) -> int:
        """Size position at pm.position_risk_pct of account value (DB-configurable)."""
        risk_pct = POSITION_RISK_PCT
        try:
            from app.database.session import get_session
            from app.database.agent_config import get_agent_config
            _db = get_session()
            try:
                risk_pct = get_agent_config(_db, "pm.position_risk_pct")
            finally:
                _db.close()
        except Exception:
            pass
        risk_dollars = account_value * risk_pct
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
        """Attempt to load the latest swing + intraday models from DB."""
        from app.database.models import ModelVersion
        from app.database.session import get_session
        import os

        db = get_session()
        swing_loaded = intraday_loaded = False
        try:
            for model_name, model_obj in [
                ("swing", self.model),
                ("intraday", self.intraday_model),
            ]:
                latest = (
                    db.query(ModelVersion)
                    .filter_by(model_name=model_name, status="ACTIVE")
                    .order_by(ModelVersion.version.desc())
                    .first()
                )
                if latest and latest.model_path:
                    try:
                        import pickle
                        with open(latest.model_path, "rb") as f:
                            loaded = pickle.load(f)
                        if model_name == "swing":
                            self.model = loaded
                            swing_loaded = True
                        else:
                            self.intraday_model = loaded
                            intraday_loaded = True
                        self.logger.info(
                            "Loaded %s model v%d (%s)",
                            model_name, latest.version, type(loaded).__name__,
                        )
                    except Exception as exc:
                        self.logger.warning("Could not load %s model: %s", model_name, exc)
                else:
                    self.logger.info("No %s model in DB yet", model_name)
        finally:
            db.close()
        return swing_loaded or intraday_loaded


# Module-level singleton (lazy — no connections at import time)
portfolio_manager = PortfolioManager()
