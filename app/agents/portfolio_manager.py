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
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

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
POSITION_RISK_PCT = 0.02       # base risk per trade (2% of strategy budget)
INTRADAY_SCAN_MINUTE = 45      # 09:45 ET for intraday scan
SWING_PREMARKET_HOUR = 8       # 08:00 ET: run swing model analysis pre-market
SWING_SEND_HOUR = 9            # 09:50 ET: send swing proposals after open volatility settles
SWING_SEND_MINUTE = 50

# ── Capital allocation ────────────────────────────────────────────────────────
SWING_BUDGET_PCT    = 0.70     # 70% of account reserved for swing trades
INTRADAY_BUDGET_PCT = 0.30     # 30% of account reserved for intraday trades
GROSS_EXPOSURE_CAP  = 0.80     # never deploy more than 80% of account at once

# Confidence-to-size multiplier: prob → scalar applied to base position size.
# Clipped to [0.5x, 2.0x] so a single high-confidence trade can't dominate.
def _confidence_scalar(prob: float) -> float:
    """Linear scale: prob=MIN_CONFIDENCE → 0.5x, prob=1.0 → 2.0x."""
    lo, hi = MIN_CONFIDENCE, 1.0
    return float(np.clip(0.5 + 1.5 * (prob - lo) / max(hi - lo, 1e-6), 0.5, 2.0))


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
        self._analyzed_today: bool = False       # 08:00 pre-market analysis done
        self._selected_today: bool = False       # 09:50 proposals sent
        self._selected_intraday_today: bool = False
        self._retrained_today: bool = False
        self._last_date: Optional[str] = None
        self._swing_proposals: List[Dict[str, Any]] = []  # cached from 08:00 analysis

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
                now = datetime.now(ET)
                today = now.strftime("%Y-%m-%d")

                # Reset daily flags at midnight
                if today != self._last_date:
                    self._analyzed_today = False
                    self._selected_today = False
                    self._selected_intraday_today = False
                    self._retrained_today = False
                    self._last_date = today
                    self._swing_proposals = []

                is_weekday = now.weekday() < 5

                # 08:00: pre-market swing model analysis (score + rank, don't send yet)
                if (
                    is_weekday
                    and now.hour == SWING_PREMARKET_HOUR
                    and now.minute == 0
                    and not self._analyzed_today
                ):
                    await self._analyze_swing_premarket()
                    self._analyzed_today = True

                # 09:50: send cached swing proposals after open volatility settles
                if (
                    is_weekday
                    and now.hour == SWING_SEND_HOUR
                    and now.minute == SWING_SEND_MINUTE
                    and not self._selected_today
                ):
                    await self._send_swing_proposals()
                    self._selected_today = True

                # 09:45: intraday model selection
                if (
                    is_weekday
                    and now.hour == MARKET_OPEN_HOUR
                    and now.minute == INTRADAY_SCAN_MINUTE
                    and not self._selected_intraday_today
                ):
                    try:
                        await self.select_intraday_instruments()
                    except Exception as _e:
                        self.logger.warning("Intraday scan skipped: %s", _e)
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

    def _fetch_swing_features(self) -> Dict[str, Dict[str, float]]:
        """Fetch bars + engineer features concurrently. Run in a thread via asyncio.to_thread."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch_one(symbol: str):
            try:
                bars = self._alpaca.get_bars(symbol, timeframe="1D", limit=300)
                if bars.empty:
                    return symbol, None
                feats = self.feature_engineer.engineer_features(symbol, bars, fetch_fundamentals=False)
                return symbol, feats
            except Exception as e:
                self.logger.debug("Skipping %s: %s", symbol, e)
                return symbol, None

        features_by_symbol: Dict[str, Dict[str, float]] = {}
        symbols = self._get_universe()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_one, s): s for s in symbols}
            for future in as_completed(futures):
                symbol, feats = future.result()
                if feats is not None:
                    features_by_symbol[symbol] = feats
        return features_by_symbol

    async def _analyze_swing_premarket(self):
        """
        08:00 ET: Score all universe stocks using yesterday's daily bars.
        Results cached in self._swing_proposals — not sent to Risk Manager yet.
        Sending is deferred to 09:50 so we enter after the open volatility spike.
        """
        self.logger.info("Pre-market swing analysis starting (08:00)...")

        if not self.model.is_trained:
            self.logger.warning("No trained model — pre-market analysis skipped")
            await self.log_decision("SELECTION_SKIPPED", reasoning={"reason": "model not trained"})
            return

        # Run blocking network calls in a thread so the event loop stays free
        features_by_symbol = await asyncio.to_thread(self._fetch_swing_features)
        self.logger.info("Swing feature fetch complete — %d symbols", len(features_by_symbol))

        if not features_by_symbol:
            self.logger.warning("Could not build features for any symbol — skipping")
            await self.log_decision("SELECTION_SKIPPED", reasoning={"reason": "no features computed"})
            return

        symbols = list(features_by_symbol.keys())
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
            self.logger.info("Model scored %d symbols — max prob=%.3f min prob=%.3f",
                             len(probabilities), max(probabilities), min(probabilities))
        except Exception as e:
            self.logger.error("Model prediction failed: %s", e)
            await self.log_decision("SELECTION_SKIPPED", reasoning={"reason": f"prediction error: {e}"})
            return

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

        target_upside = self._fetch_target_upside(symbols)
        boosted_probs = []
        for sym, prob in zip(symbols, probabilities):
            upside = target_upside.get(sym, 0.0)
            boost = min(0.05, max(0.0, upside * 0.15)) if upside > 0.10 else 0.0
            boosted_probs.append(float(prob) + boost)

        ranked = sorted(zip(symbols, boosted_probs), key=lambda x: x[1], reverse=True)
        selected = [(sym, prob) for sym, prob in ranked if prob >= min_conf][:top_n]

        self.logger.info(
            "Pre-market analysis complete — %d candidates: %s (proposals held until 09:50)",
            len(selected), [s for s, _ in selected],
        )

        self._swing_proposals = await self._build_proposals(selected)

        await self.log_decision(
            "SWING_PREMARKET_ANALYSIS",
            reasoning={
                "candidates": [{"symbol": s, "confidence": round(float(p), 4)} for s, p in selected],
                "total_evaluated": len(symbols),
                "send_time": "09:50 ET",
            },
        )

    async def _send_swing_proposals(self):
        """
        09:50 ET: Forward cached swing proposals to Risk Manager.
        Open volatility has settled; entries here get cleaner fills.
        Falls back to running a fresh analysis if 08:00 run was missed.
        """
        if not self._swing_proposals:
            self.logger.warning("No pre-market analysis cached — running analysis now")
            await self._analyze_swing_premarket()

        if not self._swing_proposals:
            self.logger.warning("Still no swing proposals after fallback analysis — skipping")
            return

        self.logger.info("Sending %d swing proposals to Risk Manager (09:50)...", len(self._swing_proposals))
        for proposal in self._swing_proposals:
            self.send_message(TRADE_PROPOSALS_QUEUE, proposal)
            self.logger.info(
                "Proposal sent: %s @ $%.2f (confidence=%.2f)",
                proposal["symbol"], proposal["entry_price"], proposal["confidence"],
            )

        await self.log_decision(
            "INSTRUMENTS_SELECTED",
            reasoning={
                "selected": [
                    {"symbol": p["symbol"], "confidence": round(float(p["confidence"]), 4)}
                    for p in self._swing_proposals
                ],
                "sent_at": "09:50 ET",
            },
        )
        self._swing_proposals = []

    async def select_instruments(self):
        """Public entry point for manual/forced selection — runs full analysis + send immediately."""
        await self._analyze_swing_premarket()
        await self._send_swing_proposals()

    # ─── Intraday Selection ───────────────────────────────────────────────────

    async def select_intraday_instruments(self):
        """Run intraday model on 5-min bars, send intraday proposals."""
        self.logger.info("Selecting intraday instruments (09:45 scan)...")

        if not self.intraday_model.is_trained:
            self.logger.warning("No intraday model available — skipping intraday scan")
            return

        def _fetch_intraday_features() -> Dict[str, Dict[str, float]]:
            from app.ml.intraday_features import compute_intraday_features, MIN_BARS
            result: Dict[str, Dict[str, float]] = {}
            for symbol in self._get_universe()[:50]:
                try:
                    bars = self._alpaca.get_bars(symbol, timeframe="5Min", limit=78)
                    if bars is None or bars.empty or len(bars) < MIN_BARS:
                        continue
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
                        result[symbol] = feats
                except Exception as exc:
                    self.logger.debug("Intraday feature skip %s: %s", symbol, exc)
            return result

        features_by_symbol = await asyncio.to_thread(_fetch_intraday_features)

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
            quantity = self._calculate_quantity(
                price, account_value, trade_type="intraday", confidence=float(confidence)
            )
            proposal: Dict[str, Any] = {
                "symbol": symbol,
                "direction": "BUY",
                "quantity": quantity,
                "entry_price": price,
                "confidence": float(confidence),
                "stop_loss": round(price * 0.995, 2),
                "profit_target": round(price * 1.01, 2),
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

            quantity = self._calculate_quantity(
                price, account_value, trade_type="swing", confidence=float(confidence)
            )
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

    def _get_deployed_by_type(self) -> Dict[str, float]:
        """
        Return {trade_type: deployed_dollars} for open positions.
        Uses Alpaca positions tagged with trade_type in their client_order_id.
        Falls back to 0 on any error so sizing degrades gracefully.
        """
        deployed = {"swing": 0.0, "intraday": 0.0, "total": 0.0}
        try:
            positions = self._alpaca.get_positions()  # list of position dicts
            for pos in positions:
                market_val = abs(float(pos.get("market_value") or 0))
                deployed["total"] += market_val
                # Trade type tagged in metadata; default to swing for legacy positions
                tt = pos.get("trade_type") or "swing"
                if tt in deployed:
                    deployed[tt] += market_val
        except Exception:
            pass
        return deployed

    def _calculate_quantity(
        self,
        price: float,
        account_value: float,
        trade_type: str = "swing",
        confidence: float = MIN_CONFIDENCE,
    ) -> int:
        """
        Size a position using three layered constraints:
          1. Strategy budget: swing uses 70%, intraday uses 30% of account
          2. Gross exposure cap: total deployed never exceeds 80% of account
          3. Confidence scalar: high-confidence signals get larger allocations

        Base position = budget_for_type × POSITION_RISK_PCT × confidence_scalar
        Then clamp to remaining headroom in both the strategy budget and gross cap.
        """
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

        # 1. Strategy budget for this trade type
        budget_pct = SWING_BUDGET_PCT if trade_type == "swing" else INTRADAY_BUDGET_PCT
        strategy_budget = account_value * budget_pct

        # 2. How much of each budget and gross cap is already deployed
        deployed = self._get_deployed_by_type()
        type_headroom  = max(0.0, strategy_budget - deployed.get(trade_type, 0.0))
        gross_headroom = max(0.0, account_value * GROSS_EXPOSURE_CAP - deployed["total"])
        available = min(type_headroom, gross_headroom)

        # 3. Base size = strategy_budget × risk_pct × confidence_scalar
        scalar = _confidence_scalar(confidence)
        base_dollars = strategy_budget * risk_pct * scalar

        # Clamp to available headroom
        position_dollars = min(base_dollars, available)

        qty = int(position_dollars / price)
        return max(qty, 1)

    # ─── Retraining ───────────────────────────────────────────────────────────

    async def _retrain(self):
        """Run training pipeline in a thread so it doesn't block the event loop."""
        self.logger.info("Starting scheduled model retraining...")
        try:
            loop = asyncio.get_event_loop()
            import functools
            version = await loop.run_in_executor(None, functools.partial(self.trainer.train_model, fetch_fundamentals=False))
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
                        from pathlib import Path
                        model_path = Path(latest.model_path)
                        model_dir = str(model_path.parent)
                        version = latest.version
                        wrapper = PortfolioSelectorModel(model_type="xgboost")
                        wrapper.load(model_dir, version, model_name=model_name)
                        loaded = wrapper
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
