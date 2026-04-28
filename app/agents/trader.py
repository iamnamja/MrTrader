"""
Trader Agent — execution engine for MrTrader.

Flow:
  1. Listen to Redis queue `trader_approved_trades` (from Risk Manager)
  2. Listen to Redis queue `trader_exit_requests` (from Portfolio Manager)
  3. Every CHECK_INTERVAL seconds, fetch daily bars and run generate_signal()
  4. On BUY signal: size position with size_position(), place market order, record trade
  5. Every tick, check open positions with check_exit() for stop/target/trail
  6. On exit signal: place market order, close trade, log P&L

Strategy is defined in app/strategy/signals.py — the single source of truth
shared with the backtesting engine.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

from app.agents.base import BaseAgent
from app.agents.circuit_breaker import circuit_breaker
from app.database.models import Order, Trade
from app.database.session import get_session
from app.strategy.position_sizer import size_position
from app.strategy.signals import check_exit, generate_signal

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

APPROVED_TRADES_QUEUE = "trader_approved_trades"
EXIT_REQUESTS_QUEUE = "trader_exit_requests"    # PM → Trader: EXIT/HOLD/EXTEND_TARGET
REEVAL_REQUESTS_QUEUE = "pm_reeval_requests"    # Trader → PM: request re-evaluation
CHECK_INTERVAL = 300      # seconds between full scan cycles
MIN_BARS = 220            # minimum daily bars required for EMA(200) + buffer
INTRADAY_FORCE_CLOSE_HOUR = 15
INTRADAY_FORCE_CLOSE_MINUTE = 45  # 3:45 PM ET force-flat all intraday positions


class Trader(BaseAgent):
    """
    Listens for Risk Manager-approved symbols, fetches daily bars,
    runs the validated strategy signals, executes entries and exits.
    """

    def __init__(self):
        super().__init__("trader")
        self.approved_symbols: Dict[str, Dict[str, Any]] = {}   # symbol → proposal
        self.active_positions: Dict[str, Dict[str, Any]] = {}   # symbol → position state
        # position state keys:
        #   entry_price, stop_price, target_price, highest_price, bars_held, trade_id,
        #   trade_type ("swing" | "intraday")
        self._force_closed_today: bool = False
        self._last_date: str = ""
        self._last_regime: str = ""   # track regime changes for stop tightening
        # Pending limit orders for swing entries (symbol → order metadata)
        self._pending_limit_orders: Dict[str, Dict[str, Any]] = {}

    # ─── Main Loop ────────────────────────────────────────────────────────────

    async def _reconcile_positions(self, alpaca) -> None:
        """
        On startup, reconcile Alpaca open positions with DB ACTIVE trades.
        Any position held by Alpaca that lacks an ACTIVE DB trade gets a synthetic
        record created so stop/target/signal are visible in the dashboard.
        Also repopulates active_positions in-memory so exit logic works correctly.
        """
        try:
            raw_positions = await asyncio.to_thread(alpaca.get_positions)
        except Exception as exc:
            self.logger.warning("Reconciliation: could not fetch Alpaca positions: %s", exc)
            return

        if not raw_positions:
            return

        db = get_session()
        try:
            db_active = {t.symbol: t for t in db.query(Trade).filter(Trade.status == "ACTIVE").all()}
        except Exception as exc:
            self.logger.warning("Reconciliation: DB query failed: %s", exc)
            db.close()
            return

        for pos in raw_positions:
            symbol = pos.get("symbol")
            if not symbol:
                continue

            qty = int(pos.get("quantity") or pos.get("qty", 0))
            avg = float(pos.get("avg_price") or pos.get("avg_entry_price", 0))

            if symbol in db_active:
                # DB record exists — reload in-memory state from DB
                t = db_active[symbol]
                if symbol not in self.active_positions:
                    self.active_positions[symbol] = {
                        "entry_price":   t.entry_price,
                        "stop_price":    t.stop_price or avg * 0.98,
                        "target_price":  t.target_price or avg * 1.06,
                        "highest_price": avg,
                        "atr":           0.0,
                        "bars_held":     t.bars_held or 0,
                        "trade_id":      t.id,
                        "trade_type":    getattr(t, "trade_type", None) or "swing",
                        "entry_date":    t.created_at.date() if t.created_at else datetime.now(ET).date(),
                    }
                    self.logger.info("Reconciled %s from DB trade id=%d", symbol, t.id)
                continue

            # No DB record — create a synthetic ACTIVE trade using generate_signal for stop/target
            if symbol in self.active_positions:
                continue  # already loaded

            self.logger.warning("Reconciliation: %s in Alpaca but no DB trade — creating synthetic record", symbol)
            try:
                from app.strategy.signals import generate_signal
                from app.database.models import TradeProposal

                # Look up trade_type from the most recent persisted proposal for this symbol
                rec_proposal = (
                    db.query(TradeProposal)
                    .filter(TradeProposal.symbol == symbol, TradeProposal.status == "APPROVED")
                    .order_by(TradeProposal.proposed_at.desc())
                    .first()
                )
                trade_type_rec = rec_proposal.trade_type if rec_proposal else None
                if not trade_type_rec:
                    # Fall back: intraday if market is open (a restart during market hours
                    # is far more likely to be an intraday session), else swing
                    now_et = datetime.now(ET)
                    trade_type_rec = "intraday" if (9 <= now_et.hour < 16) else "swing"
                    self.logger.warning(
                        "Reconciliation: no proposal record for %s — defaulting trade_type=%s",
                        symbol, trade_type_rec,
                    )

                bars = alpaca.get_bars(symbol, timeframe="1Day", limit=MIN_BARS)
                if bars is not None and not bars.empty and len(bars) >= MIN_BARS:
                    result = generate_signal(symbol, bars, ml_score=0.6, check_regime=False, check_earnings=False)
                    stop = result.stop_price if result.stop_price and result.stop_price > 0 else round(avg * 0.98, 2)
                    target = result.target_price if result.target_price and result.target_price > 0 else round(avg * 1.06, 2)
                    # NONE means no rule-based pattern fired, but entry was ML-driven
                    signal = "ML_RANK" if result.signal_type in ("NONE", None) else result.signal_type
                    atr = result.atr
                else:
                    stop = round(avg * 0.98, 2)
                    target = round(avg * 1.06, 2)
                    signal = "ML_RANK"
                    atr = 0.0

                trade = Trade(
                    symbol=symbol,
                    direction="BUY",
                    entry_price=avg,
                    quantity=qty,
                    status="ACTIVE",
                    signal_type=signal,
                    trade_type=trade_type_rec,
                    stop_price=stop,
                    target_price=target,
                    highest_price=avg,
                    bars_held=0,
                )
                db.add(trade)
                db.commit()

                self.active_positions[symbol] = {
                    "entry_price":   avg,
                    "stop_price":    stop,
                    "target_price":  target,
                    "highest_price": avg,
                    "atr":           atr,
                    "bars_held":     0,
                    "trade_id":      trade.id,
                    "trade_type":    "swing",
                    "entry_date":    datetime.now(ET).date(),
                }
                self.logger.info(
                    "Reconciled %s: created synthetic Trade id=%d stop=%.2f target=%.2f",
                    symbol, trade.id, stop, target,
                )
            except Exception as exc:
                db.rollback()
                self.logger.error("Reconciliation failed for %s: %s", symbol, exc)

        db.close()

    async def run(self):
        """Continuously consume approved trades and monitor market conditions."""
        self.logger.info("Trader Agent started")
        self.status = "running"

        # Reconcile Alpaca positions with DB on startup (handles restarts mid-session)
        try:
            from app.integrations import get_alpaca_client
            await self._reconcile_positions(get_alpaca_client())
        except Exception as exc:
            self.logger.warning("Startup reconciliation failed: %s", exc)

        while self.status == "running":
            try:
                now = datetime.now(ET)
                today = now.strftime("%Y-%m-%d")
                if today != self._last_date:
                    self._force_closed_today = False
                    self._last_date = today

                # Drain all pending approved proposals (non-blocking)
                while True:
                    proposal = await asyncio.to_thread(
                        self.get_message, APPROVED_TRADES_QUEUE, 1
                    )
                    if proposal is None:
                        break
                    symbol = proposal.get("symbol")
                    if symbol:
                        self.approved_symbols[symbol] = proposal
                        self.logger.info("Queued approved symbol: %s", symbol)

                # Drain PM exit/hold/extend requests (non-blocking)
                await self._process_exit_requests()

                # 3:45 PM ET: force-close all intraday positions
                if (
                    now.weekday() < 5
                    and now.hour == INTRADAY_FORCE_CLOSE_HOUR
                    and now.minute >= INTRADAY_FORCE_CLOSE_MINUTE
                    and not self._force_closed_today
                ):
                    await self._force_close_intraday()
                    self._force_closed_today = True

                # Check VIX / market volatility in thread pool (yfinance is sync)
                await asyncio.to_thread(circuit_breaker.check_market_volatility)

                if circuit_breaker.is_open:
                    self.logger.warning(
                        "Circuit breaker OPEN (%s) — skipping scan",
                        circuit_breaker._open_reason,
                    )
                else:
                    await self._scan_cycle()
                await asyncio.sleep(CHECK_INTERVAL)

            except asyncio.CancelledError:
                self.logger.info("Trader Agent cancelled — shutting down")
                self.status = "stopped"
                break
            except Exception as e:
                self.logger.error("Unexpected error in trader loop: %s", e, exc_info=True)
                await self.log_decision("TRADER_ERROR", reasoning={"error": str(e)})
                await asyncio.sleep(10)

    # ─── PM Communication ────────────────────────────────────────────────────

    async def _process_exit_requests(self) -> None:
        """
        Drain the trader_exit_requests queue (PM → Trader).
        Each message has: symbol, action ("EXIT"|"HOLD"|"EXTEND_TARGET"),
        reason, and optionally extend_atr (for EXTEND_TARGET).
        """
        from app.integrations import get_alpaca_client
        while True:
            msg = await asyncio.to_thread(self.get_message, EXIT_REQUESTS_QUEUE, 1)
            if msg is None:
                break
            symbol = msg.get("symbol")
            action = msg.get("action", "HOLD")
            reason = msg.get("reason", "pm_request")

            if symbol not in self.active_positions:
                # PM may send requests for positions already closed — ignore
                continue

            if action == "EXIT":
                self.logger.info("PM exit request for %s — reason: %s", symbol, reason)
                try:
                    alpaca = get_alpaca_client()
                    price = alpaca.get_latest_price(symbol) or self.active_positions[symbol]["entry_price"]
                    await self._execute_exit(symbol, price, f"PM_{reason.upper()}", alpaca)
                except Exception as exc:
                    self.logger.error("PM-requested exit failed for %s: %s", symbol, exc)

            elif action == "EXTEND_TARGET":
                extend_atr = float(msg.get("extend_atr", 0.0))
                if extend_atr > 0 and symbol in self.active_positions:
                    pos = self.active_positions[symbol]
                    old_target = pos["target_price"]
                    pos["target_price"] = round(old_target + extend_atr, 4)
                    # Persist to DB so it survives a restart
                    db = get_session()
                    try:
                        trade = db.query(Trade).filter_by(id=pos["trade_id"]).first()
                        if trade:
                            trade.target_price = pos["target_price"]
                            db.commit()
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()
                    self.logger.info(
                        "%s: target extended by %.4f ATR → $%.2f (was $%.2f)",
                        symbol, extend_atr, pos["target_price"], old_target,
                    )

            # HOLD → no action needed, PM is confirming to stay in position

    async def _send_reeval_request(self, symbol: str, reason: str, current_price: float = 0.0) -> None:
        """Ask PM to re-evaluate a position. PM will respond via trader_exit_requests."""
        pos = self.active_positions.get(symbol, {})
        entry = pos.get("entry_price") or 0.0
        pnl_pct = round((current_price - entry) / entry, 4) if entry > 0 and current_price > 0 else 0.0
        self.send_message(REEVAL_REQUESTS_QUEUE, {
            "symbol": symbol,
            "reason": reason,
            "entry_price": entry,
            "current_price": current_price,
            "current_pnl_pct": pnl_pct,
            "bars_held": pos.get("bars_held", 0),
            "trade_type": pos.get("trade_type", "swing"),
        })
        self.logger.info("Sent reeval request to PM: %s (%s)", symbol, reason)

    # ─── Scan Cycle ───────────────────────────────────────────────────────────

    async def _scan_cycle(self):
        """One pass: check entries for pending symbols, exits for active positions."""
        from app.integrations import get_alpaca_client
        alpaca = get_alpaca_client()

        # Check for regime shift → tighten stops on all open swing positions
        await self._apply_regime_stop_tightening(alpaca)

        # Check entry signals for approved-but-not-yet-entered symbols
        for symbol, proposal in list(self.approved_symbols.items()):
            if symbol in self.active_positions:
                continue
            try:
                await self._check_entry(symbol, proposal, alpaca)
            except Exception as e:
                self.logger.error("Entry check failed for %s: %s", symbol, e)

        # Poll pending limit orders (check fills, cancel unfilled at EOD)
        await self._poll_pending_limit_orders(alpaca)

        # Check exit signals for active positions
        for symbol in list(self.active_positions.keys()):
            try:
                await self._check_exit(symbol, alpaca)
            except Exception as e:
                self.logger.error("Exit check failed for %s: %s", symbol, e)

    # ─── Entry ────────────────────────────────────────────────────────────────

    async def _check_entry(self, symbol: str, proposal: Dict[str, Any], alpaca):
        """Fetch daily bars, compute entry prices via generate_signal(), enter if ML score passes.

        generate_signal() is used only for ATR-based stop/target prices and current price.
        The entry gate is the ML confidence score (>= ML_SCORE_THRESHOLD), not the
        rule-based is_buy flag — the walk-forward validation assumes ML score is sufficient.
        """
        from app.strategy.signals import ML_SCORE_THRESHOLD
        trade_type = proposal.get("trade_type", "swing")
        if circuit_breaker.is_strategy_paused(trade_type):
            self.logger.debug(
                "%s: strategy '%s' is paused — skipping entry", symbol, trade_type
            )
            return

        bars = alpaca.get_bars(symbol, timeframe="1Day", limit=MIN_BARS)
        if bars is None or bars.empty or len(bars) < MIN_BARS:
            self.logger.debug(
                "%s: only %d daily bars available (need %d)",
                symbol, len(bars) if bars is not None else 0, MIN_BARS,
            )
            return

        ml_score = proposal.get("confidence")
        if ml_score is None or ml_score < ML_SCORE_THRESHOLD:
            self.logger.debug(
                "%s: ML score %.3f below threshold %.2f — skipping",
                symbol, ml_score or 0.0, ML_SCORE_THRESHOLD,
            )
            return

        # Use generate_signal for ATR-based stop/target prices only (not as entry gate)
        result = generate_signal(symbol, bars, ml_score=ml_score, check_regime=False, check_earnings=True)

        # Size the position
        account = alpaca.get_account()
        # Use portfolio_value (not buying_power) to avoid margin inflation.
        # buying_power on a $100k paper account = $200k (2× margin) — that would
        # produce 2× more shares than intended.
        equity = float(account.get("equity", 0)) if account else 0
        cash = float(account.get("cash", 0)) if account else 0

        # In live mode, cap sizing to the capital ramp stage to limit real-money risk.
        # In paper mode, use actual account equity (matches walk-forward assumptions).
        from app.config import settings
        if settings.trading_mode == "live":
            from app.live_trading.capital_manager import capital_manager
            stage_cap = capital_manager.get_current_capital()
            cash = min(cash, stage_cap)
            equity = min(equity, stage_cap)
        # Also cap cash to actual cash (not buying_power) to avoid margin inflation
        cash = min(cash, float(account.get("cash", cash)))

        shares = size_position(
            account_equity=equity,
            available_cash=cash,
            entry_price=result.entry_price,
            stop_price=result.stop_price,
            ml_score=ml_score or 0.0,
        )
        if shares <= 0:
            self.logger.warning("%s: position sizer returned 0 shares — skipping", symbol)
            return

        await self._execute_entry(symbol, shares, result, alpaca)

    async def _execute_entry(self, symbol: str, shares: int, result, alpaca):
        """
        Place entry order:
        - Swing: limit order at 0.3% below ask (cancel if unfilled by EOD)
        - Intraday: market order for immediacy
        Records intended_price for slippage tracking.
        """
        proposal = self.approved_symbols.get(symbol, {})
        trade_type = proposal.get("trade_type", "swing")
        intended_price = result.entry_price

        if trade_type == "swing":
            # Use limit order 0.3% below ask for better execution
            limit_offset = 0.003
            try:
                from app.database.agent_config import get_agent_config
                _db = get_session()
                try:
                    limit_offset = float(get_agent_config(_db, "strategy.limit_order_offset_pct") or 0.003)
                finally:
                    _db.close()
            except Exception:
                pass

            quote = alpaca.get_quote(symbol)
            if quote and quote["ask"] > 0:
                limit_price = round(quote["ask"] * (1 - limit_offset), 2)
                intended_price = quote["ask"]
            else:
                limit_price = round(intended_price * (1 - limit_offset), 2)

            order = alpaca.place_limit_order(symbol, shares, "buy", limit_price)
            if not order:
                self.logger.error("Limit entry order failed for %s", symbol)
                return

            order_id = order.get("order_id")
            self._pending_limit_orders[symbol] = {
                "order_id": order_id,
                "shares": shares,
                "intended_price": intended_price,
                "limit_price": limit_price,
                "result": result,
                "proposal": proposal,
                "queued_at": datetime.now(ET),
            }
            self.logger.info(
                "LIMIT ORDER placed %s x%d @ $%.4f (ask=%.4f offset=%.1f%%) — awaiting fill",
                symbol, shares, limit_price, intended_price, limit_offset * 100,
            )
            return  # position is created once fill confirmed in _poll_pending_limit_orders

        # Intraday: market order
        order = alpaca.place_market_order(symbol, shares, "buy")
        if not order:
            self.logger.error("Market order failed for %s", symbol)
            return

        filled_price = alpaca.get_latest_price(symbol) or intended_price
        slippage_bps = round((filled_price - intended_price) / intended_price * 10000, 2) if intended_price > 0 else 0.0

        await self._record_entry(
            symbol, shares, filled_price, intended_price, slippage_bps,
            order.get("order_id"), result, proposal,
        )

    async def _record_entry(
        self,
        symbol: str,
        shares: int,
        filled_price: float,
        intended_price: float,
        slippage_bps: float,
        order_id: str,
        result,
        proposal: Dict[str, Any],
    ):
        """Persist a confirmed entry to DB and update active_positions."""
        trade_type = proposal.get("trade_type", "swing")
        db = get_session()
        try:
            # NONE means no rule-based pattern fired but entry was ML-driven — label it clearly
            signal_type = "ML_RANK" if result.signal_type in ("NONE", None) else result.signal_type
            trade = Trade(
                symbol=symbol,
                direction="BUY",
                entry_price=filled_price,
                quantity=shares,
                status="ACTIVE",
                signal_type=signal_type,
                trade_type=trade_type,
                stop_price=result.stop_price,
                target_price=result.target_price,
                highest_price=filled_price,
                bars_held=0,
            )
            db.add(trade)
            db.flush()

            db_order = Order(
                trade_id=trade.id,
                order_type="ENTRY",
                order_id=order_id,
                status="FILLED",
                filled_price=filled_price,
                filled_qty=shares,
                intended_price=intended_price,
                slippage_bps=slippage_bps,
            )
            db.add(db_order)

            # Link back to the persisted TradeProposal so the audit trail is complete
            proposal_id = proposal.get("_proposal_id")
            if proposal_id:
                from app.database.models import TradeProposal
                tp = db.query(TradeProposal).filter(TradeProposal.id == proposal_id).first()
                if tp:
                    tp.trade_id = trade.id

            db.commit()

            self.active_positions[symbol] = {
                "entry_price":   filled_price,
                "stop_price":    result.stop_price,
                "target_price":  result.target_price,
                "highest_price": filled_price,
                "atr":           result.atr,
                "bars_held":     0,
                "trade_id":      trade.id,
                "trade_type":    trade_type,
                "entry_date":    datetime.now(ET).date(),
            }
            self.approved_symbols.pop(symbol, None)

            self.logger.info(
                "ENTERED %s x%d @ $%.2f (intended=$%.2f slip=%.1fbps) signal=%s stop=%.2f target=%.2f",
                symbol, shares, filled_price, intended_price, slippage_bps,
                result.signal_type, result.stop_price, result.target_price,
            )
            await self.log_decision(
                "TRADE_ENTERED",
                trade_id=trade.id,
                reasoning={
                    **result.reasoning,
                    "shares": shares,
                    "filled_price": filled_price,
                    "intended_price": intended_price,
                    "slippage_bps": slippage_bps,
                },
            )
        except Exception as e:
            db.rollback()
            self.logger.error("Failed to record entry for %s: %s", symbol, e)
        finally:
            db.close()

    async def _poll_pending_limit_orders(self, alpaca) -> None:
        """
        Check fill status of pending swing limit orders.
        - Filled → record entry, move to active_positions
        - EOD or market closed → cancel unfilled orders
        """
        if not self._pending_limit_orders:
            return

        now = datetime.now(ET)
        cancel_unfilled = now.hour >= 15 and now.minute >= 45  # 3:45 PM ET cutoff

        for symbol in list(self._pending_limit_orders.keys()):
            pending = self._pending_limit_orders[symbol]
            order_id = pending["order_id"]
            try:
                status = alpaca.get_order_status(order_id)
                if status is None:
                    continue

                order_status = str(status.get("status", "")).lower()
                filled_qty = int(status.get("filled_qty") or 0)
                filled_price = status.get("filled_avg_price")

                if order_status in ("filled", "partially_filled") and filled_qty > 0 and filled_price:
                    filled_price = float(filled_price)
                    intended = pending["intended_price"]
                    slippage_bps = round((filled_price - intended) / intended * 10000, 2) if intended > 0 else 0.0
                    del self._pending_limit_orders[symbol]
                    await self._record_entry(
                        symbol=symbol,
                        shares=filled_qty,
                        filled_price=filled_price,
                        intended_price=intended,
                        slippage_bps=slippage_bps,
                        order_id=order_id,
                        result=pending["result"],
                        proposal=pending["proposal"],
                    )
                    self.logger.info(
                        "LIMIT FILLED %s x%d @ $%.4f (slip=%.1fbps)",
                        symbol, filled_qty, filled_price, slippage_bps,
                    )

                elif order_status in ("canceled", "expired", "rejected"):
                    del self._pending_limit_orders[symbol]
                    self.logger.info("Limit order %s for %s cancelled/expired — removing", order_id, symbol)

                elif cancel_unfilled:
                    alpaca.cancel_order(order_id)
                    del self._pending_limit_orders[symbol]
                    self.logger.info(
                        "EOD: cancelled unfilled limit order for %s (order=%s)", symbol, order_id
                    )

            except Exception as exc:
                self.logger.error("Error polling limit order for %s: %s", symbol, exc)

    # ─── Regime-Aware Stop Tightening ────────────────────────────────────────

    async def _apply_regime_stop_tightening(self, alpaca) -> None:
        """
        When regime shifts to HIGH, tighten open swing stops to 1×ATR from
        current price so we protect capital during sudden volatility spikes.
        Only fires on the transition (LOW/MEDIUM → HIGH), not every cycle.
        """
        try:
            from app.strategy.regime_detector import regime_detector
            regime = regime_detector.get_regime()
        except Exception:
            return

        if regime == self._last_regime:
            return  # no change — nothing to do

        prev = self._last_regime
        self._last_regime = regime

        if regime != "HIGH":
            self.logger.info("Regime changed: %s → %s", prev, regime)
            return

        # Regime just shifted to HIGH — tighten all open swing stops
        swing_positions = [
            (sym, pos) for sym, pos in self.active_positions.items()
            if pos.get("trade_type", "swing") == "swing"
        ]
        if not swing_positions:
            self.logger.info("Regime → HIGH: no open swing positions to tighten")
            return

        self.logger.warning(
            "Regime shifted %s → HIGH — tightening stops on %d swing position(s)",
            prev, len(swing_positions),
        )

        for symbol, pos in swing_positions:
            try:
                current_price = alpaca.get_latest_price(symbol)
                if current_price is None:
                    continue
                bars = alpaca.get_bars(symbol, timeframe="1Day", limit=20)
                if bars is None or bars.empty:
                    continue
                from app.strategy.signals import _calc_atr
                atr_val = _calc_atr(bars["high"], bars["low"], bars["close"], 14)
                if not atr_val:
                    continue
                tight_stop = round(current_price - 1.0 * atr_val, 4)
                if tight_stop > pos["stop_price"]:
                    old_stop = pos["stop_price"]
                    pos["stop_price"] = tight_stop
                    self.logger.warning(
                        "%s: stop tightened (regime=HIGH) $%.2f → $%.2f",
                        symbol, old_stop, tight_stop,
                    )
            except Exception as exc:
                self.logger.error("Stop tightening failed for %s: %s", symbol, exc)

        await self.log_decision(
            "REGIME_STOP_TIGHTENED",
            reasoning={
                "prev_regime": prev,
                "new_regime": regime,
                "symbols": [s for s, _ in swing_positions],
            },
        )

    # ─── Exit ─────────────────────────────────────────────────────────────────

    async def _check_exit(self, symbol: str, alpaca):
        """Fetch current price, run check_exit(), close position if triggered."""
        pos = self.active_positions[symbol]
        current_price = alpaca.get_latest_price(symbol)
        if current_price is None:
            return

        # Update highest price for trailing stop
        highest = max(pos["highest_price"], current_price)
        pos["highest_price"] = highest
        pos["bars_held"] += 1

        # Adverse-move warning: if down >3% from entry, log and tighten stop to breakeven
        pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        if pnl_pct <= -0.03 and not pos.get("_adverse_warned"):
            pos["_adverse_warned"] = True
            breakeven_stop = round(pos["entry_price"] * 0.995, 4)  # near-breakeven
            if breakeven_stop > pos["stop_price"]:
                pos["stop_price"] = breakeven_stop
            self.logger.warning(
                "%s: adverse move %.1f%% — stop moved to $%.2f",
                symbol, pnl_pct * 100, pos["stop_price"],
            )
            await self.log_decision(
                "ADVERSE_MOVE_WARNING",
                reasoning={
                    "symbol": symbol,
                    "pnl_pct": round(pnl_pct, 4),
                    "stop_price": pos["stop_price"],
                    "entry_price": pos["entry_price"],
                },
            )

        # Partial exit: when up 1×ATR, exit configurable % and move stop to breakeven
        if not pos.get("_partial_exited") and pos.get("atr", 0) > 0:
            atr_val = pos["atr"]
            profit_1r = pos["entry_price"] + atr_val
            if current_price >= profit_1r:
                await self._execute_partial_exit(symbol, current_price, atr_val, alpaca)

        # Technical weakness detection → request PM re-evaluation (swing only)
        # These don't trigger immediate exit — PM decides EXIT/HOLD/EXTEND
        if pos.get("trade_type", "swing") == "swing" and not pos.get("_reeval_sent"):
            try:
                bars = alpaca.get_bars(symbol, timeframe="1Day", limit=30)
                if bars is not None and len(bars) >= 15:
                    from app.strategy.signals import _calc_rsi
                    close = bars["close"]
                    rsi_s = _calc_rsi(close, 14)
                    rsi_now = float(rsi_s.iloc[-1])
                    rsi_prev = float(rsi_s.iloc[-2])
                    vol_now = float(bars["volume"].iloc[-1])
                    vol_avg = float(bars["volume"].iloc[-20:].mean())

                    weakness_reason = None
                    if rsi_now < rsi_prev - 10 and rsi_now < 40:
                        weakness_reason = "rsi_deteriorating"
                    elif vol_now < vol_avg * 0.4 and pnl_pct > 0:
                        weakness_reason = "volume_fading"
                    elif current_price < pos["stop_price"] * 1.005:
                        weakness_reason = "approaching_stop"

                    if weakness_reason:
                        pos["_reeval_sent"] = True
                        await self._send_reeval_request(symbol, weakness_reason, current_price)
            except Exception:
                pass  # reeval is best-effort; don't let it block exit logic

        max_hold = 20
        try:
            from app.database.agent_config import get_agent_config
            _db = get_session()
            try:
                max_hold = int(get_agent_config(_db, "strategy.max_hold_bars") or 20)
            finally:
                _db.close()
        except Exception:
            pass

        should_exit, reason, new_stop = check_exit(
            symbol=symbol,
            current_price=current_price,
            entry_price=pos["entry_price"],
            stop_price=pos["stop_price"],
            target_price=pos["target_price"],
            highest_price=highest,
            bars_held=pos["bars_held"],
            max_hold_bars=max_hold,
        )
        pos["stop_price"] = new_stop  # keep trailing stop current

        if should_exit:
            await self._execute_exit(symbol, current_price, reason, alpaca)

    async def _execute_partial_exit(self, symbol: str, current_price: float, atr_val: float, alpaca) -> None:
        """
        Exit a configurable fraction of the position at 1×ATR profit.
        Move remaining stop to breakeven. Fires at most once per position.
        """
        pos = self.active_positions[symbol]
        pos["_partial_exited"] = True  # guard — set before any await

        partial_pct = 0.50
        try:
            from app.database.session import get_session as _gs
            from app.database.agent_config import get_agent_config
            _db = _gs()
            try:
                partial_pct = float(get_agent_config(_db, "strategy.partial_exit_pct") or 0.50)
            finally:
                _db.close()
        except Exception:
            pass

        position = alpaca.get_position(symbol)
        if not position:
            return
        total_qty = int(position.get("qty", 0))
        if total_qty <= 0:
            return

        partial_qty = max(1, int(total_qty * partial_pct))
        if partial_qty >= total_qty:
            # Would exit everything — let normal exit logic handle it
            pos["_partial_exited"] = False
            return

        order = alpaca.place_market_order(symbol, partial_qty, "sell")
        if not order:
            self.logger.error("Partial exit order failed for %s", symbol)
            pos["_partial_exited"] = False
            return

        pnl = (current_price - pos["entry_price"]) * partial_qty

        # Move stop on remaining shares to near-breakeven
        breakeven_stop = round(pos["entry_price"] * 1.001, 4)
        pos["stop_price"] = max(pos["stop_price"], breakeven_stop)

        db = get_session()
        try:
            if pos.get("trade_id"):
                db_order = Order(
                    trade_id=pos["trade_id"],
                    order_type="PARTIAL_EXIT",
                    order_id=order.get("order_id"),
                    status="FILLED",
                    filled_price=current_price,
                    filled_qty=partial_qty,
                )
                db.add(db_order)
                db.commit()
        except Exception as exc:
            db.rollback()
            self.logger.error("Failed to record partial exit for %s: %s", symbol, exc)
        finally:
            db.close()

        self.logger.info(
            "PARTIAL EXIT %s: sold %d/%d shares @ $%.2f (%.0f%%) — stop → $%.2f | PnL=$%.2f",
            symbol, partial_qty, total_qty, current_price,
            partial_pct * 100, pos["stop_price"], pnl,
        )
        await self.log_decision(
            "PARTIAL_EXIT",
            trade_id=pos.get("trade_id"),
            reasoning={
                "symbol": symbol,
                "partial_qty": partial_qty,
                "total_qty": total_qty,
                "exit_price": current_price,
                "pnl": round(pnl, 2),
                "new_stop": pos["stop_price"],
            },
        )

    async def _execute_exit(self, symbol: str, current_price: float, reason: str, alpaca):
        """Place market order for exit and close the DB trade record."""
        pos = self.active_positions[symbol]
        position = alpaca.get_position(symbol)
        if not position:
            self.logger.warning("Cannot exit %s — Alpaca position not found; marking DB trade closed", symbol)
            # Still close the DB record so it doesn't stay ACTIVE forever
            trade_id = pos.get("trade_id")
            if trade_id:
                db = get_session()
                try:
                    trade = db.query(Trade).filter_by(id=trade_id).first()
                    if trade and trade.status == "ACTIVE":
                        # Try to get a real price; only fall back to entry_price if nothing available
                        actual_price = current_price or alpaca.get_latest_price(symbol) or pos["entry_price"]
                        trade.exit_price = actual_price
                        trade.pnl = (actual_price - pos["entry_price"]) * pos.get("quantity", trade.quantity or 0)
                        trade.status = "FORCE_CLOSED_NO_POSITION"
                        trade.closed_at = datetime.now(ET)
                        db.commit()
                except Exception as e:
                    db.rollback()
                    self.logger.error("Failed to close orphaned trade %s: %s", trade_id, e)
                finally:
                    db.close()
            self.active_positions.pop(symbol, None)
            return

        qty = int(position.get("qty", 0))
        if qty <= 0:
            self.active_positions.pop(symbol, None)
            return

        order = alpaca.place_market_order(symbol, qty, "sell")
        if not order:
            self.logger.error("Exit order failed for %s", symbol)
            return

        pnl = (current_price - pos["entry_price"]) * qty
        trade_id = pos.get("trade_id")

        db = get_session()
        try:
            if trade_id:
                trade = db.query(Trade).filter_by(id=trade_id).first()
                if trade:
                    trade.exit_price = current_price
                    trade.pnl = pnl
                    trade.status = "CLOSED"
                    trade.closed_at = datetime.now(ET)
                    trade.bars_held = pos["bars_held"]

            db_order = Order(
                trade_id=trade_id,
                order_type="EXIT",
                order_id=order.get("order_id"),
                status="FILLED",
                filled_price=current_price,
                filled_qty=qty,
            )
            db.add(db_order)
            db.commit()

            trade_type = self.active_positions.get(symbol, {}).get("trade_type", "swing")
            self.active_positions.pop(symbol, None)

            # Update circuit breaker with win/loss (strategy-level tracking)
            circuit_breaker.record_trade_result(won=(pnl > 0), strategy=trade_type)

            # Compliance: record PDT round-trip (intraday) and settlement / wash sale
            try:
                from app.agents.compliance import compliance_tracker
                entry_date = pos.get("entry_date")
                today = datetime.now(ET).date()
                # PDT: day trade = opened and closed same calendar day (intraday always qualifies)
                if trade_type == "intraday" or (
                    entry_date and entry_date == today
                ):
                    compliance_tracker.record_day_trade(symbol)
                # Settlement: record proceeds as unsettled (T+1)
                proceeds = current_price * qty
                compliance_tracker.record_sale_proceeds(proceeds)
                compliance_tracker.sweep_settled()
                # Wash sale: flag if closed at a loss
                if pnl < 0:
                    compliance_tracker.record_loss_close(symbol)
            except Exception as _ce:
                self.logger.debug("Compliance post-trade update failed: %s", _ce)

            # Phase 22: record signal quality telemetry
            try:
                from app.agents.performance_monitor import performance_monitor
                sig_type = "UNKNOWN"
                if trade_id:
                    _sig_db = get_session()
                    try:
                        _t = _sig_db.query(Trade).filter_by(id=trade_id).first()
                        sig_type = (_t.signal_type or "UNKNOWN") if _t else "UNKNOWN"
                    finally:
                        _sig_db.close()
                performance_monitor.record_trade_result(sig_type, pnl)
            except Exception:
                pass

            # Release intraday slot in risk manager
            if trade_type == "intraday":
                try:
                    from app.agents.risk_manager import risk_manager
                    risk_manager.on_intraday_position_closed()
                except Exception:
                    pass

            self.logger.info(
                "EXITED %s @ $%.2f | reason=%s | PnL=$%.2f",
                symbol, current_price, reason, pnl,
            )
            await self.log_decision(
                "TRADE_EXITED",
                trade_id=trade_id,
                reasoning={
                    "symbol": symbol,
                    "exit_price": current_price,
                    "pnl": pnl,
                    "reason": reason,
                    "bars_held": pos["bars_held"],
                },
            )
        except Exception as e:
            db.rollback()
            self.logger.error("Failed to record exit for %s: %s", symbol, e)
        finally:
            db.close()

    # ─── Intraday Force Close ─────────────────────────────────────────────────

    async def _force_close_intraday(self):
        """
        Force-close all active intraday positions at 3:45 PM ET.
        Prevents overnight exposure from intraday trades.
        """
        intraday_symbols = [
            sym for sym, pos in self.active_positions.items()
            if pos.get("trade_type") == "intraday"
        ]
        if not intraday_symbols:
            return

        self.logger.warning(
            "3:45 PM force-close: closing %d intraday position(s): %s",
            len(intraday_symbols), intraday_symbols,
        )

        from app.integrations import get_alpaca_client
        alpaca = get_alpaca_client()

        for symbol in intraday_symbols:
            try:
                await self._execute_exit(symbol, alpaca.get_latest_price(symbol) or 0,
                                         "FORCE_CLOSE_EOD", alpaca)
            except Exception as exc:
                self.logger.error("Force-close failed for %s: %s", symbol, exc)

        # Notify risk manager to reset intraday counter
        try:
            from app.agents.risk_manager import risk_manager
            risk_manager.reset_intraday_count()
        except Exception:
            pass

        await self.log_decision(
            "INTRADAY_FORCE_CLOSED",
            reasoning={"symbols": intraday_symbols, "count": len(intraday_symbols)},
        )


# Module-level singleton
trader = Trader()
