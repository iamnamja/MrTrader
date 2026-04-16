"""
Trader Agent — execution engine for MrTrader.

Flow:
  1. Listen to Redis queue `trader_approved_trades` (from Risk Manager)
  2. Every CHECK_INTERVAL seconds, fetch daily bars and run generate_signal()
  3. On BUY signal: size position with size_position(), place market order, record trade
  4. Every tick, check open positions with check_exit() for stop/target/trail
  5. On exit signal: place market order, close trade, log P&L

Strategy is defined in app/strategy/signals.py — the single source of truth
shared with the backtesting engine.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from app.agents.base import BaseAgent
from app.database.models import Order, Trade
from app.database.session import get_session
from app.strategy.signals import generate_signal, check_exit
from app.strategy.position_sizer import size_position
from app.agents.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)

APPROVED_TRADES_QUEUE = "trader_approved_trades"
CHECK_INTERVAL = 300      # seconds between full scan cycles
MIN_BARS = 220            # minimum daily bars required for EMA(200) + buffer


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
        #   entry_price, stop_price, target_price, highest_price, bars_held, trade_id

    # ─── Main Loop ────────────────────────────────────────────────────────────

    async def run(self):
        """Continuously consume approved trades and monitor market conditions."""
        self.logger.info("Trader Agent started")
        self.status = "running"

        while self.status == "running":
            try:
                # Drain all pending approved proposals (non-blocking)
                while True:
                    proposal = self.get_message(APPROVED_TRADES_QUEUE, timeout=1)
                    if proposal is None:
                        break
                    symbol = proposal.get("symbol")
                    if symbol:
                        self.approved_symbols[symbol] = proposal
                        self.logger.info("Queued approved symbol: %s", symbol)

                # Check VIX / market volatility (cached, won't hammer yfinance)
                circuit_breaker.check_market_volatility()

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

    # ─── Scan Cycle ───────────────────────────────────────────────────────────

    async def _scan_cycle(self):
        """One pass: check entries for pending symbols, exits for active positions."""
        from app.integrations import get_alpaca_client
        alpaca = get_alpaca_client()

        # Check entry signals for approved-but-not-yet-entered symbols
        for symbol, proposal in list(self.approved_symbols.items()):
            if symbol in self.active_positions:
                continue
            try:
                await self._check_entry(symbol, proposal, alpaca)
            except Exception as e:
                self.logger.error("Entry check failed for %s: %s", symbol, e)

        # Check exit signals for active positions
        for symbol in list(self.active_positions.keys()):
            try:
                await self._check_exit(symbol, alpaca)
            except Exception as e:
                self.logger.error("Exit check failed for %s: %s", symbol, e)

    # ─── Entry ────────────────────────────────────────────────────────────────

    async def _check_entry(self, symbol: str, proposal: Dict[str, Any], alpaca):
        """Fetch daily bars, run generate_signal(), enter on BUY."""
        bars = alpaca.get_bars(symbol, timeframe="1Day", limit=MIN_BARS)
        if bars is None or bars.empty or len(bars) < MIN_BARS:
            self.logger.debug(
                "%s: only %d daily bars available (need %d)",
                symbol, len(bars) if bars is not None else 0, MIN_BARS,
            )
            return

        result = generate_signal(symbol, bars)
        if not result.is_buy:
            return

        # Size the position
        account = alpaca.get_account()
        equity = float(account.get("equity", 0)) if account else 0
        cash = float(account.get("cash", 0)) if account else 0

        shares = size_position(
            account_equity=equity,
            available_cash=cash,
            entry_price=result.entry_price,
            stop_price=result.stop_price,
        )
        if shares <= 0:
            self.logger.warning("%s: position sizer returned 0 shares — skipping", symbol)
            return

        await self._execute_entry(symbol, shares, result, alpaca)

    async def _execute_entry(self, symbol: str, shares: int, result, alpaca):
        """Place market order and record the trade."""
        order = alpaca.place_market_order(symbol, shares, "buy")
        if not order:
            self.logger.error("Market order failed for %s", symbol)
            return

        filled_price = alpaca.get_latest_price(symbol) or result.entry_price

        db = get_session()
        try:
            trade = Trade(
                symbol=symbol,
                direction="BUY",
                entry_price=filled_price,
                quantity=shares,
                status="ACTIVE",
                signal_type=result.signal_type,
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
                order_id=order.get("order_id"),
                status="FILLED",
                filled_price=filled_price,
                filled_qty=shares,
            )
            db.add(db_order)
            db.commit()

            self.active_positions[symbol] = {
                "entry_price":   filled_price,
                "stop_price":    result.stop_price,
                "target_price":  result.target_price,
                "highest_price": filled_price,
                "bars_held":     0,
                "trade_id":      trade.id,
            }
            self.approved_symbols.pop(symbol, None)

            self.logger.info(
                "ENTERED %s x%d @ $%.2f  signal=%s  stop=%.2f  target=%.2f",
                symbol, shares, filled_price, result.signal_type,
                result.stop_price, result.target_price,
            )
            await self.log_decision(
                "TRADE_ENTERED",
                trade_id=trade.id,
                reasoning={
                    **result.reasoning,
                    "shares": shares,
                    "filled_price": filled_price,
                },
            )
        except Exception as e:
            db.rollback()
            self.logger.error("Failed to record entry for %s: %s", symbol, e)
        finally:
            db.close()

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

        should_exit, reason, new_stop = check_exit(
            symbol=symbol,
            current_price=current_price,
            entry_price=pos["entry_price"],
            stop_price=pos["stop_price"],
            target_price=pos["target_price"],
            highest_price=highest,
            bars_held=pos["bars_held"],
        )
        pos["stop_price"] = new_stop  # keep trailing stop current

        if should_exit:
            await self._execute_exit(symbol, current_price, reason, alpaca)

    async def _execute_exit(self, symbol: str, current_price: float, reason: str, alpaca):
        """Place market order for exit and close the DB trade record."""
        pos = self.active_positions[symbol]
        position = alpaca.get_position(symbol)
        if not position:
            self.logger.warning("Cannot exit %s — Alpaca position not found", symbol)
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
                    trade.closed_at = datetime.utcnow()
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

            self.active_positions.pop(symbol, None)

            # Update circuit breaker with win/loss
            circuit_breaker.record_trade_result(won=(pnl > 0))

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


# Module-level singleton
trader = Trader()
