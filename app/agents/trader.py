"""
Trader Agent — execution engine for MrTrader.

Flow:
  1. Listen to Redis queue `trader_approved_trades` (from Risk Manager)
  2. Every CHECK_INTERVAL seconds, scan approved trades for entry signals
  3. On entry signal (2+ confirmations): place market order, record trade
  4. Every tick, scan active positions for exit signals
  5. On exit signal: place limit order, close trade, log P&L
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.database.models import Order, Trade
from app.database.session import get_session
from app.indicators.technical import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    is_overbought,
    is_oversold,
    price_near_band,
)

logger = logging.getLogger(__name__)

APPROVED_TRADES_QUEUE = "trader_approved_trades"
CHECK_INTERVAL = 300          # seconds between market condition checks
TAKE_PROFIT_PCT = 0.02        # exit when up 2%
STOP_LOSS_PCT = 0.01          # exit when down 1%
MAX_HOLD_HOURS = 4            # time-based exit after 4 hours
MIN_CONFIRMATIONS = 2         # signals needed to enter


class Trader(BaseAgent):
    """
    Listens for Risk Manager-approved trade proposals, monitors market
    conditions via technical indicators, executes entries and exits.
    """

    def __init__(self):
        super().__init__("trader")
        self.approved_trades: Dict[str, Dict[str, Any]] = {}   # symbol → proposal
        self.active_positions: Dict[str, float] = {}            # symbol → entry_price
        self.active_trade_ids: Dict[str, int] = {}              # symbol → Trade.id
        self.entry_times: Dict[str, datetime] = {}              # symbol → when entered

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
                        self.approved_trades[symbol] = proposal
                        self.logger.info("Queued approved trade: %s", symbol)

                await self._check_market_conditions()
                await asyncio.sleep(CHECK_INTERVAL)

            except asyncio.CancelledError:
                self.logger.info("Trader Agent cancelled — shutting down")
                self.status = "stopped"
                break
            except Exception as e:
                self.logger.error("Unexpected error in trader loop: %s", e, exc_info=True)
                await self.log_decision("TRADER_ERROR", reasoning={"error": str(e)})
                await asyncio.sleep(10)

    # ─── Market Condition Check ───────────────────────────────────────────────

    async def _check_market_conditions(self):
        """Check entries for pending proposals and exits for active positions."""
        from app.integrations import get_alpaca_client
        alpaca = get_alpaca_client()

        # Check entry signals
        for symbol, proposal in list(self.approved_trades.items()):
            if symbol in self.active_positions:
                continue  # already in this position
            try:
                should_enter, signals = await self._check_entry_signal(symbol, proposal, alpaca)
                if should_enter:
                    await self._execute_entry(symbol, proposal, alpaca, signals)
            except Exception as e:
                self.logger.error("Entry check failed for %s: %s", symbol, e)

        # Check exit signals
        for symbol in list(self.active_positions.keys()):
            try:
                should_exit, reason = await self._check_exit_signal(symbol, alpaca)
                if should_exit:
                    await self._execute_exit(symbol, reason, alpaca)
            except Exception as e:
                self.logger.error("Exit check failed for %s: %s", symbol, e)

    # ─── Entry Signal Detection ───────────────────────────────────────────────

    async def _check_entry_signal(
        self,
        symbol: str,
        proposal: Dict[str, Any],
        alpaca,
    ) -> tuple[bool, List[str]]:
        """
        Return (should_enter, confirmed_signals).
        Requires MIN_CONFIRMATIONS signals across three strategies.
        """
        bars = alpaca.get_bars(symbol, timeframe="5Min", limit=60)
        if bars.empty or len(bars) < 26:
            return False, []

        prices = bars["close"].tolist()
        highs = bars["high"].tolist()
        lows = bars["low"].tolist()
        volumes = bars["volume"].tolist()
        current_price = prices[-1]
        direction = proposal.get("direction", "BUY").upper()

        signals: List[str] = []

        rsi = calculate_rsi(prices)
        ema_20 = calculate_ema(prices, period=20)
        macd_result = calculate_macd(prices)
        bb_result = calculate_bollinger_bands(prices)
        avg_volume = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
        current_volume = volumes[-1]

        if direction == "BUY":
            # Strategy 1 — Momentum: RSI oversold + price above EMA
            if rsi is not None and ema_20 is not None:
                if rsi < 30 and current_price > ema_20:
                    signals.append("momentum_rsi_ema")

            # Strategy 2 — MACD bullish crossover
            if macd_result is not None:
                macd_line, signal_line, _ = macd_result
                if macd_line > signal_line:
                    signals.append("macd_bullish")

            # Strategy 3 — Bollinger Band mean reversion
            if bb_result is not None:
                _, _, lower_band = bb_result
                if price_near_band(current_price, lower_band) and rsi is not None and rsi < 40:
                    signals.append("bollinger_lower")

            # Strategy 4 — Breakout: 20-bar high + above-average volume
            if len(prices) >= 20:
                high_20 = max(highs[-20:])
                if current_price >= high_20 and current_volume > avg_volume:
                    signals.append("breakout_high")

        else:  # SELL / SHORT
            # Strategy 1 — Momentum: RSI overbought + price below EMA
            if rsi is not None and ema_20 is not None:
                if rsi > 70 and current_price < ema_20:
                    signals.append("momentum_rsi_ema")

            # Strategy 2 — MACD bearish crossover
            if macd_result is not None:
                macd_line, signal_line, _ = macd_result
                if macd_line < signal_line:
                    signals.append("macd_bearish")

            # Strategy 3 — Bollinger Band mean reversion
            if bb_result is not None:
                upper_band, _, _ = bb_result
                if price_near_band(current_price, upper_band) and rsi is not None and rsi > 60:
                    signals.append("bollinger_upper")

            # Strategy 4 — Breakout: 20-bar low + above-average volume
            if len(prices) >= 20:
                low_20 = min(lows[-20:])
                if current_price <= low_20 and current_volume > avg_volume:
                    signals.append("breakout_low")

        confirmed = len(signals) >= MIN_CONFIRMATIONS
        if confirmed:
            self.logger.info(
                "Entry signal confirmed for %s (%s): %s",
                symbol, direction, signals,
            )
        else:
            self.logger.debug(
                "No entry yet for %s — signals so far: %s", symbol, signals
            )

        return confirmed, signals

    # ─── Exit Signal Detection ────────────────────────────────────────────────

    async def _check_exit_signal(
        self, symbol: str, alpaca
    ) -> tuple[bool, str]:
        """Return (should_exit, reason)."""
        current_price = alpaca.get_latest_price(symbol)
        entry_price = self.active_positions.get(symbol)
        entry_time = self.entry_times.get(symbol)

        if current_price is None or entry_price is None or entry_time is None:
            return False, ""

        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct >= TAKE_PROFIT_PCT:
            return True, f"take_profit ({pnl_pct*100:.2f}%)"

        if pnl_pct <= -STOP_LOSS_PCT:
            return True, f"stop_loss ({pnl_pct*100:.2f}%)"

        hold_time = datetime.utcnow() - entry_time
        if hold_time >= timedelta(hours=MAX_HOLD_HOURS):
            return True, f"time_limit ({hold_time})"

        return False, ""

    # ─── Order Execution ──────────────────────────────────────────────────────

    async def _execute_entry(
        self,
        symbol: str,
        proposal: Dict[str, Any],
        alpaca,
        signals: List[str],
    ):
        """Place a market order for entry and record the trade."""
        quantity = proposal.get("quantity", 1)
        direction = proposal.get("direction", "BUY").upper()
        stop_loss = proposal.get("stop_loss")

        order = alpaca.place_market_order(symbol, quantity, direction.lower())
        if not order:
            self.logger.error("Market order failed for %s", symbol)
            return

        current_price = alpaca.get_latest_price(symbol) or proposal.get("entry_price", 0)

        db = get_session()
        try:
            trade = Trade(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                quantity=quantity,
                status="ACTIVE",
            )
            db.add(trade)
            db.flush()  # get trade.id

            db_order = Order(
                trade_id=trade.id,
                order_type="ENTRY",
                order_id=order.get("order_id"),
                status="FILLED",
                filled_price=current_price,
                filled_qty=quantity,
            )
            db.add(db_order)
            db.commit()

            self.active_positions[symbol] = current_price
            self.active_trade_ids[symbol] = trade.id
            self.entry_times[symbol] = datetime.utcnow()
            # Remove from pending once entered
            self.approved_trades.pop(symbol, None)

            self.logger.info(
                "ENTERED %s %s x%d @ $%.2f | stop_loss=$%s | signals=%s",
                symbol, direction, quantity, current_price, stop_loss, signals,
            )
            await self.log_decision(
                "TRADE_ENTERED",
                trade_id=trade.id,
                reasoning={
                    "symbol": symbol,
                    "direction": direction,
                    "price": current_price,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "signals": signals,
                },
            )
        except Exception as e:
            db.rollback()
            self.logger.error("Failed to record entry for %s: %s", symbol, e)
        finally:
            db.close()

    async def _execute_exit(self, symbol: str, reason: str, alpaca):
        """Place a limit order for exit and close the trade record."""
        entry_price = self.active_positions.get(symbol)
        current_price = alpaca.get_latest_price(symbol)
        position = alpaca.get_position(symbol)

        if not position or current_price is None:
            self.logger.warning("Cannot exit %s — position not found", symbol)
            return

        qty = int(position["qty"])
        # Limit slightly above market to favour fills on sells
        limit_price = round(current_price * 0.999, 2)

        order = alpaca.place_limit_order(symbol, qty, "sell", limit_price)
        if not order:
            self.logger.error("Limit exit order failed for %s", symbol)
            return

        pnl = (current_price - entry_price) * qty if entry_price else None
        trade_id = self.active_trade_ids.get(symbol)

        db = get_session()
        try:
            if trade_id:
                trade = db.query(Trade).filter_by(id=trade_id).first()
                if trade:
                    trade.exit_price = current_price
                    trade.pnl = pnl
                    trade.status = "CLOSED"
                    trade.closed_at = datetime.utcnow()

            db_order = Order(
                trade_id=trade_id,
                order_type="EXIT",
                order_id=order.get("order_id"),
                status="PENDING",  # limit order may not fill immediately
                filled_price=limit_price,
                filled_qty=qty,
            )
            db.add(db_order)
            db.commit()

            # Clean up memory
            self.active_positions.pop(symbol, None)
            self.active_trade_ids.pop(symbol, None)
            self.entry_times.pop(symbol, None)

            self.logger.info(
                "EXITED %s @ $%.2f | reason=%s | PnL=$%.2f",
                symbol, current_price, reason, pnl or 0,
            )
            await self.log_decision(
                "TRADE_EXITED",
                trade_id=trade_id,
                reasoning={
                    "symbol": symbol,
                    "exit_price": current_price,
                    "pnl": pnl,
                    "reason": reason,
                },
            )
        except Exception as e:
            db.rollback()
            self.logger.error("Failed to record exit for %s: %s", symbol, e)
        finally:
            db.close()


# Module-level singleton
trader = Trader()
