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
UTC_TZ = ZoneInfo("UTC")

APPROVED_TRADES_QUEUE = "trader_approved_trades"
EXIT_REQUESTS_QUEUE = "trader_exit_requests"    # PM → Trader: EXIT/HOLD/EXTEND_TARGET
REEVAL_REQUESTS_QUEUE = "pm_reeval_requests"    # Trader → PM: request re-evaluation
CHECK_INTERVAL = 300      # seconds between full scan cycles
MIN_BARS = 220            # minimum daily bars required for EMA(200) + buffer
INTRADAY_FORCE_CLOSE_HOUR = 15
INTRADAY_FORCE_CLOSE_MINUTE = 45  # 3:45 PM ET force-flat all intraday positions

# Canonical exit reasons written to trades.exit_reason
_EXIT_REASON_MAP = {
    "STOP": "stop_hit",
    "STOP_HIT": "stop_hit",
    "TARGET": "target_hit",
    "TARGET_HIT": "target_hit",
    "TIME": "time_exit",
    "TIME_EXIT": "time_exit",
    "EOD": "eod_intraday",
    "EOD_INTRADAY": "eod_intraday",
    "PM_EXIT": "pm_review",
    "PM_HOLD": "pm_review",
    "PM_APPROACHING_STOP": "pm_review",
    "NEWS": "news_exit",
    "NEWS_EXIT": "news_exit",
    "KILL_SWITCH": "kill_switch",
    "MANUAL": "manual",
    "PARTIAL": "partial_exit",
}


def _normalise_exit_reason(reason: str) -> str:
    key = (reason or "").upper().strip()
    if key in _EXIT_REASON_MAP:
        return _EXIT_REASON_MAP[key]
    # Pattern-based normalization for dynamic PM reasons
    if key.startswith("PM_EARNINGS_IN_") or key.startswith("EARNINGS_IN_"):
        return "pm_earnings_proximity"
    if key.startswith("PM_NIS_") or key.startswith("NIS_"):
        return "pm_nis_exit"
    if key.startswith("PM_SCORE_DEGRADED") or key.startswith("SCORE_DEGRADED"):
        return "pm_score_degraded"
    if key.startswith("PM_NEGATIVE_NEWS") or key.startswith("NEGATIVE_NEWS"):
        return "pm_news_exit"
    return reason.lower() if reason else "unknown"


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
        # Symbols discarded today after hitting max entry quality rejections.
        # Cleared at midnight reset. Prevents re-approval from new PM proposal
        # batches from bypassing the 3-strike discard.
        self._daily_discarded_symbols: set = set()
        # Per-symbol quality rejection counts, persisted across PM re-proposals.
        # PM sends fresh proposals every 30 min — without this, _reject_count resets
        # and symbols never hit the 3-strike discard.
        self._daily_quality_rejections: Dict[str, int] = {}
        self._last_mid_recon_slot: int = -1  # Phase 78d: track 15-min reconciliation slots

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

        # Note: empty raw_positions is a *valid* response — we still need to run
        # ghost detection (DB ACTIVE trades that have no Alpaca counterpart).
        if raw_positions is None:
            return
        raw_positions = raw_positions or []

        db = get_session()
        try:
            db_active = {t.symbol: t for t in db.query(Trade).filter(Trade.status == "ACTIVE").all()}
        except Exception as exc:
            self.logger.warning("Reconciliation: DB query failed: %s", exc)
            db.close()
            return

        # Build the symbol set Alpaca currently holds — used for ghost detection below.
        alpaca_symbols = {
            (p.get("symbol") or "").upper()
            for p in raw_positions
            if p.get("symbol")
        }

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
                    entry_date = t.created_at.date() if t.created_at else datetime.now(ET).date()
                    # Recalculate bars_held from entry_date so restarts don't reset the counter.
                    # DB value is only written on close so it's always 0 for active trades.
                    days_since_entry = (datetime.now(ET).date() - entry_date).days
                    bars_held = max(t.bars_held or 0, days_since_entry)
                    _rec_dir = getattr(t, "direction", "BUY") or "BUY"
                    # Cross-check DB direction against Alpaca qty sign; use already-resolved qty
                    _alpaca_short = qty < 0
                    if _alpaca_short and _rec_dir != "SELL_SHORT":
                        self.logger.warning(
                            "Reconcile %s: DB direction=%s but Alpaca qty<0 — correcting to SELL_SHORT",
                            symbol, _rec_dir,
                        )
                        _rec_dir = "SELL_SHORT"
                    elif not _alpaca_short and _rec_dir == "SELL_SHORT":
                        self.logger.warning(
                            "Reconcile %s: DB direction=SELL_SHORT but Alpaca qty>0 — correcting to BUY",
                            symbol,
                        )
                        _rec_dir = "BUY"
                    _rec_short = _rec_dir == "SELL_SHORT"
                    _rec_stop = t.stop_price or round(avg * (1.02 if _rec_short else 0.98), 2)
                    _rec_tgt = t.target_price or round(avg * (0.94 if _rec_short else 1.06), 2)
                    self.active_positions[symbol] = {
                        "entry_price":   t.entry_price,
                        "stop_price":    _rec_stop,
                        "target_price":  _rec_tgt,
                        "highest_price": avg,
                        "atr":           0.0,
                        "bars_held":     bars_held,
                        "trade_id":      t.id,
                        "trade_type":    getattr(t, "trade_type", None) or "swing",
                        "entry_date":    entry_date,
                        "direction":     _rec_dir,
                        "proposal_uuid": getattr(t, "proposal_id", None),
                        "shares":        int(t.quantity or 0),
                    }
                    self.logger.info("Reconciled %s from DB trade id=%d dir=%s", symbol, t.id, _rec_dir)
                continue

            # No DB record — create a synthetic ACTIVE trade using generate_signal for stop/target
            if symbol in self.active_positions:
                continue  # already loaded

            # Before creating a new record, check if a same-day trade already exists.
            # This handles restart-after-partial-exit: remaining shares still open in Alpaca
            # but the original Trade row was already closed. Reuse it to avoid double-counting P&L.
            today_start = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_utc = today_start.astimezone(UTC_TZ).replace(tzinfo=None)
            existing_today = (
                db.query(Trade)
                .filter(Trade.symbol == symbol, Trade.created_at >= today_start_utc)
                .order_by(Trade.id.desc())
                .first()
            )
            if existing_today:
                self.logger.warning(
                    "Reconciliation: %s has existing trade id=%d today — treating as remnant, not creating new record",
                    symbol, existing_today.id,
                )
                _rem_dir = getattr(existing_today, "direction", "BUY") or "BUY"
                _rem_short = _rem_dir == "SELL_SHORT"
                _rem_stop = existing_today.stop_price or round(avg * (1.02 if _rem_short else 0.98), 2)
                _rem_tgt = existing_today.target_price or round(avg * (0.94 if _rem_short else 1.06), 2)
                # Recompute partial P&L from the immutable Order ledger rather than
                # trusting trade.pnl which may have been zeroed by a prior reconcile cycle.
                from app.database.models import recompute_partial_pnl
                _entry_px = float(existing_today.entry_price or avg)
                _prior_pnl = recompute_partial_pnl(db, existing_today.id, _entry_px, _rem_dir)
                # If ledger returns 0 but trade.pnl has a value, fall back to trade.pnl
                # (handles legacy rows written before Order-ledger fix was deployed)
                if _prior_pnl == 0.0 and existing_today.pnl:
                    _prior_pnl = float(existing_today.pnl)
                self.active_positions[symbol] = {
                    "entry_price":   _entry_px,
                    "stop_price":    float(_rem_stop),
                    "target_price":  float(_rem_tgt),
                    "highest_price": avg,
                    "atr":           0.0,
                    "bars_held":     existing_today.bars_held or 0,
                    "trade_id":      existing_today.id,
                    "trade_type":    getattr(existing_today, "trade_type", None) or "swing",
                    "shares":        abs(qty),
                    "entry_date":    (
                        existing_today.created_at.date()
                        if existing_today.created_at else datetime.now(ET).date()
                    ),
                    "direction":     _rem_dir,
                    "proposal_uuid": getattr(existing_today, "proposal_id", None),
                    "_partial_exited": True,
                    "_partial_pnl":    _prior_pnl,
                }
                existing_today.status = "ACTIVE"
                existing_today.quantity = abs(qty)
                existing_today.exit_price = None
                # Restore trade.pnl from the Order-ledger recompute so it stays consistent
                existing_today.pnl = _prior_pnl if _prior_pnl != 0.0 else None
                existing_today.closed_at = None
                db.commit()
                continue

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
                    # Default to swing for unknown positions — never intraday.
                    # Defaulting to intraday caused ghost positions (e.g. pre-existing
                    # Alpaca paper account holdings) to be force-closed at 3:45 PM
                    # every session, even though MrTrader never opened them.
                    trade_type_rec = "swing"
                    self.logger.warning(
                        "Reconciliation: no proposal record for %s — defaulting trade_type=swing",
                        symbol,
                    )

                # Detect short from Alpaca: negative qty indicates a short position
                _syn_dir = "SELL_SHORT" if qty < 0 else "BUY"
                _syn_short = _syn_dir == "SELL_SHORT"

                # generate_signal is long-only; for shorts use direction-aware fallbacks directly
                atr = 0.0
                signal = "ML_RANK"
                if not _syn_short:
                    bars = alpaca.get_bars(symbol, timeframe="1Day", limit=MIN_BARS)
                    if bars is not None and not bars.empty and len(bars) >= MIN_BARS:
                        result = generate_signal(
                            symbol, bars, ml_score=0.6, check_regime=False, check_earnings=False
                        )
                        stop = (result.stop_price if result.stop_price and result.stop_price > 0
                                else round(avg * 0.98, 2))
                        target = (result.target_price if result.target_price and result.target_price > 0
                                  else round(avg * 1.06, 2))
                        signal = "ML_RANK" if result.signal_type in ("NONE", None) else result.signal_type
                        atr = result.atr
                    else:
                        stop = round(avg * 0.98, 2)
                        target = round(avg * 1.06, 2)
                else:
                    stop = round(avg * 1.02, 2)   # 2% above entry for short stop
                    target = round(avg * 0.94, 2)  # 6% below entry for short target

                trade = Trade(
                    symbol=symbol,
                    direction=_syn_dir,
                    entry_price=avg,
                    quantity=abs(qty),
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
                    "direction":     _syn_dir,
                    "shares":        abs(qty),
                }
                self.logger.info(
                    "Reconciled %s: created synthetic Trade id=%d stop=%.2f target=%.2f",
                    symbol, trade.id, stop, target,
                )
            except Exception as exc:
                db.rollback()
                self.logger.error("Reconciliation failed for %s: %s", symbol, exc)

        # ── Ghost detection: ACTIVE in DB but NOT in Alpaca ──────────────────
        # Symmetrical with the "untracked" branch above. Protects against newly
        # placed market orders whose fills haven't yet propagated to Alpaca by
        # requiring the trade to be at least `reconcile.ghost_min_age_minutes`
        # old (default 5 min).
        try:
            from app.database.agent_config import get_agent_config
            from app.database.models import AuditLog
            try:
                ghost_min_age_minutes = int(get_agent_config(db, "reconcile.ghost_min_age_minutes") or 5)
            except Exception:
                ghost_min_age_minutes = 5
            from datetime import timedelta
            ghost_cutoff = datetime.utcnow() - timedelta(minutes=ghost_min_age_minutes)

            for symbol, trade in db_active.items():
                if symbol.upper() in alpaca_symbols:
                    continue
                age_anchor = trade.created_at or datetime.utcnow()
                if age_anchor > ghost_cutoff:
                    continue  # too new — Alpaca fill may not have propagated
                self.logger.warning(
                    "GHOST POSITION: Trade#%d %s ACTIVE in DB but not in Alpaca "
                    "(Alpaca has %d position(s); trade older than %d min) — marking RECONCILE_GHOST",
                    trade.id, symbol, len(alpaca_symbols), ghost_min_age_minutes,
                )
                trade.status = "RECONCILE_GHOST"
                db.add(AuditLog(
                    action="RECONCILE_GHOST_POSITION",
                    details={
                        "trade_id": trade.id,
                        "symbol": symbol,
                        "reason": "Active in DB but no matching Alpaca position (periodic reconcile)",
                        "alpaca_position_count": len(alpaca_symbols),
                        "ghost_min_age_minutes": ghost_min_age_minutes,
                        "detected_at": datetime.utcnow().isoformat(),
                        "source": "trader._reconcile_positions",
                    },
                    timestamp=datetime.utcnow(),
                ))
                # Drop in-memory state so exit logic stops watching it
                self.active_positions.pop(symbol, None)
            db.commit()
        except Exception as exc:
            db.rollback()
            self.logger.error("Ghost detection in _reconcile_positions failed: %s", exc)

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

        # Phase 78b: reload any pending limit orders from DB so we resume polling
        try:
            self._reload_pending_limits_from_db()
        except Exception as exc:
            self.logger.warning("Could not reload pending limits from DB: %s", exc)

        while self.status == "running":
            try:
                now = datetime.now(ET)
                today = now.strftime("%Y-%m-%d")
                if today != self._last_date:
                    self._force_closed_today = False
                    self._last_date = today
                    self._daily_discarded_symbols.clear()
                    self._daily_quality_rejections.clear()

                # Drain all pending approved proposals (non-blocking)
                while True:
                    proposal = await asyncio.to_thread(
                        self.get_message, APPROVED_TRADES_QUEUE, 1
                    )
                    if proposal is None:
                        break
                    symbol = proposal.get("symbol")
                    if symbol:
                        # Discard stale proposals — approved more than 30 min ago
                        approved_at_str = proposal.get("approved_at")
                        if approved_at_str:
                            try:
                                from datetime import timezone
                                approved_at = datetime.fromisoformat(approved_at_str).replace(tzinfo=timezone.utc)
                                age_minutes = (datetime.now(timezone.utc) - approved_at).total_seconds() / 60
                                if age_minutes > 30:
                                    self.logger.info(
                                        "Discarding stale proposal for %s (approved %.0f min ago)",
                                        symbol, age_minutes,
                                    )
                                    continue
                            except Exception:
                                pass
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
                    from app.live_trading.kill_switch import kill_switch
                    if kill_switch.is_active:
                        self.logger.warning("Kill switch ACTIVE — skipping scan; cancelling pending limit orders")
                        self.approved_symbols.clear()
                        if self._pending_limit_orders:
                            try:
                                from app.integrations import get_alpaca_client
                                _ks_alpaca = get_alpaca_client()
                                for _sym, _pend in list(self._pending_limit_orders.items()):
                                    try:
                                        _ks_alpaca.cancel_order(_pend["order_id"])
                                        self.logger.warning(
                                            "KS: cancelled pending limit %s (%s)", _pend["order_id"], _sym
                                        )
                                    except Exception as _ce:
                                        self.logger.error("KS: cancel pending limit %s failed: %s", _sym, _ce)
                                self._pending_limit_orders.clear()
                            except Exception as _ke:
                                self.logger.error("KS: could not cancel pending limit orders: %s", _ke)
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

            # WITHDRAW: PM wants to cancel a pending approval before entry
            if action == "WITHDRAW":
                if symbol in self.approved_symbols:
                    self.approved_symbols.pop(symbol)
                    self.logger.info("PM withdrew pending approval for %s — reason: %s", symbol, reason)
                continue

            if symbol not in self.active_positions:
                # PM may send requests for positions already closed — ignore
                continue

            if action == "EXIT":
                self.logger.info("PM exit request for %s — reason: %s", symbol, reason)
                try:
                    alpaca = get_alpaca_client()
                    _q = alpaca.get_quote(symbol)
                    price = ((_q["mid"] if _q else None)
                             or alpaca.get_latest_price(symbol)
                             or self.active_positions[symbol]["entry_price"])
                    await self._execute_exit(symbol, price, f"PM_{reason.upper()}", alpaca)
                except Exception as exc:
                    self.logger.error("PM-requested exit failed for %s: %s", symbol, exc)

            elif action == "EXTEND_TARGET":
                extend_atr = float(msg.get("extend_atr", 0.0))
                if extend_atr > 0 and symbol in self.active_positions:
                    pos = self.active_positions[symbol]
                    old_target = pos["target_price"]
                    # Shorts: target is below entry — extending means moving further down
                    _ext_sign = -1 if pos.get("direction") == "SELL_SHORT" else 1
                    new_target = round(old_target + _ext_sign * extend_atr, 4)
                    # Persist to DB so it survives a restart — but validate first.
                    db = get_session()
                    try:
                        from app.startup_reconciler import write_target_stop
                        trade = db.query(Trade).filter_by(id=pos["trade_id"]).first()
                        if trade:
                            wrote = write_target_stop(
                                trade, target_price=new_target,
                                written_by="trader.extend_target",
                                reason=msg.get("reason", "pm_extend"),
                            )
                            if wrote:
                                db.commit()
                                pos["target_price"] = new_target
                                self.logger.info(
                                    "%s: target extended by %.4f ATR → $%.2f (was $%.2f)",
                                    symbol, extend_atr, new_target, old_target,
                                )
                            else:
                                db.rollback()
                                self.logger.error(
                                    "%s: EXTEND_TARGET REJECTED (extend_atr=%.4f, "
                                    "proposed=$%.4f, current=$%.4f) — see ERROR above",
                                    symbol, extend_atr, new_target, old_target,
                                )
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()

            # HOLD → no action needed, PM is confirming to stay in position

    async def _send_reeval_request(self, symbol: str, reason: str, current_price: float = 0.0) -> None:
        """Ask PM to re-evaluate a position. PM will respond via trader_exit_requests."""
        pos = self.active_positions.get(symbol, {})
        entry = pos.get("entry_price") or 0.0
        _is_short = pos.get("direction") == "SELL_SHORT"
        if entry > 0 and current_price > 0:
            pnl_pct = round(
                (entry - current_price) / entry if _is_short else (current_price - entry) / entry,
                4,
            )
        else:
            pnl_pct = 0.0
        self.send_message(REEVAL_REQUESTS_QUEUE, {
            "symbol": symbol,
            "reason": reason,
            "entry_price": entry,
            "current_price": current_price,
            "current_pnl_pct": pnl_pct,
            "bars_held": pos.get("bars_held", 0),
            "trade_type": pos.get("trade_type", "swing"),
            "direction": pos.get("direction", "BUY"),
        })
        self.logger.info("Sent reeval request to PM: %s (%s)", symbol, reason)

    # ─── Scan Cycle ───────────────────────────────────────────────────────────

    async def _scan_cycle(self):
        """One pass: check entries for pending symbols, exits for active positions."""
        from app.integrations import get_alpaca_client
        alpaca = get_alpaca_client()

        # Phase 78d: mid-session reconciliation every 15 min during market hours
        now_et = datetime.now(ET)
        if now_et.weekday() < 5 and 9 <= now_et.hour < 16:
            recon_slot = (now_et.hour * 60 + now_et.minute) // 15
            if recon_slot != self._last_mid_recon_slot:
                self._last_mid_recon_slot = recon_slot
                try:
                    from app.startup_reconciler import reconcile
                    from app.database.session import get_session as _gs
                    _db = _gs()
                    try:
                        await asyncio.to_thread(reconcile, alpaca, _db)
                    finally:
                        _db.close()
                except Exception as _re:
                    self.logger.debug("Mid-session reconciliation failed: %s", _re)

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

    @staticmethod
    def _pead_sized_stop_target(entry_price, proposal):
        """PEAD live-sizing fidelity: scale the proposal's OWN stop/target (0.5×/1.5× ATR_norm,
        set by the PM) — expressed as a fraction of its scan price — onto the live entry, so
        risk-per-share (and thus position size) matches the PEAD backtest instead of
        generate_signal's wider swing ATR stop. Preserving the *fraction* keeps a small scan→now
        drift (bounded by the entry-quality gate) from distorting sizing, and preserves the
        long/short side. Returns (stop, target); either may be None if unavailable."""
        p_entry = proposal.get("entry_price")
        p_stop = proposal.get("stop_loss")
        p_tgt = proposal.get("profit_target")
        if not (entry_price and entry_price > 0 and p_entry and p_entry > 0):
            return None, None
        stop = round(entry_price * (p_stop / p_entry), 2) if (p_stop and p_stop > 0) else None
        tgt = round(entry_price * (p_tgt / p_entry), 2) if (p_tgt and p_tgt > 0) else None
        return stop, tgt

    async def _check_entry(self, symbol: str, proposal: Dict[str, Any], alpaca):
        """Fetch daily bars, compute entry prices via generate_signal(), enter if ML score passes.

        generate_signal() is used only for ATR-based stop/target prices and current price.
        The entry gate is the ML confidence score (>= ML_SCORE_THRESHOLD), not the
        rule-based is_buy flag — the walk-forward validation assumes ML score is sufficient.
        """
        from app.strategy.signals import ML_SCORE_THRESHOLD
        trade_type = proposal.get("trade_type", "swing")

        if symbol in self.active_positions:
            self.logger.info("%s: already in active_positions — skipping duplicate entry", symbol)
            self.approved_symbols.pop(symbol, None)
            # Fix 1b: write final status so UI shows reason instead of "Queued"
            proposal_uuid = proposal.get("proposal_uuid")
            if proposal_uuid:
                try:
                    from app.database.session import get_session
                    from app.database.models import ProposalLog
                    _db = get_session()
                    try:
                        for _row in _db.query(ProposalLog).filter(
                            ProposalLog.proposal_uuid == proposal_uuid
                        ).all():
                            _row.trader_status = "DUPLICATE_HELD"
                            _row.trader_reason = "Symbol already in active_positions"
                            _row.trader_decided_at = datetime.now(ET)
                        _db.commit()
                    finally:
                        _db.close()
                except Exception as _e:
                    self.logger.debug("Could not write DUPLICATE_HELD status: %s", _e)
            return

        if symbol in self._pending_limit_orders:
            self.logger.info("%s: limit order already pending — skipping duplicate entry", symbol)
            self.approved_symbols.pop(symbol, None)
            proposal_uuid = proposal.get("proposal_uuid")
            if proposal_uuid:
                try:
                    from app.database.session import get_session
                    from app.database.models import ProposalLog
                    _db = get_session()
                    try:
                        for _row in _db.query(ProposalLog).filter(
                            ProposalLog.proposal_uuid == proposal_uuid
                        ).all():
                            _row.trader_status = "DUPLICATE_HELD"
                            _row.trader_reason = "Limit order already pending for symbol"
                            _row.trader_decided_at = datetime.now(ET)
                        _db.commit()
                    finally:
                        _db.close()
                except Exception as _e:
                    self.logger.debug("Could not write DUPLICATE_HELD status: %s", _e)
            return

        if symbol in self._daily_discarded_symbols:
            self.logger.info("%s: discarded earlier today — rejecting re-approval", symbol)
            self.approved_symbols.pop(symbol, None)
            return

        # Intraday only: no new entries after 3:00 PM ET (force-close at 3:45 PM)
        now_et = datetime.now(ET)
        if (trade_type == "intraday" and now_et.weekday() < 5
                and (now_et.hour > 15 or (now_et.hour == 15 and now_et.minute >= 0))):
            self.logger.info(
                "%s: no new intraday entries after 3:00 PM ET (%02d:%02d) — skipping",
                symbol, now_et.hour, now_et.minute,
            )
            self.approved_symbols.pop(symbol, None)
            self._release_intraday_slot(trade_type)
            return

        if circuit_breaker.is_strategy_paused(trade_type):
            self.logger.debug(
                "%s: strategy '%s' is paused — skipping entry", symbol, trade_type
            )
            self._release_intraday_slot(trade_type)
            return

        # ── Macro/market gate ─────────────────────────────────────────────────
        # Block entries when market conditions are structurally bad today.
        # Premarket module caches SPY intraday drawdown for 5 min — cheap to call.
        try:
            from app.agents.premarket import premarket_intel
            if trade_type == "intraday" and premarket_intel.is_intraday_blocked():
                macro_flags = list(premarket_intel.macro_flags.keys())
                self.logger.warning(
                    "%s: intraday macro gate BLOCKED (SPY pre-mkt=%.1f%%, macro=%s) — skipping",
                    symbol,
                    premarket_intel.spy_premarket_pct * 100,
                    macro_flags,
                )
                self.approved_symbols.pop(symbol, None)
                self._release_intraday_slot(trade_type)
                await self.log_decision("ENTRY_BLOCKED_MACRO", reasoning={
                    "symbol": symbol, "trade_type": "intraday",
                    "reason": f"macro gate: {macro_flags}",
                })
                return
            if trade_type == "swing" and premarket_intel.is_swing_blocked():
                macro_flags = list(premarket_intel.macro_flags.keys())
                block_reason = f"Macro gate: {', '.join(macro_flags) if macro_flags else 'SPY drawdown/FOMC'}"
                self.logger.warning(
                    "%s: swing macro gate BLOCKED (%s) — skipping",
                    symbol, block_reason,
                )
                self.approved_symbols.pop(symbol, None)
                await self.log_decision("ENTRY_BLOCKED_MACRO", reasoning={
                    "symbol": symbol, "trade_type": "swing",
                    "reason": block_reason,
                })
                # Write back so Execution column shows reason instead of "Queued"
                proposal_uuid = proposal.get("proposal_uuid")
                if proposal_uuid:
                    try:
                        from app.database.session import get_session
                        from app.database.models import ProposalLog
                        db = get_session()
                        try:
                            for pl_row in db.query(ProposalLog).filter(
                                ProposalLog.proposal_uuid == proposal_uuid
                            ).all():
                                pl_row.trader_status = "MACRO_BLOCKED"
                                pl_row.trader_reason = block_reason
                                pl_row.trader_decided_at = datetime.now(ET)
                            db.commit()
                        finally:
                            db.close()
                    except Exception:
                        pass
                return
        except Exception as exc:
            self.logger.debug("Macro gate check failed (non-fatal): %s", exc)

        bars = alpaca.get_bars(symbol, timeframe="1Day", limit=MIN_BARS)
        if bars is None or bars.empty or len(bars) < MIN_BARS:
            self.logger.debug(
                "%s: only %d daily bars available (need %d)",
                symbol, len(bars) if bars is not None else 0, MIN_BARS,
            )
            self._release_intraday_slot(trade_type)
            return

        ml_score = proposal.get("confidence")
        if ml_score is None or ml_score < ML_SCORE_THRESHOLD:
            # Observability fix: this was the ONLY entry gate that rejected silently (DEBUG +
            # no trader_status), so RM-approved proposals below the Trader's threshold sat
            # looking "Queued" with no reason — the #1 cause of "why aren't trades firing?".
            # Now it logs at INFO and writes a terminal status like every other gate, and the
            # proposal is discarded (ml_score is fixed, so it can never clear on a later pass).
            reason = (f"ML score {ml_score:.3f} < entry threshold {ML_SCORE_THRESHOLD:.2f}"
                      if ml_score is not None else "no ML score on proposal")
            self.logger.info("%s: %s — rejecting entry", symbol, reason)
            self.approved_symbols.pop(symbol, None)
            self._release_intraday_slot(trade_type)
            await self.log_decision("ENTRY_REJECTED_ML_SCORE", reasoning={
                "symbol": symbol, "trade_type": trade_type,
                "ml_score": round(ml_score, 4) if ml_score is not None else None,
                "threshold": ML_SCORE_THRESHOLD, "reason": reason,
            })
            proposal_uuid = proposal.get("proposal_uuid")
            if proposal_uuid:
                try:
                    from app.database.session import get_session
                    from app.database.models import ProposalLog
                    _db = get_session()
                    try:
                        for _row in _db.query(ProposalLog).filter(
                            ProposalLog.proposal_uuid == proposal_uuid
                        ).all():
                            _row.trader_status = "REJECTED_ML_SCORE"
                            _row.trader_reason = reason
                            _row.trader_decided_at = datetime.now(ET)
                        _db.commit()
                    finally:
                        _db.close()
                except Exception as _e:
                    self.logger.debug("Could not write REJECTED_ML_SCORE status: %s", _e)
            return

        # Use generate_signal for ATR-based stop/target prices only (not as entry gate)
        result = generate_signal(symbol, bars, ml_score=ml_score, check_regime=False, check_earnings=True)

        # PEAD live-sizing FIDELITY (Opus live-path sweep finding): generate_signal returns the
        # *swing* 2.5×ATR stop; sizing PEAD off it under-sizes PEAD vs its backtest (PEAD's own
        # stop is the tighter 0.5×ATR), and a no-signal generate_signal returns stop=0 (degenerate
        # sizing). Override result.stop/target with the proposal's own stop DISTANCE (as a % of its
        # scan price) scaled to the live entry. NOTE: this affects ONLY position SIZING
        # (size_position reads result.stop_price); the entry order + Trade record already use the
        # proposal's stop/target (via _write_pending_fill), and result.entry_price (live) is left
        # untouched so the entry-quality gate is unaffected. LONG-only today: size_position returns
        # 0 for stop>=entry, so short PEAD (stop>entry) won't size until size_position is made
        # abs(entry-stop)-aware — fix before enabling pm.pead_enable_shorts (currently false).
        if (proposal.get("selector") or "").lower() == "pead":
            _ps, _pt = self._pead_sized_stop_target(result.entry_price, proposal)
            if _ps is not None:
                result.stop_price = _ps
            if _pt is not None:
                result.target_price = _pt

        # ── Real-time entry quality check ─────────────────────────────────────
        # Validate current market conditions before committing capital.
        # PM scored this symbol hours ago; Trader verifies the moment is still right.
        current_price = alpaca.get_latest_price(symbol)
        if current_price and result.entry_price > 0:
            quote = None
            intraday_bars_5m = None
            try:
                quote = alpaca.get_quote(symbol)
            except Exception:
                pass
            try:
                intraday_bars_5m = alpaca.get_bars(symbol, timeframe="5Min", limit=78)
            except Exception:
                pass

            from app.strategy.entry_quality import check_entry_quality
            eq = check_entry_quality(
                symbol=symbol,
                signal_price=result.entry_price,
                current_price=current_price,
                trade_type=trade_type,
                quote=quote,
                intraday_bars=intraday_bars_5m,
                selector=(proposal.get("selector") or ""),
            )
            if not eq.approved:
                self.logger.info(
                    "%s: entry quality check FAILED (%s) — skipping; "
                    "run=%.2f%% spread=%.3f%% momentum=%.2f%%",
                    symbol, eq.reason,
                    eq.price_run_pct * 100, eq.spread_pct * 100, eq.momentum_slope * 100,
                )
                # Track rejections using a day-level counter that survives PM re-proposals.
                # Without _daily_quality_rejections, each fresh PM proposal resets
                # _reject_count to 0 and the symbol never hits the 3-strike discard.
                self._daily_quality_rejections[symbol] = (
                    self._daily_quality_rejections.get(symbol, 0) + 1
                )
                proposal["_reject_count"] = self._daily_quality_rejections[symbol]
                max_retries = 3
                discard = "price_run" in eq.reason or proposal["_reject_count"] >= max_retries
                await self.log_decision("ENTRY_REJECTED_QUALITY", reasoning={
                    "symbol": symbol,
                    "reason": eq.reason,
                    "price_run_pct": round(eq.price_run_pct * 100, 2),
                    "spread_pct": round(eq.spread_pct * 100, 3),
                    "momentum_slope_pct": round(eq.momentum_slope * 100, 2),
                    "volume_ratio": round(eq.volume_ratio, 2),
                    "trade_type": trade_type,
                    "reject_count": proposal["_reject_count"],
                    "discarded": discard,
                })
                # Write trader outcome back to ProposalLog
                proposal_uuid = proposal.get("proposal_uuid")
                if proposal_uuid:
                    try:
                        from app.database.session import get_session
                        from app.database.models import ProposalLog
                        _reason_map = {
                            "price_run": f"Price ran {eq.price_run_pct*100:.1f}% from signal",
                            "spread": f"Spread {eq.spread_pct*100:.1f}% too wide",
                            "adverse_move": f"Adverse move {eq.price_run_pct*100:.1f}% from signal",
                        }
                        trader_reason = next(
                            (v for k, v in _reason_map.items() if k in eq.reason),
                            eq.reason,
                        )
                        db = get_session()
                        try:
                            for pl_row in db.query(ProposalLog).filter(
                                ProposalLog.proposal_uuid == proposal_uuid
                            ).all():
                                pl_row.trader_status = "DISCARDED" if discard else "QUALITY_REJECTED"
                                pl_row.trader_reason = trader_reason
                                pl_row.trader_decided_at = datetime.now(ET)
                            db.commit()
                        finally:
                            db.close()
                    except Exception:
                        pass
                if discard:
                    self.logger.info(
                        "%s: discarding proposal after %d rejection(s) (%s)",
                        symbol, proposal["_reject_count"], eq.reason,
                    )
                    self.approved_symbols.pop(symbol, None)
                    self._daily_discarded_symbols.add(symbol)
                # Position was never entered — release the RM intraday slot
                self._release_intraday_slot(trade_type)
                return

        # Final guard: check Alpaca live position before entering.
        # Prevents re-entry after a restart where active_positions was cleared but
        # the Alpaca position survived (e.g. after the DB was incorrectly closed by
        # a stale reconciler run).
        try:
            _live_pos = alpaca.get_position(symbol)
            if _live_pos:
                _live_qty = abs(int(_live_pos.get("qty", 0) or 0))
                if _live_qty > 0:
                    self.logger.warning(
                        "%s: Alpaca already holds %d shares — skipping entry to prevent duplicate position",
                        symbol, _live_qty,
                    )
                    self.approved_symbols.pop(symbol, None)
                    self._release_intraday_slot(trade_type)
                    return
        except Exception as _lp_exc:
            self.logger.debug("%s: live-position pre-entry check failed (non-fatal): %s", symbol, _lp_exc)

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
            self._release_intraday_slot(trade_type)
            return

        self.logger.info(
            "ENTRY PROCEEDING %s | type=%s ml_score=%.3f price=$%.2f stop=$%.2f target=$%.2f shares=%d",
            symbol, trade_type, ml_score or 0.0,
            result.entry_price, result.stop_price, result.target_price, shares,
        )
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
        is_short = proposal.get("direction") == "SELL_SHORT"

        proposal_uuid = proposal.get("proposal_uuid")
        # signal_type describes the SIGNAL MECHANISM that generate_signal() classified
        # (EMA_CROSSOVER | RSI_DIP | ML_RANK | NONE), NOT the PM selector. PEAD/factor/etc.
        # entries are ML-driven with no EMA/RSI crossover, so generate_signal returns
        # NONE → we record "ML_RANK". This is correct and intentional: the [ML_RANK] tag
        # means "ML-ranked signal", not "swing-ML-selector". Strategy-source attribution
        # (PEAD live-vs-backtest, etc.) lives in the dedicated Trade.selector column
        # (set below from proposal["selector"]) and is what _compute_pead_eod_stats and
        # the PEAD tracker filter on. Do NOT overload signal_type with the selector — the
        # signal_attribution buckets and the EMA/RSI/ML_RANK taxonomy depend on it.
        signal_type = "ML_RANK" if result.signal_type in ("NONE", None) else result.signal_type

        # ── Write PENDING_FILL before touching Alpaca ─────────────────────────
        # Survives any restart between order placement and fill confirmation.
        pending_trade = self._write_pending_fill(
            symbol, shares, intended_price, result, proposal, signal_type,
        )
        if pending_trade is None:
            self.logger.error("Could not write PENDING_FILL for %s — aborting entry", symbol)
            self._release_intraday_slot(trade_type)
            return
        pending_trade_id = pending_trade

        # ── PEAD marketable entry routing (owner decision #2) ──────────────────
        # The validated +0.546 backtest assumes next-open fills. The standard swing
        # below-ask limit (offset below ask) only fills the names that DON'T run —
        # an adverse-selection trap that systematically drops the high-drift winners.
        # For the PEAD selector, route a MARKETABLE limit (ask + offset) so fills
        # track the backtest's next-open assumption. Scoped to selector=="pead";
        # swing/intraday routing is byte-identical below.
        is_pead = proposal.get("selector") == "pead"
        if trade_type == "swing" and is_pead:
            marketable_offset = 0.001  # 10bps THROUGH the touch to cross the spread
            quote = alpaca.get_quote(symbol)
            if is_short:
                # Short PEAD: marketable sell — cross at bid - offset (aggressive)
                if quote and quote.get("bid", 0) > 0:
                    limit_price = round(quote["bid"] * (1 - marketable_offset), 2)
                    intended_price = quote["bid"]
                else:
                    limit_price = round(intended_price * (1 - marketable_offset), 2)
                order_side = "sell"
            else:
                # Long PEAD: marketable buy — cross at ask + offset (aggressive)
                if quote and quote.get("ask", 0) > 0:
                    limit_price = round(quote["ask"] * (1 + marketable_offset), 2)
                    intended_price = quote["ask"]
                else:
                    limit_price = round(intended_price * (1 + marketable_offset), 2)
                order_side = "buy"

            try:
                order = alpaca.place_limit_order(
                    symbol, shares, order_side, limit_price, client_order_id=proposal_uuid,
                )
            except Exception as exc:
                self.logger.error("PEAD marketable entry order failed for %s: %s", symbol, exc)
                self._cancel_pending_fill(pending_trade_id)
                return

            order_id = order.get("order_id")
            self._update_pending_fill_order_id(pending_trade_id, order_id)

            pending_entry = {
                "order_id": order_id,
                "trade_id": pending_trade_id,
                "shares": shares,
                "intended_price": intended_price,
                "limit_price": limit_price,
                "result": result,
                "proposal": proposal,
                "queued_at": datetime.now(ET),
                "requote_count": 0,
                "escalated": False,
            }
            self._pending_limit_orders[symbol] = pending_entry
            self._save_pending_limit_db(symbol, pending_entry, result)
            self.logger.info(
                "PEAD MARKETABLE order placed %s x%d @ $%.4f (%s touch=%.4f offset=%.1f%%) — crossing spread",
                symbol, shares, limit_price, order_side, intended_price, marketable_offset * 100,
            )
            return  # fill confirmed in _poll_pending_limit_orders

        if trade_type == "swing":
            # Use limit order ~10bps below ask for better execution (configurable)
            limit_offset = 0.001
            try:
                from app.database.agent_config import get_agent_config
                _db = get_session()
                try:
                    limit_offset = float(get_agent_config(_db, "strategy.limit_order_offset_pct") or 0.001)
                finally:
                    _db.close()
            except Exception:
                pass

            quote = alpaca.get_quote(symbol)
            if is_short:
                # Short entry: sell at bid + small offset (more aggressive = worse fill for us)
                if quote and quote.get("bid", 0) > 0:
                    limit_price = round(quote["bid"] * (1 + limit_offset), 2)
                    intended_price = quote["bid"]
                else:
                    limit_price = round(intended_price * (1 + limit_offset), 2)
                order_side = "sell"
            else:
                if quote and quote["ask"] > 0:
                    limit_price = round(quote["ask"] * (1 - limit_offset), 2)
                    intended_price = quote["ask"]
                else:
                    limit_price = round(intended_price * (1 - limit_offset), 2)
                order_side = "buy"

            try:
                order = alpaca.place_limit_order(
                    symbol, shares, order_side, limit_price, client_order_id=proposal_uuid,
                )
            except Exception as exc:
                self.logger.error("Limit entry order failed for %s: %s", symbol, exc)
                self._cancel_pending_fill(pending_trade_id)
                return

            order_id = order.get("order_id")
            # Update the PENDING_FILL record with the real Alpaca order ID
            self._update_pending_fill_order_id(pending_trade_id, order_id)

            pending_entry = {
                "order_id": order_id,
                "trade_id": pending_trade_id,
                "shares": shares,
                "intended_price": intended_price,
                "limit_price": limit_price,
                "result": result,
                "proposal": proposal,
                "queued_at": datetime.now(ET),
                "requote_count": 0,
                "escalated": False,
            }
            self._pending_limit_orders[symbol] = pending_entry
            # Phase 78b: persist to DB so a restart can reload this entry
            self._save_pending_limit_db(symbol, pending_entry, result)
            self.logger.info(
                "LIMIT ORDER placed %s x%d @ $%.4f (ask=%.4f offset=%.1f%%) — awaiting fill",
                symbol, shares, limit_price, intended_price, limit_offset * 100,
            )
            return  # position is confirmed in _poll_pending_limit_orders

        # Intraday: market order
        # Capture pre-trade mid as intended_price for accurate slippage measurement (Phase 76)
        try:
            _pre_quote = alpaca.get_quote(symbol)
            if _pre_quote and _pre_quote.get("mid", 0) > 0:
                intended_price = _pre_quote["mid"]
        except Exception:
            pass

        try:
            order = alpaca.place_market_order(
                symbol, shares, "sell" if is_short else "buy",
                client_order_id=proposal_uuid,
            )
        except Exception as exc:
            self.logger.error("Market order failed for %s: %s", symbol, exc)
            self._cancel_pending_fill(pending_trade_id)
            self._release_intraday_slot(trade_type)
            return

        # Poll for actual fill price instead of using next-bar price (Phase 76)
        order_id = order.get("order_id")
        filled_price = intended_price
        for _attempt in range(6):
            import time as _t
            _t.sleep(1)
            try:
                _status = alpaca.get_order_status(order_id)
                if _status and _status.get("filled_avg_price"):
                    filled_price = float(_status["filled_avg_price"])
                    break
            except Exception:
                pass

        # Slippage is direction-aware: for shorts, a lower fill price is worse (less proceeds)
        # so slippage = intended - filled (positive = bad, negative = good, same sign as longs).
        if intended_price > 0:
            if is_short:
                slippage_bps = round((intended_price - filled_price) / intended_price * 10000, 2)
            else:
                slippage_bps = round((filled_price - intended_price) / intended_price * 10000, 2)
        else:
            slippage_bps = 0.0

        await self._record_entry(
            symbol, shares, filled_price, intended_price, slippage_bps,
            order_id, result, proposal, pending_trade_id=pending_trade_id,
        )

    def _write_pending_fill(
        self, symbol: str, shares: int, intended_price: float,
        result, proposal: Dict[str, Any], signal_type: str,
    ) -> int | None:
        """Write PENDING_FILL Trade to DB before placing the order. Returns trade.id or None."""
        trade_type = proposal.get("trade_type", "swing")
        proposal_uuid = proposal.get("proposal_uuid")
        _direction = proposal.get("direction", "BUY")
        _is_short = _direction == "SELL_SHORT"
        # Accept both key conventions: stop_price (RM path) and stop_loss (PM direct path)
        _stop = (proposal.get("stop_price") or proposal.get("stop_loss") or result.stop_price or 0.0)
        _target = (proposal.get("target_price") or proposal.get("profit_target") or result.target_price or 0.0)
        if not _stop or _stop <= 0:
            # Short: stop above entry; long: stop below entry
            _stop = round(intended_price * 1.05, 2) if _is_short else round(intended_price * 0.98, 2)
        if not _target or _target <= 0:
            _target = round(intended_price * 0.95, 2) if _is_short else round(intended_price * 1.06, 2)
        db = get_session()
        try:
            trade = Trade(
                symbol=symbol,
                direction=_direction,
                entry_price=intended_price,
                quantity=shares,
                status="PENDING_FILL",
                signal_type=signal_type,
                trade_type=trade_type,
                stop_price=_stop,
                target_price=_target,
                highest_price=intended_price,
                bars_held=0,
                proposal_id=proposal_uuid,
                selector=(proposal.get("selector") or ""),
            )
            db.add(trade)
            db.commit()
            db.refresh(trade)
            self.logger.info("PENDING_FILL written for %s (trade_id=%d)", symbol, trade.id)
            return trade.id
        except Exception as exc:
            db.rollback()
            self.logger.error("Failed to write PENDING_FILL for %s: %s", symbol, exc)
            return None
        finally:
            db.close()

    def _update_pending_fill_order_id(self, trade_id: int, alpaca_order_id: str) -> None:
        """Store the Alpaca order ID on the PENDING_FILL trade record."""
        db = get_session()
        try:
            trade = db.query(Trade).filter_by(id=trade_id).first()
            if trade:
                trade.alpaca_order_id = alpaca_order_id
                db.commit()
        except Exception as exc:
            db.rollback()
            self.logger.error("Failed to update alpaca_order_id for trade %d: %s", trade_id, exc)
        finally:
            db.close()

    def _release_intraday_slot(self, trade_type: str) -> None:
        """Decrement the RM intraday slot counter when an approved intraday proposal is not filled."""
        if trade_type != "intraday":
            return
        try:
            from app.agents.risk_manager import risk_manager
            risk_manager.on_intraday_position_closed()
        except Exception:
            pass

    def _cancel_pending_fill(self, trade_id: int) -> None:
        """Mark a PENDING_FILL trade as CANCELLED when order placement fails."""
        db = get_session()
        try:
            trade = db.query(Trade).filter_by(id=trade_id, status="PENDING_FILL").first()
            if trade:
                trade.status = "CANCELLED"
                db.commit()
                self.logger.info("PENDING_FILL trade %d marked CANCELLED (order failed)", trade_id)
        except Exception as exc:
            db.rollback()
            self.logger.error("Failed to cancel pending fill trade %d: %s", trade_id, exc)
        finally:
            db.close()

    # ── Phase 78b: persist/reload pending limit orders ─────────────────────────

    def _save_pending_limit_db(self, symbol: str, pending: Dict[str, Any], result) -> None:
        """Upsert a PendingLimitOrder row so it survives restarts."""
        from app.database.models import PendingLimitOrder
        db = get_session()
        try:
            row = db.query(PendingLimitOrder).filter_by(symbol=symbol).first()
            if row is None:
                row = PendingLimitOrder(symbol=symbol)
                db.add(row)
            row.order_id = pending["order_id"]
            row.trade_id = pending.get("trade_id")
            row.shares = pending["shares"]
            row.limit_price = pending["limit_price"]
            row.intended_price = pending["intended_price"]
            row.stop_price = result.stop_price
            row.target_price = result.target_price
            row.atr = getattr(result, "atr", None)
            row.trade_type = pending.get("proposal", {}).get("trade_type", "swing")
            row.signal_type = getattr(result, "signal_type", None)
            row.direction = pending.get("proposal", {}).get("direction", "BUY")
            row.requote_count = int(pending.get("requote_count", 0) or 0)
            row.escalated = bool(pending.get("escalated", False))
            db.commit()
        except Exception as exc:
            db.rollback()
            self.logger.debug("Could not persist pending limit for %s: %s", symbol, exc)
        finally:
            db.close()

    def _delete_pending_limit_db(self, symbol: str) -> None:
        """Remove the PendingLimitOrder row on fill/cancel."""
        from app.database.models import PendingLimitOrder
        db = get_session()
        try:
            row = db.query(PendingLimitOrder).filter_by(symbol=symbol).first()
            if row:
                db.delete(row)
                db.commit()
        except Exception as exc:
            db.rollback()
            self.logger.debug("Could not delete pending limit for %s: %s", symbol, exc)
        finally:
            db.close()

    def _reload_pending_limits_from_db(self) -> None:
        """
        On startup: load any today's PendingLimitOrder rows into _pending_limit_orders.
        Called once during reconcile so active limit orders survive restarts.
        """
        from app.database.models import PendingLimitOrder
        db = get_session()
        try:
            rows = db.query(PendingLimitOrder).all()
            if not rows:
                return
            for row in rows:
                if row.symbol in self._pending_limit_orders:
                    continue  # already loaded
                # Reconstruct a minimal SignalResult-like object

                class _FakeResult:
                    stop_price = row.stop_price
                    target_price = row.target_price
                    atr = row.atr or 1.0
                    signal_type = row.signal_type or "ML_RANK"
                    reasoning = {}
                self._pending_limit_orders[row.symbol] = {
                    "order_id": row.order_id,
                    "trade_id": row.trade_id,
                    "shares": row.shares,
                    "intended_price": row.intended_price,
                    "limit_price": row.limit_price,
                    "result": _FakeResult(),
                    "proposal": {
                        "trade_type": row.trade_type,
                        "direction": getattr(row, "direction", "BUY") or "BUY",
                    },
                    "queued_at": row.created_at,
                    "requote_count": int(getattr(row, "requote_count", 0) or 0),
                    "escalated": bool(getattr(row, "escalated", False) or False),
                }
                self.logger.info(
                    "Reloaded pending limit order for %s (order=%s) from DB",
                    row.symbol, row.order_id,
                )
        except Exception as exc:
            self.logger.warning("Could not reload pending limits from DB: %s", exc)
        finally:
            db.close()

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
        pending_trade_id: int | None = None,
    ):
        """Promote a PENDING_FILL trade to ACTIVE and record the fill details."""
        # Guard: reject placeholder/test order IDs that should never reach production.
        # Real Alpaca order IDs are UUIDs (e.g. "3ee1153a-a426-43c3-...").
        # IDs starting with "order-" are test fixtures from conftest.py / scripts.
        if order_id and str(order_id).startswith("order-"):
            self.logger.error(
                "_record_entry blocked for %s: fake order_id '%s' (starts with 'order-'). "
                "This indicates a test fixture leaked into production. Aborting.",
                symbol, order_id,
            )
            return

        trade_type = proposal.get("trade_type", "swing")
        signal_type = "ML_RANK" if result.signal_type in ("NONE", None) else result.signal_type
        db = get_session()
        try:
            # Update the existing PENDING_FILL record — don't create a duplicate
            trade = db.query(Trade).filter_by(id=pending_trade_id, status="PENDING_FILL").first() \
                if pending_trade_id else None

            if trade:
                trade.entry_price = filled_price
                trade.quantity = shares
                trade.status = "ACTIVE"
                trade.highest_price = filled_price
                trade.alpaca_order_id = order_id
            else:
                # Fallback: no PENDING_FILL found (e.g. reconciler path) — create fresh
                _fb_dir = proposal.get("direction", "BUY")
                _fb_short = _fb_dir == "SELL_SHORT"
                _fb_stop = (proposal.get("stop_price") or proposal.get("stop_loss")
                            or result.stop_price
                            or round(filled_price * (1.05 if _fb_short else 0.98), 2))
                _fb_target = (proposal.get("target_price") or proposal.get("profit_target")
                              or result.target_price
                              or round(filled_price * (0.95 if _fb_short else 1.06), 2))
                trade = Trade(
                    symbol=symbol, direction=_fb_dir,
                    entry_price=filled_price,
                    quantity=shares, status="ACTIVE", signal_type=signal_type,
                    trade_type=trade_type, stop_price=_fb_stop,
                    target_price=_fb_target, highest_price=filled_price,
                    bars_held=0, alpaca_order_id=order_id,
                    proposal_id=proposal.get("proposal_uuid"),
                    selector=(proposal.get("selector") or ""),
                )
                db.add(trade)

            db.flush()

            # Phase 2d: stamp walk-forward predicted P&L at entry (non-fatal)
            try:
                from app.ml.walk_forward_stats import get_predicted_pnl as _wf_pnl
                trade.sim_predicted_pnl = _wf_pnl(trade_type, db_session=db)
            except Exception as _wf_err:
                self.logger.debug("sim_predicted_pnl lookup failed (non-fatal): %s", _wf_err)

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

            # Link back to TradeProposal audit trail
            proposal_db_id = proposal.get("_proposal_id")
            if proposal_db_id:
                from app.database.models import TradeProposal
                tp = db.query(TradeProposal).filter(TradeProposal.id == proposal_db_id).first()
                if tp:
                    tp.trade_id = trade.id

            # Write trade_id and trader outcome back to unified ProposalLog
            proposal_uuid = proposal.get("proposal_uuid")
            if proposal_uuid:
                from app.database.models import ProposalLog, ProposalEvent
                for pl_row in db.query(ProposalLog).filter(ProposalLog.proposal_uuid == proposal_uuid).all():
                    pl_row.trade_id = trade.id
                    pl_row.trader_status = "FILLED"
                    pl_row.trader_reason = f"Filled @ ${filled_price:.2f}"
                    pl_row.trader_decided_at = datetime.now(ET)
                db.add(ProposalEvent(
                    proposal_uuid=proposal_uuid,
                    event_time=datetime.now(ET),
                    actor="trader",
                    event_type="ORDER_PLACED",
                    details={
                        "trade_id": trade.id,
                        "symbol": symbol,
                        "filled_price": filled_price,
                        "intended_price": intended_price,
                        "slippage_bps": slippage_bps,
                        "shares": shares,
                        "alpaca_order_id": order_id,
                    },
                ))

            db.commit()

            _pos_dir = proposal.get("direction", "BUY")
            _pos_short = _pos_dir == "SELL_SHORT"
            _pos_stop = (proposal.get("stop_price") or proposal.get("stop_loss")
                         or result.stop_price
                         or round(filled_price * (1.05 if _pos_short else 0.98), 2))
            _pos_target = (proposal.get("target_price") or proposal.get("profit_target")
                           or result.target_price
                           or round(filled_price * (0.95 if _pos_short else 1.06), 2))
            _pos_entry: dict = {
                "entry_price":   filled_price,
                "stop_price":    _pos_stop,
                "target_price":  _pos_target,
                "highest_price": filled_price,
                "atr":           result.atr,
                "bars_held":     0,
                "trade_id":      trade.id,
                "trade_type":    trade_type,
                "entry_date":    datetime.now(ET).date(),
                "direction":     _pos_dir,
                "proposal_uuid": proposal.get("proposal_uuid"),
                "shares":        shares,
            }
            # Propagate per-proposal hold cap (e.g. PEAD hold-5) into the live position
            _mhd = proposal.get("max_hold_days")
            if _mhd and int(_mhd) > 0:
                _pos_entry["max_hold_days"] = int(_mhd)
            self.active_positions[symbol] = _pos_entry
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
        Check fill status of pending swing limit orders with a state machine:
          WAITING   → still fresh, no drift
          REQUOTE   → age > requote_age OR drift > requote_drift_bps
          ESCALATE  → past eod_escalation cutoff: cancel + place marketable limit
          CANCEL    → past cancel cutoff: cancel and give up
          FILLED / EXPIRED → handled inline as before
        """
        if not self._pending_limit_orders:
            return

        # Single config read per poll (not per symbol)
        from app.database.agent_config import get_agent_config
        from app.database.session import SessionLocal
        offset_pct = 0.001
        requote_age_min = 30
        requote_drift_bps = 20.0
        max_requotes = 3
        esc_hour = 15
        esc_minute = 15
        cancel_hour = 15
        cancel_minute = 45
        try:
            _db = SessionLocal()
            try:
                offset_pct = float(get_agent_config(_db, "strategy.limit_order_offset_pct") or 0.001)
                requote_age_min = int(get_agent_config(_db, "strategy.limit_order_requote_age_minutes") or 30)
                requote_drift_bps = float(get_agent_config(_db, "strategy.limit_order_requote_drift_bps") or 20)
                max_requotes = int(get_agent_config(_db, "strategy.limit_order_max_requotes") or 3)
                esc_hour = int(get_agent_config(_db, "strategy.limit_order_eod_escalation_hour") or 15)
                esc_minute = int(get_agent_config(_db, "strategy.limit_order_eod_escalation_minute") or 15)
                cancel_hour = int(get_agent_config(_db, "strategy.limit_order_cancel_hour") or 15)
                cancel_minute = int(get_agent_config(_db, "strategy.limit_order_cancel_minute") or 45)
            finally:
                _db.close()
        except Exception as _cfg_exc:
            self.logger.debug("Limit-order config read failed (using defaults): %s", _cfg_exc)

        now = datetime.now(ET)
        past_cancel_cutoff = (now.hour > cancel_hour) or (now.hour == cancel_hour and now.minute >= cancel_minute)
        past_escalation_cutoff = (now.hour > esc_hour) or (now.hour == esc_hour and now.minute >= esc_minute)

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

                # ── FILLED / PARTIALLY_FILLED ─────────────────────────────────
                if order_status in ("filled", "partially_filled") and filled_qty > 0 and filled_price:
                    filled_price = float(filled_price)
                    intended = pending["intended_price"]
                    _pend_short = (
                        pending.get("direction") == "SELL_SHORT"
                        or (pending.get("proposal") or {}).get("direction") == "SELL_SHORT"
                    )
                    if intended > 0:
                        if _pend_short:
                            slippage_bps = round((intended - filled_price) / intended * 10000, 2)
                        else:
                            slippage_bps = round((filled_price - intended) / intended * 10000, 2)
                    else:
                        slippage_bps = 0.0

                    # Phase 78a: cancel unfilled remainder on partial fills to prevent
                    # silent second fills creating untracked additional shares.
                    if order_status == "partially_filled":
                        try:
                            alpaca.cancel_order(order_id)
                            self.logger.info(
                                "PARTIAL_FILL %s: filled %d shares — cancelled remainder",
                                symbol, filled_qty,
                            )
                            await self.log_decision("PARTIAL_FILL_REMAINDER_CANCELLED", reasoning={
                                "symbol": symbol, "filled_qty": filled_qty, "order_id": order_id,
                            })
                        except Exception as _pf_exc:
                            self.logger.warning("Could not cancel partial remainder for %s: %s", symbol, _pf_exc)

                    del self._pending_limit_orders[symbol]
                    self._delete_pending_limit_db(symbol)
                    await self._record_entry(
                        symbol=symbol,
                        shares=filled_qty,
                        filled_price=filled_price,
                        intended_price=intended,
                        slippage_bps=slippage_bps,
                        order_id=order_id,
                        result=pending["result"],
                        proposal=pending["proposal"],
                        pending_trade_id=pending.get("trade_id"),
                    )
                    self.logger.info(
                        "LIMIT FILLED %s x%d @ $%.4f (slip=%.1fbps)",
                        symbol, filled_qty, filled_price, slippage_bps,
                    )
                    continue

                # ── EXPIRED / CANCELED / REJECTED (externally) ────────────────
                if order_status in ("canceled", "expired", "rejected"):
                    del self._pending_limit_orders[symbol]
                    self._delete_pending_limit_db(symbol)
                    self._cancel_pending_fill(pending.get("trade_id"))
                    self.logger.info("Limit order %s for %s cancelled/expired — removing", order_id, symbol)
                    continue

                # ── CANCEL cutoff: hard EOD cancel ────────────────────────────
                if past_cancel_cutoff:
                    try:
                        alpaca.cancel_order(order_id)
                    except Exception as _cx:
                        self.logger.warning("EOD cancel failed for %s: %s", symbol, _cx)
                    del self._pending_limit_orders[symbol]
                    self._delete_pending_limit_db(symbol)
                    self._cancel_pending_fill(pending.get("trade_id"))
                    self.logger.info(
                        "EOD: cancelled unfilled limit order for %s (order=%s)", symbol, order_id
                    )
                    await self.log_decision("LIMIT_EOD_CANCEL", reasoning={
                        "symbol": symbol, "order_id": order_id,
                        "limit_price": pending.get("limit_price"),
                        "requote_count": int(pending.get("requote_count", 0) or 0),
                        "escalated": bool(pending.get("escalated", False)),
                    })
                    continue

                # ── ESCALATE: past escalation cutoff and not yet escalated ────
                already_escalated = bool(pending.get("escalated", False))
                if past_escalation_cutoff and not already_escalated:
                    quote = alpaca.get_quote(symbol)
                    _esc_is_short = pending.get("proposal", {}).get("direction") == "SELL_SHORT"
                    _esc_side_price = quote.get("bid", 0) if _esc_is_short else quote.get("ask", 0)
                    if not quote or _esc_side_price <= 0:
                        self.logger.warning(
                            "Escalation skipped for %s — no quote available", symbol,
                        )
                    else:
                        try:
                            alpaca.cancel_order(order_id)
                        except Exception as _cx:
                            self.logger.warning("Escalation cancel failed for %s: %s", symbol, _cx)
                        ask = float(quote["ask"])
                        _esc_is_short = pending.get("proposal", {}).get("direction") == "SELL_SHORT"
                        # Marketable limit: for longs buy 5bps through ask; for shorts sell 5bps through bid
                        if _esc_is_short:
                            _bid = float(quote.get("bid") or ask)
                            new_limit = round(_bid * (1 - 0.0005), 2)
                            _esc_side = "sell"
                        else:
                            new_limit = round(ask * (1 + 0.0005), 2)
                            _esc_side = "buy"
                        try:
                            new_order = alpaca.place_limit_order(
                                symbol, pending["shares"], _esc_side, new_limit,
                            )
                            new_order_id = new_order.get("order_id")
                            old_limit = pending["limit_price"]
                            pending["order_id"] = new_order_id
                            pending["limit_price"] = new_limit
                            pending["intended_price"] = ask
                            pending["queued_at"] = datetime.now(ET)
                            pending["escalated"] = True
                            self._update_pending_fill_order_id(pending.get("trade_id"), new_order_id)
                            self._save_pending_limit_db(symbol, pending, pending["result"])
                            self.logger.info(
                                "LIMIT ESCALATED %s: ask=%.4f new_limit=%.4f (was=%.4f) order=%s",
                                symbol, ask, new_limit, old_limit, new_order_id,
                            )
                            await self.log_decision("LIMIT_EOD_ESCALATION", reasoning={
                                "symbol": symbol,
                                "old_order_id": order_id,
                                "new_order_id": new_order_id,
                                "old_limit": old_limit,
                                "new_limit": new_limit,
                                "ask": ask,
                                "requote_count": int(pending.get("requote_count", 0) or 0),
                            })
                        except Exception as _ex:
                            self.logger.error("Escalation place_limit_order failed for %s: %s", symbol, _ex)
                    continue

                # ── REQUOTE check: drift OR age ───────────────────────────────
                queued_at = pending.get("queued_at")
                age_minutes = 0.0
                if queued_at is not None:
                    try:
                        # queued_at may be naive (from DB) or tz-aware (from in-memory)
                        if queued_at.tzinfo is None:
                            age_minutes = (now.replace(tzinfo=None) - queued_at).total_seconds() / 60.0
                        else:
                            age_minutes = (now - queued_at).total_seconds() / 60.0
                    except Exception:
                        age_minutes = 0.0

                requote_count = int(pending.get("requote_count", 0) or 0)
                age_trigger = age_minutes >= requote_age_min
                drift_bps = 0.0
                drift_trigger = False
                fresh_ask = None
                if requote_count < max_requotes and not already_escalated:
                    quote = alpaca.get_quote(symbol)
                    _rq_is_short_check = pending.get("proposal", {}).get("direction") == "SELL_SHORT"
                    _rq_ref_price = quote.get("bid", 0) if _rq_is_short_check else quote.get("ask", 0)
                    if quote and _rq_ref_price > 0:
                        fresh_ask = float(quote["ask"])  # kept for compat; actual limit uses side-aware price below
                        old_limit = float(pending["limit_price"])
                        if old_limit > 0:
                            drift_bps = (float(_rq_ref_price) - old_limit) / old_limit * 10000.0
                        drift_trigger = drift_bps >= requote_drift_bps

                if requote_count < max_requotes and not already_escalated and (age_trigger or drift_trigger):
                    if fresh_ask is None:
                        # Could not get a quote — skip this cycle
                        continue
                    try:
                        alpaca.cancel_order(order_id)
                    except Exception as _cx:
                        self.logger.warning("Re-quote cancel failed for %s: %s", symbol, _cx)
                    _rq_is_short = pending.get("proposal", {}).get("direction") == "SELL_SHORT"
                    if _rq_is_short:
                        _fresh_bid = float(alpaca.get_quote(symbol).get("bid") or fresh_ask)
                        new_limit = round(_fresh_bid * (1 + offset_pct), 2)
                        _rq_side = "sell"
                    else:
                        new_limit = round(fresh_ask * (1 - offset_pct), 2)
                        _rq_side = "buy"
                    try:
                        new_order = alpaca.place_limit_order(
                            symbol, pending["shares"], _rq_side, new_limit,
                        )
                        new_order_id = new_order.get("order_id")
                        old_limit = pending["limit_price"]
                        pending["order_id"] = new_order_id
                        pending["limit_price"] = new_limit
                        pending["intended_price"] = fresh_ask
                        pending["queued_at"] = datetime.now(ET)
                        pending["requote_count"] = requote_count + 1
                        self._update_pending_fill_order_id(pending.get("trade_id"), new_order_id)
                        self._save_pending_limit_db(symbol, pending, pending["result"])
                        self.logger.info(
                            "LIMIT REQUOTE %s #%d: ask=%.4f new_limit=%.4f (was=%.4f) "
                            "age=%.1fm drift=%.1fbps order=%s",
                            symbol, requote_count + 1, fresh_ask, new_limit, old_limit,
                            age_minutes, drift_bps, new_order_id,
                        )
                        await self.log_decision("LIMIT_REQUOTE", reasoning={
                            "symbol": symbol,
                            "old_order_id": order_id,
                            "new_order_id": new_order_id,
                            "old_limit": old_limit,
                            "new_limit": new_limit,
                            "ask": fresh_ask,
                            "age_minutes": round(age_minutes, 1),
                            "drift_bps": round(drift_bps, 1),
                            "requote_count": requote_count + 1,
                            "trigger": "age" if age_trigger else "drift",
                        })
                    except Exception as _ex:
                        self.logger.error("Re-quote place_limit_order failed for %s: %s", symbol, _ex)
                    continue

                # else: WAITING — nothing to do this poll

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
                _q = alpaca.get_quote(symbol)
                current_price = (_q["mid"] if _q else None) or alpaca.get_latest_price(symbol)
                if current_price is None:
                    continue
                bars = alpaca.get_bars(symbol, timeframe="1Day", limit=20)
                if bars is None or bars.empty:
                    continue
                from app.strategy.signals import _calc_atr
                atr_val = _calc_atr(bars["high"], bars["low"], bars["close"], 14)
                if not atr_val:
                    continue
                _rt_is_short = pos.get("direction") == "SELL_SHORT"
                if _rt_is_short:
                    # Short: tighten stop downward (stop is above entry)
                    tight_stop = round(current_price + 1.0 * atr_val, 4)
                    _stop_better = tight_stop < pos["stop_price"]
                else:
                    tight_stop = round(current_price - 1.0 * atr_val, 4)
                    _stop_better = tight_stop > pos["stop_price"]
                if _stop_better:
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
        # Trend (TSMOM) sleeve positions are managed EXCLUSIVELY by the weekly
        # rebalancer (app/live_trading/trend_sleeve.py) — never stop/target/trail
        # exit them here, or the synthetic stops the reconciler attaches would
        # liquidate the sleeve mid-week and fight the rebalancer.
        if pos.get("trade_type") == "trend" or pos.get("selector") == "trend":
            return
        # Prefer live NBBO mid-quote over last minute bar — IEX minute bars can be
        # 10-30 min stale for low-volume names, causing stops/targets to miss.
        quote = alpaca.get_quote(symbol)
        current_price = (quote["mid"] if quote else None) or alpaca.get_latest_price(symbol)
        if current_price is None:
            return

        now = datetime.now(ET)
        is_short = pos.get("direction") == "SELL_SHORT"

        # For longs: track highest price for trailing stop.
        # For shorts: track lowest price (the "best" level — equivalent of highest for longs).
        if is_short:
            highest = min(pos["highest_price"], current_price)
        else:
            highest = max(pos["highest_price"], current_price)
        pos["highest_price"] = highest

        # Periodic position status log — every ~30 min so we have a paper trail
        _last_log = pos.get("_last_status_log", 0)
        if now.timestamp() - _last_log >= 1800:
            pos["_last_status_log"] = now.timestamp()
            if is_short:
                pnl_now = (pos["entry_price"] - current_price) / pos["entry_price"] * 100
            else:
                pnl_now = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
            self.logger.info(
                "POSITION STATUS %s | price=$%.2f entry=$%.2f stop=$%.2f target=$%.2f | "
                "pnl=%.1f%% bars_held=%d",
                symbol, current_price, pos["entry_price"],
                pos.get("stop_price", 0), pos.get("target_price", 0),
                pnl_now, pos.get("bars_held", 0),
            )

        # bars_held = trading DAYS held, not heartbeat ticks.
        # Only increment when the calendar date advances (once per day).
        today_date = now.date()
        if pos.get("_last_bar_date") != today_date:
            pos["_last_bar_date"] = today_date
            pos["bars_held"] += 1
            # Persist so bars_held survives restarts (DB value only written on close otherwise)
            try:
                from app.database.session import get_session as _gs
                with _gs() as _db:
                    _t = _db.query(Trade).filter_by(id=pos["trade_id"]).first()
                    if _t:
                        _t.bars_held = pos["bars_held"]
                        _db.commit()
            except Exception:
                pass

        # Adverse-move warning: down >3% for longs, up >3% for shorts
        if is_short:
            pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]
        else:
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        if pnl_pct <= -0.03 and not pos.get("_adverse_warned"):
            pos["_adverse_warned"] = True
            if is_short:
                # For shorts: stop is above entry; tighten toward entry
                breakeven_stop = round(pos["entry_price"] * 1.005, 4)
                if breakeven_stop < pos["stop_price"]:
                    pos["stop_price"] = breakeven_stop
            else:
                breakeven_stop = round(pos["entry_price"] * 0.995, 4)
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

        # ── Dynamic adjustments: partial exit, stop trail, target extension ────
        if pos.get("atr", 0) > 0 and not pos.get("_partial_exited"):
            from app.strategy.signals import check_dynamic_adjustments
            vix_now = 0.0
            try:
                from app.agents.circuit_breaker import circuit_breaker as _cb
                vix_now = float(_cb._last_vix or 0.0)
            except Exception:
                pass

            adj = check_dynamic_adjustments(
                symbol=symbol,
                current_price=current_price,
                entry_price=pos["entry_price"],
                stop_price=pos["stop_price"],
                target_price=pos["target_price"],
                highest_price=highest,
                shares=pos.get("shares", 1),
                atr=pos["atr"],
                trade_type=pos.get("trade_type", "swing"),
                vix=vix_now,
                direction=pos.get("direction", "BUY"),
            )

            # Apply stop update — for longs stop moves up; for shorts stop moves down
            _stop_improved = (
                adj.new_stop < pos["stop_price"] if is_short
                else adj.new_stop > pos["stop_price"]
            )
            if _stop_improved:
                if adj.stop_tightened:
                    db = get_session()
                    try:
                        from app.startup_reconciler import write_target_stop
                        trade = db.query(Trade).filter_by(id=pos.get("trade_id")).first()
                        if trade:
                            wrote = write_target_stop(
                                trade, stop_price=adj.new_stop,
                                written_by="trader.stop_tighten",
                                reason=adj.notes,
                            )
                            if wrote:
                                db.commit()
                                pos["stop_price"] = adj.new_stop
                                self.logger.info("%s: stop %s → $%.2f (%s)", symbol,
                                                 "tightened" if vix_now > 20 else "trailed",
                                                 adj.new_stop, adj.notes)
                            else:
                                db.rollback()
                                self.logger.error(
                                    "%s: stop tighten REJECTED ($%.4f) — see ERROR above",
                                    symbol, adj.new_stop,
                                )
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()
                else:
                    # Non-persisted trailing stop update (in-memory only — no DB write)
                    pos["stop_price"] = adj.new_stop

            # Apply target extension (longs extend upward; shorts extend downward)
            _target_extended = (
                adj.target_extended and (
                    adj.new_target < pos["target_price"] if is_short
                    else adj.new_target > pos["target_price"]
                )
            )
            if _target_extended:
                db = get_session()
                try:
                    from app.startup_reconciler import write_target_stop
                    trade = db.query(Trade).filter_by(id=pos.get("trade_id")).first()
                    if trade:
                        wrote = write_target_stop(
                            trade, target_price=adj.new_target,
                            written_by="trader.target_extend",
                            reason=adj.notes,
                        )
                        if wrote:
                            db.commit()
                            pos["target_price"] = adj.new_target
                            self.logger.info("%s: target extended → $%.2f (%s)",
                                             symbol, adj.new_target, adj.notes)
                        else:
                            db.rollback()
                            self.logger.error(
                                "%s: target extend REJECTED ($%.4f) — see ERROR above",
                                symbol, adj.new_target,
                            )
                except Exception:
                    db.rollback()
                finally:
                    db.close()

            # Execute partial exit if triggered
            if adj.partial_exit and adj.partial_exit_qty > 0:
                await self._execute_partial_exit(
                    symbol, current_price, pos["atr"], alpaca
                )

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
                    elif (is_short and current_price > pos["stop_price"] * 0.995) or (
                        not is_short and current_price < pos["stop_price"] * 1.005
                    ):
                        weakness_reason = "approaching_stop"

                    if weakness_reason:
                        pos["_reeval_sent"] = True
                        await self._send_reeval_request(symbol, weakness_reason, current_price)
            except Exception:
                pass  # reeval is best-effort; don't let it block exit logic

        # Per-position max_hold overrides global config (e.g. PEAD hold-5)
        max_hold = pos.get("max_hold_days") or 20
        try:
            from app.database.agent_config import get_agent_config
            _db = get_session()
            try:
                global_max = int(get_agent_config(_db, "strategy.max_hold_bars") or 20)
                if not pos.get("max_hold_days"):
                    max_hold = global_max
            finally:
                _db.close()
        except Exception:
            pass

        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        during_market_hours = market_open <= now <= market_close

        should_exit, reason, new_stop = check_exit(
            symbol=symbol,
            current_price=current_price,
            entry_price=pos["entry_price"],
            stop_price=pos["stop_price"],
            target_price=pos["target_price"],
            highest_price=highest,
            bars_held=pos["bars_held"],
            max_hold_bars=max_hold,
            direction=pos.get("direction", "BUY"),
        )
        pos["stop_price"] = new_stop  # keep trailing stop current

        # Never exit during pre-market / after-hours — only stop/target during market hours
        if should_exit and not during_market_hours:
            self.logger.debug(
                "%s: exit signal '%s' suppressed outside market hours (%s ET)",
                symbol, reason, now.strftime("%H:%M"),
            )
            should_exit = False

        if should_exit:
            self.logger.info(
                "EXIT TRIGGERED %s | reason=%s price=$%.2f stop=$%.2f target=$%.2f",
                symbol, reason, current_price, pos.get("stop_price", 0), pos.get("target_price", 0),
            )
            await self._execute_exit(symbol, current_price, reason, alpaca)
        else:
            self.logger.debug(
                "EXIT CHECK %s | price=$%.2f stop=$%.2f target=$%.2f — holding",
                symbol, current_price, pos.get("stop_price", 0), pos.get("target_price", 0),
            )

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
        raw_qty = int(position.get("qty", 0))
        total_qty = abs(raw_qty)
        if total_qty == 0:
            return

        partial_qty = max(1, int(total_qty * partial_pct))
        if partial_qty >= total_qty:
            # Would exit everything — let normal exit logic handle it
            pos["_partial_exited"] = False
            return

        _partial_is_short = pos.get("direction") == "SELL_SHORT"
        _partial_side = "buy" if _partial_is_short else "sell"
        order = alpaca.place_market_order(symbol, partial_qty, _partial_side)
        if not order:
            self.logger.error("Partial exit order failed for %s", symbol)
            pos["_partial_exited"] = False
            return

        if _partial_is_short:
            pnl = (pos["entry_price"] - current_price) * partial_qty
            # Move stop toward entry for shorts: stop is above entry, clamp it down toward entry
            breakeven_stop = round(pos["entry_price"] * 1.001, 4)
            pos["stop_price"] = min(pos["stop_price"], breakeven_stop)
        else:
            pnl = (current_price - pos["entry_price"]) * partial_qty
            # Move stop on remaining shares to near-breakeven (just below entry for longs)
            breakeven_stop = round(pos["entry_price"] * 0.999, 4)
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
                # Accumulate partial pnl on the trade record so final pnl is complete
                trade = db.query(Trade).filter_by(id=pos["trade_id"]).first()
                if trade:
                    trade.pnl = (trade.pnl or 0.0) + pnl
                    pos["_partial_pnl"] = (pos.get("_partial_pnl") or 0.0) + pnl
                db.commit()
        except Exception as exc:
            db.rollback()
            self.logger.error("Failed to record partial exit for %s: %s", symbol, exc)
        finally:
            db.close()

        # Reg T / T+1: partial long-sale proceeds are unsettled until next business day
        if not _partial_is_short:
            from app.agents.compliance import compliance_tracker
            compliance_tracker.record_sale_proceeds(current_price * partial_qty)

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
                        _fc_qty = pos.get("shares", trade.quantity or 0)
                        _fc_dir = pos.get("direction", getattr(trade, "direction", "BUY") or "BUY")
                        if _fc_dir == "SELL_SHORT":
                            _final_leg = (pos["entry_price"] - actual_price) * _fc_qty
                        else:
                            _final_leg = (actual_price - pos["entry_price"]) * _fc_qty
                        # Accumulate any partial-exit P&L.  Use in-memory cache first;
                        # fall back to Order ledger for robustness on stale in-memory state.
                        from app.database.models import recompute_partial_pnl
                        _partial = pos.get("_partial_pnl") or recompute_partial_pnl(
                            db, trade.id, pos["entry_price"], _fc_dir
                        )
                        trade.pnl = _final_leg + _partial
                        trade.status = "FORCE_CLOSED_NO_POSITION"
                        trade.exit_reason = "force_closed_no_position"
                        trade.closed_at = datetime.now(ET)
                        db.commit()
                except Exception as e:
                    db.rollback()
                    self.logger.error("Failed to close orphaned trade %s: %s", trade_id, e)
                finally:
                    db.close()
            self.active_positions.pop(symbol, None)
            return

        raw_qty = int(position.get("qty", 0))
        qty = abs(raw_qty)
        if qty == 0:
            self.active_positions.pop(symbol, None)
            return

        # Longs exit with "sell"; shorts cover with "buy"
        exit_side = "buy" if pos.get("direction") == "SELL_SHORT" else "sell"
        order = alpaca.place_market_order(symbol, qty, exit_side)
        if not order:
            self.logger.error("Exit order failed for %s", symbol)
            return

        if pos.get("direction") == "SELL_SHORT":
            final_leg_pnl = (pos["entry_price"] - current_price) * qty
        else:
            final_leg_pnl = (current_price - pos["entry_price"]) * qty
        partial_pnl = pos.get("_partial_pnl") or 0.0
        pnl = final_leg_pnl + partial_pnl
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
                    trade.exit_reason = _normalise_exit_reason(reason)

            db_order = Order(
                trade_id=trade_id,
                order_type="EXIT",
                order_id=order.get("order_id"),
                status="FILLED",
                filled_price=current_price,
                filled_qty=qty,
            )
            db.add(db_order)

            # Append TRADE_CLOSED event to ProposalEvent log
            proposal_uuid = pos.get("proposal_uuid")
            if proposal_uuid:
                from app.database.models import ProposalEvent
                entry = pos.get("entry_price", 0)
                if entry > 0:
                    _closed_is_short = pos.get("direction") == "SELL_SHORT"
                    _raw = (entry - current_price) if _closed_is_short else (current_price - entry)
                    pnl_pct = round(_raw / entry * 100, 3)
                else:
                    pnl_pct = None
                db.add(ProposalEvent(
                    proposal_uuid=proposal_uuid,
                    event_time=datetime.now(ET),
                    actor="trader",
                    event_type="TRADE_CLOSED",
                    details={
                        "exit_price": current_price,
                        "pnl": round(pnl, 2),
                        "pnl_pct": pnl_pct,
                        "exit_reason": _normalise_exit_reason(reason),
                        "bars_held": pos.get("bars_held", 0),
                        "alpaca_order_id": order.get("order_id"),
                    },
                ))

            db.commit()

            # Phase 2d: log live-vs-sim shortfall at close (non-fatal)
            try:
                if trade_id and pnl is not None:
                    _trade_type_2d = pos.get("trade_type", "swing")
                    from app.ml.walk_forward_stats import get_predicted_pnl as _wf_pnl2
                    _predicted = _wf_pnl2(_trade_type_2d)
                    if _predicted is not None:
                        _shortfall = round(pnl - _predicted * (pos.get("shares", qty) or qty), 4)
                        self.logger.info(
                            "2d shortfall %s trade#%d: live_pnl=%.2f predicted_per_share=%.4f shortfall=%.2f",
                            symbol, trade_id, pnl, _predicted, _shortfall,
                        )
            except Exception as _sf_err:
                self.logger.debug("shortfall log failed (non-fatal): %s", _sf_err)

            trade_type = self.active_positions.get(symbol, {}).get("trade_type", "swing")
            self.active_positions.pop(symbol, None)
            self.approved_symbols.pop(symbol, None)  # prevent re-entry from stale proposal

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
                # Settlement: only long closes are sale proceeds (short covers are cash outflows)
                if pos.get("direction") != "SELL_SHORT":
                    proceeds = current_price * qty
                    compliance_tracker.record_sale_proceeds(proceeds)
                compliance_tracker.sweep_settled()
                # Wash sale: flag if closed at a loss
                if pnl < 0:
                    compliance_tracker.record_loss_close(symbol, direction=pos.get("direction", "BUY"))
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

        from app.integrations import get_alpaca_client
        alpaca = get_alpaca_client()

        # Fix 2: fetch live Alpaca positions once; use it to distinguish real vs stale DB rows
        try:
            alpaca_positions = {p["symbol"] for p in (alpaca.get_positions() or [])}
        except Exception as exc:
            self.logger.warning("force_close: could not fetch Alpaca positions — assuming all real: %s", exc)
            alpaca_positions = None  # None = unknown; proceed conservatively

        # Also check DB for intraday trades not yet in in-memory state (e.g. set via manual DB update)
        stale_ghost_symbols: list = []
        try:
            from app.database.session import SessionLocal
            from app.database.models import Trade
            db = SessionLocal()
            try:
                db_intraday = db.query(Trade).filter(
                    Trade.status == "ACTIVE",
                    Trade.trade_type == "intraday",
                ).all()
                for t in db_intraday:
                    if t.symbol not in intraday_symbols:
                        # Fix 2: if Alpaca has no position for this symbol, it's a stale ghost — mark it, don't trade it
                        if alpaca_positions is not None and t.symbol not in alpaca_positions:
                            self.logger.warning(
                                "force_close: %s is ACTIVE intraday in DB but NOT in Alpaca — marking RECONCILE_GHOST",
                                t.symbol,
                            )
                            stale_ghost_symbols.append(t.symbol)
                            t.status = "RECONCILE_GHOST"
                            t.exit_reason = "force_close_ghost_no_alpaca_position"
                        else:
                            self.logger.warning(
                                "force_close: %s in DB as intraday ACTIVE but not in active_positions — adding",
                                t.symbol,
                            )
                            intraday_symbols.append(t.symbol)
                            # Seed a minimal in-memory entry so _execute_exit can close it
                            if t.symbol not in self.active_positions:
                                self.active_positions[t.symbol] = {
                                    "trade_type": "intraday",
                                    "entry_price": float(t.entry_price or 0),
                                    "stop_price": float(t.stop_price or 0),
                                    "target_price": float(t.target_price or 0),
                                    "shares": int(t.quantity or 0),
                                    "trade_id": t.id,
                                    "direction": getattr(t, "direction", "BUY") or "BUY",
                                }
                db.commit()
            finally:
                db.close()
        except Exception as exc:
            self.logger.warning("force_close: DB check failed: %s", exc)

        if stale_ghost_symbols:
            self.logger.warning(
                "force_close: marked %d stale ghost(s) as RECONCILE_GHOST (no Alpaca position): %s",
                len(stale_ghost_symbols), stale_ghost_symbols,
            )

        # Always reset the RM intraday counter — includes slots consumed by ghost rows
        try:
            from app.agents.risk_manager import risk_manager
            risk_manager.reset_intraday_count()
        except Exception:
            pass

        if not intraday_symbols:
            return

        self.logger.warning(
            "3:45 PM force-close: closing %d intraday position(s): %s",
            len(intraday_symbols), intraday_symbols,
        )

        exited: list[str] = []
        for symbol in intraday_symbols:
            try:
                await self._execute_exit(symbol, alpaca.get_latest_price(symbol) or 0,
                                         "FORCE_CLOSE_EOD", alpaca)
                exited.append(symbol)
            except Exception as exc:
                self.logger.error("Force-close failed for %s: %s", symbol, exc)

        # Only log symbols that actually exited — avoids ghost entries for positions
        # that no longer exist in Alpaca (stale in-memory state).
        if exited:
            await self.log_decision(
                "INTRADAY_FORCE_CLOSED",
                reasoning={"symbols": exited, "count": len(exited)},
            )


# Module-level singleton
trader = Trader()
