"""
Compliance & Regulatory Guardrails — Phase 21.

Tracks:
  21.1  PDT (Pattern Day Trader) rule — 3 round-trips per 5-business-day window
         when account equity < $25,000. Blocks new intraday entries at count=2.
  21.2  Wash sale awareness — flags re-entry within 30 calendar days of a loss close.
         Warning only (does not hard-block); visible in trade record and dashboard.
  21.3  Settlement tracking (Reg T, T+1) — distinguishes settled vs unsettled cash.
         RM uses settled cash for buying power to prevent free-riding.
  21.4  Symbol-level halt list — instant block of individual tickers without restart.
         Complements existing strategy-level pauses in circuit_breaker.py.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PDT_EQUITY_THRESHOLD = 25_000.0   # below this, PDT rules apply
PDT_MAX_DAY_TRADES = 3            # round-trips allowed per 5-business-day window
PDT_WARN_AT = 2                   # warn (and block new intraday) when count reaches this
PDT_WINDOW_DAYS = 5               # rolling business-day window

WASH_SALE_DAYS = 30               # flag re-entry within this many calendar days
SETTLEMENT_DAYS = 1               # T+1 settlement (since May 2024)


def _business_days_back(n: int) -> date:
    """Return the date n business days before today."""
    d = date.today()
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri
            count += 1
    return d


class ComplianceTracker:
    """
    Singleton compliance tracker shared across agents.

    Thread-safe (all mutable state guarded by a Lock).
    State is in-memory; DB is queried on demand for historical checks.
    """

    def __init__(self):
        self._lock = Lock()

        # PDT state — reset each calendar day
        # day_trades: {date_str: [symbol, ...]} (each entry = one round-trip)
        self._day_trades: Dict[str, List[str]] = {}

        # Wash sale tracking — {symbol: last_loss_close_date}
        self._loss_closes: Dict[str, date] = {}

        # Unsettled cash entries — [(settle_date, amount)]
        self._unsettled: List[Tuple[date, float]] = []

        # Symbol halt list — {symbol: reason}
        self._halted_symbols: Dict[str, str] = {}

    # ─── 21.1  PDT Rule ───────────────────────────────────────────────────────

    def record_day_trade(self, symbol: str) -> None:
        """Record a completed round-trip (buy + sell same session) for PDT tracking."""
        today = date.today().isoformat()
        with self._lock:
            self._day_trades.setdefault(today, []).append(symbol)
        logger.info("PDT: day trade recorded — %s (today total: %d)",
                    symbol, self.day_trade_count_today())

    def day_trade_count_today(self) -> int:
        """Number of round-trips completed today."""
        today = date.today().isoformat()
        with self._lock:
            return len(self._day_trades.get(today, []))

    def day_trade_count_window(self) -> int:
        """
        Total round-trips in the rolling 5-business-day window.
        Includes today.
        """
        cutoff = _business_days_back(PDT_WINDOW_DAYS)
        with self._lock:
            total = 0
            for date_str, trades in self._day_trades.items():
                try:
                    d = date.fromisoformat(date_str)
                    if d >= cutoff:
                        total += len(trades)
                except ValueError:
                    continue
        return total

    def is_pdt_blocked(self, account_equity: float) -> Tuple[bool, str]:
        """
        Return (blocked, reason).
        Blocked if equity < $25k AND rolling day-trade count >= PDT_WARN_AT.
        """
        if account_equity >= PDT_EQUITY_THRESHOLD:
            return False, f"equity ${account_equity:,.0f} ≥ PDT threshold — PDT check skipped"
        count = self.day_trade_count_window()
        if count >= PDT_WARN_AT:
            return True, (
                f"PDT limit: {count} day trades in 5-day window "
                f"(limit {PDT_MAX_DAY_TRADES}, warn at {PDT_WARN_AT}) — "
                f"blocking new intraday entries until window resets"
            )
        return False, f"PDT: {count}/{PDT_MAX_DAY_TRADES} day trades in window — OK"

    def load_day_trades_from_db(self, db) -> None:
        """Populate in-memory PDT state from DB on startup."""
        from app.database.models import Trade
        cutoff = _business_days_back(PDT_WINDOW_DAYS)
        try:
            trades = (
                db.query(Trade)
                .filter(Trade.status == "CLOSED")
                .filter(Trade.closed_at >= datetime.combine(cutoff, datetime.min.time()))
                .all()
            )
            with self._lock:
                for t in trades:
                    if t.closed_at is None or t.entry_price is None or t.exit_price is None:
                        continue
                    close_date = t.closed_at.date().isoformat()
                    # Day trade = opened and closed same calendar day
                    if t.created_at and t.created_at.date() == t.closed_at.date():
                        self._day_trades.setdefault(close_date, []).append(t.symbol)
            logger.info("PDT state loaded: %d day trades in window", self.day_trade_count_window())
        except Exception as exc:
            logger.warning("Could not load PDT state from DB: %s", exc)

    # ─── 21.2  Wash Sale Awareness ────────────────────────────────────────────

    def record_loss_close(self, symbol: str, close_date: Optional[date] = None) -> None:
        """Record that symbol was closed at a loss today (or given date)."""
        d = close_date or date.today()
        with self._lock:
            self._loss_closes[symbol] = d
        logger.info("Wash sale: recorded loss close for %s on %s", symbol, d)

    def check_wash_sale(self, symbol: str) -> Tuple[bool, str]:
        """
        Return (is_wash_sale_window, message).
        True if the symbol was sold at a loss within WASH_SALE_DAYS calendar days.
        This is a warning — caller decides whether to block or proceed.
        """
        with self._lock:
            loss_date = self._loss_closes.get(symbol)
        if loss_date is None:
            return False, f"{symbol}: no recent loss close — wash sale clear"
        days_since = (date.today() - loss_date).days
        if days_since <= WASH_SALE_DAYS:
            return True, (
                f"WASH SALE WARNING: {symbol} was closed at a loss {days_since} days ago "
                f"(on {loss_date}). Re-entry within {WASH_SALE_DAYS} days may disallow "
                f"the loss deduction. Proceeding — flag for tax review."
            )
        return False, f"{symbol}: loss close {days_since} days ago — outside wash sale window"

    def load_loss_closes_from_db(self, db) -> None:
        """Populate wash sale state from DB on startup."""
        from app.database.models import Trade
        cutoff = date.today() - timedelta(days=WASH_SALE_DAYS)
        try:
            trades = (
                db.query(Trade)
                .filter(Trade.status == "CLOSED")
                .filter(Trade.pnl < 0)
                .filter(Trade.closed_at >= datetime.combine(cutoff, datetime.min.time()))
                .all()
            )
            with self._lock:
                for t in trades:
                    if t.closed_at:
                        self._loss_closes[t.symbol] = t.closed_at.date()
            logger.info("Wash sale state loaded: %d loss positions", len(self._loss_closes))
        except Exception as exc:
            logger.warning("Could not load wash sale state from DB: %s", exc)

    # ─── 21.3  Settlement Tracking (Reg T) ───────────────────────────────────

    def record_sale_proceeds(self, amount: float, trade_date: Optional[date] = None) -> None:
        """
        Record sale proceeds as unsettled cash.
        Equities settle T+1 (since May 2024).
        """
        settle_date = (trade_date or date.today()) + timedelta(days=SETTLEMENT_DAYS)
        with self._lock:
            self._unsettled.append((settle_date, amount))
        logger.debug("Settlement: $%.2f to settle on %s", amount, settle_date)

    def unsettled_cash(self) -> float:
        """Total cash that has not yet settled."""
        today = date.today()
        with self._lock:
            return sum(amt for settle_dt, amt in self._unsettled if settle_dt > today)

    def settled_buying_power(self, total_buying_power: float) -> float:
        """
        Return buying power from settled funds only.
        = total_buying_power - unsettled_cash
        Never returns negative.
        """
        return max(0.0, total_buying_power - self.unsettled_cash())

    def sweep_settled(self) -> None:
        """Remove entries whose settlement date has passed (call periodically)."""
        today = date.today()
        with self._lock:
            self._unsettled = [(d, a) for d, a in self._unsettled if d > today]

    # ─── 21.4  Symbol-Level Halt List ────────────────────────────────────────

    def halt_symbol(self, symbol: str, reason: str) -> None:
        """Add a symbol to the halt list. Immediate effect — no restart needed."""
        with self._lock:
            self._halted_symbols[symbol] = reason
        logger.warning("SYMBOL HALTED: %s — %s", symbol, reason)

    def resume_symbol(self, symbol: str) -> None:
        """Remove a symbol from the halt list."""
        with self._lock:
            self._halted_symbols.pop(symbol, None)
        logger.info("Symbol %s removed from halt list", symbol)

    def is_symbol_halted(self, symbol: str) -> Tuple[bool, str]:
        """Return (halted, reason)."""
        with self._lock:
            reason = self._halted_symbols.get(symbol)
        if reason:
            return True, f"{symbol} is halted: {reason}"
        return False, f"{symbol}: not halted"

    @property
    def halted_symbols(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._halted_symbols)

    def status(self) -> dict:
        """Return a summary dict for dashboard/API."""
        return {
            "pdt_day_trades_today": self.day_trade_count_today(),
            "pdt_day_trades_window": self.day_trade_count_window(),
            "pdt_equity_threshold": PDT_EQUITY_THRESHOLD,
            "unsettled_cash": self.unsettled_cash(),
            "wash_sale_symbols": list(self._loss_closes.keys()),
            "halted_symbols": self.halted_symbols,
        }


# Module-level singleton
compliance_tracker = ComplianceTracker()
