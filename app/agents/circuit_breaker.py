"""
Circuit Breaker — auto-pauses trading when the system is in distress.

Three triggers:
  1. Consecutive losing trades  : 3 losses in a row → OPEN (trading paused)
  2. High market volatility      : VIX > 30 (fetched via yfinance) → OPEN
  3. Alpaca network errors       : > 5 errors in 10 minutes → OPEN

State machine:
  CLOSED  → normal operation
  OPEN    → trading paused
  HALF_OPEN (future)  → optional recovery probe

Usage:
    circuit_breaker.record_trade_result(won=False)
    circuit_breaker.record_network_error()
    if circuit_breaker.is_open:
        return  # don't trade

The orchestrator calls circuit_breaker.check_market_volatility() once per
scan cycle (not on every trade).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

CONSECUTIVE_LOSS_LIMIT = 3
VIX_PAUSE_THRESHOLD = 30.0
NETWORK_ERROR_LIMIT = 5
NETWORK_ERROR_WINDOW_SECONDS = 600   # 10 minutes
VIX_CHECK_INTERVAL_SECONDS = 300     # re-fetch VIX at most every 5 min

# Strategy-level circuit breaker: pause a strategy when win rate drops
STRATEGY_WIN_RATE_WINDOW = 20        # rolling N trades to evaluate win rate
STRATEGY_WIN_RATE_FLOOR  = 0.40      # below this → pause the strategy


class CircuitBreaker:
    def __init__(self):
        self._lock = Lock()
        self._is_open = False
        self._open_reason: Optional[str] = None
        self._open_at: Optional[datetime] = None

        # Consecutive loss tracking (global)
        self._consecutive_losses = 0

        # Network error tracking
        self._error_timestamps: list[float] = []

        # VIX cache
        self._last_vix_check: float = 0.0
        self._last_vix: Optional[float] = None

        # Per-strategy win/loss history and pause state
        # strategy → {"results": [bool, ...], "paused": bool, "paused_reason": str|None}
        self._strategy_state: dict = {
            "swing":    {"results": [], "paused": False, "paused_reason": None},
            "intraday": {"results": [], "paused": False, "paused_reason": None},
        }

    # ── State ──────────────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_closed(self) -> bool:
        return not self._is_open

    def status(self) -> dict:
        strategy_status = {
            s: {
                "paused": state["paused"],
                "paused_reason": state["paused_reason"],
                "recent_win_rate": self._strategy_win_rate(s),
                "trades_tracked": len(state["results"]),
            }
            for s, state in self._strategy_state.items()
        }
        return {
            "is_open":            self._is_open,
            "open_reason":        self._open_reason,
            "open_at":            self._open_at.isoformat() if self._open_at else None,
            "consecutive_losses": self._consecutive_losses,
            "recent_errors":      self._count_recent_errors(),
            "last_vix":           self._last_vix,
            "strategies":         strategy_status,
        }

    def is_strategy_paused(self, strategy: str) -> bool:
        """Return True if the given strategy (swing/intraday) is individually paused."""
        return self._strategy_state.get(strategy, {}).get("paused", False)

    def pause_strategy(self, strategy: str, reason: str = "manual") -> None:
        """Manually pause a specific strategy without affecting others."""
        with self._lock:
            if strategy in self._strategy_state:
                self._strategy_state[strategy]["paused"] = True
                self._strategy_state[strategy]["paused_reason"] = reason
                logger.warning("Strategy '%s' PAUSED — reason: %s", strategy, reason)

    def resume_strategy(self, strategy: str) -> None:
        """Resume a paused strategy."""
        with self._lock:
            if strategy in self._strategy_state:
                self._strategy_state[strategy]["paused"] = False
                self._strategy_state[strategy]["paused_reason"] = None
                self._strategy_state[strategy]["results"].clear()
                logger.info("Strategy '%s' RESUMED", strategy)

    def _strategy_win_rate(self, strategy: str) -> Optional[float]:
        results = self._strategy_state.get(strategy, {}).get("results", [])
        if len(results) < 5:
            return None  # not enough data
        window = results[-STRATEGY_WIN_RATE_WINDOW:]
        return round(sum(window) / len(window), 3)

    # ── Triggers ───────────────────────────────────────────────────────────────

    def record_trade_result(self, won: bool, strategy: str = "swing"):
        """
        Call after every closed trade.
        Tracks both the global consecutive-loss counter and per-strategy win rate.
        """
        with self._lock:
            # Global consecutive loss tracking
            if won:
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1
                if self._consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
                    self._trip(f"consecutive_losses={self._consecutive_losses}")

            # Per-strategy win rate tracking
            if strategy in self._strategy_state:
                results = self._strategy_state[strategy]["results"]
                results.append(bool(won))
                # Keep only last N results
                if len(results) > STRATEGY_WIN_RATE_WINDOW * 2:
                    self._strategy_state[strategy]["results"] = results[-STRATEGY_WIN_RATE_WINDOW:]

                # Check if strategy should be paused
                win_rate = self._strategy_win_rate(strategy)
                if (
                    win_rate is not None
                    and win_rate < STRATEGY_WIN_RATE_FLOOR
                    and not self._strategy_state[strategy]["paused"]
                ):
                    reason = f"win_rate={win_rate:.0%}<{STRATEGY_WIN_RATE_FLOOR:.0%}_over_{STRATEGY_WIN_RATE_WINDOW}_trades"
                    self._strategy_state[strategy]["paused"] = True
                    self._strategy_state[strategy]["paused_reason"] = reason
                    logger.warning(
                        "Strategy '%s' auto-paused — %s", strategy, reason
                    )

    def record_network_error(self):
        """Call each time an Alpaca API call fails."""
        with self._lock:
            now = time.monotonic()
            self._error_timestamps.append(now)
            if self._count_recent_errors() > NETWORK_ERROR_LIMIT:
                self._trip(
                    f"network_errors>{NETWORK_ERROR_LIMIT} in {NETWORK_ERROR_WINDOW_SECONDS}s"
                )

    def check_market_volatility(self) -> bool:
        """
        Fetch VIX from yfinance and trip if > threshold.
        Returns True if the breaker tripped.
        Caches the result for VIX_CHECK_INTERVAL_SECONDS to avoid hammering.
        """
        now = time.monotonic()
        if now - self._last_vix_check < VIX_CHECK_INTERVAL_SECONDS:
            return False

        try:
            import yfinance as yf
            vix_df = yf.download("^VIX", period="1d", progress=False, auto_adjust=True, timeout=10)
            if not vix_df.empty:
                vix = float(vix_df["Close"].values.flat[-1])
                self._last_vix = vix
                self._last_vix_check = now
                if vix > VIX_PAUSE_THRESHOLD:
                    with self._lock:
                        self._trip(f"VIX={vix:.1f}>{VIX_PAUSE_THRESHOLD}")
                    return True
        except Exception as exc:
            logger.warning("Could not fetch VIX: %s", exc)

        return False

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, reason: str = "manual_reset"):
        """Manually close the circuit breaker (human intervention)."""
        with self._lock:
            if self._is_open:
                logger.warning(
                    "Circuit breaker RESET — was: %s | reason: %s",
                    self._open_reason, reason,
                )
            self._is_open = False
            self._open_reason = None
            self._open_at = None
            self._consecutive_losses = 0
            self._error_timestamps.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _trip(self, reason: str):
        """Open the circuit breaker (must be called under self._lock)."""
        if self._is_open:
            return  # already open
        self._is_open = True
        self._open_reason = reason
        self._open_at = datetime.utcnow()
        logger.error(
            "CIRCUIT BREAKER TRIPPED — trading paused | reason: %s", reason
        )

    def _count_recent_errors(self) -> int:
        cutoff = time.monotonic() - NETWORK_ERROR_WINDOW_SECONDS
        self._error_timestamps = [t for t in self._error_timestamps if t >= cutoff]
        return len(self._error_timestamps)


# Module-level singleton
circuit_breaker = CircuitBreaker()
