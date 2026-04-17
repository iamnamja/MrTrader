"""
Unified signal generation — the single source of truth for entry/exit logic.

This module is imported by BOTH the backtesting engine (app/backtest/backtest.py)
and the live Trader agent (app/agents/trader.py) so the strategy is always identical.

Strategy (validated in Phase 7 backtest: 56.7% win rate, +1.27% avg trade):
  Trend filter : price > EMA(200)  AND  close > close[-63]  (3-month momentum)
  Signal A     : EMA(20) crosses above EMA(50)  AND  50 < RSI < 70
  Signal B     : RSI dips below 45 then recovers  AND  price > EMA(20)
  Stop         : entry - 2.5 × ATR(14)
  Target       : entry + 4.0 × ATR(14)
  Trail        : activates after +4% gain, trails at 3% below highest close
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Strategy parameters (must stay in sync with backtest.py params) ───────────
RSI_PERIOD = 14
RSI_DIP_ENTRY = 45          # RSI level for dip-and-recovery signal
EMA_FAST = 20
EMA_SLOW = 50
EMA_TREND = 200
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5
ATR_TARGET_MULT = 4.0
TRAIL_ACTIVATION = 0.04     # start trailing once up 4%
TRAIL_PCT = 0.03            # trail 3% below highest close
MOMENTUM_LOOKBACK = 63      # bars (~3 months on daily)


@dataclass
class SignalResult:
    """Outcome of generate_signal() for a single symbol."""

    action: str             # "BUY" | "HOLD"
    signal_type: str        # "EMA_CROSSOVER" | "RSI_DIP" | "NONE"
    entry_price: float
    stop_price: float       # initial hard stop
    target_price: float     # profit target
    atr: float
    position_size: int = 0  # filled by position_sizer; 0 means not yet sized
    reasoning: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.action == "BUY"

    @property
    def risk_per_share(self) -> float:
        return max(self.entry_price - self.stop_price, 0.01)

    @property
    def reward_per_share(self) -> float:
        return max(self.target_price - self.entry_price, 0.01)

    @property
    def risk_reward(self) -> float:
        return self.reward_per_share / self.risk_per_share


def generate_signal(
    symbol: str,
    bars: pd.DataFrame,
    check_earnings: bool = True,
) -> SignalResult:
    """
    Compute a trading signal from an OHLCV DataFrame of daily bars.

    Args:
        symbol:          ticker (used only for logging)
        bars:            DataFrame with lowercase columns: open, high, low, close, volume
                         Index should be datetime.  Minimum ~210 rows for EMA(200).
        check_earnings:  if True, suppress BUY signals during earnings blackout window.
                         Set False in backtesting to avoid live API calls.

    Returns:
        SignalResult with action="BUY" if entry criteria met, else "HOLD".
    """
    _no_signal = _make_no_signal(bars)

    if bars is None or len(bars) < EMA_TREND + 10:
        logger.debug("%s: insufficient bars (%d)", symbol, len(bars) if bars is not None else 0)
        return _no_signal

    close = bars["close"]
    high = bars["high"]
    low = bars["low"]

    # ── Indicators ────────────────────────────────────────────────────────────
    ema_fast_s = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow_s = close.ewm(span=EMA_SLOW, adjust=False).mean()
    ema_trend_s = close.ewm(span=EMA_TREND, adjust=False).mean()
    rsi_s = _calc_rsi(close, RSI_PERIOD)
    atr_val = _calc_atr(high, low, close, ATR_PERIOD)

    if atr_val is None or atr_val <= 0:
        return _no_signal

    price = float(close.iloc[-1])
    ema_fast = float(ema_fast_s.iloc[-1])
    ema_slow = float(ema_slow_s.iloc[-1])
    ema_trend = float(ema_trend_s.iloc[-1])
    rsi_now = float(rsi_s.iloc[-1])
    rsi_prev = float(rsi_s.iloc[-2]) if len(rsi_s) >= 2 else rsi_now
    ema_fast_prev = float(ema_fast_s.iloc[-2]) if len(ema_fast_s) >= 2 else ema_fast
    ema_slow_prev = float(ema_slow_s.iloc[-2]) if len(ema_slow_s) >= 2 else ema_slow

    # ── Trend filter ──────────────────────────────────────────────────────────
    trend_ok = price > ema_trend
    momentum_ok = (
        len(close) > MOMENTUM_LOOKBACK and
        price > float(close.iloc[-(MOMENTUM_LOOKBACK + 1)])
    )
    if not (trend_ok and momentum_ok):
        logger.debug(
            "%s: trend filter failed — trend_ok=%s momentum_ok=%s",
            symbol, trend_ok, momentum_ok,
        )
        return _no_signal

    # ── Entry signals ─────────────────────────────────────────────────────────
    ema_crossover = (
        ema_fast > ema_slow and
        ema_fast_prev <= ema_slow_prev and
        50 < rsi_now < 70
    )
    rsi_dip = (
        rsi_prev < RSI_DIP_ENTRY and
        rsi_now >= RSI_DIP_ENTRY and
        price > ema_fast
    )

    if not (ema_crossover or rsi_dip):
        return _no_signal

    # Earnings blackout: suppress BUY signals near earnings releases
    if check_earnings:
        try:
            from app.strategy.earnings_filter import is_earnings_blackout
            if is_earnings_blackout(symbol):
                logger.info("%s: BUY suppressed — earnings blackout", symbol)
                return _no_signal
        except Exception as exc:
            logger.debug("Earnings filter error for %s: %s", symbol, exc)

    signal_type = "EMA_CROSSOVER" if ema_crossover else "RSI_DIP"
    stop = round(price - ATR_STOP_MULT * atr_val, 4)
    target = round(price + ATR_TARGET_MULT * atr_val, 4)

    reasoning = {
        "price": round(price, 4),
        "ema_fast": round(ema_fast, 4),
        "ema_slow": round(ema_slow, 4),
        "ema_trend": round(ema_trend, 4),
        "rsi": round(rsi_now, 2),
        "atr": round(atr_val, 4),
        "stop": stop,
        "target": target,
        "trend_ok": trend_ok,
        "momentum_ok": momentum_ok,
        "signal_type": signal_type,
    }
    logger.info(
        "%s: BUY signal [%s]  price=%.2f  stop=%.2f  target=%.2f  RSI=%.1f",
        symbol, signal_type, price, stop, target, rsi_now,
    )
    return SignalResult(
        action="BUY",
        signal_type=signal_type,
        entry_price=price,
        stop_price=stop,
        target_price=target,
        atr=atr_val,
        reasoning=reasoning,
    )


def check_exit(
    symbol: str,
    current_price: float,
    entry_price: float,
    stop_price: float,
    target_price: float,
    highest_price: float,
    bars_held: int,
    min_hold_bars: int = 3,
) -> tuple[bool, str, float]:
    """
    Check whether an open position should be exited.

    Returns:
        (should_exit, reason, updated_stop_price)
    """
    # Never exit within the minimum hold period
    if bars_held < min_hold_bars:
        return False, "", stop_price

    # Update trailing stop
    pnl_pct = (current_price - entry_price) / entry_price
    new_stop = stop_price
    if pnl_pct >= TRAIL_ACTIVATION:
        trail_stop = highest_price * (1 - TRAIL_PCT)
        if trail_stop > stop_price:
            new_stop = trail_stop

    if current_price >= target_price:
        return True, f"target_hit (P&L={pnl_pct*100:+.1f}%)", new_stop
    if current_price <= new_stop:
        return True, f"stop_hit (P&L={pnl_pct*100:+.1f}%)", new_stop

    return False, "", new_stop


# ── Internal helpers ──────────────────────────────────────────────────────────

def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(100)


def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> Optional[float]:
    if len(close) < period + 1:
        return None
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    val = float(atr.iloc[-1])
    return val if not np.isnan(val) else None


def _make_no_signal(bars: Optional[pd.DataFrame]) -> SignalResult:
    price = float(bars["close"].iloc[-1]) if bars is not None and len(bars) > 0 else 0.0
    return SignalResult(
        action="HOLD", signal_type="NONE",
        entry_price=price, stop_price=0.0, target_price=0.0, atr=0.0,
    )
