"""
Real-time entry quality filter — called by Trader at execution time.

PM picks candidates hours in advance. By the time Trader is about to enter,
conditions may have changed. This module checks whether "right now" is
actually a good moment to execute the trade.

Checks (all must pass for BUY to proceed):
  1. Price run   — stock hasn't already moved >1.5% past the signal price
  2. Intraday momentum — 5-min trend not sharply against the daily signal
  3. Spread/liquidity — bid-ask spread ≤ 0.5% of mid-price
  4. Volume context  — current bar volume not suspiciously low (< 30% of avg)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

MAX_PRICE_RUN_PCT = 0.015       # don't enter if price > 1.5% above signal price
MAX_ADVERSE_MOVE_PCT = 0.010    # don't enter if price > 1.0% below signal (gap down)
MAX_SPREAD_PCT = 0.005          # skip if bid-ask spread > 0.5% of mid
MIN_VOLUME_RATIO = 0.30         # skip if current bar volume < 30% of 20-bar avg
INTRADAY_BARS_NEEDED = 12       # min 5-min bars to assess intraday momentum (1 hour)
INTRADAY_MOMENTUM_LOOKBACK = 6  # bars for short-term slope (30 min)


@dataclass
class EntryQualityResult:
    approved: bool
    reason: str          # empty string if approved; failure reason if rejected
    price_run_pct: float = 0.0
    spread_pct: float = 0.0
    momentum_slope: float = 0.0
    volume_ratio: float = 1.0


def check_entry_quality(
    symbol: str,
    signal_price: float,
    current_price: float,
    trade_type: str,
    quote: Optional[dict] = None,
    intraday_bars: Optional[pd.DataFrame] = None,
) -> EntryQualityResult:
    """
    Evaluate whether current market conditions support entering a trade.

    Args:
        symbol:        ticker
        signal_price:  price at which the ML/technical signal fired (from generate_signal)
        current_price: live price right now
        trade_type:    "swing" or "intraday"
        quote:         dict with "bid" and "ask" keys (optional)
        intraday_bars: DataFrame of 5-min OHLCV bars for today (optional)

    Returns:
        EntryQualityResult — .approved=True if all checks pass
    """
    if signal_price <= 0 or current_price <= 0:
        return EntryQualityResult(approved=False, reason="invalid_prices")

    # ── Check 1: price run ────────────────────────────────────────────────────
    price_run_pct = (current_price - signal_price) / signal_price
    if price_run_pct > MAX_PRICE_RUN_PCT:
        logger.info(
            "%s entry rejected: price run +%.2f%% above signal $%.2f (current $%.2f)",
            symbol, price_run_pct * 100, signal_price, current_price,
        )
        return EntryQualityResult(
            approved=False,
            reason=f"price_run_{price_run_pct*100:.1f}pct",
            price_run_pct=price_run_pct,
        )

    # For swing, we're more tolerant of small adverse moves (noise); for intraday be stricter
    adverse_threshold = MAX_ADVERSE_MOVE_PCT if trade_type == "intraday" else MAX_ADVERSE_MOVE_PCT * 1.5
    if price_run_pct < -adverse_threshold:
        logger.info(
            "%s entry rejected: adverse move %.2f%% from signal (current $%.2f, signal $%.2f)",
            symbol, price_run_pct * 100, current_price, signal_price,
        )
        return EntryQualityResult(
            approved=False,
            reason=f"adverse_move_{abs(price_run_pct)*100:.1f}pct",
            price_run_pct=price_run_pct,
        )

    # ── Check 2: spread / liquidity ───────────────────────────────────────────
    spread_pct = 0.0
    if quote and quote.get("bid", 0) > 0 and quote.get("ask", 0) > 0:
        bid, ask = float(quote["bid"]), float(quote["ask"])
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid > 0 else 0.0
        if spread_pct > MAX_SPREAD_PCT:
            logger.info(
                "%s entry rejected: spread %.3f%% > max %.3f%%",
                symbol, spread_pct * 100, MAX_SPREAD_PCT * 100,
            )
            return EntryQualityResult(
                approved=False,
                reason=f"spread_{spread_pct*100:.2f}pct",
                spread_pct=spread_pct,
                price_run_pct=price_run_pct,
            )

    # ── Check 3: intraday momentum (only when bars available) ─────────────────
    momentum_slope = 0.0
    if intraday_bars is not None and len(intraday_bars) >= INTRADAY_BARS_NEEDED:
        closes = intraday_bars["close"].tail(INTRADAY_MOMENTUM_LOOKBACK).values
        if len(closes) >= 2:
            # Simple slope: (last - first) / first
            momentum_slope = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0.0

        # For intraday trades: require momentum to not be sharply negative
        # For swing trades: more forgiving — we're entering for multi-day hold
        momentum_threshold = -0.008 if trade_type == "intraday" else -0.015
        if momentum_slope < momentum_threshold:
            logger.info(
                "%s entry rejected: intraday momentum slope %.2f%% too negative for %s",
                symbol, momentum_slope * 100, trade_type,
            )
            return EntryQualityResult(
                approved=False,
                reason=f"momentum_{momentum_slope*100:.1f}pct",
                price_run_pct=price_run_pct,
                spread_pct=spread_pct,
                momentum_slope=momentum_slope,
            )

    # ── Check 4: volume context ───────────────────────────────────────────────
    volume_ratio = 1.0
    if intraday_bars is not None and len(intraday_bars) >= 20:
        avg_vol = float(intraday_bars["volume"].tail(20).mean())
        current_vol = float(intraday_bars["volume"].iloc[-1]) if len(intraday_bars) > 0 else avg_vol
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        if volume_ratio < MIN_VOLUME_RATIO:
            logger.info(
                "%s entry rejected: volume ratio %.2f (current bar is low-volume)",
                symbol, volume_ratio,
            )
            return EntryQualityResult(
                approved=False,
                reason=f"low_volume_{volume_ratio:.2f}x",
                price_run_pct=price_run_pct,
                spread_pct=spread_pct,
                momentum_slope=momentum_slope,
                volume_ratio=volume_ratio,
            )

    logger.debug(
        "%s entry quality OK: run=%.2f%% spread=%.3f%% momentum=%.2f%% vol_ratio=%.2f",
        symbol, price_run_pct * 100, spread_pct * 100, momentum_slope * 100, volume_ratio,
    )
    return EntryQualityResult(
        approved=True,
        reason="",
        price_run_pct=price_run_pct,
        spread_pct=spread_pct,
        momentum_slope=momentum_slope,
        volume_ratio=volume_ratio,
    )
