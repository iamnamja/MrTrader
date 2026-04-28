"""
ATR-based position sizer — single source of truth for trade sizing.

Imported by both the backtesting engine and the live Trader agent.

Sizing rule: risk 2% of account per trade.
  shares = floor( (account_equity * risk_fraction) / (entry - stop) )
  Capped at 90% of available cash.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

RISK_FRACTION = 0.02          # risk 2% of account per trade
CASH_CAP = 0.90               # never commit more than 90% of cash to one trade
MAX_POSITION_PCT = 0.10       # never put more than 10% of account equity in one position

# Conviction multiplier bounds (applied to risk_fraction based on ML score)
_CONVICTION_MIN = 0.75        # low-confidence floor
_CONVICTION_MAX = 1.25        # high-confidence ceiling
_SCORE_LOW = 0.55             # ML score at which _CONVICTION_MIN applies
_SCORE_HIGH = 0.75            # ML score at which _CONVICTION_MAX applies


def conviction_multiplier(ml_score: float) -> float:
    """
    Linear scale: ml_score ≤ 0.55 → 0.75×, ml_score ≥ 0.75 → 1.25×.
    Clipped to [_CONVICTION_MIN, _CONVICTION_MAX].
    """
    if ml_score <= _SCORE_LOW:
        return _CONVICTION_MIN
    if ml_score >= _SCORE_HIGH:
        return _CONVICTION_MAX
    slope = (_CONVICTION_MAX - _CONVICTION_MIN) / (_SCORE_HIGH - _SCORE_LOW)
    return round(_CONVICTION_MIN + slope * (ml_score - _SCORE_LOW), 4)


def size_position(
    account_equity: float,
    available_cash: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float = RISK_FRACTION,
    ml_score: float = 0.0,
) -> int:
    """
    Compute integer share count for a new position.

    Args:
        account_equity:  total account value (used for risk %)
        available_cash:  buying power available right now
        entry_price:     planned entry (current market price)
        stop_price:      initial ATR-based hard stop
        risk_fraction:   fraction of equity to risk (default 2%)
        ml_score:        ML model confidence (0–1). When > 0, applies a
                         conviction multiplier to risk_fraction (0.75–1.25×).

    Returns:
        Integer share count >= 0. Returns 0 if inputs are invalid
        or if the position would be unaffordable.
    """
    if entry_price <= 0 or stop_price >= entry_price or account_equity <= 0:
        logger.debug(
            "size_position: invalid inputs entry=%.4f stop=%.4f equity=%.2f",
            entry_price, stop_price, account_equity,
        )
        return 0

    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return 0

    # Apply conviction multiplier when ML score provided
    effective_risk_fraction = risk_fraction
    if ml_score > 0:
        effective_risk_fraction = risk_fraction * conviction_multiplier(ml_score)

    dollar_risk = account_equity * effective_risk_fraction
    risk_based_shares = int(dollar_risk / risk_per_share)

    # Cap at 90% of available cash
    max_affordable = int(available_cash * CASH_CAP / entry_price)
    # Cap at 10% of account equity per position (prevents oversized bets on tight-stop stocks)
    max_position = int(account_equity * MAX_POSITION_PCT / entry_price)
    shares = min(risk_based_shares, max_affordable, max_position)

    if shares <= 0:
        logger.debug(
            "size_position: result=0 (risk_based=%d, max_affordable=%d)",
            risk_based_shares, max_affordable,
        )
    else:
        logger.debug(
            "size_position: %s shares  risk_based=%d  affordable=%d  "
            "entry=%.2f  stop=%.2f  equity=%.0f  ml_score=%.3f  conviction=%.2f×",
            shares, risk_based_shares, max_affordable, entry_price, stop_price,
            account_equity, ml_score, conviction_multiplier(ml_score) if ml_score > 0 else 1.0,
        )

    return max(shares, 0)
