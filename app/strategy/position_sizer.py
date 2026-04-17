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


def size_position(
    account_equity: float,
    available_cash: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float = RISK_FRACTION,
) -> int:
    """
    Compute integer share count for a new position.

    Args:
        account_equity:  total account value (used for risk %)
        available_cash:  buying power available right now
        entry_price:     planned entry (current market price)
        stop_price:      initial ATR-based hard stop
        risk_fraction:   fraction of equity to risk (default 2%)

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

    dollar_risk = account_equity * risk_fraction
    risk_based_shares = int(dollar_risk / risk_per_share)

    # Cap at 90% of available cash
    max_affordable = int(available_cash * CASH_CAP / entry_price)
    shares = min(risk_based_shares, max_affordable)

    if shares <= 0:
        logger.debug(
            "size_position: result=0 (risk_based=%d, max_affordable=%d)",
            risk_based_shares, max_affordable,
        )
    else:
        logger.debug(
            "size_position: %s shares  risk_based=%d  affordable=%d  "
            "entry=%.2f  stop=%.2f  equity=%.0f",
            shares, risk_based_shares, max_affordable, entry_price, stop_price, account_equity,
        )

    return max(shares, 0)
