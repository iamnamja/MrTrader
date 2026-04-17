"""
Portfolio heat management.

"Heat" = total risk capital exposed across all open positions.
Each position risks (entry_price - stop_price) * qty dollars.

If stop_price is unavailable we estimate risk as FALLBACK_RISK_PCT of trade cost.

Hard limit: total heat must stay below MAX_PORTFOLIO_HEAT_PCT of account value.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

MAX_PORTFOLIO_HEAT_PCT = 0.06   # 6% max total risk
FALLBACK_RISK_PCT = 0.02        # 2% assumed risk when stop unknown


def _position_risk(pos: Dict[str, Any]) -> float:
    """Dollar risk for a single position dict."""
    qty = float(pos.get("qty", pos.get("quantity", 0)))
    entry = float(pos.get("entry_price", pos.get("avg_entry_price", 0)))
    stop = pos.get("stop_price")

    if stop is not None and entry > 0 and float(stop) < entry:
        return (entry - float(stop)) * qty
    # Fallback: assume 2% risk on market value
    market_val = float(pos.get("market_value", entry * qty))
    return market_val * FALLBACK_RISK_PCT


def get_portfolio_heat(positions: List[Dict[str, Any]], account_value: float) -> float:
    """Return current portfolio heat as a fraction of account value (0.0 – 1.0)."""
    if account_value <= 0:
        return 0.0
    total_risk = sum(_position_risk(p) for p in positions)
    return total_risk / account_value


def validate_portfolio_heat(
    new_trade_risk: float,
    positions: List[Dict[str, Any]],
    account_value: float,
    max_heat_pct: float = MAX_PORTFOLIO_HEAT_PCT,
) -> Tuple[bool, str]:
    """
    Return (ok, message).  ok=False blocks the trade.

    new_trade_risk: dollar risk of the proposed new trade (entry - stop) * qty.
    """
    current_heat = get_portfolio_heat(positions, account_value)
    if account_value <= 0:
        return False, "Account value is zero or negative"

    new_heat = current_heat + new_trade_risk / account_value
    if new_heat > max_heat_pct:
        return (
            False,
            f"Portfolio heat would reach {new_heat * 100:.1f}%, "
            f"exceeds {max_heat_pct * 100:.0f}% limit "
            f"(current {current_heat * 100:.1f}%)",
        )
    return (
        True,
        f"Portfolio heat OK ({new_heat * 100:.1f}% / {max_heat_pct * 100:.0f}%)",
    )
