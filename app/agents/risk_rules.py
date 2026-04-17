"""
Risk validation rules for the Risk Manager Agent.

Each rule is a standalone function that returns (is_valid: bool, message: str).
Thresholds live in RiskLimits and can be overridden during backtesting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RiskLimits:
    """Configurable risk thresholds - tune via backtesting (Phase 7)."""

    MAX_POSITION_SIZE_PCT: float = 0.05       # 5%  – max single position / account value
    MAX_SECTOR_CONCENTRATION_PCT: float = 0.20  # 20% – max one sector / account value
    MAX_DAILY_LOSS_PCT: float = 0.02           # 2%  – max daily loss / account value
    MAX_ACCOUNT_DRAWDOWN_PCT: float = 0.05     # 5%  – max (peak-current) / peak
    MAX_OPEN_POSITIONS: int = 5                # absolute position count limit
    MAX_PORTFOLIO_HEAT_PCT: float = 0.06       # 6%  – total risk across all positions
    NORMAL_VOLATILITY_ATR_RATIO: float = 0.02  # base ATR/price ratio for stop-loss calc
    STOP_LOSS_BASE_PCT: float = 0.02           # 2% stop loss at normal volatility


# ─── Rule 1: Buying Power ─────────────────────────────────────────────────────

def validate_buying_power(
    trade_cost: float,
    available_buying_power: float,
    limits: RiskLimits = None,
) -> Tuple[bool, str]:
    """Ensure sufficient buying power exists before placing a trade."""
    if limits is None:
        limits = RiskLimits()

    if trade_cost > available_buying_power:
        return (
            False,
            f"Insufficient buying power: need ${trade_cost:,.2f}, "
            f"have ${available_buying_power:,.2f}",
        )
    return True, f"Buying power OK (${available_buying_power:,.2f} available)"


# ─── Rule 2: Max Position Size ────────────────────────────────────────────────

def validate_position_size(
    proposed_cost: float,
    account_value: float,
    limits: RiskLimits = None,
) -> Tuple[bool, str]:
    """
    Ensure a single position stays within MAX_POSITION_SIZE_PCT of account value.

    Note: we check only the proposed trade cost against account value so that
    partial positions (e.g. adding to an existing one) are evaluated correctly
    by the caller after combining old + new exposure.
    """
    if limits is None:
        limits = RiskLimits()

    if account_value <= 0:
        return False, "Account value is zero or negative"

    position_pct = proposed_cost / account_value
    max_pct = limits.MAX_POSITION_SIZE_PCT

    if position_pct > max_pct:
        return (
            False,
            f"Position would be {position_pct * 100:.1f}% of account, "
            f"exceeds {max_pct * 100:.0f}% limit",
        )
    return (
        True,
        f"Position size OK ({position_pct * 100:.1f}% of account)",
    )


# ─── Rule 3: Sector Concentration ─────────────────────────────────────────────

def validate_sector_concentration(
    proposed_cost: float,
    current_sector_value: float,
    account_value: float,
    sector: str = "UNKNOWN",
    limits: RiskLimits = None,
) -> Tuple[bool, str]:
    """Ensure no single sector exceeds MAX_SECTOR_CONCENTRATION_PCT."""
    if limits is None:
        limits = RiskLimits()

    if account_value <= 0:
        return False, "Account value is zero or negative"

    total_sector_value = current_sector_value + proposed_cost
    sector_pct = total_sector_value / account_value
    max_pct = limits.MAX_SECTOR_CONCENTRATION_PCT

    if sector_pct > max_pct:
        return (
            False,
            f"Sector '{sector}' would reach {sector_pct * 100:.1f}% of account, "
            f"exceeds {max_pct * 100:.0f}% limit",
        )
    return (
        True,
        f"Sector concentration OK ('{sector}' at {sector_pct * 100:.1f}%)",
    )


# ─── Rule 4: Daily Loss Limit ─────────────────────────────────────────────────

def validate_daily_loss(
    daily_pnl: float,
    account_value: float,
    limits: RiskLimits = None,
) -> Tuple[bool, str]:
    """
    Block new trades when today's realized losses exceed MAX_DAILY_LOSS_PCT.
    daily_pnl is negative for losses (e.g. -500 means $500 loss).
    """
    if limits is None:
        limits = RiskLimits()

    if account_value <= 0:
        return False, "Account value is zero or negative"

    loss_pct = abs(min(daily_pnl, 0)) / account_value
    max_loss_pct = limits.MAX_DAILY_LOSS_PCT

    if loss_pct >= max_loss_pct:
        return (
            False,
            f"Daily loss limit reached: ${abs(daily_pnl):,.2f} "
            f"({loss_pct * 100:.2f}% of account, limit {max_loss_pct * 100:.0f}%)",
        )
    return (
        True,
        f"Daily loss OK (${abs(daily_pnl):,.2f} / {loss_pct * 100:.2f}% used)",
    )


# ─── Rule 5: Account Drawdown ─────────────────────────────────────────────────

def validate_account_drawdown(
    current_equity: float,
    peak_equity: float,
    limits: RiskLimits = None,
) -> Tuple[bool, str]:
    """
    Block new trades when account drawdown exceeds MAX_ACCOUNT_DRAWDOWN_PCT.
    Drawdown = (peak - current) / peak.
    """
    if limits is None:
        limits = RiskLimits()

    if peak_equity <= 0:
        return False, "Peak equity is zero or negative"

    drawdown_pct = (peak_equity - current_equity) / peak_equity
    max_dd_pct = limits.MAX_ACCOUNT_DRAWDOWN_PCT

    if drawdown_pct >= max_dd_pct:
        return (
            False,
            f"Account drawdown {drawdown_pct * 100:.2f}% exceeds "
            f"{max_dd_pct * 100:.0f}% limit (peak ${peak_equity:,.2f}, "
            f"current ${current_equity:,.2f})",
        )
    return (
        True,
        f"Drawdown OK ({drawdown_pct * 100:.2f}%)",
    )


# ─── Rule 6: Open Position Count ─────────────────────────────────────────────

def validate_open_positions(
    current_position_count: int,
    limits: RiskLimits = None,
) -> Tuple[bool, str]:
    """Prevent opening more than MAX_OPEN_POSITIONS simultaneous positions."""
    if limits is None:
        limits = RiskLimits()

    max_pos = limits.MAX_OPEN_POSITIONS

    if current_position_count >= max_pos:
        return (
            False,
            f"Open position limit reached: {current_position_count}/{max_pos}",
        )
    return (
        True,
        f"Position count OK ({current_position_count}/{max_pos})",
    )


# ─── Rule 7: Portfolio Heat ──────────────────────────────────────────────────

def validate_portfolio_heat(
    new_trade_risk: float,
    positions: List[Dict],
    account_value: float,
    limits: RiskLimits = None,
) -> Tuple[bool, str]:
    """Ensure total portfolio risk stays below MAX_PORTFOLIO_HEAT_PCT."""
    if limits is None:
        limits = RiskLimits()
    from app.strategy.portfolio_heat import validate_portfolio_heat as _check
    return _check(new_trade_risk, positions, account_value, limits.MAX_PORTFOLIO_HEAT_PCT)


# ─── Rule 8: Dynamic Stop Loss ────────────────────────────────────────────────

def calculate_dynamic_stop_loss(
    entry_price: float,
    atr: Optional[float] = None,
    limits: RiskLimits = None,
) -> float:
    """
    Calculate a volatility-adjusted stop-loss price.

    If ATR is provided, the stop is scaled by ATR / (entry_price * NORMAL_VOLATILITY_ATR_RATIO).
    At normal volatility (ATR/price == NORMAL_VOLATILITY_ATR_RATIO) this equals STOP_LOSS_BASE_PCT.
    Without ATR, falls back to the base stop-loss percentage.
    """
    if limits is None:
        limits = RiskLimits()

    if atr is not None and entry_price > 0:
        atr_ratio = atr / entry_price
        normal_ratio = limits.NORMAL_VOLATILITY_ATR_RATIO
        # Scale stop pct proportionally: higher ATR → wider stop
        stop_pct = limits.STOP_LOSS_BASE_PCT * (atr_ratio / normal_ratio)
    else:
        stop_pct = limits.STOP_LOSS_BASE_PCT

    stop_loss = entry_price * (1 - stop_pct)
    return round(stop_loss, 2)


# ─── Composite helper ─────────────────────────────────────────────────────────

def get_sector_exposure(
    positions: List[Dict],
    sector_map: Dict[str, str],
) -> Dict[str, float]:
    """
    Compute total market value per sector from a list of position dicts.

    positions: list of {symbol, market_value, ...}
    sector_map: {symbol: sector_name}
    Returns: {sector_name: total_market_value}
    """
    exposure: Dict[str, float] = {}
    for pos in positions:
        symbol = pos.get("symbol", "")
        sector = sector_map.get(symbol, "UNKNOWN")
        exposure[sector] = exposure.get(sector, 0.0) + float(pos.get("market_value", 0))
    return exposure
