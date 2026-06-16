"""
Risk validation rules for the Risk Manager Agent.

Each rule is a standalone function that returns (is_valid: bool, message: str).
Thresholds live in RiskLimits and can be overridden during backtesting.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple  # noqa: F401

logger = logging.getLogger(__name__)


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
    max_spread_pct: float = 0.005              # 0.5% max bid-ask spread (IEX quotes > 2% treated as stale)
    max_adtv_pct: float = 0.01                 # 1% max trade cost as fraction of 20d ADTV
    max_correlation: float = 0.75              # max 60-day return correlation with open positions
    max_portfolio_beta: float = 1.30           # max portfolio beta vs SPY
    high_beta_threshold: float = 1.20          # beta above which a stock is "high-beta"
    max_factor_concentration: float = 0.60     # max fraction of portfolio in same sector
    ls_net_exposure_pct: float = 0.40          # target net long exposure (long-short)/NAV
    ls_net_exposure_tolerance: float = 0.15    # allowed deviation from target before blocking
    max_short_notional_pct: float = 0.75       # hard cap on total short notional / NAV

    @classmethod
    def from_db(cls, db) -> "RiskLimits":
        """Build RiskLimits pulling values from DB config, falling back to defaults."""
        try:
            from app.database.agent_config import get_agent_config
            _d = cls()  # defaults

            def _get(key, default):
                v = get_agent_config(db, key)
                return default if v is None else v

            return cls(
                MAX_POSITION_SIZE_PCT=_get("risk.max_position_size_pct", _d.MAX_POSITION_SIZE_PCT),
                MAX_SECTOR_CONCENTRATION_PCT=_get("risk.max_sector_concentration_pct", _d.MAX_SECTOR_CONCENTRATION_PCT),
                MAX_DAILY_LOSS_PCT=_get("risk.max_daily_loss_pct", _d.MAX_DAILY_LOSS_PCT),
                MAX_ACCOUNT_DRAWDOWN_PCT=_get("risk.max_account_drawdown_pct", _d.MAX_ACCOUNT_DRAWDOWN_PCT),
                MAX_OPEN_POSITIONS=_get("risk.max_open_positions", _d.MAX_OPEN_POSITIONS),
                MAX_PORTFOLIO_HEAT_PCT=_get("risk.max_portfolio_heat_pct", _d.MAX_PORTFOLIO_HEAT_PCT),
                NORMAL_VOLATILITY_ATR_RATIO=_get("risk.normal_volatility_atr_ratio", _d.NORMAL_VOLATILITY_ATR_RATIO),
                STOP_LOSS_BASE_PCT=_get("risk.stop_loss_base_pct", _d.STOP_LOSS_BASE_PCT),
                max_spread_pct=_get("risk.max_spread_pct", _d.max_spread_pct),
                max_adtv_pct=_get("risk.max_adtv_pct", _d.max_adtv_pct),
                max_correlation=_get("risk.max_correlation", _d.max_correlation),
                max_portfolio_beta=_get("risk.max_portfolio_beta", _d.max_portfolio_beta),
                high_beta_threshold=_get("risk.high_beta_threshold", _d.high_beta_threshold),
                max_factor_concentration=_get("risk.max_factor_concentration", _d.max_factor_concentration),
                ls_net_exposure_pct=_get("pm.ls_net_exposure_pct", _d.ls_net_exposure_pct),
                ls_net_exposure_tolerance=_get("pm.ls_net_exposure_tolerance", _d.ls_net_exposure_tolerance),
                max_short_notional_pct=_get("pm.ls_max_short_notional_pct", _d.max_short_notional_pct),
            )
        except Exception as exc:
            logger.warning(
                "RiskLimits.from_db failed — using hardcoded defaults. "
                "Check DB config for corrupt/missing values: %s",
                exc, exc_info=True,
            )
            return cls()


# ─── Rule 1: Buying Power ─────────────────────────────────────────────────────

def validate_buying_power(
    trade_cost: float,
    available_buying_power: float,
    limits: RiskLimits = None,
    direction: str = "BUY",
) -> Tuple[bool, str]:
    """Ensure sufficient buying power exists before placing a trade.

    For short sales, Reg T requires 150% of notional as margin, so effective cost
    is multiplied by 1.5 when checking available buying power.
    """
    if limits is None:
        limits = RiskLimits()

    effective_cost = trade_cost * 1.5 if direction == "SELL_SHORT" else trade_cost
    if effective_cost > available_buying_power:
        return (
            False,
            f"Insufficient buying power: need ${effective_cost:,.2f}"
            f"{' (150% short margin)' if direction == 'SELL_SHORT' else ''}, "
            f"have ${available_buying_power:,.2f}",
        )
    return True, f"Buying power OK (${available_buying_power:,.2f} available)"


# ─── Rule 2: Max Position Size ────────────────────────────────────────────────

def validate_position_size(
    proposed_cost: float,
    account_value: float,
    limits: RiskLimits = None,
    max_pct_override: float = None,
) -> Tuple[bool, str]:
    """
    Ensure a single position stays within MAX_POSITION_SIZE_PCT of account value.

    Note: we check only the proposed trade cost against account value so that
    partial positions (e.g. adding to an existing one) are evaluated correctly
    by the caller after combining old + new exposure.

    max_pct_override: per-trade ceiling override (e.g. the B4 PEAD ramp's
    pm.pead_max_position_pct = 10%, which intentionally exceeds the global 5%).
    """
    if limits is None:
        limits = RiskLimits()

    if account_value <= 0:
        return False, "Account value is zero or negative"

    position_pct = proposed_cost / account_value
    max_pct = (float(max_pct_override) if max_pct_override is not None
               else limits.MAX_POSITION_SIZE_PCT)

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
    direction: str = "BUY",
) -> Tuple[bool, str]:
    """Ensure no single sector exceeds MAX_SECTOR_CONCENTRATION_PCT.

    Direction-aware: a SELL_SHORT in the same sector reduces gross net exposure,
    so it is added as a negative contribution and evaluated on absolute sector pct.
    """
    if limits is None:
        limits = RiskLimits()

    if account_value <= 0:
        return False, "Account value is zero or negative"

    # Shorts reduce net sector exposure; use signed cost so hedging is allowed.
    signed_cost = -proposed_cost if direction == "SELL_SHORT" else proposed_cost
    total_sector_value = current_sector_value + signed_cost
    sector_pct = abs(total_sector_value) / account_value
    max_pct = limits.MAX_SECTOR_CONCENTRATION_PCT

    if sector_pct > max_pct:
        return (
            False,
            f"Sector '{sector}' would reach {sector_pct * 100:.1f}% (net) of account, "
            f"exceeds {max_pct * 100:.0f}% limit",
        )
    return (
        True,
        f"Sector concentration OK ('{sector}' at {sector_pct * 100:.1f}% net)",
    )


# ─── Rule 3b: Correlation Risk ───────────────────────────────────────────────

def validate_correlation_risk(
    symbol: str,
    open_symbols: List[str],
    account_value: float,
    position_values: Dict[str, float],
    lookback_days: int = 30,
    limits: RiskLimits = None,
    proposed_direction: str = "BUY",
    position_directions: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str]:
    """
    Reject if the proposed symbol is too highly correlated with an existing
    position that already represents a meaningful slice of the portfolio.

    Correlation is sign-adjusted for direction: a short proposed against an
    existing long is a hedge (effective correlation is negated) and should not
    be rejected on correlation grounds. Only same-direction high correlation is
    penalised.

    Correlation is computed on daily close returns over the last `lookback_days`.
    Fails open (returns True) if data is unavailable.
    """
    if limits is None:
        limits = RiskLimits()

    if not open_symbols:
        return True, "No open positions — correlation check skipped"

    try:
        import pandas as pd
        import yfinance as yf

        tickers = list({symbol} | set(open_symbols))
        raw = yf.download(
            tickers,
            period=f"{lookback_days + 5}d",
            progress=False,
            auto_adjust=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["close"]
        else:
            closes = raw

        closes = closes.dropna(how="all")
        returns = closes.pct_change().dropna()

        if symbol not in returns.columns or len(returns) < 10:
            return True, f"Correlation check skipped — insufficient data for {symbol}"

        sym_returns = returns[symbol]
        _prop_is_short = proposed_direction == "SELL_SHORT"
        for existing in open_symbols:
            if existing not in returns.columns:
                continue
            pos_value = position_values.get(existing, 0.0)
            pos_pct = pos_value / account_value if account_value > 0 else 0.0
            if pos_pct < 0.05:
                continue  # small positions don't drive sector risk
            corr = float(sym_returns.corr(returns[existing]))
            # Sign-adjust: opposite directions mean the correlation is effectively negated
            # (a short against a correlated long is a hedge, not a concentration risk)
            _pos_dir = (position_directions or {}).get(existing, "long")
            _pos_is_short = _pos_dir in ("short", "SELL_SHORT")
            if _prop_is_short != _pos_is_short:
                corr = -corr  # opposite directions → hedging relationship
            if corr > limits.max_correlation:
                return (
                    False,
                    f"Correlation {symbol}/{existing} = {corr:.2f} > {limits.max_correlation:.2f} "
                    f"({existing} is {pos_pct*100:.1f}% of portfolio)",
                )

        return True, f"Correlation risk OK for {symbol}"

    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug(
            "Correlation check failed (fail-open): %s", exc
        )
        return True, f"Correlation check skipped (data unavailable): {exc}"


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


# ─── Rule 9: Net Exposure Gate ────────────────────────────────────────────────

def validate_net_exposure(
    proposed_notional: float,
    proposed_direction: str,
    current_long_notional: float,
    current_short_notional: float,
    account_value: float,
    limits: RiskLimits = None,
) -> Tuple[bool, str]:
    """
    Reject entries that push net exposure beyond target ± tolerance.

    Net exposure = (long_notional - short_notional) / account_value.
    Adding a long increases net; adding a short decreases net.
    Fails open (returns True) when account_value is zero or negative.
    """
    if limits is None:
        limits = RiskLimits()
    if account_value <= 0:
        return True, "Net exposure check skipped — account_value <= 0"

    delta = proposed_notional if proposed_direction != "SELL_SHORT" else -proposed_notional
    new_net = (current_long_notional - current_short_notional + delta) / account_value
    lo = limits.ls_net_exposure_pct - limits.ls_net_exposure_tolerance
    hi = limits.ls_net_exposure_pct + limits.ls_net_exposure_tolerance
    if lo <= new_net <= hi:
        return True, f"Net exposure {new_net:.1%} within target band [{lo:.1%}, {hi:.1%}]"
    return (
        False,
        f"Net exposure {new_net:.1%} outside target band [{lo:.1%}, {hi:.1%}] — "
        f"entry would push portfolio off target {limits.ls_net_exposure_pct:.1%}",
    )


# ─── Rule 10: Short Notional Cap ─────────────────────────────────────────────

def validate_short_notional(
    proposed_notional: float,
    current_short_notional: float,
    account_value: float,
    limits: RiskLimits = None,
) -> Tuple[bool, str]:
    """
    Hard cap on total short notional as a fraction of NAV.

    Prevents unlimited leverage on the short side regardless of net-exposure target.
    """
    if limits is None:
        limits = RiskLimits()
    if account_value <= 0:
        return True, "Short notional check skipped — account_value <= 0"
    new_short_pct = (current_short_notional + proposed_notional) / account_value
    if new_short_pct <= limits.max_short_notional_pct:
        return True, f"Short notional {new_short_pct:.1%} within cap {limits.max_short_notional_pct:.1%}"
    return (
        False,
        f"Short notional {new_short_pct:.1%} would exceed cap {limits.max_short_notional_pct:.1%}",
    )


# ─── Rule 8: Dynamic Stop Loss ────────────────────────────────────────────────

def calculate_dynamic_stop_loss(
    entry_price: float,
    atr: Optional[float] = None,
    limits: RiskLimits = None,
    direction: str = "BUY",
) -> float:
    """
    Calculate a volatility-adjusted stop-loss price.

    For longs (default): stop is below entry (entry * (1 - stop_pct)).
    For shorts (direction="SELL_SHORT"): stop is above entry (entry * (1 + stop_pct)).

    If ATR is provided, stop pct scales with ATR / (entry * NORMAL_VOLATILITY_ATR_RATIO).
    """
    if limits is None:
        limits = RiskLimits()

    if atr is not None and entry_price > 0:
        atr_ratio = atr / entry_price
        normal_ratio = limits.NORMAL_VOLATILITY_ATR_RATIO
        stop_pct = limits.STOP_LOSS_BASE_PCT * (atr_ratio / normal_ratio)
    else:
        stop_pct = limits.STOP_LOSS_BASE_PCT

    if direction == "SELL_SHORT":
        stop_loss = entry_price * (1 + stop_pct)
    else:
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
    from app.live_trading.cash_sleeve import CASH_ETFS  # T-bills aren't a risk sector
    exposure: Dict[str, float] = {}
    for pos in positions:
        symbol = pos.get("symbol", "")
        if symbol in CASH_ETFS:
            continue  # P1-1 cash sleeve must not pollute the UNKNOWN sector bucket
        sector = sector_map.get(symbol, "UNKNOWN")
        exposure[sector] = exposure.get(sector, 0.0) + abs(float(pos.get("market_value", 0)))
    return exposure
