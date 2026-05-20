"""
Agent configuration store — runtime-tunable parameters for PM / Risk / Trader.

All values live in the `configuration` DB table under namespaced keys
(e.g. "pm.min_confidence", "risk.max_position_size_pct").

Agents call get_agent_config() at decision time so changes take effect
without a restart.  Hardcoded module constants remain as the fallback
default if the DB row has never been written.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Schema: all tunable parameters ────────────────────────────────────────────
# Each entry: key, default, type, min, max, description, group
CONFIG_SCHEMA: List[Dict[str, Any]] = [
    # Portfolio Manager
    {
        "key": "pm.top_n_stocks",
        "default": 10,
        "type": "int",
        "min": 1,
        "max": 30,
        "description": "Maximum stocks to select per cycle",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.min_confidence",
        "default": 0.55,
        "type": "float",
        "min": 0.5,
        "max": 0.95,
        "description": "Minimum ML model probability to propose a trade",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.position_risk_pct",
        "default": 0.02,
        "type": "float",
        "min": 0.005,
        "max": 0.05,
        "description": "Fraction of account to risk per trade (position sizing)",
        "group": "Portfolio Manager",
    },
    # Risk Manager
    {
        "key": "risk.max_position_size_pct",
        "default": 0.05,
        "type": "float",
        "min": 0.01,
        "max": 0.20,
        "description": "Max single position as % of account value",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_sector_concentration_pct",
        "default": 0.20,
        "type": "float",
        "min": 0.05,
        "max": 0.50,
        "description": "Max sector exposure as % of account value",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_daily_loss_pct",
        "default": 0.02,
        "type": "float",
        "min": 0.005,
        "max": 0.10,
        "description": "Max daily loss before blocking new trades",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_account_drawdown_pct",
        "default": 0.05,
        "type": "float",
        "min": 0.01,
        "max": 0.20,
        "description": "Max peak-to-trough drawdown before blocking new trades",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_open_positions",
        "default": 5,
        "type": "int",
        "min": 1,
        "max": 20,
        "description": "Maximum simultaneous open positions",
        "group": "Risk Manager",
    },
    {
        "key": "pm.exit_threshold",
        "default": 0.35,
        "type": "float",
        "min": 0.10,
        "max": 0.55,
        "description": "Re-score below this threshold triggers PM exit signal",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.swing_selector",
        "default": "pead_quality_short",
        "type": "str",
        "description": (
            "Swing selection method: 'ml_model' uses LambdaRank/XGBoost scores; "
            "'factor_portfolio' uses momentum+quality composite; "
            "'pead' uses PEAD scorer (EPS surprise, hold-5); "
            "'quality_short' uses QualityShortScorer (shorts-only); "
            "'pead_quality_short' combines PEAD + QualityShort (Phase I default)"
        ),
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_enable_shorts",
        "default": "false",
        "type": "str",
        "description": (
            "Enable PEAD short entries (EPS surprise < -5%). "
            "Requires margin-enabled Alpaca account and short order routing to be wired. "
            "Default false = longs only."
        ),
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_net_exposure_pct",
        "default": 0.40,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "description": (
            "Target net long exposure (long_mkt_val - short_mkt_val) / NAV. "
            "0.40 = 40% net long (directional L/S). 1.0 = long-only."
        ),
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_top_n_long",
        "default": 20,
        "type": "int",
        "min": 5,
        "max": 50,
        "description": "Number of long candidates from factor composite score (top-N).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_top_n_short",
        "default": 15,
        "type": "int",
        "min": 5,
        "max": 50,
        "description": "Number of short candidates from factor composite score (bottom-N).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_borrow_cost_annual_pct",
        "default": 0.005,
        "type": "float",
        "min": 0.0,
        "max": 0.10,
        "description": "Annualised borrow cost for short positions (deducted daily). 0.005 = 0.5%/yr.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_net_exposure_tolerance",
        "default": 0.15,
        "type": "float",
        "min": 0.05,
        "max": 0.40,
        "description": (
            "Allowed deviation from ls_net_exposure_pct before new entries are blocked. "
            "0.15 means entries blocked if net exposure would be outside [target-15%, target+15%]."
        ),
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_max_short_notional_pct",
        "default": 0.75,
        "type": "float",
        "min": 0.10,
        "max": 1.50,
        "description": "Hard cap on total short notional as fraction of NAV. 0.75 = 75% NAV max short.",
        "group": "Portfolio Manager",
    },
    # Trader / Strategy
    {
        "key": "strategy.partial_exit_pct",
        "default": 0.50,
        "type": "float",
        "min": 0.10,
        "max": 0.90,
        "description": "Fraction of position to exit at 1×ATR profit (partial exit)",
        "group": "Strategy",
    },
    {
        "key": "strategy.max_hold_bars",
        "default": 20,
        "type": "int",
        "min": 5,
        "max": 60,
        "description": "Safety-net max hold in daily bars before forced exit",
        "group": "Strategy",
    },
    # Risk Intelligence (Phase 19)
    {
        "key": "risk.max_correlation",
        "default": 0.75,
        "type": "float",
        "min": 0.30,
        "max": 0.99,
        "description": "Max 60-day return correlation with any open position before rejecting entry",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_portfolio_beta",
        "default": 1.30,
        "type": "float",
        "min": 0.50,
        "max": 3.00,
        "description": "Max portfolio beta vs SPY before blocking high-beta new entries",
        "group": "Risk Manager",
    },
    {
        "key": "risk.high_beta_threshold",
        "default": 1.20,
        "type": "float",
        "min": 0.50,
        "max": 2.50,
        "description": "Beta above which a new position is considered high-beta",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_factor_concentration",
        "default": 0.60,
        "type": "float",
        "min": 0.20,
        "max": 0.90,
        "description": "Max fraction of portfolio capital in the same sector/factor",
        "group": "Risk Manager",
    },
    # Execution quality
    {
        "key": "risk.max_spread_pct",
        "default": 0.005,
        "type": "float",
        "min": 0.0005,
        "max": 0.02,
        "description": "Max bid-ask spread as fraction of mid-price before rejecting entry",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_adtv_pct",
        "default": 0.01,
        "type": "float",
        "min": 0.001,
        "max": 0.05,
        "description": "Max trade cost as fraction of 20-day ADTV (liquidity gate)",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_portfolio_heat_pct",
        "default": 0.06,
        "type": "float",
        "min": 0.02,
        "max": 0.20,
        "description": "Max total portfolio heat (sum of position risks) as fraction of account value",
        "group": "Risk Manager",
    },
    {
        "key": "risk.normal_volatility_atr_ratio",
        "default": 0.02,
        "type": "float",
        "min": 0.005,
        "max": 0.10,
        "description": "Base ATR/price ratio used when computing adaptive stop-loss distance",
        "group": "Risk Manager",
    },
    {
        "key": "risk.stop_loss_base_pct",
        "default": 0.02,
        "type": "float",
        "min": 0.005,
        "max": 0.10,
        "description": "Default stop-loss percentage at normal volatility",
        "group": "Risk Manager",
    },
    {
        "key": "strategy.limit_order_offset_pct",
        "default": 0.001,
        "type": "float",
        "min": 0.0001,
        "max": 0.01,
        "description": "Limit order placed this % below ask for swing entries (10bps default)",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_requote_age_minutes",
        "default": 30,
        "type": "int",
        "min": 5,
        "max": 120,
        "description": "Re-quote a swing limit order if it has been unfilled for this many minutes",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_requote_drift_bps",
        "default": 20.0,
        "type": "float",
        "min": 5.0,
        "max": 100.0,
        "description": "Re-quote a swing limit order if the ask has drifted more than this many bps from our limit",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_max_requotes",
        "default": 3,
        "type": "int",
        "min": 0,
        "max": 10,
        "description": "Maximum number of times a swing limit order may be re-quoted before falling through to escalation/cancel",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_eod_escalation_hour",
        "default": 15,
        "type": "int",
        "min": 14,
        "max": 15,
        "description": "Hour (ET, 24h) at which unfilled swing limits escalate to a marketable limit",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_eod_escalation_minute",
        "default": 15,
        "type": "int",
        "min": 0,
        "max": 59,
        "description": "Minute (ET) at which unfilled swing limits escalate to a marketable limit",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_cancel_hour",
        "default": 15,
        "type": "int",
        "min": 14,
        "max": 16,
        "description": "Hour (ET) after which unfilled swing limit orders are cancelled outright",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_cancel_minute",
        "default": 45,
        "type": "int",
        "min": 0,
        "max": 59,
        "description": "Minute (ET) after which unfilled swing limit orders are cancelled outright",
        "group": "Strategy",
    },
    {
        "key": "strategy.ema_fast",
        "default": 20,
        "type": "int",
        "min": 5,
        "max": 50,
        "description": "Fast EMA period for crossover signal",
        "group": "Strategy",
    },
    {
        "key": "strategy.ema_slow",
        "default": 50,
        "type": "int",
        "min": 20,
        "max": 200,
        "description": "Slow EMA period for crossover signal",
        "group": "Strategy",
    },
    {
        "key": "strategy.rsi_period",
        "default": 14,
        "type": "int",
        "min": 5,
        "max": 30,
        "description": "RSI calculation period",
        "group": "Strategy",
    },
    {
        "key": "strategy.rsi_dip_entry",
        "default": 45,
        "type": "int",
        "min": 20,
        "max": 60,
        "description": "RSI level that triggers dip-and-recovery entry",
        "group": "Strategy",
    },
    {
        "key": "strategy.atr_stop_mult",
        "default": 2.5,
        "type": "float",
        "min": 1.0,
        "max": 5.0,
        "description": "ATR multiplier for stop-loss placement",
        "group": "Strategy",
    },
    {
        "key": "strategy.atr_target_mult",
        "default": 4.0,
        "type": "float",
        "min": 1.5,
        "max": 10.0,
        "description": "ATR multiplier for profit-target placement",
        "group": "Strategy",
    },
    {
        "key": "strategy.trail_activation_pct",
        "default": 0.04,
        "type": "float",
        "min": 0.01,
        "max": 0.15,
        "description": "% gain before trailing stop activates",
        "group": "Strategy",
    },
    {
        "key": "strategy.trail_pct",
        "default": 0.03,
        "type": "float",
        "min": 0.01,
        "max": 0.10,
        "description": "Trailing stop distance below highest close",
        "group": "Strategy",
    },
    # Reconciliation
    {
        "key": "reconcile.ghost_min_age_minutes",
        "default": 5,
        "type": "int",
        "min": 1,
        "max": 60,
        "description": "Minimum trade age in minutes before it can be marked as a ghost",
        "group": "Reconciliation",
    },
    {
        "key": "reconcile.interval_minutes",
        "default": 5,
        "type": "int",
        "min": 1,
        "max": 30,
        "description": "Frequency of periodic position reconciliation during market hours",
        "group": "Reconciliation",
    },
]

_DEFAULTS: Dict[str, Any] = {s["key"]: s["default"] for s in CONFIG_SCHEMA}


def get_agent_config(db, key: str) -> Any:
    """
    Read one config value from DB, falling back to the schema default.
    Coerces the stored value to the declared type.
    """
    from app.database.config_store import get_config
    schema = next((s for s in CONFIG_SCHEMA if s["key"] == key), None)
    raw = get_config(db, f"agent.{key}")
    if raw is None:
        return _DEFAULTS.get(key)
    try:
        if schema and schema["type"] == "int":
            return int(raw)
        if schema and schema["type"] == "float":
            return float(raw)
        return raw
    except (TypeError, ValueError):
        return _DEFAULTS.get(key)


def get_all_agent_config(db) -> Dict[str, Any]:
    """Return all config values as a flat dict (DB values override defaults)."""
    return {s["key"]: get_agent_config(db, s["key"]) for s in CONFIG_SCHEMA}


def set_agent_config(db, key: str, value: Any) -> None:
    """Write one config value, validating against schema bounds."""
    schema = next((s for s in CONFIG_SCHEMA if s["key"] == key), None)
    if schema is None:
        raise ValueError(f"Unknown config key: {key}")

    # Coerce + range-check
    try:
        if schema["type"] == "int":
            value = int(value)
        elif schema["type"] == "float":
            value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for {key}: {exc}") from exc

    if "min" in schema and value < schema["min"]:
        raise ValueError(f"{key} must be >= {schema['min']}")
    if "max" in schema and value > schema["max"]:
        raise ValueError(f"{key} must be <= {schema['max']}")

    from app.database.config_store import set_config
    set_config(db, f"agent.{key}", value, description=schema.get("description", ""))
    logger.info("Agent config updated: %s = %s", key, value)
