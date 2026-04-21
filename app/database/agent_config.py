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
    # Execution quality
    {
        "key": "risk.max_spread_pct",
        "default": 0.0015,
        "type": "float",
        "min": 0.0005,
        "max": 0.01,
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
        "key": "strategy.limit_order_offset_pct",
        "default": 0.003,
        "type": "float",
        "min": 0.001,
        "max": 0.01,
        "description": "Limit order placed this % below ask for swing entries",
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
