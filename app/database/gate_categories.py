"""
Gate category classification for decision_audit rows.

Categories:
  alpha      — signal-driven blocks whose counterfactual P&L tells us if the gate adds value
               (NIS, earnings, macro events, negative news)
  quality    — entry-condition blocks that filter poor setups
               (gap chase, spread, momentum, volume)
  risk       — portfolio-risk blocks that are worth tracking but secondary
               (correlation, sector, beta, heat)
  structural — capacity/regulatory blocks with no counterfactual meaning
               (buying power, PDT, kill switch, drawdown limits)
  scan       — whole-scan abstentions; tracked in scan_abstentions, not decision_audit
"""
from __future__ import annotations

# Maps block_reason prefix → gate_category.
# block_reason may have trailing detail after a colon (e.g. "earnings_gate: earnings_in_1d")
# so we match on the prefix before any colon or space.
_PREFIX_MAP: dict[str, str] = {
    # Alpha gates
    "nis_block_entry": "alpha",
    "negative_news": "alpha",
    "earnings_gate": "alpha",
    "macro_event_window": "alpha",
    # Quality / entry-condition gates
    "gap_chase": "quality",
    "spread_too_wide": "quality",
    "no_momentum": "quality",
    "low_volume": "quality",
    "bid_ask_spread": "quality",
    "adtv_liquidity": "quality",
    # Risk gates
    "correlation_risk": "risk",
    "correlation": "risk",
    "sector_concentration": "risk",
    "factor_concentration": "risk",
    "portfolio_heat": "risk",
    "beta_exposure": "risk",
    "position_size": "risk",
    # Structural gates — no counterfactual
    "buying_power": "structural",
    "gross_exposure_cap": "structural",
    "strategy_budget_cap": "structural",
    "open_positions": "structural",
    "intraday_position_cap": "structural",
    "pdt": "structural",
    "kill_switch": "structural",
    "symbol_halt": "structural",
    "daily_loss": "structural",
    "account_drawdown": "structural",
    "account_fetch": "structural",
    # Scan gates — recorded in scan_abstentions, not here
    "gate1a_spy_range": "scan",
    "gate1c_meltup": "scan",
    "gate1b_score_spread": "scan",
}

# Categories where we backfill counterfactual outcomes
CALIBRATABLE_CATEGORIES = frozenset({"alpha", "quality", "risk"})


def classify_gate(block_reason: str | None) -> str:
    """Return gate_category string for a given block_reason value."""
    if not block_reason:
        return "structural"
    # Match on prefix before first colon, space, or digit
    key = block_reason.split(":")[0].split(" ")[0].rstrip("_0123456789")
    if key in _PREFIX_MAP:
        return _PREFIX_MAP[key]
    # Fallback: unknown gate treated as structural (safe — no counterfactual)
    return "structural"


def needs_outcome_backfill(gate_category: str | None) -> bool:
    """True if we should fetch a counterfactual price outcome for this gate."""
    return gate_category in CALIBRATABLE_CATEGORIES
