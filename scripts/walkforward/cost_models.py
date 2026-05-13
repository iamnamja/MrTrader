"""
cost_models.py — Transaction cost models for walk-forward simulation.

Usage:
    cost = FixedBpsCostModel(round_trip_bps=5)
    pct = cost.cost_pct   # 0.00025
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FixedBpsCostModel:
    """Fixed round-trip cost in basis points (entry + exit combined)."""
    round_trip_bps: float = 5.0

    @property
    def cost_pct(self) -> float:
        return self.round_trip_bps / 10_000 / 2


def cost_from_turnover(turnover_fraction: float, bps_per_side: float = 5.0) -> float:
    """Compute cost fraction from portfolio turnover and one-way bps.

    Args:
        turnover_fraction: fraction of portfolio that changed hands (0.0–1.0+).
                           E.g. fully replacing the portfolio = 1.0.
        bps_per_side:      transaction cost per trade in basis points (one side).
                           Default 5 bps matches the WF simulation default.
    Returns:
        Cost as a fraction of portfolio value (e.g. 0.0005 for 5 bps on 1.0 turnover).
    """
    return turnover_fraction * bps_per_side / 10_000


@dataclass
class SpreadCostModel:
    """Half-spread + market impact model (for future day-trading use)."""
    half_spread_bps: float = 5.0
    impact_bps: float = 0.0

    @property
    def cost_pct(self) -> float:
        return (self.half_spread_bps + self.impact_bps) / 10_000
