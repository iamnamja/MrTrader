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


@dataclass
class SpreadCostModel:
    """Half-spread + market impact model (for future day-trading use)."""
    half_spread_bps: float = 5.0
    impact_bps: float = 0.0

    @property
    def cost_pct(self) -> float:
        return (self.half_spread_bps + self.impact_bps) / 10_000
