"""
Dataclasses for the News Intelligence Service (NIS).

Two-tier design:
  Tier 1 — MacroIntelligence: event-level risk/direction for economic calendar events.
  Tier 2 — NewsSignal: per-symbol scored news signal fed into PM scoring overlay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

# ── Shared types ──────────────────────────────────────────────────────────────

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
Direction = Literal["BULLISH", "BEARISH", "NEUTRAL"]
ScorerTier = Literal["rules", "haiku", "cached", "fallback"]
ActionPolicy = Literal[
    "ignore",
    "watch",
    "size_down_light",   # 0.75×
    "size_down_heavy",   # 0.50×
    "size_up_light",     # 1.15×
    "block_entry",
    "exit_review",
]

# ── Tier 1: Macro Intelligence ─────────────────────────────────────────────────

@dataclass
class MacroEventSignal:
    """
    LLM-scored interpretation of a single economic calendar event.
    Replaces the binary FOMC/NFP block in premarket.py.
    """
    event_type: str                      # "FOMC", "NFP", "CPI", etc.
    event_time: str                      # "14:00 ET"
    risk_level: RiskLevel                # LOW / MEDIUM / HIGH
    direction: Direction                 # expected market direction post-event
    sizing_factor: float                 # 0.5–1.0 applied to all new entries today
    block_new_entries: bool              # True only for HIGH risk uncertain events
    consensus_summary: str               # "Hold at 5.25% — 94% priced in"
    rationale: str                       # one sentence explanation
    scorer_tier: ScorerTier
    evaluated_at: datetime
    schema_version: str = "v1"
    prompt_version: str = "macro_v1"
    already_priced_in: bool = False      # True when estimate == prior

    @property
    def effective_sizing_factor(self) -> float:
        """Sizing factor clamped to [0.3, 1.0]."""
        return max(0.3, min(1.0, self.sizing_factor))


@dataclass
class MacroContext:
    """
    Aggregated Tier 1 output for the full trading day.
    Produced once at ~9 AM, consumed by PM + Trader throughout the day.
    """
    as_of: datetime
    events_today: list[MacroEventSignal] = field(default_factory=list)
    block_new_entries: bool = False       # True if ANY event is HIGH-risk uncertain
    global_sizing_factor: float = 1.0    # product of all event sizing factors
    overall_risk: RiskLevel = "LOW"
    rationale: str = ""
    schema_version: str = "v1"

    @classmethod
    def neutral(cls) -> "MacroContext":
        """Return a no-op context used when NIS is unavailable."""
        return cls(as_of=datetime.utcnow(), rationale="NIS unavailable — no adjustment")


# ── Tier 2: Stock-Level News Signal ───────────────────────────────────────────

@dataclass
class NewsSignal:
    """
    LLM-scored news signal for a specific symbol.
    Produced at 9 AM premarket and refreshed during 30-min scans for new articles.
    """
    symbol: str
    evaluated_at: datetime
    direction_score: float               # [-1.0, 1.0] negative=bearish
    materiality_score: float             # [0.0, 1.0]
    downside_risk_score: float           # [0.0, 1.0]
    upside_catalyst_score: float         # [0.0, 1.0]
    confidence: float                    # [0.0, 1.0]
    already_priced_in_score: float       # [0.0, 1.0]
    action_policy: ActionPolicy
    sizing_multiplier: float             # computed from scores, [0.3, 1.25]
    rationale: str
    top_headlines: list[str] = field(default_factory=list)
    scorer_tier: ScorerTier = "fallback"
    stale: bool = False
    schema_version: str = "v1"
    prompt_version: str = "stock_v1"

    @classmethod
    def neutral(cls, symbol: str) -> "NewsSignal":
        """No-news baseline — no adjustment to sizing or direction."""
        return cls(
            symbol=symbol,
            evaluated_at=datetime.utcnow(),
            direction_score=0.0,
            materiality_score=0.0,
            downside_risk_score=0.0,
            upside_catalyst_score=0.0,
            confidence=0.0,
            already_priced_in_score=0.0,
            action_policy="ignore",
            sizing_multiplier=1.0,
            rationale="No recent news",
            scorer_tier="fallback",
        )

    def news_score_overlay(
        self,
        upside_weight: float = 0.10,
        downside_weight: float = 0.15,
    ) -> float:
        """
        Additive overlay on the ML model score (Section 11.1 of NIS spec).
        Positive = good news boosts signal, negative = bad news weakens it.
        """
        overlay = (
            upside_weight * self.upside_catalyst_score
            - downside_weight * self.downside_risk_score
        ) * self.confidence * (1.0 - 0.5 * self.already_priced_in_score)
        return round(overlay, 4)
