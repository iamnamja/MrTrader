"""
Capital manager — governs the gradual capital ramp during live trading.

Stages:
  1  →  $1,000   (3 days)   minimal exposure, test execution
  2  →  $2,500   (3 days)   small increase, monitor fills
  3  →  $5,000   (5 days)   medium exposure, verify risk controls
  4  →  $10,000  (5 days)   half capital, monitor drawdown
  5  →  $20,000  (ongoing)  full capital
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

_CFG_STAGE_INDEX = "capital_ramp.stage_index"
_CFG_STAGE_START = "capital_ramp.stage_start"

logger = logging.getLogger(__name__)


@dataclass
class CapitalStage:
    stage: int           # 1-indexed label
    capital: float       # approved capital for this stage ($)
    duration_days: Optional[int]  # None = no expiry (final stage)

    def elapsed_days(self, start_time: datetime) -> int:
        return (datetime.utcnow() - start_time).days if start_time else 0

    def is_complete(self, start_time: datetime) -> bool:
        if self.duration_days is None:
            return False  # final stage never "completes"
        return self.elapsed_days(start_time) >= self.duration_days

    def health_ok(self, max_drawdown_pct: float, daily_loss_pct: float) -> bool:
        """Stage fails if drawdown > 3% or single-day loss > 2%."""
        return max_drawdown_pct <= 3.0 and daily_loss_pct <= 2.0


class CapitalManager:
    """
    Track the current live-trading capital stage and advance it when
    risk thresholds and time requirements are met.
    """

    STAGES: List[CapitalStage] = [
        CapitalStage(1, 1_000,  3),
        CapitalStage(2, 2_500,  3),
        CapitalStage(3, 5_000,  5),
        CapitalStage(4, 10_000, 5),
        CapitalStage(5, 20_000, None),
    ]

    def __init__(self):
        self._stage_index = 0        # index into STAGES list
        self._stage_start: Optional[datetime] = None
        self._paused = False

    # ── Read-only state ────────────────────────────────────────────────────────

    @property
    def current_stage(self) -> CapitalStage:
        return self.STAGES[self._stage_index]

    @property
    def is_at_max(self) -> bool:
        return self._stage_index >= len(self.STAGES) - 1

    def get_current_capital(self) -> float:
        return self.current_stage.capital

    def get_all_stages(self) -> List[Dict[str, Any]]:
        return [
            {
                "stage":        s.stage,
                "capital":      s.capital,
                "duration_days": s.duration_days,
                "is_current":   (i == self._stage_index),
                "days_elapsed": s.elapsed_days(self._stage_start) if i == self._stage_index else None,
            }
            for i, s in enumerate(self.STAGES)
        ]

    # ── Stage management ──────────────────────────────────────────────────────

    def start(self):
        """Call when live trading begins (after approval)."""
        self._stage_index = 0
        self._stage_start = datetime.utcnow()
        logger.warning("Capital ramp started at Stage 1 ($1,000)")

    def can_advance(self, max_drawdown_pct: float, daily_loss_pct: float) -> bool:
        if self.is_at_max or self._paused or self._stage_start is None:
            return False
        stage = self.current_stage
        return (
            stage.is_complete(self._stage_start) and
            stage.health_ok(max_drawdown_pct, daily_loss_pct)
        )

    def advance(self) -> Dict[str, Any]:
        """Advance to the next capital stage (unconditional — caller must guard)."""
        if self.is_at_max:
            return {"status": "already_at_max", "stage": self.current_stage.stage,
                    "capital": self.current_stage.capital}

        self._stage_index += 1
        self._stage_start = datetime.utcnow()
        new = self.current_stage

        logger.warning(
            "CAPITAL ADVANCED to Stage %d — $%s",
            new.stage, f"{new.capital:,.0f}",
        )
        return {
            "status": "advanced",
            "stage": new.stage,
            "capital": new.capital,
            "effective_at": self._stage_start.isoformat(),
        }

    def check_health(self, max_drawdown_pct: float, daily_loss_pct: float) -> bool:
        """Return False (and log) if risk thresholds are breached."""
        ok = self.current_stage.health_ok(max_drawdown_pct, daily_loss_pct)
        if not ok:
            logger.error(
                "Capital stage health BREACH — drawdown=%.2f%% daily_loss=%.2f%%",
                max_drawdown_pct, daily_loss_pct,
            )
        return ok

    def pause(self):
        self._paused = True
        logger.warning("Capital increases PAUSED")

    def resume(self):
        self._paused = False
        logger.info("Capital increases RESUMED")


# Module-level singleton
capital_manager = CapitalManager()
