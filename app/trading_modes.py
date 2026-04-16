"""
Trading mode management — paper vs. live.

The mode is seeded from settings.trading_mode at startup and can be
switched at runtime via the API approval workflow.
"""
import logging
from enum import Enum
from typing import Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class TradingModeManager:
    """Manage the current trading mode (paper vs. live)."""

    def __init__(self):
        self._mode = TradingMode(settings.trading_mode)
        logger.info("Trading mode initialised: %s", self._mode.value)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def is_paper(self) -> bool:
        return self._mode == TradingMode.PAPER

    @property
    def is_live(self) -> bool:
        return self._mode == TradingMode.LIVE

    # ── Transitions ───────────────────────────────────────────────────────────

    def switch_to_live(self):
        """Switch to live trading — irreversible in this session."""
        if self._mode == TradingMode.LIVE:
            logger.warning("Already in LIVE mode, no change.")
            return
        logger.warning("*** SWITCHING TO LIVE TRADING — REAL MONEY AT RISK ***")
        self._mode = TradingMode.LIVE

    def switch_to_paper(self):
        """Fall back to paper trading (e.g. after a kill-switch event)."""
        logger.info("Switching back to PAPER trading mode.")
        self._mode = TradingMode.PAPER

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self._mode.value,
            "is_paper": self.is_paper,
            "is_live": self.is_live,
        }


# Module-level singleton
mode_manager = TradingModeManager()
