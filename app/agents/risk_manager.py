"""
Risk Manager Agent — the sole gatekeeper for all trade execution.

Flow:
  1. Listen to Redis queue `trade_proposals`
  2. Run all risk rules against each proposal
  3. APPROVED → forward to `trader_approved_trades` queue
     REJECTED → log rejection with full reasoning, drop the proposal
"""

import asyncio
import logging
from datetime import date
from typing import Any, Dict, Optional, Tuple

from app.agents.base import BaseAgent
from app.agents.risk_rules import (
    RiskLimits,
    calculate_dynamic_stop_loss,
    get_sector_exposure,
    validate_account_drawdown,
    validate_buying_power,
    validate_daily_loss,
    validate_open_positions,
    validate_portfolio_heat,
    validate_position_size,
    validate_sector_concentration,
)
from app.database.models import RiskMetric
from app.database.session import get_session

logger = logging.getLogger(__name__)

TRADE_PROPOSALS_QUEUE = "trade_proposals"
APPROVED_TRADES_QUEUE = "trader_approved_trades"


class RiskManager(BaseAgent):
    """
    Continuous listener that validates trade proposals against risk rules
    and either forwards them for execution or rejects them with reasoning.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        super().__init__("risk_manager")
        self.limits = limits or RiskLimits()
        # Sector mapping: symbol → sector (extend as needed; fallback = UNKNOWN)
        self._sector_map: Dict[str, str] = {}
        # Track equity high-water mark for drawdown calculation
        self._peak_equity: Optional[float] = None

    # ─── Main Loop ────────────────────────────────────────────────────────────

    async def run(self):
        """Continuously consume trade_proposals and validate them."""
        self.logger.info("Risk Manager started — listening on '%s'", TRADE_PROPOSALS_QUEUE)
        self.status = "running"

        while self.status == "running":
            try:
                proposal = self.get_message(TRADE_PROPOSALS_QUEUE, timeout=5)
                if proposal is None:
                    await asyncio.sleep(0)  # yield control
                    continue

                self.logger.info(
                    "Received proposal: %s %s x%s @ $%s from %s",
                    proposal.get("symbol"),
                    proposal.get("direction"),
                    proposal.get("quantity"),
                    proposal.get("entry_price"),
                    proposal.get("source_agent", "unknown"),
                )

                is_approved, reasoning = await self._validate_trade(proposal)

                if is_approved:
                    await self._approve(proposal, reasoning)
                else:
                    await self._reject(proposal, reasoning)

            except asyncio.CancelledError:
                self.logger.info("Risk Manager cancelled — shutting down")
                self.status = "stopped"
                break
            except Exception as e:
                self.logger.error("Unexpected error in risk manager loop: %s", e, exc_info=True)
                await self.log_decision(
                    "RISK_MANAGER_ERROR",
                    reasoning={"error": str(e)},
                )
                # Brief pause before retrying to avoid tight error loops
                await asyncio.sleep(1)

    # ─── Approval / Rejection ─────────────────────────────────────────────────

    async def _approve(self, proposal: Dict[str, Any], reasoning: Dict[str, Any]) -> None:
        from datetime import datetime

        approved_proposal = {
            **proposal,
            "stop_loss": reasoning.get("stop_loss"),
            "approved_at": datetime.utcnow().isoformat(),
        }
        self.send_message(APPROVED_TRADES_QUEUE, approved_proposal)
        self.logger.info(
            "APPROVED: %s %s — stop_loss=$%s",
            proposal.get("symbol"),
            proposal.get("direction"),
            reasoning.get("stop_loss"),
        )
        await self.log_decision("TRADE_APPROVED", reasoning=reasoning)

    async def _reject(self, proposal: Dict[str, Any], reasoning: Dict[str, Any]) -> None:
        self.logger.warning(
            "REJECTED: %s %s — %s",
            proposal.get("symbol"),
            proposal.get("direction"),
            reasoning.get("failed_rule"),
        )
        await self.log_decision("TRADE_REJECTED", reasoning=reasoning)

    # ─── Validation Engine ────────────────────────────────────────────────────

    async def _validate_trade(
        self, proposal: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all risk checks. Return (True, full_reasoning) on approval,
        or (False, reasoning_with_failed_rule) on first failure.
        """
        reasoning: Dict[str, Any] = {"checks": [], "proposal": proposal}

        # ── Fetch live account + position data ───────────────────────────────
        try:
            account, positions = self._fetch_account_state()
        except Exception as e:
            msg = f"Cannot fetch account data: {e}"
            self.logger.error(msg)
            reasoning["failed_rule"] = "account_fetch"
            reasoning["checks"].append({"rule": "account_fetch", "ok": False, "msg": msg})
            return False, reasoning

        account_value = account["portfolio_value"]
        buying_power = account["buying_power"]
        current_equity = account["equity"]

        # Update high-water mark
        if self._peak_equity is None or current_equity > self._peak_equity:
            self._peak_equity = current_equity

        symbol = proposal["symbol"]
        quantity = proposal["quantity"]
        entry_price = proposal["entry_price"]
        trade_cost = quantity * entry_price

        # ── Rule 1: Buying Power ──────────────────────────────────────────────
        ok, msg = validate_buying_power(trade_cost, buying_power, self.limits)
        reasoning["checks"].append({"rule": "buying_power", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "buying_power"
            return False, reasoning

        # ── Rule 2: Position Size ─────────────────────────────────────────────
        ok, msg = validate_position_size(trade_cost, account_value, self.limits)
        reasoning["checks"].append({"rule": "position_size", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "position_size"
            return False, reasoning

        # ── Rule 3: Sector Concentration ──────────────────────────────────────
        sector = self._sector_map.get(symbol, "UNKNOWN")
        sector_exposure = get_sector_exposure(positions, self._sector_map)
        current_sector_value = sector_exposure.get(sector, 0.0)
        ok, msg = validate_sector_concentration(
            trade_cost, current_sector_value, account_value, sector, self.limits
        )
        reasoning["checks"].append({"rule": "sector_concentration", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "sector_concentration"
            return False, reasoning

        # ── Rule 4: Daily Loss ────────────────────────────────────────────────
        daily_pnl = self._get_daily_pnl()
        ok, msg = validate_daily_loss(daily_pnl, account_value, self.limits)
        reasoning["checks"].append({"rule": "daily_loss", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "daily_loss"
            return False, reasoning

        # ── Rule 5: Account Drawdown ──────────────────────────────────────────
        ok, msg = validate_account_drawdown(current_equity, self._peak_equity, self.limits)
        reasoning["checks"].append({"rule": "account_drawdown", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "account_drawdown"
            return False, reasoning

        # ── Rule 6: Open Positions ────────────────────────────────────────────
        ok, msg = validate_open_positions(len(positions), self.limits)
        reasoning["checks"].append({"rule": "open_positions", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "open_positions"
            return False, reasoning

        # ── Rule 7: Portfolio Heat ────────────────────────────────────────────
        atr = proposal.get("atr")
        stop_loss_est = calculate_dynamic_stop_loss(entry_price, atr=atr, limits=self.limits)
        new_trade_risk = (entry_price - stop_loss_est) * quantity
        ok, msg = validate_portfolio_heat(new_trade_risk, positions, account_value, self.limits)
        reasoning["checks"].append({"rule": "portfolio_heat", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "portfolio_heat"
            return False, reasoning

        # ── Rule 8: Dynamic Stop Loss ─────────────────────────────────────────
        atr = proposal.get("atr")  # optional; caller can supply ATR
        stop_loss = calculate_dynamic_stop_loss(entry_price, atr=atr, limits=self.limits)
        stop_msg = f"Stop loss set at ${stop_loss:.2f}"
        reasoning["checks"].append({"rule": "stop_loss", "ok": True, "msg": stop_msg})
        reasoning["stop_loss"] = stop_loss

        return True, reasoning

    # ─── Data Helpers ─────────────────────────────────────────────────────────

    def _fetch_account_state(self):
        """Fetch live account info and positions from Alpaca."""
        from app.integrations import get_alpaca_client
        client = get_alpaca_client()
        account = client.get_account()
        positions = client.get_positions()
        return account, positions

    def _get_daily_pnl(self) -> float:
        """Return today's realized P&L from the database (0.0 if no record yet)."""
        db = get_session()
        try:
            today = str(date.today())
            metric = db.query(RiskMetric).filter_by(date=today).first()
            return float(metric.daily_pnl) if metric and metric.daily_pnl is not None else 0.0
        finally:
            db.close()

    def update_sector_map(self, sector_map: Dict[str, str]) -> None:
        """Update the symbol→sector mapping (called externally or at startup)."""
        self._sector_map.update(sector_map)
        self.logger.info("Sector map updated: %d symbols", len(self._sector_map))

    def pause(self) -> None:
        """Pause processing new proposals (e.g. on drawdown breach)."""
        self.status = "paused"
        self.logger.warning("Risk Manager PAUSED")

    def resume(self) -> None:
        """Resume processing after a pause."""
        self.status = "running"
        self.logger.info("Risk Manager RESUMED")


# Module-level singleton
risk_manager = RiskManager()
