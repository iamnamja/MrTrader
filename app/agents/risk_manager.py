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
MAX_INTRADAY_POSITIONS = 3   # hard cap on concurrent intraday trades
GROSS_EXPOSURE_CAP    = 0.80  # never deploy > 80% of account across all strategies
SWING_BUDGET_PCT      = 0.70  # swing strategy budget fraction
INTRADAY_BUDGET_PCT   = 0.30  # intraday strategy budget fraction


class RiskManager(BaseAgent):
    """
    Continuous listener that validates trade proposals against risk rules
    and either forwards them for execution or rejects them with reasoning.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        super().__init__("risk_manager")
        self.limits = limits or RiskLimits()
        from app.utils.constants import SECTOR_MAP
        self._sector_map: Dict[str, str] = dict(SECTOR_MAP)
        self._peak_equity: Optional[float] = None
        # Count of open intraday positions (updated on approve/exit messages)
        self._open_intraday_count: int = 0

    # ─── Main Loop ────────────────────────────────────────────────────────────

    async def run(self):
        """Continuously consume trade_proposals and validate them."""
        self.logger.info("Risk Manager started — listening on '%s'", TRADE_PROPOSALS_QUEUE)
        self.status = "running"

        while self.status == "running":
            try:
                proposal = await asyncio.to_thread(
                    self.get_message, TRADE_PROPOSALS_QUEUE, 3
                )
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

        if proposal.get("trade_type") == "intraday":
            self._open_intraday_count += 1

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
        # Optional AI veto explanation (non-blocking)
        try:
            from app.ai.claude_client import explain_risk_veto
            ai_explanation = explain_risk_veto(
                symbol=proposal.get("symbol", "?"),
                failed_rule=reasoning.get("failed_rule", "unknown"),
                veto_reason=reasoning.get("message", ""),
                proposal=proposal,
            )
            if ai_explanation:
                reasoning["ai_veto_explanation"] = ai_explanation
        except Exception:
            pass
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

        # Reload limits from DB so UI changes take effect without restart
        try:
            from app.database.session import get_session
            _db = get_session()
            try:
                self.limits = RiskLimits.from_db(_db)
            finally:
                _db.close()
        except Exception:
            pass

        symbol = proposal.get("symbol", "")
        quantity = proposal.get("quantity", 0)
        entry_price = proposal.get("entry_price", 0.0)
        trade_cost = quantity * entry_price

        # ── Rule 0: Intraday position cap ────────────────────────────────────
        if proposal.get("trade_type") == "intraday":
            if self._open_intraday_count >= MAX_INTRADAY_POSITIONS:
                msg = (f"Intraday position cap reached "
                       f"({self._open_intraday_count}/{MAX_INTRADAY_POSITIONS})")
                reasoning["failed_rule"] = "intraday_position_cap"
                reasoning["checks"].append({"rule": "intraday_position_cap", "ok": False,
                                            "msg": msg})
                return False, reasoning
            reasoning["checks"].append({"rule": "intraday_position_cap", "ok": True,
                                        "msg": f"intraday count={self._open_intraday_count}"})

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

        if self._peak_equity is None or current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # ── Rule 0b: Gross exposure cap (80% of account) ─────────────────────
        total_deployed = sum(
            abs(float(p.get("market_value") or 0)) for p in positions
        )
        gross_pct = (total_deployed + trade_cost) / max(account_value, 1.0)
        if gross_pct > GROSS_EXPOSURE_CAP:
            msg = (f"Gross exposure cap: adding ${trade_cost:,.0f} would bring "
                   f"total to {gross_pct:.1%} (limit {GROSS_EXPOSURE_CAP:.0%})")
            reasoning["failed_rule"] = "gross_exposure_cap"
            reasoning["checks"].append({"rule": "gross_exposure_cap", "ok": False, "msg": msg})
            return False, reasoning
        reasoning["checks"].append({
            "rule": "gross_exposure_cap", "ok": True,
            "msg": f"gross={gross_pct:.1%} (limit {GROSS_EXPOSURE_CAP:.0%})"
        })

        # ── Rule 0c: Strategy budget cap ─────────────────────────────────────
        trade_type = proposal.get("trade_type", "swing")
        budget_pct = SWING_BUDGET_PCT if trade_type == "swing" else INTRADAY_BUDGET_PCT
        strategy_budget = account_value * budget_pct
        type_deployed = sum(
            abs(float(p.get("market_value") or 0))
            for p in positions
            if (p.get("trade_type") or "swing") == trade_type
        )
        type_pct = (type_deployed + trade_cost) / max(account_value, 1.0)
        if type_pct > budget_pct:
            msg = (f"{trade_type} budget cap: adding ${trade_cost:,.0f} would use "
                   f"{type_pct:.1%} of account (limit {budget_pct:.0%})")
            reasoning["failed_rule"] = "strategy_budget_cap"
            reasoning["checks"].append({"rule": "strategy_budget_cap", "ok": False, "msg": msg})
            return False, reasoning
        reasoning["checks"].append({
            "rule": "strategy_budget_cap", "ok": True,
            "msg": f"{trade_type} deployed={type_pct:.1%} (limit {budget_pct:.0%})"
        })

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
        atr = proposal.get("atr")
        if proposal.get("trade_type") == "intraday":
            # Intraday: use tight 0.5% stop from proposal or 0.5% of entry
            intraday_stop = proposal.get("stop_loss") or round(entry_price * 0.995, 2)
            stop_loss = min(intraday_stop, entry_price * 0.995)
        else:
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

    def on_intraday_position_closed(self) -> None:
        """Call when an intraday position is closed to free the slot."""
        self._open_intraday_count = max(0, self._open_intraday_count - 1)

    def reset_intraday_count(self) -> None:
        """Reset daily intraday counter (call at EOD after force-close)."""
        self._open_intraday_count = 0

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
