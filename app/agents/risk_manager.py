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
import time
from datetime import date
from typing import Any, Dict, Optional, Tuple

from app.agents.base import BaseAgent
from app.agents.risk_rules import (
    RiskLimits,
    calculate_dynamic_stop_loss,
    validate_correlation_risk,
    get_sector_exposure,
    validate_account_drawdown,
    validate_buying_power,
    validate_daily_loss,
    validate_open_positions,
    validate_portfolio_heat,
    validate_position_size,
    validate_sector_concentration,
)
from app.database.models import RiskMetric, TradeProposal, ProposalLog, ProposalEvent
from app.database.session import get_session

logger = logging.getLogger(__name__)

TRADE_PROPOSALS_QUEUE = "trade_proposals"
APPROVED_TRADES_QUEUE = "trader_approved_trades"
MAX_INTRADAY_POSITIONS = 3    # hard cap on concurrent intraday trades
GROSS_EXPOSURE_CAP = 0.80     # never deploy > 80% of account across all strategies
SWING_BUDGET_PCT = 0.70       # swing strategy budget fraction
INTRADAY_BUDGET_PCT = 0.30    # intraday strategy budget fraction


class RiskManager(BaseAgent):
    """
    Continuous listener that validates trade proposals against risk rules
    and either forwards them for execution or rejects them with reasoning.
    """

    # How often (seconds) to reload RiskLimits from DB.  Allows operator config
    # changes to take effect quickly without hammering the DB on every proposal.
    LIMITS_TTL: float = 60.0

    def __init__(self, limits: Optional[RiskLimits] = None):
        super().__init__("risk_manager")
        self.limits = limits or RiskLimits()
        self._limits_loaded_at: float = 0.0  # monotonic clock; 0 forces first-proposal load
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

        # Load persisted peak equity on startup so drawdown rule survives restarts
        self._load_peak_equity()
        # Restore open intraday position count from DB so the cap survives restarts
        self._load_intraday_count()

        while self.status == "running":
            try:
                proposal = await asyncio.to_thread(
                    self.get_message, TRADE_PROPOSALS_QUEUE, 3
                )
                if proposal is None:
                    await asyncio.sleep(0)  # yield control
                    continue

                # Kill switch: reject all proposals immediately — no new trades allowed
                from app.live_trading.kill_switch import kill_switch
                if kill_switch.is_active:
                    self.logger.warning(
                        "Kill switch ACTIVE — rejecting proposal for %s",
                        proposal.get("symbol"),
                    )
                    db_proposal = await asyncio.to_thread(self._persist_proposal, proposal)
                    await self._reject(proposal, {"failed_rule": "kill_switch",
                                                  "message": "Kill switch is active"}, db_proposal)
                    continue

                self.logger.info(
                    "Received proposal: %s %s x%s @ $%s from %s",
                    proposal.get("symbol"),
                    proposal.get("direction"),
                    proposal.get("quantity"),
                    proposal.get("entry_price"),
                    proposal.get("source_agent", "unknown"),
                )

                db_proposal = await asyncio.to_thread(self._persist_proposal, proposal)
                is_approved, reasoning = await self._validate_trade(proposal)

                if is_approved:
                    await self._approve(proposal, reasoning, db_proposal)
                else:
                    await self._reject(proposal, reasoning, db_proposal)

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

    # ─── Proposal persistence ─────────────────────────────────────────────────

    def _persist_proposal(self, proposal: Dict[str, Any]) -> Optional[Any]:
        """Write proposal to DB immediately on receipt; returns the ORM object."""
        from datetime import datetime as _dt
        db = get_session()
        try:
            rec = TradeProposal(
                symbol=proposal.get("symbol", ""),
                trade_type=proposal.get("trade_type", "swing"),
                direction=proposal.get("direction", "BUY"),
                entry_price=proposal.get("entry_price"),
                confidence=proposal.get("confidence"),
                ml_score=proposal.get("ml_score") or proposal.get("confidence"),
                source_agent=proposal.get("source_agent", "portfolio_manager"),
                status="PENDING",
                proposed_at=_dt.utcnow(),
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)
            return rec
        except Exception as exc:
            db.rollback()
            self.logger.warning("Failed to persist proposal for %s: %s", proposal.get("symbol"), exc)
            return None
        finally:
            db.close()

    def _update_proposal_status(self, proposal_id: Optional[int], status: str, reason: str = "") -> None:
        if proposal_id is None:
            return
        from datetime import datetime as _dt
        db = get_session()
        try:
            rec = db.query(TradeProposal).filter(TradeProposal.id == proposal_id).first()
            if rec:
                rec.status = status
                rec.reject_reason = reason or None
                rec.decided_at = _dt.utcnow()
                db.commit()
        except Exception as exc:
            db.rollback()
            self.logger.warning("Failed to update proposal %s: %s", proposal_id, exc)
        finally:
            db.close()

    def _writeback_proposal_log(
        self,
        proposal_uuid: Optional[str],
        rm_status: str,
        rm_reason: str,
        rm_rule: str,
        rm_inputs: Dict[str, Any],
    ) -> None:
        """Write RM decision back to ProposalLog row (keyed by proposal_uuid) and append event."""
        if not proposal_uuid:
            return
        from datetime import datetime as _dt
        now = _dt.utcnow()
        db = get_session()
        try:
            rows = db.query(ProposalLog).filter(ProposalLog.proposal_uuid == proposal_uuid).all()
            for row in rows:
                row.rm_status = rm_status
                row.rm_reason = rm_reason[:255] if rm_reason else None
                row.rm_rule = rm_rule[:50] if rm_rule else None
                row.rm_inputs = rm_inputs
                row.rm_decided_at = now
            event_type = "RM_APPROVED" if rm_status == "APPROVED" else "RM_REJECTED"
            db.add(ProposalEvent(
                proposal_uuid=proposal_uuid,
                event_time=now,
                actor="risk_manager",
                event_type=event_type,
                details={
                    "rm_status": rm_status,
                    "rm_reason": rm_reason,
                    "rm_rule": rm_rule,
                    "rm_inputs": rm_inputs,
                },
            ))
            db.commit()
        except Exception as exc:
            db.rollback()
            self.logger.debug("ProposalLog writeback failed for %s: %s", proposal_uuid, exc)
        finally:
            db.close()

    # ─── Approval / Rejection ─────────────────────────────────────────────────

    async def _approve(self, proposal: Dict[str, Any], reasoning: Dict[str, Any], db_proposal=None) -> None:
        from datetime import datetime

        if proposal.get("trade_type") == "intraday":
            self._open_intraday_count += 1

        if db_proposal:
            await asyncio.to_thread(self._update_proposal_status, db_proposal.id, "APPROVED")

        # Write back to unified ProposalLog
        rm_inputs = {k: reasoning.get(k) for k in (
            "portfolio_heat", "open_positions", "open_intraday", "cash_available",
            "max_position_pct", "daily_loss_limit_hit",
        ) if reasoning.get(k) is not None}
        await asyncio.to_thread(
            self._writeback_proposal_log,
            proposal.get("proposal_uuid"),
            "APPROVED", "", "", rm_inputs,
        )

        approved_proposal = {
            **proposal,
            "stop_loss": reasoning.get("stop_loss"),
            "approved_at": datetime.utcnow().isoformat(),
            "_proposal_id": db_proposal.id if db_proposal else None,
        }
        self.send_message(APPROVED_TRADES_QUEUE, approved_proposal)
        self.logger.info(
            "APPROVED: %s %s — stop_loss=$%s",
            proposal.get("symbol"),
            proposal.get("direction"),
            reasoning.get("stop_loss"),
        )
        await self.log_decision("TRADE_APPROVED", reasoning=reasoning)

    async def _reject(self, proposal: Dict[str, Any], reasoning: Dict[str, Any], db_proposal=None) -> None:
        failed_rule = reasoning.get("failed_rule", "unknown")
        self.logger.warning(
            "REJECTED: %s %s — %s",
            proposal.get("symbol"),
            proposal.get("direction"),
            failed_rule,
        )
        if db_proposal:
            await asyncio.to_thread(self._update_proposal_status, db_proposal.id, "REJECTED", failed_rule)

        # Write back to unified ProposalLog
        rm_inputs = {k: reasoning.get(k) for k in (
            "portfolio_heat", "open_positions", "open_intraday", "cash_available",
            "max_position_pct", "daily_loss_limit_hit",
        ) if reasoning.get(k) is not None}
        await asyncio.to_thread(
            self._writeback_proposal_log,
            proposal.get("proposal_uuid"),
            "REJECTED",
            reasoning.get("message", failed_rule),
            failed_rule,
            rm_inputs,
        )
        # Optional AI veto explanation (non-blocking)
        try:
            from app.ai.claude_client import explain_risk_veto
            ai_explanation = explain_risk_veto(
                symbol=proposal.get("symbol", "?"),
                failed_rule=failed_rule,
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

        # Reload limits from DB at most every LIMITS_TTL seconds.
        # This lets operator config changes take effect quickly (< 1 minute) while
        # avoiding a DB round-trip on every proposal in a busy market session.
        _now = time.monotonic()
        if _now - self._limits_loaded_at >= self.LIMITS_TTL:
            try:
                _db = get_session()
                try:
                    self.limits = RiskLimits.from_db(_db)
                    self._limits_loaded_at = _now
                finally:
                    _db.close()
            except Exception as exc:
                logger.warning("Could not reload RiskLimits from DB: %s", exc)

        symbol = proposal.get("symbol", "")
        quantity = proposal.get("quantity", 0)
        entry_price = proposal.get("entry_price", 0.0)
        direction = proposal.get("direction", "BUY")
        trade_cost = quantity * entry_price

        # ── Rule 0: Symbol halt check ────────────────────────────────────────
        from app.agents.compliance import compliance_tracker
        halted, halt_reason = compliance_tracker.is_symbol_halted(symbol)
        reasoning["checks"].append({"rule": "symbol_halt", "ok": not halted, "msg": halt_reason})
        if halted:
            reasoning["failed_rule"] = "symbol_halt"
            return False, reasoning

        # ── Rule 0: Earnings calendar gate ───────────────────────────────────
        try:
            from app.calendars.earnings import earnings_calendar
            trade_type = proposal.get("trade_type", "swing")
            earnings_risk = earnings_calendar.get_earnings_risk(symbol, trade_type)
            blocked = earnings_risk.block_intraday if trade_type == "intraday" else earnings_risk.block_swing
            msg = earnings_risk.reason or (
                f"earnings in {earnings_risk.days_until}d" if earnings_risk.days_until is not None else "ok"
            )
            reasoning["checks"].append({"rule": "earnings_gate", "ok": not blocked, "msg": msg})
            if blocked:
                reasoning["failed_rule"] = "earnings_gate"
                return False, reasoning
        except Exception as _e:
            reasoning["checks"].append({"rule": "earnings_gate", "ok": True, "msg": f"skipped: {_e}"})

        # ── Rule 0: Macro event window gate ──────────────────────────────────
        try:
            from app.calendars.macro import macro_calendar
            macro_ctx = macro_calendar.get_context()
            if macro_ctx.block_new_entries:
                event_names = ", ".join(e.event_type for e in macro_ctx.events_today)
                msg = f"within event window: {event_names}"
                reasoning["checks"].append({"rule": "macro_event_window", "ok": False, "msg": msg})
                reasoning["failed_rule"] = "macro_event_window"
                return False, reasoning
            reasoning["checks"].append({
                "rule": "macro_event_window", "ok": True,
                "msg": macro_ctx.next_event or "no events",
            })
        except Exception as _e:
            reasoning["checks"].append({"rule": "macro_event_window", "ok": True, "msg": f"skipped: {_e}"})

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
        current_equity = account["equity"]

        # ── Rule 0a: PDT check (intraday only, equity < $25k) ────────────────
        if proposal.get("trade_type") == "intraday":
            pdt_blocked, pdt_msg = compliance_tracker.is_pdt_blocked(current_equity)
            reasoning["checks"].append({"rule": "pdt", "ok": not pdt_blocked, "msg": pdt_msg})
            if pdt_blocked:
                reasoning["failed_rule"] = "pdt"
                return False, reasoning

        # Use settled buying power only (Reg T: T+1 settlement)
        raw_buying_power = account["buying_power"]
        buying_power = compliance_tracker.settled_buying_power(raw_buying_power)
        if buying_power < raw_buying_power:
            self.logger.debug(
                "Buying power adjusted for unsettled cash: $%.2f → $%.2f",
                raw_buying_power, buying_power,
            )

        if self._peak_equity is None or current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._persist_peak_equity(current_equity)

        # ── Rule 0b: Gross exposure cap (80% of account) ─────────────────────
        # T-bill cash-sleeve positions (P1-1) are cash-equivalents, NOT risk — exclude them
        # so parking idle cash never inflates risk gross and starves trend/PEAD/swing.
        from app.live_trading.cash_sleeve import CASH_ETFS
        total_deployed = sum(
            abs(float(p.get("market_value") or 0))
            for p in positions if p.get("symbol") not in CASH_ETFS
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
        # Alpaca positions don't carry trade_type — resolve from DB
        try:
            from app.database.models import Trade as _BudgetTrade
            _bdb = get_session()
            try:
                _type_map = {
                    t.symbol: (t.trade_type or "swing")
                    for t in _bdb.query(_BudgetTrade)
                    .filter(_BudgetTrade.status.in_(("ACTIVE", "PENDING_FILL")))
                    .all()
                }
            finally:
                _bdb.close()
        except Exception:
            _type_map = {}
        type_deployed = sum(
            abs(float(p.get("market_value") or 0))
            for p in positions
            if p.get("symbol") not in CASH_ETFS   # T-bills aren't a strategy budget
            and _type_map.get(p.get("symbol"), "swing") == trade_type
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
        ok, msg = validate_buying_power(
            trade_cost, buying_power, self.limits, direction=proposal.get("direction", "BUY")
        )
        reasoning["checks"].append({"rule": "buying_power", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "buying_power"
            return False, reasoning

        # ── Rule 2: Position Size ─────────────────────────────────────────────
        # B4: PEAD positions use the (higher) pm.pead_max_position_pct ceiling so the
        # aggressive paper-ramp isn't silently clipped to the global 5%. Other
        # selectors keep the global limit. Falls back to the global on any error.
        _pos_override = None
        if proposal.get("selector") == "pead":
            try:
                from app.database.session import get_session as _gs_rm
                from app.database.agent_config import get_agent_config as _gac_rm
                _db_rm = _gs_rm()
                try:
                    _pos_override = float(_gac_rm(_db_rm, "pm.pead_max_position_pct"))
                finally:
                    _db_rm.close()
            except Exception:
                _pos_override = None
        ok, msg = validate_position_size(
            trade_cost, account_value, self.limits, max_pct_override=_pos_override
        )
        reasoning["checks"].append({"rule": "position_size", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "position_size"
            return False, reasoning

        # ── Rule 3: Sector Concentration ──────────────────────────────────────
        sector = self._sector_map.get(symbol, "UNKNOWN")
        sector_exposure = get_sector_exposure(positions, self._sector_map)
        current_sector_value = sector_exposure.get(sector, 0.0)
        ok, msg = validate_sector_concentration(
            trade_cost, current_sector_value, account_value, sector, self.limits,
            direction=direction,
        )
        reasoning["checks"].append({"rule": "sector_concentration", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "sector_concentration"
            return False, reasoning

        # ── Rule 3b: Correlation Risk ─────────────────────────────────────────
        open_symbols = [p["symbol"] for p in positions if p.get("symbol") != symbol]
        position_values = {
            p["symbol"]: abs(float(p.get("market_value") or (float(p.get("qty") or 0) * entry_price)))
            for p in positions if p.get("symbol")
        }
        # Direction map for sign-adjusted correlation: long+short pairs hedge, not concentrate
        _pos_directions = {
            p["symbol"]: (
                p.get("side") or ("short" if float(p.get("qty") or p.get("quantity") or 0) < 0 else "long")
            )
            for p in positions if p.get("symbol")
        }
        ok, msg = validate_correlation_risk(
            symbol, open_symbols, account_value, position_values, limits=self.limits,
            proposed_direction=proposal.get("direction", "BUY"),
            position_directions=_pos_directions,
        )
        reasoning["checks"].append({"rule": "correlation_risk", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "correlation_risk"
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
        # T-bill cash-sleeve holdings (P1-1) are not risk positions — don't let them
        # consume MAX_OPEN_POSITIONS slots and block a legitimate entry.
        _risk_pos_count = len([p for p in positions if p.get("symbol") not in CASH_ETFS])
        ok, msg = validate_open_positions(_risk_pos_count, self.limits)
        reasoning["checks"].append({"rule": "open_positions", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "open_positions"
            return False, reasoning

        # ── Rule 7: Portfolio Heat ────────────────────────────────────────────
        atr = proposal.get("atr")
        _prop_dir = proposal.get("direction", "BUY")
        stop_loss_est = calculate_dynamic_stop_loss(
            entry_price, atr=atr, limits=self.limits, direction=_prop_dir,
        )
        # Risk is always positive: abs(entry - stop) * qty regardless of direction
        new_trade_risk = abs(entry_price - stop_loss_est) * quantity
        ok, msg = validate_portfolio_heat(new_trade_risk, positions, account_value, self.limits)
        reasoning["checks"].append({"rule": "portfolio_heat", "ok": ok, "msg": msg})
        if not ok:
            reasoning["failed_rule"] = "portfolio_heat"
            return False, reasoning

        # ── Rule 8a: Bid-Ask Spread (execution quality) ───────────────────────
        if proposal.get("trade_type") == "swing":
            try:
                max_spread = self.limits.max_spread_pct
                from app.integrations import get_alpaca_client
                quote = get_alpaca_client().get_quote(symbol)
                if quote is not None:
                    spread_pct = quote["spread_pct"]
                    # Spreads > 2% are stale IEX quotes, not real market spreads — ignore
                    if spread_pct <= 0.02 and spread_pct > max_spread:
                        msg = (f"Bid-ask spread too wide: {spread_pct*100:.3f}% "
                               f"(limit {max_spread*100:.3f}%)")
                        reasoning["failed_rule"] = "bid_ask_spread"
                        reasoning["checks"].append({"rule": "bid_ask_spread", "ok": False, "msg": msg})
                        return False, reasoning
                    reasoning["checks"].append({
                        "rule": "bid_ask_spread", "ok": True,
                        "msg": f"spread={spread_pct*100:.3f}% (limit {max_spread*100:.3f}%)"
                    })
                else:
                    reasoning["checks"].append({
                        "rule": "bid_ask_spread", "ok": True,
                        "msg": "quote unavailable — skipped",
                    })
            except Exception as exc:
                self.logger.debug("Spread check error for %s: %s", symbol, exc)
                reasoning["checks"].append({
                    "rule": "bid_ask_spread", "ok": True,
                    "msg": f"spread check skipped: {exc}",
                })

        # ── Rule 8b: ADTV Liquidity Gate ─────────────────────────────────────
        if proposal.get("trade_type") == "swing":
            try:
                max_adtv_pct = self.limits.max_adtv_pct
                from app.integrations import get_alpaca_client
                bars_df = get_alpaca_client().get_bars(symbol, timeframe="1D", limit=20)
                if bars_df is not None and len(bars_df) >= 5:
                    adtv = float((bars_df["close"] * bars_df["volume"]).mean())
                    if adtv > 0:
                        adtv_pct = trade_cost / adtv
                        if adtv_pct > max_adtv_pct:
                            msg = (f"Trade size too large vs ADTV: "
                                   f"${trade_cost:,.0f} = {adtv_pct*100:.2f}% of ADTV "
                                   f"(limit {max_adtv_pct*100:.1f}%)")
                            reasoning["failed_rule"] = "adtv_liquidity"
                            reasoning["checks"].append({"rule": "adtv_liquidity", "ok": False, "msg": msg})
                            return False, reasoning
                        reasoning["checks"].append({
                            "rule": "adtv_liquidity", "ok": True,
                            "msg": f"trade={adtv_pct*100:.2f}% of ADTV (limit {max_adtv_pct*100:.1f}%)"
                        })
                else:
                    reasoning["checks"].append({
                        "rule": "adtv_liquidity", "ok": True,
                        "msg": "insufficient ADTV data — skipped",
                    })
            except Exception as exc:
                self.logger.debug("ADTV check error for %s: %s", symbol, exc)
                reasoning["checks"].append({"rule": "adtv_liquidity", "ok": True, "msg": f"ADTV check skipped: {exc}"})

        # ── Rule 9: Correlation Check (swing only) ───────────────────────────
        if proposal.get("trade_type") == "swing" and positions:
            corr_result = await asyncio.to_thread(
                self._check_correlation, symbol, positions, self.limits.max_correlation,
                proposal.get("direction", "BUY"),
            )
            reasoning["checks"].append({"rule": "correlation", **corr_result})
            if not corr_result["ok"]:
                reasoning["failed_rule"] = "correlation"
                return False, reasoning

        # ── Rule 10: Beta-Adjusted Exposure (swing only) ─────────────────────
        if proposal.get("trade_type") == "swing":
            beta_result = await asyncio.to_thread(
                self._check_beta_exposure, symbol, trade_cost, positions,
                account_value, self.limits, proposal,
            )
            reasoning["checks"].append({"rule": "beta_exposure", **beta_result})
            if not beta_result["ok"]:
                reasoning["failed_rule"] = "beta_exposure"
                return False, reasoning

        # ── Rule 11: Factor/Sector Concentration (swing only) ────────────────
        if proposal.get("trade_type") == "swing":
            factor_result = self._check_factor_concentration(
                symbol, trade_cost, positions, account_value,
                self.limits.max_factor_concentration, proposal,
            )
            reasoning["checks"].append({"rule": "factor_concentration", **factor_result})
            if not factor_result["ok"]:
                reasoning["failed_rule"] = "factor_concentration"
                return False, reasoning

        # ── Rule 8: Dynamic Stop Loss ─────────────────────────────────────────
        atr = proposal.get("atr")
        _rm_dir = proposal.get("direction", "BUY")
        _rm_short = _rm_dir == "SELL_SHORT"
        if proposal.get("trade_type") == "intraday":
            # Intraday: use proposal stop or tight default (direction-aware)
            if _rm_short:
                intraday_stop = proposal.get("stop_loss") or round(entry_price * 1.005, 2)
                # Clamp: stop must not be looser (higher) than 0.5% above entry
                stop_loss = min(intraday_stop, entry_price * 1.005)
            else:
                intraday_stop = proposal.get("stop_loss") or round(entry_price * 0.995, 2)
                # Clamp: stop must not be looser (lower) than 0.5% below entry; use max to enforce floor
                stop_loss = max(intraday_stop, entry_price * 0.995)
        else:
            stop_loss = calculate_dynamic_stop_loss(
                entry_price, atr=atr, limits=self.limits, direction=_rm_dir,
            )
        stop_msg = f"Stop loss set at ${stop_loss:.2f}"
        reasoning["checks"].append({"rule": "stop_loss", "ok": True, "msg": stop_msg})
        reasoning["stop_loss"] = stop_loss

        # ── Wash sale warning (non-blocking) ─────────────────────────────────
        is_wash, wash_msg = compliance_tracker.check_wash_sale(symbol, proposed_direction=_rm_dir)
        reasoning["checks"].append({"rule": "wash_sale", "ok": True, "msg": wash_msg})
        if is_wash:
            self.logger.warning("WASH SALE: %s", wash_msg)
            reasoning["wash_sale_warning"] = wash_msg

        return True, reasoning

    # ─── Phase 19: Risk Intelligence Helpers ─────────────────────────────────

    def _check_correlation(
        self, symbol: str, positions: list, max_corr: float,
        proposed_direction: str = "BUY",
    ) -> dict:
        """
        Compute 60-day return correlation between proposed symbol and each open
        position. Returns ok=False with the worst offender if any exceed max_corr.
        Skips check (ok=True) if bars unavailable.

        Direction-aware: a short proposal against a long position is a hedge —
        effective correlation is negated so opposite-direction pairs don't
        incorrectly trigger the concentration limit.
        """
        try:
            import numpy as np
            from app.integrations import get_alpaca_client
            client = get_alpaca_client()

            bars_new = client.get_bars(symbol, timeframe="1Day", limit=65)
            if bars_new is None or len(bars_new) < 20:
                return {"ok": True, "msg": "correlation: insufficient bars — skipped"}

            returns_new = bars_new["close"].pct_change().dropna().values
            worst_corr, worst_sym = 0.0, ""

            # Build direction map for existing positions
            _existing_dirs = {
                p["symbol"]: (
                    p.get("side") or ("short" if float(p.get("qty") or p.get("quantity") or 0) < 0 else "long")
                )
                for p in positions if p.get("symbol")
            }
            _prop_is_short = proposed_direction == "SELL_SHORT"

            open_syms = [p["symbol"] for p in positions if p.get("symbol") != symbol]
            for open_sym in open_syms:
                try:
                    bars_open = client.get_bars(open_sym, timeframe="1Day", limit=65)
                    if bars_open is None or len(bars_open) < 20:
                        continue
                    returns_open = bars_open["close"].pct_change().dropna().values
                    n = min(len(returns_new), len(returns_open))
                    if n < 10:
                        continue
                    corr = float(np.corrcoef(returns_new[-n:], returns_open[-n:])[0, 1])
                    # Opposite directions hedge each other — negate effective correlation
                    _existing_is_short = _existing_dirs.get(open_sym) in ("short", "SELL_SHORT")
                    if _prop_is_short != _existing_is_short:
                        corr = -corr
                    if corr > worst_corr:
                        worst_corr, worst_sym = corr, open_sym
                except Exception:
                    continue

            if worst_corr > max_corr:
                return {
                    "ok": False,
                    "msg": (f"Correlation {worst_corr:.2f} with {worst_sym} "
                            f"exceeds limit {max_corr:.2f}"),
                }
            return {
                "ok": True,
                "msg": f"max correlation={worst_corr:.2f} with {worst_sym or 'none'} (limit {max_corr:.2f})",
            }
        except Exception as exc:
            self.logger.debug("Correlation check error for %s: %s", symbol, exc)
            return {"ok": True, "msg": f"correlation check skipped: {exc}"}

    def _compute_beta(self, symbol: str, lookback: int = 252) -> float:
        """
        Compute beta of symbol vs SPY using OLS regression on daily returns.
        Returns 1.0 on failure (neutral assumption).
        """
        try:
            import numpy as np
            from app.integrations import get_alpaca_client
            client = get_alpaca_client()
            bars_sym = client.get_bars(symbol, timeframe="1Day", limit=lookback + 5)
            bars_spy = client.get_bars("SPY", timeframe="1Day", limit=lookback + 5)
            if bars_sym is None or bars_spy is None:
                return 1.0
            r_sym = bars_sym["close"].pct_change().dropna().values
            r_spy = bars_spy["close"].pct_change().dropna().values
            n = min(len(r_sym), len(r_spy))
            if n < 30:
                return 1.0
            r_sym, r_spy = r_sym[-n:], r_spy[-n:]
            cov = float(np.cov(r_sym, r_spy)[0, 1])
            var = float(np.var(r_spy))
            return round(cov / var, 4) if var > 0 else 1.0
        except Exception:
            return 1.0

    def _check_beta_exposure(
        self, symbol: str, trade_cost: float, positions: list,
        account_value: float, limits, proposal: dict = None,
    ) -> dict:
        """
        Compute current portfolio beta. If portfolio beta > max_portfolio_beta
        and the proposed stock's beta > high_beta_threshold, reject.
        """
        try:
            # Portfolio beta = Σ(signed_position_value × beta_i) / account_value
            # Shorts have negative market_value from Alpaca → subtract beta (they hedge)
            portfolio_beta = 0.0
            for pos in positions:
                pos_sym = pos.get("symbol", "")
                pos_val = float(pos.get("market_value") or 0)  # signed: negative for shorts
                pos_beta = self._compute_beta(pos_sym, lookback=63)
                portfolio_beta += pos_val * pos_beta
            portfolio_beta = portfolio_beta / max(account_value, 1.0)

            new_beta = self._compute_beta(symbol, lookback=63)
            # For SELL_SHORT, the proposed trade reduces beta exposure
            _beta_sign = -1 if (proposal or {}).get("direction") == "SELL_SHORT" else 1
            prospective_beta = (
                portfolio_beta * account_value + _beta_sign * trade_cost * new_beta
            ) / max(account_value + trade_cost, 1.0)

            if (prospective_beta > limits.max_portfolio_beta
                    and new_beta > limits.high_beta_threshold):
                return {
                    "ok": False,
                    "msg": (f"Portfolio beta {portfolio_beta:.2f} > limit {limits.max_portfolio_beta:.2f} "
                            f"and {symbol} beta={new_beta:.2f} > {limits.high_beta_threshold:.2f}"),
                }
            return {
                "ok": True,
                "msg": (f"portfolio_beta={portfolio_beta:.2f} prospective={prospective_beta:.2f} "
                        f"{symbol}_beta={new_beta:.2f}"),
            }
        except Exception as exc:
            self.logger.debug("Beta check error for %s: %s", symbol, exc)
            return {"ok": True, "msg": f"beta check skipped: {exc}"}

    def _check_factor_concentration(
        self, symbol: str, trade_cost: float, positions: list,
        account_value: float, max_factor_conc: float, proposal: dict = None,
    ) -> dict:
        """
        Use sector as a factor proxy. Reject if adding this position would push
        NET sector exposure above max_factor_concentration of account value.
        Shorts subtract from sector exposure (market-neutral books don't double-count).
        """
        try:
            new_sector = self._sector_map.get(symbol, "UNKNOWN")
            # Proposed trade: shorts reduce sector exposure
            _conc_sign = -1 if (proposal or {}).get("direction") == "SELL_SHORT" else 1
            sector_value = _conc_sign * trade_cost
            for pos in positions:
                pos_sym = pos.get("symbol", "")
                if self._sector_map.get(pos_sym, "UNKNOWN") == new_sector:
                    # Use signed market_value: Alpaca returns negative for shorts
                    sector_value += float(pos.get("market_value") or 0)

            sector_pct = sector_value / max(account_value, 1.0)
            if abs(sector_pct) > max_factor_conc:
                return {
                    "ok": False,
                    "msg": (f"Factor/sector '{new_sector}' concentration {sector_pct:.1%} "
                            f"> limit {max_factor_conc:.1%}"),
                }
            return {
                "ok": True,
                "msg": f"sector '{new_sector}' concentration={sector_pct:.1%} (limit {max_factor_conc:.1%})",
            }
        except Exception as exc:
            self.logger.debug("Factor concentration check error for %s: %s", symbol, exc)
            return {"ok": True, "msg": f"factor check skipped: {exc}"}

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

    # ── Phase 82: Persist peak equity across restarts ─────────────────────────

    def _load_peak_equity(self) -> None:
        """Restore peak equity high-water mark from DB on startup."""
        try:
            from app.database.config_store import get_config
            db = get_session()
            try:
                val = get_config(db, "risk.peak_equity")
                if val is not None:
                    self._peak_equity = float(val)
                    self.logger.info("Peak equity restored from DB: $%.2f", self._peak_equity)
            finally:
                db.close()
        except Exception as exc:
            self.logger.warning("Could not load peak equity from DB: %s", exc)

    def _persist_peak_equity(self, value: float) -> None:
        """Persist new peak equity high-water mark to DB."""
        try:
            from app.database.config_store import set_config
            db = get_session()
            try:
                set_config(db, "risk.peak_equity", value, "Drawdown high-water mark")
            finally:
                db.close()
        except Exception as exc:
            self.logger.debug("Could not persist peak equity: %s", exc)

    def _load_intraday_count(self) -> None:
        """Restore open intraday position count from DB so the cap survives restarts."""
        try:
            from app.database.models import Trade as _Trade
            db = get_session()
            try:
                count = (
                    db.query(_Trade)
                    .filter(
                        _Trade.status.in_(("ACTIVE", "PENDING_FILL")),
                        _Trade.trade_type == "intraday",
                    )
                    .count()
                )
                self._open_intraday_count = count
                if count > 0:
                    self.logger.info("Intraday count restored from DB: %d open", count)
            finally:
                db.close()
        except Exception as exc:
            self.logger.warning("Could not restore intraday count: %s", exc)

    def reload_limits(self) -> None:
        """Force an immediate reload of RiskLimits from DB (bypasses TTL).

        Call this from the admin API after changing risk config so changes take
        effect on the next proposal rather than waiting up to LIMITS_TTL seconds.
        """
        try:
            _db = get_session()
            try:
                self.limits = RiskLimits.from_db(_db)
                self._limits_loaded_at = time.monotonic()
                self.logger.info("RiskLimits reloaded from DB (admin-triggered)")
            finally:
                _db.close()
        except Exception as exc:
            self.logger.warning("reload_limits failed: %s", exc)

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
