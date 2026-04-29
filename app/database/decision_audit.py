"""
Repository functions for the decision_audit table (Phase 61).

PM calls write_decision() at every enter/block choice so we can later
answer: did each gate (news, macro, correlation) block winners or losers?
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def write_decision(
    symbol: str,
    strategy: str,
    final_decision: str,
    *,
    model_score: Optional[float] = None,
    size_multiplier: float = 1.0,
    block_reason: Optional[str] = None,
    news_signal=None,           # NewsSignal dataclass or None
    macro_context=None,         # MacroContext dataclass or None
    top_features: Optional[dict] = None,  # {feature: value} top model inputs at decision time
) -> None:
    """
    Persist one PM decision row.  Fails silently — never blocks trading.

    final_decision: 'enter' | 'block' | 'size_down' | 'exit_review'
    """
    try:
        from app.database.session import get_session
        from app.database.models import DecisionAudit

        row = DecisionAudit(
            id=str(uuid.uuid4()),
            decided_at=datetime.now(timezone.utc),
            symbol=symbol,
            strategy=strategy,
            model_score=model_score,
            final_decision=final_decision,
            size_multiplier=size_multiplier,
            block_reason=block_reason,
        )

        if news_signal is not None:
            row.news_action_policy = getattr(news_signal, "action_policy", None)
            row.news_direction_score = getattr(news_signal, "direction_score", None)
            row.news_materiality = getattr(news_signal, "materiality_score", None)
            row.news_sizing_multiplier = getattr(news_signal, "sizing_multiplier", None)
            row.news_rationale = getattr(news_signal, "rationale", None)

        if macro_context is not None:
            row.macro_risk_level = getattr(macro_context, "overall_risk", None)
            row.macro_sizing_factor = getattr(macro_context, "global_sizing_factor", None)

        if top_features:
            row.top_features = top_features

        with get_session() as db:
            db.add(row)
            db.commit()

    except Exception as exc:
        logger.debug("decision_audit write failed (non-fatal): %s", exc)


def backfill_outcomes(lookback_days: int = 7) -> int:
    """
    EOD script: fill outcome_pnl_pct for decision_audit rows where a trade was entered.
    Returns number of rows updated.
    """
    updated = 0
    try:
        from app.database.session import get_session
        from app.database.models import DecisionAudit, Trade
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        with get_session() as db:
            rows = (
                db.query(DecisionAudit)
                .filter(
                    DecisionAudit.final_decision == "enter",
                    DecisionAudit.outcome_pnl_pct.is_(None),
                    DecisionAudit.decided_at >= cutoff,
                )
                .all()
            )

            for row in rows:
                trade = (
                    db.query(Trade)
                    .filter(
                        Trade.symbol == row.symbol,
                        Trade.created_at >= row.decided_at - timedelta(minutes=30),
                        Trade.status == "CLOSED",
                    )
                    .order_by(Trade.created_at)
                    .first()
                )
                if trade and trade.pnl is not None and trade.entry_price:
                    row.outcome_pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
                    updated += 1

            db.commit()

    except Exception as exc:
        logger.warning("decision_audit backfill failed: %s", exc)

    return updated


def gate_performance_summary() -> list[dict]:
    """
    Return aggregate stats per block_reason — the calibration query.
    Shows whether each gate is blocking winners or losers.
    """
    try:
        from app.database.session import get_session
        from app.database.models import DecisionAudit
        from sqlalchemy import func

        with get_session() as db:
            rows = (
                db.query(
                    DecisionAudit.block_reason,
                    func.count(DecisionAudit.id).label("count"),
                    func.avg(DecisionAudit.outcome_pnl_pct).label("avg_pnl_pct"),
                )
                .filter(DecisionAudit.final_decision == "block")
                .group_by(DecisionAudit.block_reason)
                .all()
            )
            return [
                {
                    "block_reason": r.block_reason,
                    "count": r.count,
                    "avg_pnl_pct": round(float(r.avg_pnl_pct or 0), 4),
                }
                for r in rows
            ]
    except Exception as exc:
        logger.warning("gate_performance_summary failed: %s", exc)
        return []
