"""
Phase 74 — NIS & Decision Audit API endpoints.

GET /api/nis/macro          — today's MacroContext (risk level, sizing, rationale)
GET /api/nis/signals        — cached NewsSignal per symbol (from morning digest)
GET /api/nis/cost           — LLM cost summary from llm_call_log
GET /api/decision-audit/summary   — gate_performance_summary() (block reason → avg P&L)
GET /api/decision-audit/recent    — last N decision_audit rows
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from app.database.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nis", tags=["nis"])
audit_router = APIRouter(prefix="/api/decision-audit", tags=["decision-audit"])


# ── NIS: Macro Context ────────────────────────────────────────────────────────

@router.get("/macro")
def get_macro_context() -> Dict[str, Any]:
    """Return today's NIS Tier 1 MacroContext as JSON."""
    try:
        from app.agents.premarket import premarket_intel
        ctx = premarket_intel.macro_context
        if ctx is None:
            return {"status": "not_yet_run", "detail": "Premarket routine has not run yet today"}
        return {
            "as_of": ctx.as_of.isoformat() if hasattr(ctx.as_of, "isoformat") else str(ctx.as_of),
            "overall_risk": ctx.overall_risk,
            "block_new_entries": ctx.block_new_entries,
            "global_sizing_factor": ctx.global_sizing_factor,
            "rationale": ctx.rationale,
            "events_today": [
                {
                    "event_type": e.event_type,
                    "event_time": e.event_time,
                    "risk_level": e.risk_level,
                    "direction": e.direction,
                    "sizing_factor": e.sizing_factor,
                    "block_new_entries": e.block_new_entries,
                    "consensus_summary": e.consensus_summary,
                    "rationale": e.rationale,
                    "already_priced_in": e.already_priced_in,
                }
                for e in ctx.events_today
            ],
        }
    except Exception as exc:
        logger.warning("GET /api/nis/macro failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


# ── NIS: Cached Stock Signals ─────────────────────────────────────────────────

@router.get("/signals")
def get_stock_signals() -> Dict[str, Any]:
    """Return all currently cached NewsSignal entries from the in-memory stock cache."""
    try:
        from app.news.intelligence_service import _stock_cache
        import time as _time
        now = _time.time()
        signals = []
        for symbol, sig in sorted(_stock_cache.items()):
            age_s = int(now - sig.evaluated_at.replace(tzinfo=timezone.utc).timestamp()
                        if sig.evaluated_at.tzinfo else
                        now - sig.evaluated_at.timestamp())
            signals.append({
                "symbol": symbol,
                "action_policy": sig.action_policy,
                "direction_score": sig.direction_score,
                "materiality_score": sig.materiality_score,
                "downside_risk_score": sig.downside_risk_score,
                "upside_catalyst_score": sig.upside_catalyst_score,
                "confidence": sig.confidence,
                "already_priced_in_score": sig.already_priced_in_score,
                "sizing_multiplier": sig.sizing_multiplier,
                "rationale": sig.rationale,
                "top_headlines": sig.top_headlines,
                "scorer_tier": sig.scorer_tier,
                "age_seconds": age_s,
                "evaluated_at": sig.evaluated_at.isoformat() if hasattr(sig.evaluated_at, "isoformat") else str(sig.evaluated_at),
            })
        return {
            "count": len(signals),
            "signals": signals,
            "blocked": [s["symbol"] for s in signals if s["action_policy"] == "block_entry"],
            "sized_down": [s["symbol"] for s in signals if "size_down" in s["action_policy"]],
        }
    except Exception as exc:
        logger.warning("GET /api/nis/signals failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


# ── NIS: LLM Cost Summary ─────────────────────────────────────────────────────

@router.get("/cost")
def get_llm_cost(days: int = Query(default=7, ge=1, le=90)) -> Dict[str, Any]:
    """Return LLM cost summary from llm_call_log for the last N days."""
    try:
        from app.database.models import LLMCallLog
        from sqlalchemy import func
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with get_session() as db:
            rows = (
                db.query(
                    LLMCallLog.call_type,
                    func.count(LLMCallLog.id).label("calls"),
                    func.sum(LLMCallLog.cost_usd).label("total_cost"),
                    func.sum(LLMCallLog.input_tokens).label("input_tokens"),
                    func.sum(LLMCallLog.output_tokens).label("output_tokens"),
                    func.avg(LLMCallLog.latency_ms).label("avg_latency_ms"),
                    func.sum(LLMCallLog.cache_hit.cast("integer")).label("cache_hits"),
                )
                .filter(LLMCallLog.called_at >= cutoff)
                .group_by(LLMCallLog.call_type)
                .all()
            )
            total_cost = sum(float(r.total_cost or 0) for r in rows)
            return {
                "period_days": days,
                "total_cost_usd": round(total_cost, 4),
                "daily_avg_usd": round(total_cost / days, 4),
                "by_call_type": [
                    {
                        "call_type": r.call_type,
                        "calls": r.calls,
                        "cost_usd": round(float(r.total_cost or 0), 4),
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "avg_latency_ms": round(float(r.avg_latency_ms or 0)),
                        "cache_hit_rate": round(int(r.cache_hits or 0) / max(r.calls, 1), 3),
                    }
                    for r in rows
                ],
            }
    except Exception as exc:
        logger.warning("GET /api/nis/cost failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


# ── Decision Audit: Gate Performance ─────────────────────────────────────────

@audit_router.get("/summary")
def get_audit_summary() -> Dict[str, Any]:
    """
    Return gate_performance_summary — for each block_reason, how many trades
    were blocked and what was the avg realized P&L of those missed trades.
    Requires ~2 weeks of data to be meaningful.
    """
    try:
        from app.database.decision_audit import gate_performance_summary
        rows = gate_performance_summary()
        return {
            "gate_summary": rows,
            "interpretation": "Positive avg_pnl_pct = gate blocked winners (bad). Negative = gate blocked losers (good).",
        }
    except Exception as exc:
        logger.warning("GET /api/decision-audit/summary failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


@audit_router.get("/recent")
def get_recent_decisions(
    limit: int = Query(default=50, ge=1, le=500),
    strategy: Optional[str] = Query(default=None),
    final_decision: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return the most recent decision_audit rows, optionally filtered."""
    try:
        from app.database.models import DecisionAudit
        with get_session() as db:
            q = db.query(DecisionAudit).order_by(DecisionAudit.decided_at.desc())
            if strategy:
                q = q.filter(DecisionAudit.strategy == strategy)
            if final_decision:
                q = q.filter(DecisionAudit.final_decision == final_decision)
            rows = q.limit(limit).all()
            return {
                "count": len(rows),
                "decisions": [
                    {
                        "id": r.id,
                        "decided_at": r.decided_at.isoformat(),
                        "symbol": r.symbol,
                        "strategy": r.strategy,
                        "final_decision": r.final_decision,
                        "model_score": r.model_score,
                        "size_multiplier": r.size_multiplier,
                        "block_reason": r.block_reason,
                        "news_action_policy": r.news_action_policy,
                        "news_materiality": r.news_materiality,
                        "macro_risk_level": r.macro_risk_level,
                        "outcome_pnl_pct": r.outcome_pnl_pct,
                    }
                    for r in rows
                ],
            }
    except Exception as exc:
        logger.warning("GET /api/decision-audit/recent failed: %s", exc)
        return {"status": "error", "detail": str(exc)}
