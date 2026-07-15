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
import time
from datetime import datetime, timezone, timedelta, date
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from app.database.session import get_session
from app.news.macro_polarity import classify_outcome

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nis", tags=["nis"])
audit_router = APIRouter(prefix="/api/decision-audit", tags=["decision-audit"])

# ── Macro-actuals display back-fill ───────────────────────────────────────────
# The Macro-Intel snapshot is built once premarket (~09:00 ET), so an event released LATER that
# day (e.g. 08:30 jobless claims whose actual FMP posts with a lag) stays actual=None -> "Pending".
# This DISPLAY-ONLY helper re-fetches today's released events (include_past_today=True — a path the
# trading gate does NOT use) and back-fills the actual on read, cached to avoid hammering FMP.
# It changes NOTHING about the cached trading decision (risk/sizing/block); it only fills the actual.
_actuals_cache: Dict[str, tuple] = {}   # {today_iso: (fetched_at_epoch, {event_type: {...}})}
_ACTUALS_TTL_SEC = 600                   # 10 min


def _todays_event_actuals() -> Dict[str, Dict[str, Any]]:
    """Fresh FMP metadata for the display window (today + 1-day look-ahead), keyed by event_type,
    cached 10 min: the released actual/estimate/prior AND the full ISO event_time_utc.

    The event_time_utc back-fill matters because a snapshot built by older code (or served from the
    DB before today's premarket re-runs) carries only a date-less "HH:MM UTC" string, so the UI
    cannot tell a same-day print from a next-day look-ahead one. Re-deriving the full timestamp from
    the live calendar lets the display show the real ET date without waiting for the next premarket.

    Never raises (returns {} on any failure -> the display degrades to the snapshot's own fields)."""
    today = date.today().isoformat()
    cached = _actuals_cache.get(today)
    if cached and (time.time() - cached[0]) < _ACTUALS_TTL_SEC:
        return cached[1]
    out: Dict[str, Dict[str, Any]] = {}
    try:
        from app.news.sources.fmp_source import fetch_economic_calendar
        # days_ahead=1 so the look-ahead events that hold the block (e.g. tomorrow's PPI) are covered
        # too — their full date is exactly what the operator needs to see they are not yet due.
        events = fetch_economic_calendar(days_ahead=1, min_impact="low",
                                         include_past_today=True) or []
        for e in events:
            et = e.get("event_type")
            if not et:
                continue
            rec = out.setdefault(et, {})
            evt_time = e.get("event_time")
            if evt_time is not None and not rec.get("event_time_utc"):
                rec["event_time_utc"] = (evt_time.isoformat() if hasattr(evt_time, "isoformat")
                                         else str(evt_time))
            if e.get("actual") is not None:
                rec.update({"actual": e.get("actual"), "estimate": e.get("estimate"),
                            "prior": e.get("prior")})
    except Exception:
        logger.debug("macro actuals back-fill fetch failed (non-fatal)", exc_info=True)
    _actuals_cache[today] = (time.time(), out)
    return out


def _enrich_event(ev: Dict[str, Any], meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Back-fill a still-Pending event's actual + its missing full timestamp from the fresh fetch,
    then (re)classify the outcome."""
    got = meta.get(ev.get("event_type"))
    if got:
        if ev.get("actual") is None and got.get("actual") is not None:
            ev = {**ev, "actual": got["actual"],
                  "estimate": ev.get("estimate") if ev.get("estimate") is not None else got.get("estimate"),
                  "prior": ev.get("prior") if ev.get("prior") is not None else got.get("prior")}
        # supply the full ISO timestamp when the snapshot only has the date-less display string
        if not ev.get("event_time_utc") and got.get("event_time_utc"):
            ev = {**ev, "event_time_utc": got["event_time_utc"]}
    return {**ev, **classify_outcome(ev.get("event_type"), ev.get("actual"), ev.get("estimate"))}


# ── NIS: Macro Context ────────────────────────────────────────────────────────

@router.get("/macro")
def get_macro_context() -> Dict[str, Any]:
    """Return today's NIS Tier 1 MacroContext as JSON.

    Primary source: in-memory premarket_intel (populated at 09:00 ET).
    Fallback: most recent nis_macro_snapshots DB row (survives server restarts).
    """
    try:
        from app.agents.premarket import premarket_intel
        ctx = premarket_intel.macro_context
        if ctx is not None:
            try:
                _is_today = ctx.as_of.date() == date.today() if hasattr(ctx.as_of, "date") else True
            except Exception:
                _is_today = True
            # Back-fill actuals for events released after the ~09:00 snapshot (today only).
            actuals = _todays_event_actuals() if _is_today else {}
            return {
                "as_of": ctx.as_of.isoformat() if hasattr(ctx.as_of, "isoformat") else str(ctx.as_of),
                "overall_risk": ctx.overall_risk,
                "block_new_entries": ctx.block_new_entries,
                "global_sizing_factor": ctx.global_sizing_factor,
                "rationale": ctx.rationale,
                "source": "live",
                "snapshot_date": date.today().isoformat(),
                "is_today": _is_today,
                "events_today": [
                    _enrich_event({
                        "event_type": e.event_type,
                        "event_name": getattr(e, "event_name", ""),
                        "event_time": e.event_time,
                        "event_time_utc": getattr(e, "event_time_utc", "") or "",
                        "risk_level": e.risk_level,
                        "direction": e.direction,
                        "sizing_factor": e.sizing_factor,
                        "block_new_entries": e.block_new_entries,
                        "consensus_summary": e.consensus_summary,
                        "rationale": e.rationale,
                        "already_priced_in": e.already_priced_in,
                        "actual": e.actual,
                        "estimate": e.estimate,
                        "prior": e.prior,
                    }, actuals)
                    for e in ctx.events_today
                ],
            }

        # Fallback: latest DB snapshot
        from app.database.models import NisMacroSnapshot
        with get_session() as db:
            snap = db.query(NisMacroSnapshot).order_by(NisMacroSnapshot.snapshot_date.desc()).first()
        if snap is not None:
            _snap_today = snap.snapshot_date == date.today()
            actuals = _todays_event_actuals() if _snap_today else {}
            return {
                "as_of": snap.as_of.isoformat(),
                "overall_risk": snap.overall_risk,
                "block_new_entries": snap.block_new_entries,
                "global_sizing_factor": snap.global_sizing_factor,
                "rationale": snap.rationale,
                "source": "db_snapshot",
                "snapshot_date": snap.snapshot_date.isoformat(),
                "is_today": _snap_today,
                "events_today": [_enrich_event(dict(ev), actuals) for ev in (snap.events_json or [])],
            }

        return {"status": "not_yet_run", "detail": "Premarket routine has not run yet today"}
    except Exception as exc:
        logger.warning("GET /api/nis/macro failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


@router.get("/macro-history")
def get_macro_history(
    days: int = Query(default=1, ge=1, le=14),
    limit: int = Query(default=50, ge=1, le=500),
) -> Dict[str, Any]:
    """Return the timestamped NIS macro re-assessment lineage (newest-first) for the last N ET days.

    Backs the 'Today's NIS Assessment' history: each premarket + post-event re-evaluation is its own
    `nis_macro_history` row (already persisted by persist_nis_macro_snapshot), so this replays how
    risk/sizing/block evolved through the day (was → is → new) with the trigger that caused each
    change. Read-only; never affects trading."""
    try:
        from app.database.models import NisMacroHistory
        # snapshot_date is written as the UTC date (persist_nis_macro_snapshot), so filter on the same
        # basis. During US trading hours UTC-date == ET-date, so "today" matches the operator's view;
        # this just avoids a 1-day boundary mismatch overnight.
        start_date = datetime.now(timezone.utc).date() - timedelta(days=days - 1)
        with get_session() as db:
            rows = (db.query(NisMacroHistory)
                    .filter(NisMacroHistory.snapshot_date >= start_date)
                    .order_by(NisMacroHistory.as_of.desc())
                    .limit(limit)
                    .all())
            return {
                "count": len(rows),
                "history": [
                    {
                        "as_of": r.as_of.isoformat() if r.as_of else None,
                        "snapshot_date": r.snapshot_date.isoformat() if r.snapshot_date else None,
                        "trigger_source": r.trigger_source,
                        "trigger_event_type": r.trigger_event_type,
                        "trigger_event_name": r.trigger_event_name,
                        "overall_risk": r.overall_risk,
                        "block_new_entries": r.block_new_entries,
                        "global_sizing_factor": r.global_sizing_factor,
                        "rationale": r.rationale,
                        "n_events": len(r.events_json or []),
                    }
                    for r in rows
                ],
            }
    except Exception as exc:
        logger.warning("GET /api/nis/macro-history failed: %s", exc)
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


@audit_router.get("/gate-calibration")
def get_gate_calibration() -> Dict[str, Any]:
    """
    Rich gate calibration report: per-gate counterfactual P&L, by-category summary,
    and scan abstention history with SPY outcomes.
    """
    try:
        from app.database.decision_audit import gate_calibration_report
        return gate_calibration_report()
    except Exception as exc:
        logger.warning("GET /api/decision-audit/gate-calibration failed: %s", exc)
        return {"per_gate": [], "by_category": [], "scan_abstentions": [], "error": str(exc)}


@audit_router.post("/backfill-outcomes")
def trigger_backfill_outcomes(lookback_days: int = Query(default=7, ge=1, le=30)) -> Dict[str, Any]:
    """Manually trigger the gate outcome backfill job (normally runs nightly)."""
    try:
        from app.database.decision_audit import backfill_gate_outcomes, backfill_scan_abstention_outcomes
        gate_result = backfill_gate_outcomes(lookback_days=lookback_days)
        scan_result = backfill_scan_abstention_outcomes(lookback_days=lookback_days)
        return {"gate_outcomes": gate_result, "scan_abstentions": scan_result}
    except Exception as exc:
        logger.warning("POST /api/decision-audit/backfill-outcomes failed: %s", exc)
        return {"error": str(exc)}


@audit_router.get("/recent")
def get_recent_decisions(
    limit: int = Query(default=50, ge=1, le=500),
    strategy: Optional[str] = Query(default=None),
    final_decision: Optional[str] = Query(default=None),
    days: int = Query(default=1, ge=0, le=30),
) -> Dict[str, Any]:
    """Return the most recent decision_audit rows, optionally filtered.

    `days` scopes by decided_at to the last N ET calendar days (default 1 = TODAY only).
    Without it, /recent returned the last `limit` rows across ALL sessions, so a stale skip
    batch (e.g. a kill-switch window or a prior server instance) would surface forever — the
    "phantom AAPL swing / pm_skip: kill_switch with the switch now off" symptom. days=0 disables
    the date filter (full forensic history)."""
    try:
        from app.database.models import DecisionAudit
        with get_session() as db:
            q = db.query(DecisionAudit).order_by(DecisionAudit.decided_at.desc())
            if days > 0:
                from datetime import datetime, timedelta, timezone
                from zoneinfo import ZoneInfo
                _et = ZoneInfo("America/New_York")
                start_et = (datetime.now(_et) - timedelta(days=days - 1)).replace(
                    hour=0, minute=0, second=0, microsecond=0)
                q = q.filter(DecisionAudit.decided_at >= start_et.astimezone(timezone.utc))
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
                        # F10: surface the two sizing dimensions separately so the SIZE DOWN
                        # panel can fold the macro factor in alongside the news factor.
                        "news_sizing_multiplier": r.news_sizing_multiplier,
                        "macro_sizing_factor": r.macro_sizing_factor,
                        "block_reason": r.block_reason,
                        "news_action_policy": r.news_action_policy,
                        "news_materiality": r.news_materiality,
                        "macro_risk_level": r.macro_risk_level,
                        "outcome_pnl_pct": r.outcome_pnl_pct,
                        "outcome_1d_pct": r.outcome_1d_pct,
                        "vol_targeting_mult": r.vol_targeting_mult,
                        "regime_sizing_mult": r.regime_sizing_mult,
                        "regime_label": r.regime_label_at_decision,
                        "top_features": r.top_features,
                        "gate_category": r.gate_category,
                        "price_at_decision": float(r.price_at_decision) if r.price_at_decision else None,
                    }
                    for r in rows
                ],
            }
    except Exception as exc:
        logger.warning("GET /api/decision-audit/recent failed: %s", exc)
        return {"status": "error", "detail": str(exc)}
