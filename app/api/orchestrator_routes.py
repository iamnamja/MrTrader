"""
Orchestrator API — control and inspect the agent orchestration layer.
"""

import asyncio
import logging
import pytz
from datetime import datetime, timedelta, time as dtime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.orchestrator import orchestrator
from app.scheduler import scheduler
# NOTE: app.control_bridge is imported LAZILY inside each control route (like routes.py),
# never at module level — so a future bridge-import error can never stop the FastAPI app
# from booting (trading must not be held hostage to the control-bridge module).

router = APIRouter(prefix="/api/orchestrator", tags=["orchestrator"])
logger = logging.getLogger(__name__)

# ── In-memory session log (last 200 entries) ──────────────────────────────────
_session_log: list[Dict[str, Any]] = []
_MAX_LOG = 200


def _log(level: str, message: str, detail: dict | None = None):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "message": message,
        "detail": detail or {},
    }
    _session_log.insert(0, entry)
    if len(_session_log) > _MAX_LOG:
        _session_log.pop()


# ── Market helpers ─────────────────────────────────────────────────────────────

_ET = pytz.timezone("America/New_York")
_MARKET_OPEN = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)


def _now_et() -> datetime:
    """Return current datetime in ET. Extracted for testability."""
    return datetime.now(tz=_ET)


def _market_status() -> Dict[str, Any]:
    """Determine whether the US equity market is currently open."""
    now_et = _now_et()
    is_weekday = now_et.weekday() < 5
    cur_time = now_et.time().replace(tzinfo=None)
    in_hours = _MARKET_OPEN <= cur_time < _MARKET_CLOSE
    is_open = is_weekday and in_hours

    if is_open:
        # Minutes until 16:00 today
        close_h, close_m = _MARKET_CLOSE.hour, _MARKET_CLOSE.minute
        mins_to_close = (close_h - now_et.hour) * 60 + (close_m - now_et.minute)
        next_event: Dict[str, Any] = {"event": "market_close", "minutes": mins_to_close}
    else:
        # Walk forward to find next weekday open
        candidate = now_et.date()
        if cur_time >= _MARKET_CLOSE or not is_weekday:
            candidate += timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        next_event = {"event": "market_open", "date": str(candidate), "time": "09:30 ET"}

    return {
        "is_open": is_open,
        "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "weekday": now_et.strftime("%A"),
        "next_event": next_event,
    }


# ── Status endpoints ──────────────────────────────────────────────────────────

@router.get("/status")
async def get_orchestrator_status():
    """Full orchestrator + market status snapshot."""
    status = orchestrator.get_status()
    status["market"] = _market_status()
    return status


@router.get("/market-status")
async def get_market_status():
    """Is the market currently open?"""
    return _market_status()


@router.get("/session-log")
async def get_session_log(limit: int = 50):
    """Recent session events (manual triggers, errors, health events)."""
    return {"entries": _session_log[:min(limit, _MAX_LOG)], "total": len(_session_log)}


@router.get("/jobs")
async def get_scheduled_jobs():
    """List all APScheduler jobs with next run times."""
    jobs = scheduler.get_jobs()
    return {
        "count": len(jobs),
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": str(job.next_run_time) if job.next_run_time else None,
                "paused": job.next_run_time is None,
            }
            for job in jobs
        ],
    }


# ── Trading controls ──────────────────────────────────────────────────────────

@router.post("/pause-trading")
async def pause_trading():
    """Emergency pause — stop portfolio selection and agent triggers."""
    from app.control_bridge import bridge_or_none, CMD_PAUSE
    bridged = bridge_or_none(CMD_PAUSE)
    if bridged is not None:
        _log("WARNING", "Trading pause forwarded to daemon")
        return bridged
    orchestrator.pause_trading()
    _log("WARNING", "Trading paused by user")
    return {"status": "trading_paused"}


@router.post("/resume-trading")
async def resume_trading():
    """Resume trading after an emergency pause."""
    from app.control_bridge import bridge_or_none, CMD_RESUME
    bridged = bridge_or_none(CMD_RESUME)
    if bridged is not None:
        _log("INFO", "Trading resume forwarded to daemon")
        return bridged
    orchestrator.resume_trading()
    _log("INFO", "Trading resumed by user")
    return {"status": "trading_resumed"}


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a scheduled job by ID."""
    from app.control_bridge import bridge_or_none, CMD_JOB_PAUSE
    bridged = bridge_or_none(CMD_JOB_PAUSE, {"job_id": job_id})
    if bridged is not None:
        return bridged
    try:
        scheduler.pause_job(job_id)
        _log("INFO", f"Job paused: {job_id}")
        return {"status": "paused", "job_id": job_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job by ID."""
    from app.control_bridge import bridge_or_none, CMD_JOB_RESUME
    bridged = bridge_or_none(CMD_JOB_RESUME, {"job_id": job_id})
    if bridged is not None:
        return bridged
    try:
        scheduler.resume_job(job_id)
        _log("INFO", f"Job resumed: {job_id}")
        return {"status": "resumed", "job_id": job_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Manual cycle trigger ──────────────────────────────────────────────────────

@router.post("/trigger-cycle")
async def trigger_one_cycle():
    """
    Manually run one full agent pipeline cycle (PM → Risk → Trader).
    Useful for testing outside market hours.
    Returns immediately; the cycle runs in the background.
    """
    from app.control_bridge import bridge_or_none, CMD_TRIGGER_CYCLE
    bridged = bridge_or_none(CMD_TRIGGER_CYCLE)
    if bridged is not None:
        _log("INFO", "Manual cycle forwarded to daemon")
        return bridged
    _log("INFO", "Manual cycle triggered by user")

    async def _run():
        try:
            pm = orchestrator.agents.get("portfolio_manager")
            if pm:
                _log("INFO", "PM: running instrument selection (v37 LambdaRank)")
                await pm.select_instruments()
                _log("INFO", "PM: proposals sent to trade_proposals queue — Risk Manager + Trader will pick up automatically")
            else:
                _log("WARNING", "PM agent not registered — start the app first")

            _log("INFO", "Manual cycle complete — monitor logs for Risk/Trader activity")
        except Exception as exc:
            logger.error("Manual cycle error: %s", exc)
            _log("ERROR", f"Manual cycle failed: {exc}")

    asyncio.create_task(_run())
    return {"status": "cycle_started", "message": "Agent pipeline running in background — check session log"}


@router.post("/trigger-swing-analysis")
async def trigger_swing_analysis():
    """Force-run today's swing premarket analysis and send proposals.

    Bypasses the time-of-day routing in select_instruments() so it works
    even after the 09:45 ET cutoff. Use when the scheduled premarket run
    failed (e.g. scorer crash) and proposals need to be recovered mid-session.
    """
    from app.control_bridge import bridge_or_none, CMD_TRIGGER_SWING
    bridged = bridge_or_none(CMD_TRIGGER_SWING)
    if bridged is not None:
        _log("INFO", "Manual swing analysis forwarded to daemon")
        return bridged
    _log("INFO", "Manual swing premarket analysis triggered by user")

    async def _run():
        try:
            pm = orchestrator.agents.get("portfolio_manager")
            if pm:
                _log("INFO", "Forcing swing premarket analysis (bypassing time-of-day gate)...")
                pm._analyzed_today = False   # reset flag so premarket reruns
                pm._selected_today = False   # reset so proposals are re-sent
                await pm._analyze_swing_premarket()
                await pm._send_swing_proposals()
                _log("INFO", "Swing analysis complete — proposals sent to queue")
            else:
                _log("WARNING", "PM agent not registered — start the app first")
        except Exception as exc:
            logger.error("Manual swing analysis error: %s", exc)
            _log("ERROR", f"Swing analysis failed: {exc}")

    asyncio.create_task(_run())
    return {"status": "swing_analysis_started", "message": "Swing premarket analysis running — check session log for proposals"}


@router.post("/trigger-retraining")
async def trigger_retraining():
    """Manually kick off ML model retraining."""
    from app.control_bridge import bridge_or_none, CMD_TRIGGER_RETRAIN
    bridged = bridge_or_none(CMD_TRIGGER_RETRAIN)
    if bridged is not None:
        _log("INFO", "Manual retraining forwarded to daemon")
        return bridged
    _log("INFO", "Manual model retraining triggered by user")

    async def _retrain():
        try:
            pm = orchestrator.agents.get("portfolio_manager")
            if pm:
                await pm._retrain()
                _log("INFO", "Retraining complete")
            else:
                _log("WARNING", "PM agent not registered")
        except Exception as exc:
            _log("ERROR", f"Retraining failed: {exc}")

    asyncio.create_task(_retrain())
    return {"status": "retraining_started", "message": "Retraining running in background"}


@router.get("/ai-briefing")
async def get_ai_briefing():
    """Generate a Claude AI daily briefing from recent pending proposals."""
    try:
        from app.database.session import get_session
        from app.database.models import AgentDecision
        from app.ai.claude_client import summarise_daily_proposals
        from datetime import date

        db = get_session()
        try:
            today = date.today().isoformat()
            decisions = (
                db.query(AgentDecision)
                .filter(
                    AgentDecision.decision_type == "TRADE_PROPOSAL",
                    AgentDecision.timestamp >= f"{today}T00:00:00",
                )
                .order_by(AgentDecision.timestamp.desc())
                .limit(15)
                .all()
            )
            proposals = [d.reasoning for d in decisions if d.reasoning]
        finally:
            db.close()

        summary = summarise_daily_proposals(proposals)
        return {
            "proposals_reviewed": len(proposals),
            "briefing": summary,
            "ai_available": summary is not None,
        }
    except Exception as exc:
        logger.error("AI briefing error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/trigger-intraday-scan")
async def trigger_intraday_scan():
    """
    Manually run an intraday scan using the full universe (use_morning_candidates=False).
    This rebuilds the morning candidates list and persists it to DB, fixing post-restart timeouts.
    Runs in the background — check logs for progress.
    """
    from app.control_bridge import bridge_or_none, CMD_TRIGGER_INTRADAY
    bridged = bridge_or_none(CMD_TRIGGER_INTRADAY)
    if bridged is not None:
        _log("INFO", "Manual intraday scan forwarded to daemon")
        return bridged
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    now = datetime.now(ET)
    _log("INFO", f"Manual intraday scan triggered ({now.strftime('%H:%M')} ET)")

    async def _run():
        try:
            pm = orchestrator.agents.get("portfolio_manager")
            if pm:
                _log("INFO", "Intraday scan: running full-universe scan to rebuild morning candidates")
                await pm.select_intraday_instruments(
                    window=(now.hour, now.minute),
                    use_morning_candidates=False,
                )
                _log("INFO", "Intraday scan complete — morning candidates rebuilt and persisted")
            else:
                _log("WARNING", "PM agent not registered")
        except Exception as exc:
            logger.error("Manual intraday scan error: %s", exc)
            _log("ERROR", f"Manual intraday scan failed: {exc}")

    asyncio.create_task(_run())
    return {"status": "intraday_scan_started", "message": "Full-universe intraday scan running — check logs (takes ~5-7 min)"}
