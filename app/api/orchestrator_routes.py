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
_MARKET_OPEN  = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)


def _now_et() -> datetime:
    """Return current datetime in ET. Extracted for testability."""
    return datetime.now(tz=_ET)


def _market_status() -> Dict[str, Any]:
    """Determine whether the US equity market is currently open."""
    now_et     = _now_et()
    is_weekday = now_et.weekday() < 5
    cur_time   = now_et.time().replace(tzinfo=None)
    in_hours   = _MARKET_OPEN <= cur_time < _MARKET_CLOSE
    is_open    = is_weekday and in_hours

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
    orchestrator.pause_trading()
    _log("WARNING", "Trading paused by user")
    return {"status": "trading_paused"}


@router.post("/resume-trading")
async def resume_trading():
    """Resume trading after an emergency pause."""
    orchestrator.resume_trading()
    _log("INFO", "Trading resumed by user")
    return {"status": "trading_resumed"}


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a scheduled job by ID."""
    try:
        scheduler.pause_job(job_id)
        _log("INFO", f"Job paused: {job_id}")
        return {"status": "paused", "job_id": job_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job by ID."""
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
    _log("INFO", "Manual cycle triggered by user")

    async def _run():
        try:
            pm = orchestrator.agents.get("portfolio_manager")
            if pm:
                _log("INFO", "PM: running instrument selection")
                await pm.select_instruments()
            else:
                _log("WARNING", "PM agent not registered")

            risk = orchestrator.agents.get("risk_manager")
            if risk:
                _log("INFO", "RiskManager: processing proposals")
                await risk.process_proposals()
            else:
                _log("WARNING", "Risk agent not registered — skipping")

            trader = orchestrator.agents.get("trader")
            if trader:
                _log("INFO", "Trader: processing approved trades")
                await trader.process_approved_trades()
            else:
                _log("WARNING", "Trader agent not registered — skipping")

            _log("INFO", "Manual cycle complete")
        except Exception as exc:
            logger.error("Manual cycle error: %s", exc)
            _log("ERROR", f"Manual cycle failed: {exc}")

    asyncio.create_task(_run())
    return {"status": "cycle_started", "message": "Agent pipeline running in background — check session log"}


@router.post("/trigger-retraining")
async def trigger_retraining():
    """Manually kick off ML model retraining."""
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
