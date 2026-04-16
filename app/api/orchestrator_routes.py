"""
Orchestrator API — control and inspect the agent orchestration layer.
"""

from fastapi import APIRouter, HTTPException

from app.orchestrator import orchestrator
from app.scheduler import scheduler

router = APIRouter(prefix="/api/orchestrator", tags=["orchestrator"])


@router.get("/status")
async def get_orchestrator_status():
    """Return orchestrator and agent status."""
    return orchestrator.get_status()


@router.get("/jobs")
async def get_scheduled_jobs():
    """List all APScheduler jobs."""
    jobs = scheduler.get_jobs()
    return {
        "count": len(jobs),
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": str(job.next_run_time) if job.next_run_time else None,
            }
            for job in jobs
        ],
    }


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a scheduled job by ID."""
    try:
        scheduler.pause_job(job_id)
        return {"status": "paused", "job_id": job_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job by ID."""
    try:
        scheduler.resume_job(job_id)
        return {"status": "resumed", "job_id": job_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/pause-trading")
async def pause_trading():
    """Emergency pause — stop portfolio selection and agent triggers."""
    orchestrator.pause_trading()
    return {"status": "trading_paused"}


@router.post("/resume-trading")
async def resume_trading():
    """Resume trading after an emergency pause."""
    orchestrator.resume_trading()
    return {"status": "trading_resumed"}
