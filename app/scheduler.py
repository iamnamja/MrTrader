"""
Agent Scheduler — APScheduler wrapper for cron and interval jobs.
"""

import logging
from typing import Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class AgentScheduler:
    """Manage agent scheduling via APScheduler."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone="America/New_York")

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Agent scheduler started")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Agent scheduler stopped")

    def schedule_daily_at_time(
        self, func: Callable, hour: int, minute: int, job_id: str
    ):
        """Run func daily at hh:mm ET on weekdays."""
        self.scheduler.add_job(
            func,
            CronTrigger(hour=hour, minute=minute, day_of_week="0-4",
                        timezone="America/New_York"),
            id=job_id,
            replace_existing=True,
            misfire_grace_time=60,
        )
        logger.info("Scheduled %s daily at %02d:%02d ET", job_id, hour, minute)

    def schedule_every_n_minutes(
        self, func: Callable, minutes: int, job_id: str
    ):
        """Run func every N minutes."""
        self.scheduler.add_job(
            func,
            "interval",
            minutes=minutes,
            id=job_id,
            replace_existing=True,
            misfire_grace_time=30,
        )
        logger.info("Scheduled %s every %d minutes", job_id, minutes)

    def get_jobs(self):
        return self.scheduler.get_jobs()

    def pause_job(self, job_id: str):
        job = self.scheduler.get_job(job_id)
        if job:
            job.pause()
            logger.info("Paused job: %s", job_id)

    def resume_job(self, job_id: str):
        job = self.scheduler.get_job(job_id)
        if job:
            job.resume()
            logger.info("Resumed job: %s", job_id)


# Module-level singleton
scheduler = AgentScheduler()
