"""
Agent Orchestrator — schedules agents, routes messages, handles failures.

Pipeline:
  PortfolioManager  →  (trade_proposals queue)
  RiskManager       →  (trader_approved_trades queue)
  Trader            →  Alpaca orders
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from app.scheduler import scheduler

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Central coordinator for all trading agents.

    Responsibilities:
    - Start / stop all agents as async tasks
    - Schedule periodic agent triggers
    - Health-check all integrations every 5 minutes
    - Emergency pause / resume trading
    """

    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, str] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    # ─── Registration ─────────────────────────────────────────────────────────

    def register_agent(self, name: str, instance) -> None:
        self.agents[name] = instance
        self.agent_status[name] = "idle"
        logger.info("Registered agent: %s", name)

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all agents and the scheduler."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting orchestrator…")
        self._running = True

        # Launch each agent's run() loop as a background task
        for name, agent in self.agents.items():
            task = asyncio.create_task(self._run_agent(name, agent), name=name)
            self._tasks[name] = task

        # Scheduler: health check every 5 min, periodic triggers
        scheduler.start()
        self._schedule_jobs()

        # Background monitor
        asyncio.create_task(self._monitor_loop(), name="orchestrator_monitor")
        logger.info("Orchestrator started — %d agents running", len(self.agents))

    async def stop(self) -> None:
        """Stop all agents and the scheduler."""
        logger.info("Stopping orchestrator…")
        self._running = False
        scheduler.stop()
        for name, task in self._tasks.items():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            logger.info("Agent %s stopped", name)
        self._tasks.clear()

    # ─── Scheduling ───────────────────────────────────────────────────────────

    def _schedule_jobs(self) -> None:
        """Register all APScheduler jobs."""

        # Health check every 5 minutes
        scheduler.schedule_every_n_minutes(
            self._health_check, minutes=5, job_id="health_check"
        )

        # Daily portfolio selection trigger at 09:30 ET (agents also self-trigger,
        # but this gives an explicit external nudge)
        scheduler.schedule_daily_at_time(
            self._trigger_portfolio_selection, hour=9, minute=30,
            job_id="portfolio_selection_trigger"
        )

        # Daily retraining trigger at 17:00 ET
        scheduler.schedule_daily_at_time(
            self._trigger_retraining, hour=17, minute=0,
            job_id="model_retraining_trigger"
        )

        # Post-market health check + session summary at 16:15 ET
        scheduler.schedule_daily_at_time(
            self._trigger_daily_summary, hour=16, minute=15,
            job_id="daily_session_summary"
        )

    # ─── Agent runner ─────────────────────────────────────────────────────────

    async def _run_agent(self, name: str, agent) -> None:
        """Run an agent's main loop, restarting on unexpected errors."""
        while self._running:
            try:
                self.agent_status[name] = "running"
                await agent.run()
                self.agent_status[name] = "stopped"
                break  # clean exit
            except asyncio.CancelledError:
                self.agent_status[name] = "stopped"
                raise
            except Exception as exc:
                self.agent_status[name] = "error"
                logger.error("Agent %s crashed: %s — restarting in 30 s", name, exc,
                             exc_info=True)
                await self._log_error(name, str(exc))
                await asyncio.sleep(30)

    # ─── Scheduled callbacks ──────────────────────────────────────────────────

    async def _trigger_portfolio_selection(self) -> None:
        """Nudge Portfolio Manager to run selection (it also triggers itself)."""
        pm = self.agents.get("portfolio_manager")
        if pm:
            logger.info("Orchestrator: triggering portfolio selection")
            try:
                await pm.select_instruments()
            except Exception as exc:
                logger.error("Portfolio selection trigger failed: %s", exc)
                await self._log_error("portfolio_manager", str(exc))

    async def _trigger_retraining(self) -> None:
        """Kick off ML model retraining in a thread."""
        logger.info("Orchestrator: triggering model retraining")
        try:
            from app.ml.training import ModelTrainer
            loop = asyncio.get_event_loop()
            trainer = ModelTrainer(n_workers=24)
            version = await loop.run_in_executor(
                None, lambda: trainer.train_model(fetch_fundamentals=False)
            )
            logger.info("Retraining complete → v%d", version)
        except Exception as exc:
            logger.error("Retraining trigger failed: %s", exc)
            await self._log_error("model_trainer", str(exc))

    async def _trigger_daily_summary(self) -> None:
        """Run post-market health check and store the daily session summary."""
        logger.info("Orchestrator: running daily session summary")
        try:
            from app.live_trading.monitoring import monitor
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, monitor.daily_session_summary)
            logger.info(
                "Daily summary stored: P&L=%.2f status=%s",
                summary["pnl_today"], summary["status"],
            )
        except Exception as exc:
            logger.error("Daily summary failed: %s", exc)

    async def _health_check(self) -> None:
        """Verify DB, Redis, and Alpaca; pause trading if any critical service is down."""
        from app.database import check_db_connection
        from app.integrations import get_alpaca_client, get_redis_queue

        db_ok = check_db_connection()
        redis_ok = get_redis_queue().health_check()

        alpaca_ok = False
        try:
            alpaca_ok = get_alpaca_client().health_check()
        except Exception:
            pass

        if not all([db_ok, redis_ok, alpaca_ok]):
            logger.warning(
                "Health check FAILED — DB=%s Redis=%s Alpaca=%s",
                db_ok, redis_ok, alpaca_ok,
            )
            if not alpaca_ok:
                logger.critical("Alpaca unavailable — pausing trading")
                self.pause_trading()
        else:
            logger.debug("Health check OK")

    # ─── Emergency controls ───────────────────────────────────────────────────

    def pause_trading(self) -> None:
        """Pause portfolio selection and retraining triggers."""
        scheduler.pause_job("portfolio_selection_trigger")
        scheduler.pause_job("model_retraining_trigger")
        # Pause agent loops via their status flag
        for name, agent in self.agents.items():
            if hasattr(agent, "status"):
                agent.status = "paused"
        logger.warning("TRADING PAUSED by orchestrator")

    def resume_trading(self) -> None:
        """Resume paused triggers and agents."""
        scheduler.resume_job("portfolio_selection_trigger")
        scheduler.resume_job("model_retraining_trigger")
        for name, agent in self.agents.items():
            if hasattr(agent, "status") and agent.status == "paused":
                agent.status = "running"
        logger.info("TRADING RESUMED by orchestrator")

    # ─── Monitor loop ─────────────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                logger.debug("Agent status: %s", self.agent_status)
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Monitor error: %s", exc)

    # ─── DB helpers ───────────────────────────────────────────────────────────

    async def _log_error(self, agent_name: str, error_message: str) -> None:
        from app.database.session import get_session

        db = get_session()
        try:
            from app.database.models import AgentDecision
            record = AgentDecision(
                agent_name=agent_name,
                decision_type="AGENT_ERROR",
                reasoning={"error": error_message},
                timestamp=datetime.utcnow(),
            )
            db.add(record)
            db.commit()
        except Exception as exc:
            logger.warning("Could not log error to DB: %s", exc)
        finally:
            db.close()

    # ─── Status ───────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        from app.integrations import get_redis_queue

        try:
            redis = get_redis_queue()
            queue_lengths = {
                "trade_proposals": redis.get_queue_length("trade_proposals"),
                "trader_approved_trades": redis.get_queue_length("trader_approved_trades"),
            }
        except Exception:
            queue_lengths = {}

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "running": self._running,
            "agents": self.agent_status,
            "scheduled_jobs": len(scheduler.get_jobs()),
            "queues": queue_lengths,
        }


# Module-level singleton
orchestrator = AgentOrchestrator()
