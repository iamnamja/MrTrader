"""
Unit tests for Phase 5: AgentScheduler and AgentOrchestrator.

All tests are pure-Python — no network, database, Redis, or Alpaca calls.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from app.orchestrator import AgentOrchestrator
from app.scheduler import AgentScheduler


# ─── Scheduler ───────────────────────────────────────────────────────────────

class TestAgentScheduler:
    def setup_method(self):
        self.sched = AgentScheduler()

    def teardown_method(self):
        if self.sched.scheduler.running:
            self.sched.stop()

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        self.sched.start()
        await asyncio.sleep(0)  # let event loop process scheduler startup
        assert self.sched.scheduler.running
        self.sched.stop()
        # AsyncIOScheduler shutdown is deferred in the event loop — verify
        # the scheduler object is no longer tracked as running by our wrapper
        # after a brief yield
        await asyncio.sleep(0.05)
        assert not self.sched.scheduler.running

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        self.sched.start()
        self.sched.start()  # second call should be a no-op
        assert self.sched.scheduler.running

    @pytest.mark.asyncio
    async def test_schedule_daily_registers_job(self):
        self.sched.start()
        self.sched.schedule_daily_at_time(lambda: None, hour=9, minute=30,
                                          job_id="test_daily")
        job_ids = [j.id for j in self.sched.get_jobs()]
        assert "test_daily" in job_ids

    @pytest.mark.asyncio
    async def test_schedule_interval_registers_job(self):
        self.sched.start()
        self.sched.schedule_every_n_minutes(lambda: None, minutes=5,
                                            job_id="test_interval")
        job_ids = [j.id for j in self.sched.get_jobs()]
        assert "test_interval" in job_ids

    @pytest.mark.asyncio
    async def test_pause_and_resume_job(self):
        self.sched.start()
        self.sched.schedule_every_n_minutes(lambda: None, minutes=5,
                                            job_id="test_pause")
        self.sched.pause_job("test_pause")
        job = self.sched.scheduler.get_job("test_pause")
        assert job.next_run_time is None  # paused jobs have no next_run_time

        self.sched.resume_job("test_pause")
        job = self.sched.scheduler.get_job("test_pause")
        assert job.next_run_time is not None

    @pytest.mark.asyncio
    async def test_pause_nonexistent_job_is_safe(self):
        self.sched.start()
        self.sched.pause_job("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_resume_nonexistent_job_is_safe(self):
        self.sched.start()
        self.sched.resume_job("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_get_jobs_returns_list(self):
        self.sched.start()
        jobs = self.sched.get_jobs()
        assert isinstance(jobs, list)

    @pytest.mark.asyncio
    async def test_replace_existing_job(self):
        self.sched.start()
        self.sched.schedule_every_n_minutes(lambda: None, minutes=5,
                                            job_id="dup")
        self.sched.schedule_every_n_minutes(lambda: None, minutes=10,
                                            job_id="dup")
        job_ids = [j.id for j in self.sched.get_jobs()]
        assert job_ids.count("dup") == 1


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class TestAgentOrchestrator:
    def setup_method(self):
        self.orch = AgentOrchestrator()

    def test_register_agent(self):
        mock_agent = MagicMock()
        self.orch.register_agent("test_agent", mock_agent)
        assert "test_agent" in self.orch.agents
        assert self.orch.agent_status["test_agent"] == "idle"

    def test_register_multiple_agents(self):
        self.orch.register_agent("a", MagicMock())
        self.orch.register_agent("b", MagicMock())
        assert set(self.orch.agents.keys()) == {"a", "b"}

    def test_get_status_structure(self):
        self.orch.register_agent("pm", MagicMock())
        with patch("app.integrations.get_redis_queue") as mock_redis:
            mock_redis.return_value.get_queue_length.return_value = 0
            status = self.orch.get_status()
        assert "timestamp" in status
        assert "running" in status
        assert "agents" in status
        assert "scheduled_jobs" in status

    def test_get_status_agents_present(self):
        self.orch.register_agent("portfolio_manager", MagicMock())
        with patch("app.integrations.get_redis_queue") as mock_redis:
            mock_redis.return_value.get_queue_length.return_value = 0
            status = self.orch.get_status()
        assert "portfolio_manager" in status["agents"]

    def test_pause_trading_pauses_scheduler_jobs(self):
        with patch.object(self.orch, "_running", True):
            mock_sched = MagicMock()
            with patch("app.orchestrator.scheduler", mock_sched):
                self.orch.pause_trading()
                mock_sched.pause_job.assert_any_call("portfolio_selection_trigger")
                mock_sched.pause_job.assert_any_call("model_retraining_trigger")

    def test_pause_trading_sets_agent_status(self):
        mock_agent = MagicMock()
        mock_agent.status = "running"
        self.orch.register_agent("trader", mock_agent)
        with patch("app.orchestrator.scheduler"):
            self.orch.pause_trading()
        assert mock_agent.status == "paused"

    def test_resume_trading_resumes_scheduler_jobs(self):
        mock_sched = MagicMock()
        with patch("app.orchestrator.scheduler", mock_sched):
            self.orch.resume_trading()
            mock_sched.resume_job.assert_any_call("portfolio_selection_trigger")
            mock_sched.resume_job.assert_any_call("model_retraining_trigger")

    def test_resume_trading_restores_agent_status(self):
        mock_agent = MagicMock()
        mock_agent.status = "paused"
        self.orch.register_agent("risk_manager", mock_agent)
        with patch("app.orchestrator.scheduler"):
            self.orch.resume_trading()
        assert mock_agent.status == "running"

    @pytest.mark.asyncio
    async def test_run_agent_handles_crash_and_restarts(self):
        """Agent that crashes once should be restarted."""
        call_count = 0

        async def flaky_run():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            # Second call: exit cleanly
            self.orch._running = False

        mock_agent = MagicMock()
        mock_agent.run = flaky_run
        self.orch.register_agent("flaky", mock_agent)
        self.orch._running = True

        with patch("app.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            with patch.object(self.orch, "_log_error", new_callable=AsyncMock):
                await self.orch._run_agent("flaky", mock_agent)

        assert call_count == 2
        assert self.orch.agent_status["flaky"] == "stopped"

    @pytest.mark.asyncio
    async def test_health_check_pauses_on_alpaca_failure(self):
        with patch("app.database.check_db_connection", return_value=True):
            with patch("app.integrations.get_redis_queue") as mock_redis:
                mock_redis.return_value.health_check.return_value = True
                with patch("app.integrations.get_alpaca_client") as mock_alpaca:
                    mock_alpaca.return_value.health_check.return_value = False
                    with patch.object(self.orch, "pause_trading") as mock_pause:
                        await self.orch._health_check()
                        mock_pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_no_pause_when_all_ok(self):
        with patch("app.database.check_db_connection", return_value=True):
            with patch("app.integrations.get_redis_queue") as mock_redis:
                mock_redis.return_value.health_check.return_value = True
                with patch("app.integrations.get_alpaca_client") as mock_alpaca:
                    mock_alpaca.return_value.health_check.return_value = True
                    with patch.object(self.orch, "pause_trading") as mock_pause:
                        await self.orch._health_check()
                        mock_pause.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_launches_agent_tasks(self):
        async def dummy_run():
            await asyncio.sleep(0)

        mock_agent = MagicMock()
        mock_agent.run = dummy_run
        self.orch.register_agent("dummy", mock_agent)

        with patch("app.orchestrator.scheduler") as mock_sched:
            mock_sched.start = MagicMock()
            mock_sched.get_jobs.return_value = []
            mock_sched.schedule_every_n_minutes = MagicMock()
            mock_sched.schedule_daily_at_time = MagicMock()
            await self.orch.start()
            assert self.orch._running is True
            assert "dummy" in self.orch._tasks
            await self.orch.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        with patch("app.orchestrator.scheduler") as mock_sched:
            mock_sched.start = MagicMock()
            mock_sched.get_jobs.return_value = []
            mock_sched.schedule_every_n_minutes = MagicMock()
            mock_sched.schedule_daily_at_time = MagicMock()
            await self.orch.start()
            await self.orch.start()  # second call should be no-op
            mock_sched.start.assert_called_once()
            await self.orch.stop()


class TestRetrainingSubprocess:
    """Phase 99 — _trigger_retraining spawns subprocess, not run_in_executor."""

    def setup_method(self):
        self.orch = AgentOrchestrator()

    @pytest.mark.asyncio
    async def test_retraining_spawns_subprocess_on_success(self):
        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock(return_value=0)

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
            patch("builtins.open", mock_open()),
        ):
            await self.orch._trigger_retraining()

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[1].endswith("retrain_cron.py")

    @pytest.mark.asyncio
    async def test_retraining_logs_error_on_nonzero_exit(self):
        mock_proc = AsyncMock()
        mock_proc.pid = 99
        mock_proc.wait = AsyncMock(return_value=1)

        errors_logged = []

        async def fake_log_error(agent, msg):
            errors_logged.append(msg)

        self.orch._log_error = fake_log_error

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("builtins.open", mock_open()),
        ):
            await self.orch._trigger_retraining()

        assert any("retrain_cron.py exited 1" in e for e in errors_logged)

    @pytest.mark.asyncio
    async def test_retraining_gate_fail_exit_code_2_not_logged_as_error(self):
        """Exit code 2 = gate failed (expected path), should warn not error."""
        mock_proc = AsyncMock()
        mock_proc.pid = 42
        mock_proc.wait = AsyncMock(return_value=2)

        errors_logged = []

        async def fake_log_error(agent, msg):
            errors_logged.append(msg)

        self.orch._log_error = fake_log_error

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("builtins.open", mock_open()),
        ):
            await self.orch._trigger_retraining()

        assert not errors_logged  # gate fail is a warning, not an error

    @pytest.mark.asyncio
    async def test_retraining_does_not_import_modeltrainer(self):
        """ModelTrainer must NOT be imported inside uvicorn — verify it's gone."""
        import importlib
        import app.orchestrator as orch_mod
        source = importlib.util.find_spec("app.orchestrator").origin
        with open(source) as f:
            content = f.read()
        assert "ModelTrainer" not in content
