"""Tests for graceful-shutdown hardening.

Background: the in-process orchestrator + joblib/loky worker pools are non-daemon,
so on Ctrl+C the interpreter could block forever joining them (requiring a force
kill). The hardening adds (1) a daemon watchdog that os._exit(0)s if shutdown wedges,
(2) explicit worker-pool teardown, and (3) tracking the orchestrator monitor loop so
stop() actually cancels it.
"""
import asyncio

import pytest

from app.main import _start_shutdown_watchdog, _kill_worker_pools


def test_shutdown_watchdog_is_daemon_alive_and_cancellable():
    # Use a huge timeout so the os._exit branch can NEVER fire during the test.
    t = _start_shutdown_watchdog(3600.0)
    try:
        assert t.daemon is True          # daemon → keeps running during interpreter exit
        assert t.is_alive()
    finally:
        t.cancel()                       # ensure it never fires after the test


def test_kill_worker_pools_is_best_effort_no_raise():
    # Must be safe even when no loky pool exists / joblib is absent.
    _kill_worker_pools()


def test_watchdog_disarmed_under_test_mode():
    # The lifespan shutdown guards `_start_shutdown_watchdog()` behind `not is_test_mode()`
    # so TestClient teardowns (which run lifespan shutdown) never arm an os._exit(0) timer
    # inside the pytest process. This asserts the guard's premise holds under pytest.
    from app.utils.runtime import is_test_mode
    assert is_test_mode() is True


@pytest.mark.asyncio
async def test_orchestrator_stop_cancels_all_tracked_tasks_incl_monitor():
    from app.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator()
    orch._running = True

    async def _forever():
        while True:
            await asyncio.sleep(3600)

    dummy = asyncio.create_task(_forever())
    monitor = asyncio.create_task(_forever())
    orch._tasks["dummy"] = dummy
    orch._tasks["_monitor"] = monitor      # mirrors the now-tracked monitor loop

    await orch.stop()                      # bounded (5s/task); scheduler.stop() is a no-op here

    # both tasks must be terminated, and the registry cleared
    await asyncio.sleep(0)                  # let cancellations settle
    assert dummy.cancelled() or dummy.done()
    assert monitor.cancelled() or monitor.done()
    assert orch._tasks == {}
    assert orch._running is False


def test_monitor_loop_is_tracked_in_start_source():
    """Guard the fix: the monitor loop must be stored in _tasks (so stop() cancels
    it), not fire-and-forget. Asserted on source to avoid spinning a real loop."""
    import inspect
    from app.orchestrator import AgentOrchestrator
    src = inspect.getsource(AgentOrchestrator.start)
    assert '_tasks["_monitor"]' in src or "_tasks['_monitor']" in src
