"""
tradingd.py — the standalone MrTrader trading daemon (R0.2 Phase 1).

Runs the PM/RM/Trader brain (orchestrator + agents + scheduler + news monitor)
WITHOUT FastAPI. This is the process that owns trading once the deployment flips to
`MRTRADER_DAEMON_MODE=subprocess` — at which point the FastAPI app runs read-only and
this daemon does the trading (so a web restart no longer restarts trading).

    MRTRADER_DAEMON_MODE=subprocess python -m app.tradingd

SAFETY INTERLOCK: this daemon REFUSES to start unless MRTRADER_DAEMON_MODE=subprocess.
Combined with the web booting the brain only when mode != subprocess, exactly one
process ever runs the brain — the ADR's "double-running agents" risk cannot occur.

Phase 1 scope: a runnable standalone brain (the boot path + graceful shutdown). The
control-plane bridge (web → daemon kill-switch / pause / manual triggers via
Postgres/Redis) and the observability bridge are Phase 2/3 — until then, in subprocess
mode the web's control routes are managed by the daemon, not yet remotely driveable.
"""
from __future__ import annotations

# Mirror the `python -m app` launcher: scrub env vars that must come only from .env
# (os.environ otherwise overrides pydantic-settings) BEFORE any app module is imported.
import os

_SCRUB = ["INITIAL_CAPITAL"]
for _k in _SCRUB:
    if _k in os.environ:
        print(f"[tradingd] Removed stale env var {_k}={os.environ[_k]!r} — .env value will be used instead")
        del os.environ[_k]

import asyncio  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from app.config import settings  # noqa: E402
from app.trading_runtime import (  # noqa: E402
    DAEMON_MODE_SUBPROCESS,
    get_daemon_mode,
    prepare_trading_state,
    start_shutdown_watchdog,
    start_trading_brain,
    stop_trading_brain,
)


def _configure_daemon_logging() -> None:
    """Minimal stdout logging — journald/systemd (Phase 4) captures it. File-log parity
    with the web's rotating daily handler is a later phase; kept lean here on purpose."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    """Set SIGINT/SIGTERM to request a graceful stop. Falls back to signal.signal on
    platforms (Windows) where the event loop has no add_signal_handler."""
    def _request_stop() -> None:
        logging.getLogger("mrtrader.daemon").info("Stop signal received — shutting down…")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Windows ProactorEventLoop: route the OS signal back onto the loop thread.
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(_request_stop))


async def _run() -> int:
    log = logging.getLogger("mrtrader.daemon")
    commit_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    bar = "=" * 72
    log.info("\n%s\n  MRTRADER TRADING DAEMON  %s\n  mode: %s  trading_mode: %s  python: %s\n%s",
             bar, commit_now, DAEMON_MODE_SUBPROCESS, settings.trading_mode.upper(),
             sys.version.split()[0], bar)

    # ── Database (fatal) ──
    from app.database import init_db, check_db_connection
    init_db()
    if not check_db_connection():
        log.error("Cannot connect to database — daemon aborting.")
        return 1
    log.info("OK Database connection verified")

    # ── Connectivity (Redis fatal — the brain talks over Redis queues; Alpaca degradable) ──
    from app.integrations import get_alpaca_client, get_redis_queue
    if not await asyncio.to_thread(get_redis_queue().health_check):
        log.error("Redis health check failed — daemon aborting (inter-agent queues unavailable).")
        return 1
    log.info("OK Redis connection verified")
    try:
        if await asyncio.to_thread(get_alpaca_client().health_check):
            log.info("OK Alpaca connection verified")
        else:
            log.warning("Alpaca health check failed — continuing in degraded mode")
    except Exception as e:  # noqa: BLE001
        log.warning("Alpaca connectivity check failed: %s", e)

    # ── Brain preamble + boot ──
    await prepare_trading_state(log)
    news_monitor_task = await start_trading_brain(log)

    # ── R0.2 Phase 2: control bridge — drain web-issued control commands + sync the
    # kill-switch from Postgres (the web is read-only in subprocess mode). ──
    from app.control_bridge import consume_control_commands, state_sync_loop
    consumer_task = asyncio.create_task(consume_control_commands(), name="control_consumer")
    state_sync_task = asyncio.create_task(state_sync_loop(), name="state_sync")
    log.info("MrTrader trading daemon started successfully (control bridge active)")

    # ── Serve until signalled ──
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    _install_signal_handlers(loop, stop)
    try:
        await stop.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    # ── Graceful shutdown (watchdog-guarded against a wedged worker pool) ──
    log.info("%s\n  MRTRADER TRADING DAEMON SHUTDOWN\n%s", bar, bar)
    watchdog = start_shutdown_watchdog()
    for t in (consumer_task, state_sync_task):
        t.cancel()
    await asyncio.gather(consumer_task, state_sync_task, return_exceptions=True)
    await stop_trading_brain(news_monitor_task, log)
    watchdog.cancel()
    return 0


def main() -> int:
    _configure_daemon_logging()
    log = logging.getLogger("mrtrader.daemon")
    mode = get_daemon_mode()
    if mode != DAEMON_MODE_SUBPROCESS:
        log.error(
            "Refusing to start: MRTRADER_DAEMON_MODE=%r (need %r). In the default "
            "in_process mode the FastAPI app already runs the brain — starting this "
            "daemon too would double-run the agents. Set MRTRADER_DAEMON_MODE=subprocess "
            "(and run the web read-only) to use the standalone daemon.",
            mode, DAEMON_MODE_SUBPROCESS)
        return 2
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
