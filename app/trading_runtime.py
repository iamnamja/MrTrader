"""
trading_runtime.py — shared trading-brain lifecycle + process-shutdown hardening.

R0.2 Phase 1 (daemon decouple). The PM/RM/Trader "brain" (orchestrator + agents +
scheduler + news monitor) and its preamble (state restore / reconciliation / queue
flush) live here so that BOTH boot paths use the exact same code:

  • the FastAPI lifespan (app/main.py) when MRTRADER_DAEMON_MODE=in_process (DEFAULT)
  • the standalone daemon (app/tradingd.py, `python -m app.tradingd`) when =subprocess

The default mode is byte-identical to the pre-R0.2 in-process boot — the web process
boots the brain exactly as before. In subprocess mode the web stays read-only and the
daemon owns trading. The mutual-exclusion interlock (web boots unless subprocess; the
daemon boots ONLY if subprocess) guarantees exactly one process ever runs the brain —
the "double-running agents" risk from the ADR cannot occur.

This module imports NO FastAPI — the daemon must not pull in the web stack.
"""
from __future__ import annotations

import asyncio
import logging
import os

DAEMON_MODE_IN_PROCESS = "in_process"
DAEMON_MODE_SUBPROCESS = "subprocess"
_VALID_MODES = {DAEMON_MODE_IN_PROCESS, DAEMON_MODE_SUBPROCESS}

# Comfortably above the worst-case graceful sequence (notify ~5s + bounded
# orchestrator.stop ~10s + parallel task cancels ~3s) so the watchdog only fires on a
# genuine HANG, never on a slow-but-healthy shutdown.
SHUTDOWN_WATCHDOG_SECONDS = float(os.environ.get("MRTRADER_SHUTDOWN_TIMEOUT", "30"))


# ─── Daemon mode ────────────────────────────────────────────────────────────

def get_daemon_mode() -> str:
    """Resolve MRTRADER_DAEMON_MODE. Unknown/empty → in_process (fail-safe: the web
    keeps booting the brain, i.e. today's behavior — never silently leaves nothing
    running the brain)."""
    raw = (os.environ.get("MRTRADER_DAEMON_MODE") or DAEMON_MODE_IN_PROCESS).strip().lower()
    if raw not in _VALID_MODES:
        logging.getLogger(__name__).warning(
            "MRTRADER_DAEMON_MODE=%r is not one of %s — falling back to %s "
            "(web boots the brain).", raw, sorted(_VALID_MODES), DAEMON_MODE_IN_PROCESS)
        return DAEMON_MODE_IN_PROCESS
    return raw


def web_boots_brain() -> bool:
    """True unless an external daemon owns the brain. The FastAPI process boots the
    orchestrator EXCEPT in subprocess mode (where `python -m app.tradingd` does)."""
    return get_daemon_mode() != DAEMON_MODE_SUBPROCESS


# ─── Process-shutdown hardening (shared by web + daemon) ──────────────────────

def start_shutdown_watchdog(timeout_s: float = SHUTDOWN_WATCHDOG_SECONDS):
    """Arm a daemon timer that force-exits the process if graceful shutdown wedges.

    Daemon threads keep running while the main thread is blocked joining non-daemon
    threads at interpreter exit, so this fires even in the exact hang we're guarding
    against. Returns the Timer so a caller/test can cancel it. If shutdown completes
    normally first, the process exits and this daemon dies with it (never fires)."""
    import threading

    def _force() -> None:
        logging.getLogger("mrtrader.shutdown").error(
            "Graceful shutdown exceeded %.0fs — forcing exit (os._exit). Lingering "
            "non-daemon worker pool/thread suspected.", timeout_s)
        os._exit(0)

    t = threading.Timer(timeout_s, _force)
    t.daemon = True
    t.start()
    return t


def kill_worker_pools() -> None:
    """Tear down background worker pools that would otherwise block interpreter exit.
    Best-effort and non-blocking; never raises.

    NOTE: the PM's ad-hoc ThreadPoolExecutors (portfolio_manager.py) shut down with
    wait=False but their non-daemon worker threads can still linger mid-task; those
    are covered by the hard-exit watchdog rather than here (they're transient)."""
    log = logging.getLogger("mrtrader.shutdown")
    # joblib/loky reusable PROCESS pool (persists across calls — the lingering
    # `python -c exec(...)` workers). kill_workers=True + wait=False = no block.
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=False, kill_workers=True)
        log.info("Shutdown: killed loky reusable executor workers")
    except Exception:
        pass


# ─── Trading-brain preamble (run by whichever process owns the brain) ─────────

async def prepare_trading_state(log: logging.Logger) -> None:
    """Restore persisted kill-switch/capital state, reconcile broker vs DB, and flush
    stale inter-agent queues. Run ONCE by the brain-owning process before the
    orchestrator starts. Each step is best-effort (warn-and-continue) exactly as the
    pre-R0.2 lifespan did — only DB init + connectivity (done by the caller) are fatal."""
    # ── Restore persisted state ──
    try:
        from app.live_trading.kill_switch import kill_switch
        from app.live_trading.capital_manager import capital_manager
        kill_switch.load_state()
        log.info("State restored (kill_switch=%s, capital_stage=%s)",
                 kill_switch.is_active, capital_manager.current_stage.stage)
    except Exception as e:
        log.warning("State restore warning: %s", e)

    # ── Startup reconciliation (broker vs DB) ──
    try:
        from app.integrations import get_alpaca_client
        from app.startup_reconciler import reconcile
        from app.database.session import get_session
        alpaca = get_alpaca_client()
        db = get_session()
        try:
            await asyncio.to_thread(reconcile, alpaca, db)
        finally:
            db.close()
    except Exception as e:
        log.warning("Startup reconciliation skipped: %s", e)

    # ── Flush stale inter-agent queues ──
    try:
        from app.integrations import get_redis_queue
        rq = get_redis_queue()
        for qname in ["trade_proposals", "risk_approved", "exit_requests", "pm_commands"]:
            n = rq.get_queue_length(qname)
            if n > 0:
                rq.clear_queue(qname)
                log.warning("Startup: flushed %d stale message(s) from '%s'", n, qname)
    except Exception as e:
        log.warning("Startup queue flush failed (non-fatal): %s", e)


# ─── Trading-brain lifecycle ──────────────────────────────────────────────────

async def start_trading_brain(log: logging.Logger) -> "asyncio.Task":
    """Register the PM/RM/Trader agents, start the orchestrator + scheduler, and launch
    the news monitor. Returns the news-monitor task so the caller can cancel it on
    shutdown. Mirrors the pre-R0.2 in-process boot exactly."""
    from app.agents.portfolio_manager import portfolio_manager
    from app.agents.risk_manager import risk_manager
    from app.agents.trader import trader
    from app.orchestrator import orchestrator
    from app.utils.constants import SECTOR_MAP

    risk_manager.update_sector_map(SECTOR_MAP)
    orchestrator.register_agent("portfolio_manager", portfolio_manager)
    orchestrator.register_agent("risk_manager", risk_manager)
    orchestrator.register_agent("trader", trader)
    await orchestrator.start()

    # Log active model versions (retry for connection pool warmup)
    try:
        from app.database.session import get_session
        from app.database.models import ModelVersion
        for _attempt in range(3):
            db = get_session()
            try:
                for name in ("swing", "intraday"):
                    row = (
                        db.query(ModelVersion)
                        .filter_by(model_name=name, status="ACTIVE")
                        .order_by(ModelVersion.version.desc())
                        .first()
                    )
                    if row:
                        log.info("Active model: %s v%d (path=%s)", name, row.version, row.model_path)
                    else:
                        if _attempt < 2:
                            break
                        log.warning("Active model: %s — NONE found in DB", name)
                else:
                    break
            finally:
                db.close()
            await asyncio.sleep(0.5)
    except Exception as e:
        log.warning("Could not log model versions: %s", e)

    from app.agents.news_monitor import news_monitor
    news_monitor_task = asyncio.create_task(news_monitor.run(), name="news_monitor")
    log.info("Orchestrator started (news monitor running)")
    return news_monitor_task


async def stop_trading_brain(news_monitor_task, log: logging.Logger) -> None:
    """Stop the orchestrator (bounded), kill lingering worker pools, and cancel the
    news monitor. Used by the standalone daemon; the FastAPI lifespan keeps its own
    inline shutdown sequence (guarded by web_boots_brain) so the web teardown stays
    byte-identical to pre-R0.2."""
    from app.orchestrator import orchestrator
    try:
        await asyncio.wait_for(orchestrator.stop(), timeout=10.0)
    except asyncio.TimeoutError:
        log.warning("orchestrator.stop() exceeded 10s — proceeding (watchdog armed)")
    except Exception as exc:
        log.warning("orchestrator.stop() error (proceeding): %s", exc)

    kill_worker_pools()

    if news_monitor_task is not None and not news_monitor_task.done():
        news_monitor_task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(news_monitor_task, return_exceptions=True), timeout=3.0)
        except asyncio.TimeoutError:
            pass
