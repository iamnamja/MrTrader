"""
control_bridge.py — R0.2 Phase 2: the web ↔ trading-daemon control bridge.

PROBLEM (subprocess mode): when the brain runs in the standalone daemon, the FastAPI
control routes call the *web's* orchestrator singleton — which has no agents and an
empty scheduler — so pause/resume/manual-triggers do nothing to the real brain, and a
web-triggered kill-switch persists to Postgres but the daemon's in-memory kill_switch
stays stale until restart.

SOLUTION (mode-conditional, default-off):
  • in_process (DEFAULT) — the routes call the in-process orchestrator directly, exactly
    as before Phase 2. The consumer/sync loops below never run (the daemon never runs).
    Nothing changes.
  • subprocess — the routes EMIT a command onto the Redis `pm_commands` queue; the
    daemon drains it (`consume_control_commands`) and applies it to the REAL brain. The
    kill-switch (already Postgres-persisted) propagates to the daemon via a periodic
    Postgres reload (`state_sync_loop`) plus an immediate reload command on activate/reset.

This module imports no FastAPI; the daemon uses it without the web stack.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict, Optional

log = logging.getLogger("mrtrader.control_bridge")

COMMAND_QUEUE = "pm_commands"

# ── Command names (the wire protocol) ─────────────────────────────────────────
CMD_PAUSE = "pause"
CMD_RESUME = "resume"
CMD_TRIGGER_CYCLE = "trigger_cycle"
CMD_TRIGGER_SWING = "trigger_swing"
CMD_TRIGGER_RETRAIN = "trigger_retrain"
CMD_TRIGGER_INTRADAY = "trigger_intraday"
CMD_CAPITAL_ADVANCE = "capital_advance"
CMD_GO_LIVE = "go_live"                     # start the capital ramp + flip to live (daemon-owned)
CMD_JOB_PAUSE = "job_pause"
CMD_JOB_RESUME = "job_resume"
CMD_RELOAD_STATE = "reload_state"          # tell the daemon to reload kill-switch now

# triggers run long (retrain/intraday can take minutes) → the consumer dispatches them
# as background tasks (see _TRIGGER_HANDLERS) so it never blocks; the rest apply inline.
_KNOWN_COMMANDS = {
    CMD_PAUSE, CMD_RESUME, CMD_TRIGGER_CYCLE, CMD_TRIGGER_SWING, CMD_TRIGGER_RETRAIN,
    CMD_TRIGGER_INTRADAY, CMD_CAPITAL_ADVANCE, CMD_GO_LIVE, CMD_JOB_PAUSE, CMD_JOB_RESUME,
    CMD_RELOAD_STATE,
}


# ── Producer side (called by the FastAPI routes in subprocess mode) ───────────

def emit_control_command(cmd: str, payload: Optional[dict] = None) -> bool:
    """Push a control command onto the daemon's command queue. Returns False on any
    failure (the route surfaces 'queued: false' rather than raising)."""
    if cmd not in _KNOWN_COMMANDS:
        log.error("emit_control_command: unknown command %r — refusing to enqueue", cmd)
        return False
    try:
        from app.integrations import get_redis_queue
        return bool(get_redis_queue().push(COMMAND_QUEUE, {"cmd": cmd, "payload": payload or {}}))
    except Exception as exc:  # noqa: BLE001
        log.error("emit_control_command(%s) failed: %s", cmd, exc)
        return False


def bridge_or_none(cmd: str, payload: Optional[dict] = None) -> Optional[dict]:
    """Route helper. In subprocess mode, emit `cmd` to the daemon and return a 'queued'
    response dict for the route to return. In in_process mode, return None so the route
    runs its existing direct path UNCHANGED (byte-identical to pre-Phase-2)."""
    from app.trading_runtime import web_boots_brain
    if web_boots_brain():
        return None
    ok = emit_control_command(cmd, payload)
    return {"status": "queued", "via": "daemon", "command": cmd, "queued": ok,
            "detail": "brain runs in the standalone daemon (subprocess mode); "
                      "command enqueued on pm_commands"}


# ── Consumer side (runs ONLY in the daemon) ───────────────────────────────────

def _pm():
    from app.orchestrator import orchestrator
    return orchestrator.agents.get("portfolio_manager")


async def _do_trigger_cycle() -> None:
    pm = _pm()
    if pm is None:
        log.warning("trigger_cycle: PM agent not registered")
        return
    log.info("trigger_cycle: running instrument selection")
    await pm.select_instruments()
    log.info("trigger_cycle: proposals sent to queue")


async def _do_trigger_swing() -> None:
    pm = _pm()
    if pm is None:
        log.warning("trigger_swing: PM agent not registered")
        return
    log.info("trigger_swing: forcing swing premarket analysis (bypassing time-of-day gate)")
    pm._analyzed_today = False
    pm._selected_today = False
    await pm._analyze_swing_premarket()
    await pm._send_swing_proposals()
    log.info("trigger_swing: complete — proposals sent")


async def _do_trigger_retrain() -> None:
    pm = _pm()
    if pm is None:
        log.warning("trigger_retrain: PM agent not registered")
        return
    log.info("trigger_retrain: starting")
    await pm._retrain()
    log.info("trigger_retrain: complete")


async def _do_trigger_intraday() -> None:
    pm = _pm()
    if pm is None:
        log.warning("trigger_intraday: PM agent not registered")
        return
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    log.info("trigger_intraday: full-universe scan rebuilding morning candidates")
    await pm.select_intraday_instruments(window=(now.hour, now.minute), use_morning_candidates=False)
    log.info("trigger_intraday: complete")


async def _do_pause() -> None:
    from app.orchestrator import orchestrator
    orchestrator.pause_trading()
    log.warning("pause: applied to the daemon orchestrator")


async def _do_resume() -> None:
    from app.orchestrator import orchestrator
    orchestrator.resume_trading()
    log.info("resume: applied to the daemon orchestrator")


async def _do_job_pause(payload: dict) -> None:
    job_id = payload.get("job_id")
    if not job_id:
        log.warning("job_pause: missing job_id")
        return
    from app.scheduler import scheduler
    scheduler.pause_job(job_id)
    log.info("job_pause: %s", job_id)


async def _do_job_resume(payload: dict) -> None:
    job_id = payload.get("job_id")
    if not job_id:
        log.warning("job_resume: missing job_id")
        return
    from app.scheduler import scheduler
    scheduler.resume_job(job_id)
    log.info("job_resume: %s", job_id)


async def _do_capital_advance() -> None:
    from app.live_trading.capital_manager import capital_manager
    from app.live_trading.monitoring import monitor
    health = monitor.health_check()
    if not capital_manager.can_advance(health["max_drawdown_pct"], abs(min(health["pnl_today_pct"], 0))):
        log.warning("capital_advance: cannot advance (stage not complete or health breached)")
        return
    result = capital_manager.advance()
    log.warning("capital_advance: %s", result)
    try:
        from app.database.session import get_session
        from app.database.models import AuditLog
        from datetime import datetime
        db = get_session()
        try:
            db.add(AuditLog(action="CAPITAL_STAGE_ADVANCED", details=result, timestamp=datetime.utcnow()))
            db.commit()
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("capital_advance: audit write failed: %s", exc)


async def _do_go_live() -> None:
    # Runs in the daemon (which owns the authoritative capital_manager + mode_manager). The web's
    # /approval/go-live re-verifies readiness, then bridges here so the DAEMON's singletons are the
    # ones started/flipped — otherwise the web mutated its own (empty) singletons and the live brain
    # never went live.
    from app.live_trading.capital_manager import capital_manager
    from app.trading_modes import mode_manager
    capital_manager.start()
    mode_manager.switch_to_live()
    log.warning("go_live: capital ramp started (Stage %d) + mode -> live",
                capital_manager.current_stage.stage)
    try:
        from app.database.session import get_session
        from app.database.models import AuditLog
        from datetime import datetime
        db = get_session()
        try:
            db.add(AuditLog(action="GO_LIVE_ACTIVATED",
                            details={"initial_capital": capital_manager.get_current_capital(),
                                     "via": "control_bridge"},
                            timestamp=datetime.utcnow()))
            db.commit()
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("go_live: audit write failed: %s", exc)


async def _do_reload_state() -> None:
    await sync_persisted_state(log)


_INLINE_HANDLERS: Dict[str, Callable[..., Awaitable[None]]] = {
    CMD_PAUSE: _do_pause,
    CMD_RESUME: _do_resume,
    CMD_JOB_PAUSE: _do_job_pause,
    CMD_JOB_RESUME: _do_job_resume,
    CMD_CAPITAL_ADVANCE: _do_capital_advance,
    CMD_GO_LIVE: _do_go_live,
    CMD_RELOAD_STATE: _do_reload_state,
}
_TRIGGER_HANDLERS: Dict[str, Callable[[], Awaitable[None]]] = {
    CMD_TRIGGER_CYCLE: _do_trigger_cycle,
    CMD_TRIGGER_SWING: _do_trigger_swing,
    CMD_TRIGGER_RETRAIN: _do_trigger_retrain,
    CMD_TRIGGER_INTRADAY: _do_trigger_intraday,
}


async def dispatch_command(cmd: str, payload: dict, *, background: set) -> None:
    """Apply one command to the in-process brain. Long-running triggers are spawned as
    tracked background tasks so the consumer keeps draining (a 7-min retrain must not
    block a pause/kill behind it). Never raises — a bad command is logged and dropped."""
    try:
        # Operator visibility: a pause does NOT interrupt an already-running manual trigger
        # (same as in_process — the route's fire-and-forget task keeps running). Make that
        # explicit in the log rather than silently leaving a trigger live behind a pause.
        if cmd in (CMD_PAUSE, CMD_RESUME) and background:
            log.warning("%s applied while %d manual-trigger task(s) still run (pause does NOT "
                        "interrupt an in-flight trigger)", cmd, len(background))
        if cmd in _TRIGGER_HANDLERS:
            handler = _TRIGGER_HANDLERS[cmd]

            async def _runner(_h=handler, _c=cmd):
                try:
                    await _h()
                except Exception as exc:  # noqa: BLE001
                    log.error("command %s failed: %s", _c, exc)
            t = asyncio.create_task(_runner(), name=f"control:{cmd}")
            background.add(t)
            t.add_done_callback(background.discard)
        elif cmd in _INLINE_HANDLERS:
            h = _INLINE_HANDLERS[cmd]
            # job_pause/job_resume take a payload; the rest take none
            if cmd in (CMD_JOB_PAUSE, CMD_JOB_RESUME):
                await h(payload)
            else:
                await h()
        else:
            log.warning("dispatch_command: unknown command %r (dropped)", cmd)
    except Exception as exc:  # noqa: BLE001 — the consumer must never die on one bad command
        log.error("dispatch_command(%s) error: %s", cmd, exc)


async def consume_control_commands(*, poll_timeout: int = 1, max_iterations: Optional[int] = None) -> None:
    """Daemon loop: drain `pm_commands` and apply each command to the real brain. Blocking
    BLPOP runs in a thread so the event loop stays free. `max_iterations` bounds the loop
    for tests (None = forever). Resilient: one failing command never stops the loop."""
    from app.integrations import get_redis_queue
    rq = get_redis_queue()
    background: set = set()
    log.info("control-command consumer started (queue=%s)", COMMAND_QUEUE)
    iterations = 0
    try:
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            try:
                msg = await asyncio.to_thread(rq.pop, COMMAND_QUEUE, poll_timeout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("control consumer pop error: %s", exc)
                await asyncio.sleep(1)
                continue
            if not msg:
                continue
            cmd = msg.get("cmd")
            payload = msg.get("payload") or {}
            log.info("control command received: %s payload=%s", cmd, payload)
            await dispatch_command(cmd, payload, background=background)
    except asyncio.CancelledError:
        log.info("control-command consumer stopping")
        # cancel any in-flight manual-trigger tasks so they don't outlive the consumer
        for t in list(background):
            t.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        raise


# ── State sync (the daemon's Postgres poll for kill-switch) ───────────────────

async def sync_persisted_state(logger_: Optional[logging.Logger] = None) -> bool:
    """Reload the kill-switch from Postgres into the daemon's singleton so a web-triggered
    activate/reset is honored (the agents/sleeves read kill_switch.is_active live). Returns
    True if the active flag CHANGED. Never raises."""
    lg = logger_ or log
    try:
        from app.live_trading.kill_switch import kill_switch
        before = kill_switch.is_active
        kill_switch.load_state()
        after = kill_switch.is_active
        if after != before:
            lg.warning("state-sync: kill-switch active %s -> %s (reloaded from Postgres)", before, after)
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        lg.error("state-sync: kill-switch reload failed: %s", exc)
        return False


async def state_sync_loop(*, interval: float = 3.0, max_iterations: Optional[int] = None) -> None:
    """Daemon loop: periodically reload kill-switch state from Postgres (backstop for the
    immediate reload_state command). `max_iterations` bounds it for tests."""
    log.info("state-sync loop started (interval=%.1fs)", interval)
    iterations = 0
    try:
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            await sync_persisted_state(log)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        log.info("state-sync loop stopping")
        raise
