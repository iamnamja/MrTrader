"""
Agent Orchestrator — schedules agents, routes messages, handles failures.

Pipeline:
  PortfolioManager  →  (trade_proposals queue)
  RiskManager       →  (trader_approved_trades queue)
  Trader            →  Alpaca orders
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
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
        # True when trading was paused AUTOMATICALLY by the health check (Alpaca down). Lets the
        # health check auto-RESUME once Alpaca recovers — a transient blip must not permanently halt
        # trading.
        self._auto_paused = False
        # True when an OPERATOR paused trading. Tracked SEPARATELY from _auto_paused so a health-check
        # Alpaca blip during a manual pause can neither convert it to an auto-pause nor (on recovery)
        # auto-RESUME the halt the operator deliberately requested. Cleared only by an explicit resume.
        self._manual_paused = False

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

        # Background monitor — tracked in _tasks so stop() cancels it (else it
        # leaks and lingers up to its 60s sleep after _running flips False).
        self._tasks["_monitor"] = asyncio.create_task(
            self._monitor_loop(), name="orchestrator_monitor")
        logger.info("Orchestrator started — %d agents running", len(self.agents))

    async def stop(self) -> None:
        """Stop all agents and the scheduler."""
        logger.info("Stopping orchestrator…")
        self._running = False
        scheduler.stop()
        # Cancel every tracked task (agents + the monitor loop) in PARALLEL and await
        # them under a single bound — sequential 5s-per-task could blow the lifespan
        # shutdown budget when several tasks are slow to honor cancellation.
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        self._tasks.clear()
        logger.info("Orchestrator stopped (%d task(s))", len(tasks))

    # ─── Scheduling ───────────────────────────────────────────────────────────

    def _schedule_jobs(self) -> None:
        """Register all APScheduler jobs."""

        # Health check every 5 minutes
        scheduler.schedule_every_n_minutes(
            self._health_check, minutes=5, job_id="health_check"
        )

        # Dead-man liveness heartbeat every 1 min (Alpha-v10 H5). Writes a durable timestamp file the
        # EXTERNAL dead-man watchdog reads; if it goes stale the brain has died/hung. Additive + inert
        # to trading (a file write); fail-safe (write_heartbeat never raises).
        scheduler.schedule_every_n_minutes(
            self._write_heartbeat, minutes=1, job_id="dead_man_heartbeat"
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

        # Weekly trend-sleeve rebalance — registered DAILY (09:45 ET, Mon–Fri) with
        # an in-function weekday guard so pm.trend_rebalance_weekday stays live-tunable
        # (no redeploy to change the rebalance day). 09:45 is after the 09:30 selection
        # nudge and after the open settles so daily bars/quotes are stable.
        # misfire_grace_time=1800: this is a WEEKLY rebalance (Monday-only, in-handler
        # gated). With the 60s default, a >60s-late fire at 09:45 (busy loop / restart)
        # would make APScheduler DROP the whole week's rebalance. A 30-min grace lets a
        # delayed Monday fire still run; TSMOM is a slow signal so a late fire is benign.
        scheduler.schedule_daily_at_time(
            self._trigger_trend_rebalance, hour=9, minute=45,
            job_id="trend_rebalance_trigger", misfire_grace_time=1800,
        )

        # Cash/T-bill sleeve (P1-1) at 09:50 ET — AFTER trend (09:45) so the idle
        # remainder is settled. Same weekly cadence; dormant unless pm.cash_enabled=true.
        scheduler.schedule_daily_at_time(
            self._trigger_cash_rebalance, hour=9, minute=50,
            job_id="cash_rebalance_trigger", misfire_grace_time=1800,
        )

        # CH5 — post-rebalance ENFORCE verification at 11:07 ET (same weekly cadence + weekday +
        # market-open guard as the trend rebalance). Read-only: confirms the enforce gates evaluated
        # CLEAN (no spurious hold) and the CH0b scorecard captured this rebalance's per-governor
        # multipliers + ungoverned counterfactual (un-backfillable); emails a PASS/ATTENTION summary.
        # 11:07 (not 10:17) gives ample slack even if the 09:45 rebalance MISFIRES late (its
        # misfire_grace_time is 1800s → it can start as late as ~10:15), so the scorecard write is
        # always done first. No trading. Critical for the FIRST enforce Monday (2026-07-13) and a
        # durable weekly enforce-health check thereafter.
        scheduler.schedule_daily_at_time(
            self._verify_enforce_rebalance, hour=11, minute=7,
            job_id="enforce_rebalance_verify", misfire_grace_time=1800,
        )

        # Crypto trend LIVE-PAPER tracker (P3-1) at 09:55 ET — report-only OOS recorder.
        # Recomputes the rules-based crypto-trend book on live Alpaca closes weekly and
        # freezes the forward out-of-sample slice. No orders, no capital. Crypto is 24/7
        # so there is no market-open gate; weekly cadence, dormant unless enabled.
        scheduler.schedule_daily_at_time(
            self._trigger_crypto_paper_track, hour=9, minute=55,
            job_id="crypto_paper_track_trigger", misfire_grace_time=1800,
        )

        # Nightly options NBBO/spread logger at 15:55 ET (Alpha-v6 P1c slow fuse).
        # Snapshots live bid/ask for the frozen panel just before the close — the
        # only window where the chain's quotes reflect a real trading book. Misses
        # cost one observation day (the spread fitter tolerates gaps), so a modest
        # 180s grace is enough. Pure data logging: no orders, no agent state.
        scheduler.schedule_daily_at_time(
            self._trigger_options_nbbo_log, hour=15, minute=55,
            job_id="options_nbbo_logger", misfire_grace_time=180,
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
                # Re-project a STANDING pause onto the agent before it restarts, so a crash during a
                # manual/auto pause cannot silently resume trading (run() also preserves an existing
                # 'paused' status; this covers a pause issued during the 30s restart window).
                if (self._manual_paused or self._auto_paused) and hasattr(agent, "status"):
                    agent.status = "paused"
                    logger.warning("Agent %s restart: re-applying standing pause", name)

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
        """Spawn retrain_cron.py as a subprocess so training workers are isolated from uvicorn."""
        # Global retrain guard (Alpha-v4 P0): honor RETRAIN_WEEKDAY here, not just in the
        # dormant PM path. -1 disables all scheduled retrains; otherwise only the configured
        # weekday fires. This is the root-cause fix for the daily-trigger ignoring the flag —
        # without it any model in retrain_cron defaults to nightly-on. Per-model freezes
        # (SWING_ENABLED / INTRADAY_ENABLED) are the finer-grained control inside retrain_cron.
        try:
            from app.ml.retrain_config import RETRAIN_WEEKDAY
            from datetime import datetime as _dt
            try:
                from zoneinfo import ZoneInfo
                _wd = _dt.now(ZoneInfo("America/New_York")).weekday()
            except Exception:
                _wd = _dt.now().weekday()
            if RETRAIN_WEEKDAY < 0:
                logger.info("Orchestrator: scheduled retrain disabled (RETRAIN_WEEKDAY=-1) — skipping")
                return
            if _wd != RETRAIN_WEEKDAY:
                logger.debug("Orchestrator: not the retrain weekday (%d != %d) — skipping",
                             _wd, RETRAIN_WEEKDAY)
                return
        except Exception as exc:
            logger.warning("Orchestrator: retrain-weekday guard failed (proceeding): %s", exc)

        # Guard: skip retrain if a CPCV or walk-forward job is actively writing logs.
        # Running two XGBoost training jobs concurrently on this machine causes OOM.
        logs_dir = Path(__file__).resolve().parent.parent / "logs"
        _now = time.time()
        for _log in logs_dir.glob("p0_*_cpcv.log"):
            if _now - _log.stat().st_mtime < 3600:
                logger.warning(
                    "Orchestrator: skipping scheduled retrain — CPCV job active (%s written %.0f min ago)",
                    _log.name, (_now - _log.stat().st_mtime) / 60,
                )
                return
        for _log in logs_dir.glob("wf_*.log"):
            if _now - _log.stat().st_mtime < 3600:
                logger.warning(
                    "Orchestrator: skipping scheduled retrain — walk-forward job active (%s written %.0f min ago)",
                    _log.name, (_now - _log.stat().st_mtime) / 60,
                )
                return
        logger.info("Orchestrator: spawning retraining subprocess")
        script = Path(__file__).resolve().parent.parent / "scripts" / "retrain_cron.py"
        log_date = datetime.utcnow().strftime("%Y-%m-%d")
        log_path = Path(__file__).resolve().parent.parent / "logs" / f"retrain_{log_date}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "a") as log_fh:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(script),
                    stdout=log_fh, stderr=log_fh,
                )
            logger.info("Retraining subprocess started (pid=%d) — logging to %s", proc.pid, log_path)
            returncode = await proc.wait()
            if returncode == 0:
                logger.info("Retraining subprocess completed successfully")
            elif returncode == 2:
                logger.warning("Retraining subprocess: one or more models failed the gate — previous champions retained")
            else:
                logger.error("Retraining subprocess exited with code %d — see %s", returncode, log_path)
                await self._log_error("model_trainer", f"retrain_cron.py exited {returncode}")
        except Exception as exc:
            logger.error("Retraining trigger failed: %s", exc)
            await self._log_error("model_trainer", str(exc))

    async def _verify_enforce_rebalance(self) -> None:
        """CH5 post-rebalance ENFORCE verification (read-only; emails a PASS/ATTENTION summary).

        Fires daily (cron) but runs only on the trend rebalance weekday AND only when the market is
        open (same fail-closed guard as the rebalance — no point verifying a rebalance that didn't
        happen). Confirms the enforce gates evaluated clean + the CH0b scorecard captured the
        governor data (un-backfillable). Blocking DB/notifier work runs off the event loop. Never
        disrupts anything — verification only, no trading."""
        from datetime import datetime as _dt
        try:
            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
        except Exception:
            _et = None
        try:
            from app.database.session import get_session
            from app.database.agent_config import get_agent_config
            db = get_session()
            try:
                target_weekday = int(get_agent_config(db, "pm.trend_rebalance_weekday"))
            finally:
                db.close()
        except Exception:
            target_weekday = 0
        today = _dt.now(_et).date() if _et else _dt.now().date()
        if today.weekday() != target_weekday:
            return
        try:
            from app.integrations import get_alpaca_client
            clock = get_alpaca_client().get_clock()
        except Exception as exc:
            logger.debug("enforce-verify: clock check failed — skip: %s", exc)
            return
        if not clock or not clock.get("is_open"):
            return
        try:
            from scripts.verify_enforce_rebalance import run_and_report
            loop = asyncio.get_event_loop()
            rep = await loop.run_in_executor(None, run_and_report)
            logger.info("Enforce-rebalance verify: status=%s attention=%d",
                        rep.get("status"), len(rep.get("attention", [])))
        except Exception as exc:  # noqa: BLE001
            logger.warning("enforce-rebalance verify failed (non-fatal): %s", exc)

    async def _trigger_trend_rebalance(self) -> None:
        """Weekly TSMOM trend-sleeve rebalance (Alpha-v4 live wiring).

        Fires daily (cron Mon–Fri); runs only on pm.trend_rebalance_weekday AND only
        when the market is genuinely open (fail-closed on holidays — the cron has no
        holiday calendar). The executor itself is dormant unless pm.trend_enabled=true
        and honours pm.trend_shadow. Blocking Alpaca work runs off the event loop.
        """
        from datetime import datetime as _dt
        try:
            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
        except Exception:
            _et = None

        try:
            from app.database.session import get_session
            from app.database.agent_config import get_agent_config
            db = get_session()
            try:
                target_weekday = int(get_agent_config(db, "pm.trend_rebalance_weekday"))
            finally:
                db.close()
        except Exception:
            target_weekday = 0  # Monday default

        today = _dt.now(_et).date() if _et else _dt.now().date()
        if today.weekday() != target_weekday:
            logger.debug("trend rebalance: not the configured weekday (%d != %d) — skip",
                         today.weekday(), target_weekday)
            return

        # Market-open guard (fail-closed: unknown/closed -> do not trade; covers holidays)
        try:
            from app.integrations import get_alpaca_client
            clock = get_alpaca_client().get_clock()
        except Exception as exc:
            logger.warning("trend rebalance: clock check failed — skipping (fail-closed): %s", exc)
            return
        if not clock or not clock.get("is_open"):
            logger.info("trend rebalance: market not open today — skip (holiday/closed)")
            return

        logger.info("Orchestrator: triggering weekly trend rebalance")
        try:
            from app.live_trading import trend_sleeve, trend_tracker
            loop = asyncio.get_event_loop()
            # Alpha-v4 P3: recompute the regime-aware sleeve weights BEFORE the trend
            # rebalance, so the rebalance in this same tick reads the fresh effective
            # allocation. run_allocator is a no-op (returns 'disabled') unless
            # pm.allocator_enabled=true, so this is inert until the owner flips it on.
            try:
                from app.live_trading import sleeve_allocator_live
                _alloc = await loop.run_in_executor(None, sleeve_allocator_live.run_allocator)
                logger.info("Allocator recompute: status=%s scheme=%s source=%s",
                            _alloc.get("status"), _alloc.get("scheme"), _alloc.get("source"))
            except Exception as _aexc:
                logger.error("Allocator recompute failed (continuing with static weights): %s", _aexc)
            summary = await loop.run_in_executor(None, trend_sleeve.run_trend_rebalance)
            logger.info("Trend rebalance done: status=%s mode=%s approved=%d blocked=%d",
                        summary.get("status"), summary.get("mode"),
                        len(summary.get("approved", [])), len(summary.get("blocked", [])))
            # Fold the weekly rollup into the rebalance day.
            await loop.run_in_executor(None, trend_tracker.weekly_rollup)
            # P1-4: record the INTENDED book for this rebalance, then run the live-vs-sim
            # back-validation report (report-only; emails its verdict once enough live history
            # has accrued — BUILDING until then).
            try:
                from app.live_trading import back_validation
                await loop.run_in_executor(
                    None, lambda: back_validation.record_rebalance_intent(summary))
                _bv = await loop.run_in_executor(None, back_validation.weekly_report)
                logger.info("Back-validation weekly: verdict=%s n_days=%s",
                            _bv.get("verdict"), _bv.get("n_days"))
            except Exception as _bvexc:
                logger.error("Back-validation weekly report failed (continuing): %s", _bvexc)
        except Exception as exc:
            logger.error("Trend rebalance trigger failed: %s", exc)
            await self._log_error("trend_sleeve", str(exc))

    async def _trigger_cash_rebalance(self) -> None:
        """Weekly cash/T-bill sleeve rebalance (P1-1). Fires daily (cron Mon–Fri); runs only
        on pm.cash_rebalance_weekday AND when the market is open, AFTER the trend rebalance so
        the idle remainder is settled. Dormant unless pm.cash_enabled=true; honours pm.cash_shadow.
        """
        from datetime import datetime as _dt
        try:
            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
        except Exception:
            _et = None

        try:
            from app.database.session import get_session
            from app.database.agent_config import get_agent_config
            db = get_session()
            try:
                target_weekday = int(get_agent_config(db, "pm.cash_rebalance_weekday"))
            finally:
                db.close()
        except Exception:
            target_weekday = 0  # Monday default

        today = _dt.now(_et).date() if _et else _dt.now().date()
        if today.weekday() != target_weekday:
            logger.debug("cash rebalance: not the configured weekday (%d != %d) — skip",
                         today.weekday(), target_weekday)
            return

        try:
            from app.integrations import get_alpaca_client
            clock = get_alpaca_client().get_clock()
        except Exception as exc:
            logger.warning("cash rebalance: clock check failed — skipping (fail-closed): %s", exc)
            return
        if not clock or not clock.get("is_open"):
            logger.info("cash rebalance: market not open today — skip (holiday/closed)")
            return

        logger.info("Orchestrator: triggering weekly cash rebalance")
        try:
            from app.live_trading import cash_sleeve, cash_tracker
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, cash_sleeve.run_cash_rebalance)
            logger.info("Cash rebalance done: status=%s mode=%s action=%s approved=%d",
                        summary.get("status"), summary.get("mode"),
                        summary.get("action"), len(summary.get("approved", [])))
            await loop.run_in_executor(None, cash_tracker.weekly_rollup)
        except Exception as exc:
            logger.error("Cash rebalance trigger failed: %s", exc)
            await self._log_error("cash_sleeve", str(exc))

    async def _trigger_crypto_paper_track(self) -> None:
        """Weekly crypto trend LIVE-PAPER tracker (P3-1). Report-only: recomputes the
        rules-based crypto-trend book on live Alpaca closes and freezes the forward OOS
        slice. NO orders/capital, so NO market-open gate (crypto is 24/7). Fires daily,
        runs only on pm.crypto_paper_rebalance_weekday; dormant unless pm.crypto_paper_enabled.
        """
        from datetime import datetime as _dt
        try:
            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
        except Exception:
            _et = None

        try:
            from app.database.session import get_session
            from app.database.agent_config import get_agent_config
            db = get_session()
            try:
                target_weekday = int(get_agent_config(db, "pm.crypto_paper_rebalance_weekday"))
            finally:
                db.close()
        except Exception:
            target_weekday = 0  # Monday default

        today = _dt.now(_et).date() if _et else _dt.now().date()
        if today.weekday() != target_weekday:
            logger.debug("crypto paper track: not the configured weekday (%d != %d) — skip",
                         today.weekday(), target_weekday)
            return

        logger.info("Orchestrator: triggering weekly crypto live-paper track")
        try:
            from app.live_trading import crypto_paper_track
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, crypto_paper_track.run_crypto_paper_track)
            logger.info("Crypto paper track done: status=%s n_oos=%s sharpe=%s (backtest %.2f)",
                        summary.get("status"), summary.get("n_oos_days"),
                        summary.get("oos_sharpe"), summary.get("backtest_sharpe"))
            await loop.run_in_executor(None, crypto_paper_track.weekly_email, summary)
        except Exception as exc:
            logger.error("Crypto paper track trigger failed: %s", exc)
            await self._log_error("crypto_paper_track", str(exc))

    async def _trigger_options_nbbo_log(self) -> None:
        """Nightly options NBBO/spread snapshot (Alpha-v6 P1c — FUSE A).

        Appends the frozen panel's live bid/ask spread observations to
        data/options_spread_obs.parquet via scripts/log_options_nbbo.py. Blocking
        HTTP + parquet work runs off the event loop. Failure is logged (and lands
        in the error log) but never disturbs the trading agents — losing one
        observation day only delays the spread-model calibration window.
        """
        logger.info("Orchestrator: running nightly options NBBO logging")
        try:
            from scripts.log_options_nbbo import run_nbbo_logging
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, run_nbbo_logging)
            logger.info(
                "Options NBBO logging done: status=%s rows=%s dropped=%s store_total=%s",
                summary.get("status"), summary.get("rows_written"),
                summary.get("rows_dropped_no_quote"), summary.get("store_rows_total"),
            )
            # "skipped" = market holiday (the 15:55 weekday schedule still fires on
            # holidays; run_nbbo_logging's calendar gate skips them) — expected, not
            # an error. Only a real zero-row run lands in the error log.
            if summary.get("status") not in ("ok", "skipped"):
                await self._log_error("options_nbbo_logger",
                                      f"zero rows logged: {summary}")
            # P2-4: keep the spread COST SURFACE fresh as the NBBO log accrues, and flag when
            # it matures enough for a VRP verdict. Re-runs the (cheap) calibration nightly and
            # emails ONCE when the window crosses MATURE_MIN_DAYS. Report-only; never disturbs
            # trading (the calibrated model is opt-in and not wired to any live verdict yet).
            if summary.get("status") == "ok":
                try:
                    from app.options.spread_model import (
                        calibrate_from_parquet, OBS_PARQUET, MODEL_PATH, MATURE_MIN_DAYS,
                    )
                    m = await loop.run_in_executor(
                        None, lambda: calibrate_from_parquet(OBS_PARQUET))
                    await loop.run_in_executor(None, lambda: m.save(MODEL_PATH))
                    logger.info(
                        "Spread surface recalibrated: n_obs=%s days=%s window=%s..%s -> %s",
                        m.n_obs, m.n_days, m.calibrated_from, m.calibrated_through,
                        "MATURE" if m.is_mature else "BUILDING")
                    if m.is_mature:
                        from app.notifications import notifier
                        notifier.enqueue("options_spread_mature", {
                            "n_obs": m.n_obs, "n_days": m.n_days,
                            "window": f"{m.calibrated_from}..{m.calibrated_through}",
                            "mature_min_days": MATURE_MIN_DAYS,
                        }, dedup_key="options_spread_mature")  # fires once
                except Exception as _cexc:
                    logger.error("Spread recalibration failed (continuing): %s", _cexc)
        except Exception as exc:
            logger.error("Options NBBO logging failed: %s", exc)
            await self._log_error("options_nbbo_logger", str(exc))

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
            # P1-4: capture the trend sleeve's EOD state for live-vs-sim back-validation
            # (report-only; never raises; builds the daily tracking-error series).
            try:
                from app.live_trading import back_validation
                await loop.run_in_executor(None, back_validation.record_daily_snapshot)
            except Exception as _bvexc:
                logger.error("Back-validation snapshot failed (continuing): %s", _bvexc)
        except Exception as exc:
            logger.error("Daily summary failed: %s", exc)

    async def _write_heartbeat(self) -> None:
        """Dead-man liveness beat (H5): write the durable heartbeat the external watchdog reads.
        Fail-safe — write_heartbeat never raises; a failed write just looks 'stale' to the watchdog."""
        from app.live_trading.heartbeat import ping_snitch, write_heartbeat
        write_heartbeat()
        # Off-box dead-man's-snitch: ping an external check each beat so total-machine death (power/OS)
        # — the one failure the on-box watchdog dies with — is caught externally. No-op unless
        # MRTRADER_SNITCH_URL is set; fire-and-forget, never raises.
        ping_snitch()
        # H2: refresh the in-memory kill-switch state machine's heartbeat too, so its dead-man
        # backstop only escalates if THIS beat loop has been dead for minutes (never raises).
        try:
            from app.live_trading.kill_switch_state import kill_switch_sm
            kill_switch_sm.heartbeat()
        except Exception:
            pass

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
            # Auto-pause only from a RUNNING state — never when already auto- OR manually paused. A
            # manual pause is already a halt, so converting it to an auto-pause would let the recovery
            # path below auto-resume the operator's deliberate halt.
            if not alpaca_ok and not self._auto_paused and not self._manual_paused:
                logger.critical("Alpaca unavailable — pausing trading (auto; will auto-resume on "
                                "recovery)")
                self.pause_trading(auto=True)
        else:
            # All services healthy. If we auto-paused on a prior Alpaca outage, auto-RESUME now so a
            # transient blip can't permanently halt trading (the cooperative-pause loops are still
            # alive). A MANUAL pause (operator) is NEVER auto-resumed.
            if self._auto_paused and not self._manual_paused:
                logger.warning("Alpaca recovered — auto-resuming trading")
                self.resume_trading()
            else:
                logger.debug("Health check OK")

    # ─── Emergency controls ───────────────────────────────────────────────────

    def pause_trading(self, *, auto: bool = False) -> None:
        """Pause portfolio selection + retraining triggers and the agent loops.

        The agent loops are COOPERATIVE: setting status='paused' makes them idle but stay alive (the
        task is not killed), so resume_trading() actually resumes them. `auto=True` (health-check
        Alpaca-down) marks this as auto-paused so it can be auto-resumed on recovery; a manual pause
        (auto=False) sets the operator-pause latch so it is NEVER auto-resumed."""
        scheduler.pause_job("portfolio_selection_trigger")
        scheduler.pause_job("model_retraining_trigger")
        for name, agent in self.agents.items():
            if hasattr(agent, "status") and agent.status != "stopped":
                agent.status = "paused"
        if auto:
            self._auto_paused = True
        else:
            # operator pause: latch manual, and it is NOT an auto-pause
            self._manual_paused = True
            self._auto_paused = False
        logger.warning("TRADING PAUSED by orchestrator%s", " (auto)" if auto else " (manual)")

    def resume_trading(self) -> None:
        """Resume paused triggers and agents. Clears BOTH the auto- and manual-pause flags — an
        explicit resume is the only thing that lifts an operator (manual) pause."""
        scheduler.resume_job("portfolio_selection_trigger")
        scheduler.resume_job("model_retraining_trigger")
        for name, agent in self.agents.items():
            if hasattr(agent, "status") and agent.status == "paused":
                agent.status = "running"
        self._auto_paused = False
        self._manual_paused = False
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
