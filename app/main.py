import asyncio
import logging
import logging.config
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket as _WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db, check_db_connection
from app.integrations import get_alpaca_client, get_redis_queue
from app.api.orchestrator_routes import router as orchestrator_router
from app.api.routes import router as dashboard_router
from app.api.watchlist_routes import router as watchlist_router
from app.api.config_routes import router as config_router
from app.api.nis_routes import router as nis_router, audit_router
from app.api.websocket import websocket_endpoint


# ---------------------------------------------------------------------------
# Logging
#
# Design: dictConfig with disable_existing_loggers=False merges cleanly with
# uvicorn's loggers. Uvicorn installs its own handlers before our lifespan
# runs; we set propagate=True and handlers=[] on uvicorn's named loggers so
# each record bubbles to root exactly once, avoiding fan-out duplication.
# Called once inside lifespan, never at import time.
# ---------------------------------------------------------------------------

class _DailyFileHandler(logging.Handler):
    """Rotate log file at midnight, prune files older than 30 days."""

    _KEEP_DAYS = 30

    def __init__(self, log_dir: str = "logs") -> None:
        super().__init__()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(exist_ok=True)
        self._current_date: str = ""
        self._file = None
        self._open_for_today()

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _open_for_today(self) -> None:
        if self._file:
            self._file.close()
        self._current_date = self._today()
        self._file = open(
            self._log_dir / f"mrtrader_{self._current_date}.log", "a", encoding="utf-8"
        )
        files = sorted(self._log_dir.glob("mrtrader_*.log"))
        for old in files[: max(0, len(files) - self._KEEP_DAYS)]:
            try:
                old.unlink()
            except OSError:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._today() != self._current_date:
                self._open_for_today()
            self._file.write(self.format(record) + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        super().close()


def _configure_logging() -> None:
    """Replace the root handler list with exactly two handlers via dictConfig.

    Using dictConfig guarantees an exact, known handler set regardless of what
    uvicorn or imported modules added to the root logger before us. Setting
    uvicorn's loggers to handlers=[] + propagate=True means each log record
    reaches root exactly once — no duplication.
    """
    level = settings.log_level.upper()
    fmt = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": fmt},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stderr",
            },
            "daily_file": {
                "()": _DailyFileHandler,
                "formatter": "standard",
                "log_dir": "logs",
            },
        },
        "root": {"level": level, "handlers": ["console", "daily_file"]},
        # Reformat uvicorn loggers; no handlers here — they propagate to root.
        "loggers": {
            "uvicorn":        {"handlers": [], "level": level, "propagate": True},
            "uvicorn.error":  {"handlers": [], "level": level, "propagate": True},
            "uvicorn.access": {"handlers": [], "level": "WARNING", "propagate": True},
        },
    })


def _git_info() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit, branch
    except Exception:
        return "unknown", "unknown"


# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup"/"shutdown").
# Starlette guarantees exactly one entry and one exit per ASGI process.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. Logging must be configured first ──────────────────────────────────
    _configure_logging()
    log = logging.getLogger("mrtrader.startup")

    # ── 2. Banner ─────────────────────────────────────────────────────────────
    commit, branch = _git_info()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    bar = "=" * 72
    log.info(
        "\n%s\n  MRTRADER STARTUP  %s\n  git: %s  branch: %s\n"
        "  mode: %s  capital: $%s  python: %s  port: %s\n%s",
        bar, now_utc, commit, branch,
        settings.trading_mode.upper(),
        f"{settings.initial_capital:,.0f}",
        sys.version.split()[0], settings.port, bar,
    )

    # ── 3. Database ───────────────────────────────────────────────────────────
    try:
        init_db()
        log.info("OK Database initialized")
    except Exception as e:
        log.error("Database initialization failed: %s", e)
        raise
    if not check_db_connection():
        raise RuntimeError("Cannot connect to database")
    log.info("OK Database connection verified")

    # ── 4. Redis + Alpaca (parallel) ─────────────────────────────────────────
    try:
        redis_queue = get_redis_queue()
        redis_ok, alpaca_ok = await asyncio.gather(
            asyncio.to_thread(redis_queue.health_check),
            asyncio.to_thread(get_alpaca_client().health_check),
        )
        if redis_ok:
            log.info("OK Redis connection verified")
        else:
            raise RuntimeError("Redis health check failed")
        if alpaca_ok:
            log.info("OK Alpaca connection verified")
        else:
            log.warning("Alpaca health check failed — continuing in degraded mode")
    except RuntimeError:
        raise
    except Exception as e:
        log.warning("Startup connectivity check failed: %s", e)

    # ── 5. Restore persisted state ────────────────────────────────────────────
    try:
        from app.live_trading.kill_switch import kill_switch
        from app.live_trading.capital_manager import capital_manager
        kill_switch.load_state()
        log.info("State restored (kill_switch=%s, capital_stage=%s)",
                 kill_switch.is_active, capital_manager.current_stage.stage)
    except Exception as e:
        log.warning("State restore warning: %s", e)

    # ── 6. Startup reconciliation (Alpaca vs DB) ──────────────────────────────
    try:
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

    # ── 7. Flush stale inter-agent queues ─────────────────────────────────────
    try:
        rq = get_redis_queue()
        for qname in ["trade_proposals", "risk_approved", "exit_requests", "pm_commands"]:
            n = rq.get_queue_length(qname)
            if n > 0:
                rq.clear_queue(qname)
                log.warning("Startup: flushed %d stale message(s) from '%s'", n, qname)
    except Exception as e:
        log.warning("Startup queue flush failed (non-fatal): %s", e)

    # ── 8. Orchestrator + agents ──────────────────────────────────────────────
    try:
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
        app.state.news_monitor_task = asyncio.create_task(
            news_monitor.run(), name="news_monitor"
        )
        log.info("Orchestrator started (news monitor running)")
    except Exception as e:
        log.error("Orchestrator startup failed: %s", e)
        raise

    # ── 9. Background warm-ups (non-blocking) ────────────────────────────────
    app.state.warm_task = asyncio.create_task(_warm_caches(), name="cache_warmup")
    app.state.market_task = asyncio.create_task(_refresh_market_data(), name="market_data")

    log.info("MrTrader application started successfully")

    # ── Serve requests ────────────────────────────────────────────────────────
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    _log_shutdown = logging.getLogger("mrtrader.shutdown")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    _log_shutdown.info("%s\n  MRTRADER SHUTDOWN  %s\n%s", "=" * 72, now_utc, "=" * 72)

    from app.orchestrator import orchestrator
    await orchestrator.stop()

    for task_name in ("news_monitor_task", "warm_task", "market_task"):
        t = getattr(app.state, task_name, None)
        if t and not t.done():
            t.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(t), timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass


async def _warm_caches() -> None:
    log = logging.getLogger("mrtrader.startup")
    from app.api.routes import get_dashboard_summary, get_health_alias, get_market_regime
    results = await asyncio.gather(
        get_dashboard_summary(), get_health_alias(), get_market_regime(),
        return_exceptions=True,
    )
    for name, r in zip(["summary", "health", "regime"], results):
        if isinstance(r, Exception):
            log.warning("Cache warm-up skipped for %s: %s", name, r)
    log.info("OK Dashboard cache warmed")


async def _refresh_market_data() -> None:
    log = logging.getLogger("mrtrader.startup")

    def _do_refresh() -> None:
        try:
            from app.data.macro_history import update_macro_history
            update_macro_history()
        except Exception as e:
            log.warning("Macro history refresh failed: %s", e)
        try:
            from scripts.backfill_sector_etf_history import update_sector_etf_history_incremental
            update_sector_etf_history_incremental()
        except Exception as e:
            log.warning("Sector ETF refresh failed: %s", e)

    await asyncio.to_thread(_do_refresh)
    log.info("OK Market data (macro + sector ETF) refreshed")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MrTrader - Automated Trading System",
    description="AI-powered automated day trading system",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(orchestrator_router)
app.include_router(dashboard_router)
app.include_router(watchlist_router)
app.include_router(config_router)
app.include_router(nis_router)
app.include_router(audit_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    from fastapi.responses import JSONResponse
    from app.live_trading.kill_switch import kill_switch
    from app.agents.circuit_breaker import circuit_breaker

    degraded = kill_switch.is_active or circuit_breaker.is_open
    body = {
        "status": "degraded" if degraded else "healthy",
        "version": "0.1.0",
        "kill_switch": kill_switch.is_active,
        "circuit_breaker": circuit_breaker.status(),
    }
    return JSONResponse(content=body, status_code=503 if degraded else 200)


@app.get("/api/status")
async def get_status():
    db_ok = check_db_connection()
    redis_ok, alpaca_ok = await asyncio.gather(
        asyncio.to_thread(get_redis_queue().health_check),
        asyncio.to_thread(get_alpaca_client().health_check),
    )
    return {
        "status": "healthy" if all([db_ok, redis_ok]) else "degraded",
        "database": "OK connected" if db_ok else "disconnected",
        "redis": "OK connected" if redis_ok else "disconnected",
        "alpaca": "OK connected" if alpaca_ok else "error",
        "mode": settings.trading_mode,
    }


@app.get("/api/account")
async def get_account_info():
    try:
        account = get_alpaca_client().get_account()
        return {"status": "success", "data": account}
    except Exception as e:
        logger.error("Error fetching account info: %s", e)
        return {"status": "error", "message": str(e)}


@app.get("/api/positions")
async def get_positions():
    try:
        positions = get_alpaca_client().get_positions()
        return {"status": "success", "data": positions, "count": len(positions)}
    except Exception as e:
        logger.error("Error fetching positions: %s", e)
        return {"status": "error", "message": str(e)}


@app.get("/api/position/{symbol}")
async def get_position(symbol: str):
    try:
        position = get_alpaca_client().get_position(symbol.upper())
        if position:
            return {"status": "success", "data": position}
        return {"status": "not_found", "message": f"No position found for {symbol}"}
    except Exception as e:
        logger.error("Error fetching position: %s", e)
        return {"status": "error", "message": str(e)}


@app.websocket("/ws")
async def websocket_route(websocket: _WebSocket):
    await websocket_endpoint(websocket)


_REACT_DIST = "frontend/dist"
_REACT_INDEX = os.path.join(_REACT_DIST, "index.html")
_LEGACY_HTML = "frontend/dashboard.html"

_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    if os.path.isfile(_REACT_INDEX):
        with open(_REACT_INDEX) as f:
            return HTMLResponse(content=f.read(), headers=_NO_CACHE_HEADERS)
    with open(_LEGACY_HTML) as f:
        return HTMLResponse(content=f.read(), headers=_NO_CACHE_HEADERS)


@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


# Must be last — API routes above take priority.
if os.path.isdir(_REACT_DIST):
    app.mount("/", StaticFiles(directory=_REACT_DIST, html=True), name="spa")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port, reload=settings.debug)
