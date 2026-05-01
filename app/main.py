import asyncio
import logging
import logging.handlers
import os
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import WebSocket as _WebSocket

from app.config import settings
from app.database import init_db, check_db_connection
from app.integrations import get_alpaca_client, get_redis_queue
from app.api.orchestrator_routes import router as orchestrator_router
from app.api.routes import router as dashboard_router
from app.api.watchlist_routes import router as watchlist_router
from app.api.config_routes import router as config_router
from app.api.nis_routes import router as nis_router, audit_router
from app.api.websocket import websocket_endpoint


class _DailyFileHandler(logging.Handler):
    """
    Writes to logs/mrtrader_YYYY-MM-DD.log based on the current wall-clock date.

    - On startup: opens today's dated file (appends if it already exists).
    - At midnight: detects the date change on the next log write and seamlessly
      switches to a new file — no restart required, no stale data in wrong file.
    - Keeps last 30 daily files; older ones are pruned automatically.
    """

    _KEEP_DAYS = 30

    def __init__(self, log_dir: Path, fmt: logging.Formatter) -> None:
        super().__init__()
        self._log_dir = log_dir
        self.setFormatter(fmt)
        self._current_date: str = ""
        self._file = None
        self._open_for_today()

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _open_for_today(self) -> None:
        if self._file:
            self._file.close()
        self._current_date = self._today()
        path = self._log_dir / f"mrtrader_{self._current_date}.log"
        self._file = open(path, "a", encoding="utf-8")
        self._prune_old_files()

    def _prune_old_files(self) -> None:
        files = sorted(self._log_dir.glob("mrtrader_*.log"))
        for old in files[: max(0, len(files) - self._KEEP_DAYS)]:
            try:
                old.unlink()
            except OSError:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        if self._today() != self._current_date:
            self._open_for_today()
        try:
            msg = self.format(record)
            self._file.write(msg + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        super().close()


def _setup_logging() -> None:
    """Configure root logger: console + daily dated file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler (already added by uvicorn; just apply our format)
    plain_stream_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]
    if not plain_stream_handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)
    else:
        for h in plain_stream_handlers:
            h.setFormatter(fmt)

    # Daily file handler — logs/mrtrader_YYYY-MM-DD.log, rotates at midnight
    root.addHandler(_DailyFileHandler(log_dir, fmt))


def _git_info() -> tuple[str, str]:
    """Return (short_commit_hash, branch_name). Returns ('unknown', 'unknown') on error."""
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


_setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MrTrader - Automated Trading System",
    description="AI-powered automated day trading system",
    version="0.1.0",
)

# Register routers
app.include_router(orchestrator_router)
app.include_router(dashboard_router)
app.include_router(watchlist_router)
app.include_router(config_router)
app.include_router(nis_router)
app.include_router(audit_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    import sys
    from datetime import datetime, timezone

    commit, branch = _git_info()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    py_ver = sys.version.split()[0]

    banner = (
        "\n"
        "═" * 72 + "\n"
        f"  MRTRADER STARTUP  {now_utc}\n"
        f"  git: {commit}  branch: {branch}\n"
        f"  mode: {settings.trading_mode.upper()}  capital: ${settings.initial_capital:,.0f}"
        f"  python: {py_ver}  port: {settings.port}\n"
        "═" * 72
    )
    logger.info(banner)

    # Initialize database
    try:
        init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise

    # Check database connection
    if not check_db_connection():
        raise RuntimeError("Cannot connect to database")
    logger.info("✓ Database connection verified")

    # Check Redis + Alpaca concurrently (both are sync network calls)
    try:
        redis_queue = get_redis_queue()
        redis_ok, alpaca_ok = await asyncio.gather(
            asyncio.to_thread(redis_queue.health_check),
            asyncio.to_thread(get_alpaca_client().health_check),
        )
        if redis_ok:
            logger.info("✓ Redis connection verified")
        else:
            raise RuntimeError("Redis health check failed")
        if alpaca_ok:
            logger.info("✓ Alpaca connection verified")
        else:
            logger.warning("⚠ Alpaca health check failed")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"⚠ Startup check failed: {e}")

    # Restore persisted state (kill switch + capital ramp)
    try:
        from app.live_trading.kill_switch import kill_switch
        from app.live_trading.capital_manager import capital_manager
        kill_switch.load_state()
        logger.info("State restored (kill_switch=%s, capital_stage=%s)",
                    kill_switch.is_active, capital_manager.current_stage.stage)
    except Exception as e:
        logger.warning("State restore warning: %s", e)

    # Startup reconciliation (Alpaca vs DB)
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
        logger.warning("Startup reconciliation skipped: %s", e)

    # Flush stale inter-agent queue messages before agents start consuming them.
    # Proposals in Redis survive process restarts; without this, a restarted RM will
    # re-approve proposals that PM already flagged as sent today.
    try:
        from app.integrations.redis_queue import get_redis_queue
        _rq = get_redis_queue()
        for _qname in ["trade_proposals", "risk_approved", "exit_requests", "pm_commands"]:
            _n = _rq.get_queue_length(_qname)
            if _n > 0:
                _rq.clear_queue(_qname)
                logger.warning("Startup: flushed %d stale message(s) from queue '%s'", _n, _qname)
    except Exception as _e:
        logger.warning("Startup queue flush failed (non-fatal): %s", _e)

    # Start orchestrator (registers + starts all agents)
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

        # Log active model versions from DB
        try:
            from app.database.session import get_session
            from app.database.models import ModelVersion
            _db = get_session()
            try:
                for _name in ("swing", "intraday"):
                    _row = (
                        _db.query(ModelVersion)
                        .filter_by(model_name=_name, status="ACTIVE")
                        .order_by(ModelVersion.version.desc())
                        .first()
                    )
                    if _row:
                        logger.info("Active model: %s v%d (path=%s)", _name, _row.version, _row.model_path)
                    else:
                        logger.warning("Active model: %s — NONE found in DB", _name)
            finally:
                _db.close()
        except Exception as _e:
            logger.warning("Could not log model versions: %s", _e)

        # Phase 53: start live news monitor as background task
        from app.agents.news_monitor import news_monitor
        asyncio.create_task(news_monitor.run(), name="news_monitor")
        logger.info("Orchestrator started (news monitor running)")
    except Exception as e:
        logger.error("Orchestrator startup failed: %s", e)
        raise

    # Warm all dashboard caches in background — fire and forget, don't block server readiness
    async def _warm_cache():
        from app.api.routes import get_dashboard_summary, get_health_alias, get_market_regime
        results = await asyncio.gather(
            get_dashboard_summary(),
            get_health_alias(),
            get_market_regime(),
            return_exceptions=True,
        )
        names = ["summary", "health", "regime"]
        for name, r in zip(names, results):
            if isinstance(r, Exception):
                logger.warning("Cache warm-up skipped for %s: %s", name, r)
        logger.info("✓ Dashboard cache warmed")

    asyncio.create_task(_warm_cache())

    logger.info("MrTrader application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info("═" * 72)
    logger.info("  MRTRADER SHUTDOWN  %s", now_utc)
    logger.info("═" * 72)
    from app.orchestrator import orchestrator
    await orchestrator.stop()


# Health and Status Endpoints


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns 503 when the kill switch is active or the circuit breaker is open
    so that load balancers and monitoring tools can detect the degraded state.
    """
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
    """Get system status"""
    db_ok = check_db_connection()
    redis_ok, alpaca_ok = await asyncio.gather(
        asyncio.to_thread(get_redis_queue().health_check),
        asyncio.to_thread(get_alpaca_client().health_check),
    )
    alpaca_status = "✓ connected" if alpaca_ok else "✗ error"
    status = "healthy" if all([db_ok, redis_ok]) else "degraded"

    return {
        "status": status,
        "database": "✓ connected" if db_ok else "✗ disconnected",
        "redis": "✓ connected" if redis_ok else "✗ disconnected",
        "alpaca": alpaca_status,
        "mode": settings.trading_mode,
    }


@app.get("/api/account")
async def get_account_info():
    """Get Alpaca account information"""
    if get_alpaca_client is None:
        return {
            "status": "unavailable",
            "message": "Alpaca not installed (Phase 1 only, will add in Phase 2)",
        }
    try:
        alpaca = get_alpaca_client()
        account = alpaca.get_account()
        return {
            "status": "success",
            "data": account,
        }
    except Exception as e:
        logger.error(f"Error fetching account info: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/api/positions")
async def get_positions():
    """Get all open positions"""
    if get_alpaca_client is None:
        return {
            "status": "unavailable",
            "message": "Alpaca not installed (Phase 1 only)",
            "data": [],
            "count": 0,
        }
    try:
        alpaca = get_alpaca_client()
        positions = alpaca.get_positions()
        return {
            "status": "success",
            "data": positions,
            "count": len(positions),
        }
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/api/position/{symbol}")
async def get_position(symbol: str):
    """Get position for a specific symbol"""
    if get_alpaca_client is None:
        return {
            "status": "unavailable",
            "message": "Alpaca not installed (Phase 1 only)",
        }
    try:
        alpaca = get_alpaca_client()
        position = alpaca.get_position(symbol.upper())
        if position:
            return {
                "status": "success",
                "data": position,
            }
        else:
            return {
                "status": "not_found",
                "message": f"No position found for {symbol}",
            }
    except Exception as e:
        logger.error(f"Error fetching position: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MrTrader - Automated Trading System",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "status": "/api/status",
            "account": "/api/account",
            "positions": "/api/positions",
            "docs": "/docs",
            "dashboard": "/dashboard",
            "orchestrator": "/api/orchestrator/status",
            "jobs": "/api/orchestrator/jobs",
        },
    }


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


# Mount entire frontend/dist/ at root — must be LAST so API routes above take priority.
# html=True serves index.html for unknown paths (SPA fallback).
# This serves both /assets/index-*.js and any legacy /index-*.js paths correctly.
if os.path.isdir(_REACT_DIST):
    app.mount("/", StaticFiles(directory=_REACT_DIST, html=True), name="spa")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port, reload=settings.debug)
