import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db, check_db_connection
from app.integrations import get_alpaca_client, get_redis_queue
from fastapi.responses import HTMLResponse
from app.api.orchestrator_routes import router as orchestrator_router
from app.api.routes import router as dashboard_router
from app.api.websocket import websocket_endpoint

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    logger.info("Starting MrTrader application...")

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

    # Check Redis connection
    try:
        redis_queue = get_redis_queue()
        if redis_queue.health_check():
            logger.info("✓ Redis connection verified")
        else:
            raise RuntimeError("Redis health check failed")
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        raise

    # Check Alpaca connection (optional for Phase 1)
    try:
        if get_alpaca_client is not None:
            alpaca = get_alpaca_client()
            if alpaca.health_check():
                logger.info("✓ Alpaca connection verified")
            else:
                logger.warning("⚠ Alpaca health check failed (will be fixed in Phase 2)")
        else:
            logger.warning("⚠ Alpaca not installed yet (Phase 1 only, will add in Phase 2)")
    except Exception as e:
        logger.warning(f"⚠ Alpaca connection failed: {e} (Phase 1 only, will fix in Phase 2)")

    # Restore persisted state (kill switch + capital ramp)
    try:
        from app.live_trading.kill_switch import kill_switch
        from app.live_trading.capital_manager import capital_manager
        kill_switch.load_state()
        capital_manager.load_state()
        logger.info("State restored (kill_switch=%s, capital_stage=%s)",
                    kill_switch.is_active, capital_manager.current_stage.stage)
    except Exception as e:
        logger.warning("State restore warning: %s", e)

    # Startup reconciliation (Alpaca vs DB)
    try:
        from app.startup_reconciler import reconcile
        from app.database.session import get_session
        alpaca = get_alpaca_client()
        if alpaca.health_check():
            db = get_session()
            try:
                reconcile(alpaca, db)
            finally:
                db.close()
    except Exception as e:
        logger.warning("Startup reconciliation skipped: %s", e)

    # Start orchestrator (registers + starts all agents)
    try:
        from app.agents.portfolio_manager import portfolio_manager
        from app.agents.risk_manager import risk_manager
        from app.agents.trader import trader
        from app.orchestrator import orchestrator

        orchestrator.register_agent("portfolio_manager", portfolio_manager)
        orchestrator.register_agent("risk_manager", risk_manager)
        orchestrator.register_agent("trader", trader)
        await orchestrator.start()
        logger.info("Orchestrator started")
    except Exception as e:
        logger.error("Orchestrator startup failed: %s", e)
        raise

    logger.info("MrTrader application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down MrTrader application...")
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
    redis_ok = get_redis_queue().health_check()

    alpaca_ok = False
    alpaca_status = "not installed"
    if get_alpaca_client is not None:
        try:
            alpaca_ok = get_alpaca_client().health_check()
            alpaca_status = "✓ connected"
        except Exception as e:
            alpaca_status = f"✗ error: {str(e)[:50]}"
    else:
        alpaca_status = "⚠ not installed (Phase 1 only)"

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
async def websocket_route(websocket):
    await websocket_endpoint(websocket)


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    with open("frontend/dashboard.html") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port, reload=settings.debug)
