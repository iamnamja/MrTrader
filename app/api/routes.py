"""
Dashboard API routes — metrics, positions, trades, decisions, controls,
paper-trading approval workflow, and live-trading capital management.
"""

import logging
from datetime import date, datetime
from typing import List

from fastapi import APIRouter, HTTPException
from sqlalchemy import desc

from app.api.schemas import (
    AgentDecisionResponse,
    DailyMetricResponse,
    DashboardSummaryResponse,
    PositionResponse,
    SystemHealthResponse,
    TradeResponse,
)
from app.config import settings
from app.database import check_db_connection
from app.database.models import AgentDecision, AuditLog, RiskMetric, Trade
from app.database.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# ─── Lazy integrations ────────────────────────────────────────────────────────

def _alpaca():
    from app.integrations import get_alpaca_client
    return get_alpaca_client()


def _redis():
    from app.integrations import get_redis_queue
    return get_redis_queue()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _trades_today_count() -> int:
    db = get_session()
    try:
        today = date.today().isoformat()
        return (
            db.query(Trade)
            .filter(Trade.created_at >= date.fromisoformat(today))
            .count()
        )
    finally:
        db.close()


def _system_status() -> str:
    checks = [
        check_db_connection(),
        _redis().health_check(),
        _alpaca().health_check(),
    ]
    if all(checks):
        return "healthy"
    if any(checks):
        return "degraded"
    return "critical"


# ─── Summary ──────────────────────────────────────────────────────────────────

@router.get("/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary():
    """Account value, P&L, position counts, and system status at a glance."""
    try:
        alpaca = _alpaca()
        account = alpaca.get_account()

        db = get_session()
        try:
            today = date.today().isoformat()
            metric = db.query(RiskMetric).filter_by(date=today).first()
            daily_pnl = float(metric.daily_pnl) if metric and metric.daily_pnl else 0.0
        finally:
            db.close()

        account_value = float(account["portfolio_value"])
        previous_value = account_value - daily_pnl
        daily_pnl_pct = (daily_pnl / previous_value * 100) if previous_value > 0 else 0.0

        initial_capital = 20_000.0
        total_pnl = account_value - initial_capital
        total_pnl_pct = (total_pnl / initial_capital) * 100

        return DashboardSummaryResponse(
            timestamp=datetime.utcnow(),
            account_value=account_value,
            buying_power=float(account["buying_power"]),
            cash=float(account["cash"]),
            daily_pnl=daily_pnl,
            daily_pnl_pct=round(daily_pnl_pct, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            open_positions_count=len(alpaca.get_positions()),
            trades_today_count=_trades_today_count(),
            trading_mode=settings.trading_mode,
            system_status=_system_status(),
        )
    except Exception as exc:
        logger.error("Dashboard summary error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Positions ────────────────────────────────────────────────────────────────

@router.get("/positions", response_model=List[PositionResponse])
async def get_open_positions():
    """Live open positions from Alpaca."""
    try:
        raw = _alpaca().get_positions()
        result = []
        for pos in raw:
            qty = pos.get("quantity") or pos.get("qty", 0)
            avg = pos.get("avg_price") or pos.get("avg_entry_price", 0.0)
            current = pos.get("current_price")
            pnl = pos.get("pnl_unrealized") or pos.get("unrealized_pl")
            pnl_pct = None
            if pnl is not None and avg and float(avg) > 0:
                pnl_pct = round(float(pnl) / (float(avg) * int(qty)) * 100, 2)
            result.append(PositionResponse(
                symbol=pos["symbol"],
                quantity=int(qty),
                avg_price=float(avg),
                current_price=float(current) if current else None,
                pnl_unrealized=float(pnl) if pnl is not None else None,
                pnl_unrealized_pct=pnl_pct,
            ))
        return result
    except Exception as exc:
        logger.error("Positions error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Trade history ────────────────────────────────────────────────────────────

@router.get("/trades", response_model=List[TradeResponse])
async def get_trade_history(limit: int = 100, status: str = ""):
    """Recent trades from the database. Filter by status=ACTIVE|CLOSED|PENDING."""
    db = get_session()
    try:
        q = db.query(Trade).order_by(desc(Trade.created_at))
        if status:
            q = q.filter(Trade.status == status.upper())
        trades = q.limit(min(limit, 500)).all()
        return [
            TradeResponse(
                id=t.id,
                symbol=t.symbol,
                direction=t.direction,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                quantity=t.quantity,
                pnl=t.pnl,
                status=t.status,
                signal_type=getattr(t, 'signal_type', None),
                stop_price=getattr(t, 'stop_price', None),
                target_price=getattr(t, 'target_price', None),
                created_at=t.created_at,
                closed_at=t.closed_at,
            )
            for t in trades
        ]
    except Exception as exc:
        logger.error("Trade history error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


# ─── Agent decisions ──────────────────────────────────────────────────────────

@router.get("/decisions", response_model=List[AgentDecisionResponse])
async def get_agent_decisions(limit: int = 50):
    """Agent decision audit trail."""
    db = get_session()
    try:
        decisions = (
            db.query(AgentDecision)
            .order_by(desc(AgentDecision.timestamp))
            .limit(min(limit, 200))
            .all()
        )
        return [
            AgentDecisionResponse(
                id=d.id,
                agent_name=d.agent_name,
                decision_type=d.decision_type,
                trade_id=d.trade_id,
                reasoning=d.reasoning,
                timestamp=d.timestamp,
            )
            for d in decisions
        ]
    except Exception as exc:
        logger.error("Decisions error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


# ─── Daily metrics ────────────────────────────────────────────────────────────

@router.get("/metrics/daily", response_model=List[DailyMetricResponse])
async def get_daily_metrics(days: int = 30):
    """Daily P&L and drawdown for charting."""
    db = get_session()
    try:
        metrics = (
            db.query(RiskMetric)
            .order_by(RiskMetric.date)
            .limit(min(days, 365))
            .all()
        )
        return [
            DailyMetricResponse(
                date=m.date,
                daily_pnl=m.daily_pnl,
                max_drawdown=m.max_drawdown,
            )
            for m in metrics
        ]
    except Exception as exc:
        logger.error("Daily metrics error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


# ─── System health ────────────────────────────────────────────────────────────

@router.get("/health")
async def get_health_alias():
    """Health check alias for dashboard polling."""
    db_ok = check_db_connection()
    redis_ok = False
    alpaca_ok = False
    try:
        redis_ok = _redis().health_check()
    except Exception:
        pass
    try:
        alpaca_ok = _alpaca().health_check()
    except Exception:
        pass
    from app.trading_modes import mode_manager
    from app.live_trading.kill_switch import kill_switch
    status = "healthy" if all([db_ok, redis_ok]) else "degraded"
    if kill_switch.is_active:
        status = "halted"
    return {
        "status": status,
        "checks": {"database": db_ok, "redis": redis_ok, "alpaca": alpaca_ok},
        "trading_mode": mode_manager.mode.value,
        "kill_switch_active": kill_switch.is_active,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/system-health", response_model=SystemHealthResponse)
async def get_system_health():
    """Full system health check."""
    db_ok = check_db_connection()
    redis_ok = _redis().health_check()
    alpaca_ok = False
    try:
        alpaca_ok = _alpaca().health_check()
    except Exception:
        pass

    return SystemHealthResponse(
        database="OK" if db_ok else "FAIL",
        redis="OK" if redis_ok else "FAIL",
        alpaca="OK" if alpaca_ok else "FAIL",
        overall="HEALTHY" if all([db_ok, redis_ok, alpaca_ok]) else "DEGRADED",
        timestamp=datetime.utcnow().isoformat(),
    )


# ─── Controls ─────────────────────────────────────────────────────────────────

@router.post("/control/pause")
async def pause_trading():
    """Pause all trading activity."""
    from app.orchestrator import orchestrator
    orchestrator.pause_trading()
    return {"status": "trading_paused"}


@router.post("/control/resume")
async def resume_trading():
    """Resume trading after a pause."""
    from app.orchestrator import orchestrator
    orchestrator.resume_trading()
    return {"status": "trading_resumed"}


@router.post("/control/kill-switch")
async def emergency_close_all():
    """Emergency: close every open position immediately."""
    try:
        alpaca = _alpaca()
        positions = alpaca.get_positions()
        closed = []
        errors = []
        for pos in positions:
            sym = pos["symbol"]
            qty = int(pos.get("quantity") or pos.get("qty", 0))
            try:
                alpaca.place_market_order(sym, qty, "sell")
                closed.append(sym)
            except Exception as exc:
                errors.append({"symbol": sym, "error": str(exc)})
        return {"status": "kill_switch_executed", "closed": closed, "errors": errors}
    except Exception as exc:
        logger.error("Kill switch error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/control/close-position/{symbol}")
async def close_position(symbol: str):
    """Close a single position by symbol."""
    try:
        alpaca = _alpaca()
        position = alpaca.get_position(symbol.upper())
        if not position:
            raise HTTPException(status_code=404, detail=f"No open position for {symbol}")
        qty = int(position.get("quantity") or position.get("qty", 0))
        alpaca.place_market_order(symbol.upper(), qty, "sell")
        return {"status": "closed", "symbol": symbol.upper()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Close position error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Paper-trading approval workflow (Phase 8) ────────────────────────────────

@router.get("/approval/status")
async def get_approval_status():
    """Check whether the paper-trading session meets go-live criteria."""
    from app.approval_workflow import approval_workflow
    is_ready, metrics = approval_workflow.check_go_live_readiness()
    return {"is_ready": is_ready, "metrics": metrics}


@router.post("/approval/request-live")
async def request_live_approval(approved_by: str = "user"):
    """Evaluate metrics and, if criteria pass, grant go-live approval."""
    from app.approval_workflow import approval_workflow
    result = approval_workflow.request_approval(approved_by=approved_by)
    if result["status"] == "denied":
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/approval/go-live")
async def approve_and_go_live():
    """
    Full go-live switch:
      1. Re-verify all go-live criteria.
      2. Start the capital ramp at Stage 1.
      3. Flip trading mode to LIVE.
    """
    from app.approval_workflow import approval_workflow
    from app.trading_modes import mode_manager
    from app.live_trading.capital_manager import capital_manager

    is_ready, metrics = approval_workflow.check_go_live_readiness()
    if not is_ready:
        raise HTTPException(
            status_code=400,
            detail={"message": "Go-live criteria not met", "metrics": metrics},
        )

    capital_manager.start()
    mode_manager.switch_to_live()

    db = get_session()
    try:
        db.add(AuditLog(
            action="GO_LIVE_ACTIVATED",
            details={
                "activated_at": datetime.utcnow().isoformat(),
                "initial_capital": capital_manager.get_current_capital(),
                "metrics_snapshot": {k: v for k, v in metrics.items()
                                     if not isinstance(v, dict)},
            },
            timestamp=datetime.utcnow(),
        ))
        db.commit()
    finally:
        db.close()

    return {
        "status": "live_trading_enabled",
        "trading_mode": mode_manager.mode.value,
        "initial_capital": capital_manager.get_current_capital(),
        "activated_at": datetime.utcnow().isoformat(),
    }


# ─── Live trading monitoring & capital management (Phase 9) ───────────────────

@router.get("/live/status")
async def get_live_trading_status():
    """Real-time live account health plus current capital stage."""
    from app.live_trading.monitoring import monitor
    from app.live_trading.capital_manager import capital_manager
    from app.live_trading.kill_switch import kill_switch
    from app.trading_modes import mode_manager

    health = monitor.health_check()
    return {
        **health,
        "trading_mode":    mode_manager.mode.value,
        "capital":         capital_manager.get_current_capital(),
        "capital_stage":   capital_manager.current_stage.stage,
        "kill_switch_active": kill_switch.is_active,
    }


@router.get("/live/capital-stages")
async def get_capital_stages():
    """All capital ramp stages and which is currently active."""
    from app.live_trading.capital_manager import capital_manager
    return capital_manager.get_all_stages()


@router.post("/live/increase-capital")
async def request_capital_increase():
    """
    Attempt to advance to the next capital stage.
    Requires stage duration elapsed AND health thresholds met.
    """
    from app.live_trading.capital_manager import capital_manager
    from app.live_trading.monitoring import monitor

    health = monitor.health_check()
    if not capital_manager.can_advance(
        health["max_drawdown_pct"], abs(min(health["pnl_today_pct"], 0))
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Cannot advance: stage not complete or health thresholds breached",
                "health": health,
            },
        )

    result = capital_manager.advance()

    db = get_session()
    try:
        db.add(AuditLog(
            action="CAPITAL_STAGE_ADVANCED",
            details=result,
            timestamp=datetime.utcnow(),
        ))
        db.commit()
    finally:
        db.close()

    return result


@router.post("/live/kill-switch")
async def activate_kill_switch(reason: str = "User triggered"):
    """Emergency: close all positions and halt new trades."""
    from app.live_trading.kill_switch import kill_switch
    return kill_switch.activate(reason=reason)


@router.post("/live/kill-switch/reset")
async def reset_kill_switch(reason: str = "Manual reset"):
    """Re-enable trading after reviewing the kill-switch event."""
    from app.live_trading.kill_switch import kill_switch
    kill_switch.reset(reason=reason)
    return {"status": "kill_switch_reset", "reason": reason}


@router.get("/live/audit-log")
async def get_live_audit_log(limit: int = 100):
    """Recent kill-switch, capital, and alert audit entries."""
    db = get_session()
    try:
        logs = (
            db.query(AuditLog)
            .filter(AuditLog.action.in_([
                "GO_LIVE_ACTIVATED",
                "GO_LIVE_APPROVAL_GRANTED",
                "CAPITAL_STAGE_ADVANCED",
                "KILL_SWITCH_ACTIVATED",
                "KILL_SWITCH_RESET",
                "ALERT_SENT",
            ]))
            .order_by(desc(AuditLog.timestamp))
            .limit(min(limit, 500))
            .all()
        )
        return [
            {
                "action":    log.action,
                "details":   log.details,
                "timestamp": log.timestamp.isoformat(),
            }
            for log in logs
        ]
    finally:
        db.close()


@router.get("/live/readiness")
async def live_readiness_check():
    """
    Pre-flight checklist for switching to live trading.
    Returns a structured pass/fail report for all safety criteria.
    """
    from app.live_trading.readiness import ReadinessChecker
    return ReadinessChecker().run()


# ─── Analytics ────────────────────────────────────────────────────────────────

@router.get("/analytics/signal-attribution")
async def get_signal_attribution(days: int = 90):
    """Performance breakdown by signal type (EMA_CROSSOVER vs RSI_DIP)."""
    try:
        from app.analytics.signal_attribution import get_signal_attribution as _sa
        return {"days": days, "attribution": _sa(days=days)}
    except Exception as exc:
        logger.error("Signal attribution error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/analytics/drawdown")
async def get_drawdown_analytics(days: int = 90):
    """Drawdown analytics: worst sequences, per-symbol breakdown."""
    try:
        from app.analytics.drawdown_analyzer import get_drawdown_summary
        return get_drawdown_summary(days=days)
    except Exception as exc:
        logger.error("Drawdown analytics error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/analytics/earnings-blackout/{symbol}")
async def check_earnings_blackout(symbol: str):
    """Check if a symbol is in an earnings blackout window."""
    try:
        from app.strategy.earnings_filter import earnings_filter
        blackout = earnings_filter.is_blackout(symbol.upper())
        next_date = earnings_filter.next_earnings_date(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "blackout_active": blackout,
            "next_earnings_date": next_date.isoformat() if next_date else None,
        }
    except Exception as exc:
        logger.error("Earnings blackout check error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/analytics/regime")
async def get_market_regime():
    """Return current VIX-based market regime."""
    try:
        from app.strategy.regime_detector import regime_detector
        regime = regime_detector.get_regime()
        vix = regime_detector.get_vix()
        return {
            "regime": regime,
            "vix": round(vix, 2) if vix is not None else None,
            "trend_following_active": regime_detector.trend_following_active(),
            "mean_reversion_active": regime_detector.mean_reversion_active(),
            "position_size_multiplier": regime_detector.position_size_multiplier(),
        }
    except Exception as exc:
        logger.error("Regime detection error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/analytics/portfolio-heat")
async def get_portfolio_heat():
    """Return current portfolio heat (total risk as % of account value)."""
    try:
        from app.integrations import get_alpaca_client
        from app.strategy.portfolio_heat import (
            MAX_PORTFOLIO_HEAT_PCT,
            get_portfolio_heat as _get_heat,
        )
        client = get_alpaca_client()
        account = client.get_account()
        positions = client.get_positions()
        account_value = float(account.get("portfolio_value", 0))
        heat = _get_heat(positions, account_value)
        return {
            "portfolio_heat_pct": round(heat * 100, 2),
            "max_heat_pct": round(MAX_PORTFOLIO_HEAT_PCT * 100, 2),
            "headroom_pct": round((MAX_PORTFOLIO_HEAT_PCT - heat) * 100, 2),
            "positions": len(positions),
        }
    except Exception as exc:
        logger.error("Portfolio heat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
