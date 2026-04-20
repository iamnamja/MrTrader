"""
Dashboard API routes — metrics, positions, trades, decisions, controls,
paper-trading approval workflow, and live-trading capital management.
"""

import asyncio
import logging
from datetime import date, datetime
from typing import List

from fastapi import APIRouter, HTTPException
from app.api.cache import ttl_cache
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


def _win_rate_and_drawdown():
    """Compute win rate and max drawdown from all closed trades in the DB."""
    db = get_session()
    try:
        closed = db.query(Trade).filter(Trade.status == "CLOSED").all()
        if not closed:
            return None, None
        pnls = [float(t.pnl) for t in closed if t.pnl is not None]
        if not pnls:
            return None, None
        wins = sum(1 for p in pnls if p > 0)
        win_rate = round(wins / len(pnls) * 100, 1)
        # Peak-to-trough drawdown on cumulative P&L curve
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            if cum > peak:
                peak = cum
            dd = (peak - cum) / max(abs(peak), 1e-6) * 100
            if dd > max_dd:
                max_dd = dd
        return win_rate, round(max_dd, 2)
    except Exception:
        return None, None
    finally:
        db.close()


def _system_status(alpaca_ok: bool = True) -> str:
    try:
        db_ok = check_db_connection()
    except Exception:
        db_ok = False
    try:
        redis_ok = _redis().health_check()
    except Exception:
        redis_ok = False
    checks = [db_ok, redis_ok, alpaca_ok]
    if all(checks):
        return "healthy"
    if any(checks):
        return "degraded"
    return "critical"


# ─── Summary ──────────────────────────────────────────────────────────────────

ALPACA_TIMEOUT = 4.0  # seconds before giving up and returning partial data


@router.get("/summary", response_model=DashboardSummaryResponse)
@ttl_cache(seconds=10)
async def get_dashboard_summary():
    """Account value, P&L, position counts, and system status at a glance."""
    try:
        # Fire Alpaca, DB queries, and stats all in parallel — don't wait for Alpaca before starting DB work
        def _fetch_alpaca():
            try:
                alpaca = _alpaca()
                acct = alpaca.get_account()
                pos = alpaca.get_positions()
                return acct, pos
            except Exception:
                return None, []

        (account, positions), (win_rate, max_dd), trades_count = await asyncio.gather(
            asyncio.wait_for(asyncio.to_thread(_fetch_alpaca), timeout=ALPACA_TIMEOUT),
            asyncio.to_thread(_win_rate_and_drawdown),
            asyncio.to_thread(_trades_today_count),
        )

        # Daily P&L = sum of unrealized P&L on open positions (what Alpaca tracks)
        daily_pnl = sum(float(p.get("unrealized_pl", 0)) for p in (positions or []))

        if account is None:
            logger.warning("Alpaca unavailable for summary — returning DB-only data")

        try:
            sys_status = await asyncio.wait_for(
                asyncio.to_thread(_system_status, account is not None), timeout=5.0
            )
        except asyncio.TimeoutError:
            sys_status = "degraded"

        account_value = float(account["portfolio_value"]) if account else None
        buying_power = float(account["buying_power"]) if account else None
        cash = float(account["cash"]) if account else None

        if account_value is not None:
            previous_value = account_value - daily_pnl
            daily_pnl_pct = (daily_pnl / previous_value * 100) if previous_value > 0 else 0.0
            initial_capital = 20_000.0
            total_pnl = round(account_value - initial_capital, 2)
            total_pnl_pct = round((total_pnl / initial_capital) * 100, 2)
        else:
            daily_pnl_pct = total_pnl = total_pnl_pct = None

        # Capital deployed = sum of market_value across open positions
        capital_deployed = sum(
            float(p.get("market_value") or (float(p.get("avg_entry_price", 0) or p.get("avg_price", 0)) * int(p.get("qty", 0) or p.get("quantity", 0))))
            for p in (positions or [])
        )
        capital_deployed_pct = round(capital_deployed / account_value * 100, 1) if account_value else None

        # Last signal from most recent TRADE_ENTERED decision
        def _last_signal():
            db = get_session()
            try:
                d = (
                    db.query(AgentDecision)
                    .filter(AgentDecision.decision_type == "TRADE_ENTERED")
                    .order_by(desc(AgentDecision.timestamp))
                    .first()
                )
                if not d:
                    return None, None
                r = d.reasoning or {}
                sig = r.get("signal_type", "TRADE")
                age_hours = round((datetime.utcnow() - d.timestamp).total_seconds() / 3600, 1)
                return sig, age_hours
            except Exception:
                return None, None
            finally:
                db.close()

        last_signal_type, last_signal_age = await asyncio.to_thread(_last_signal)

        return DashboardSummaryResponse(
            timestamp=datetime.utcnow(),
            account_value=account_value,
            buying_power=buying_power,
            cash=cash,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            open_positions_count=len(positions),
            trades_today_count=trades_count,
            capital_deployed=round(capital_deployed, 2),
            capital_deployed_pct=capital_deployed_pct,
            last_signal_type=last_signal_type,
            last_signal_age_hours=last_signal_age,
            trading_mode=settings.trading_mode,
            system_status=sys_status,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
        )
    except Exception as exc:
        logger.error("Dashboard summary error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Positions ────────────────────────────────────────────────────────────────

@router.get("/positions", response_model=List[PositionResponse])
@ttl_cache(seconds=10)
async def get_open_positions():
    """Live open positions from Alpaca, enriched with stop/target/signal from DB."""
    try:
        raw = await asyncio.to_thread(_alpaca().get_positions)

        # Enrich with stop/target/signal from active DB trades
        def _db_trade_meta():
            db = get_session()
            try:
                trades = db.query(Trade).filter(Trade.status == "ACTIVE").all()
                return {t.symbol: t for t in trades}
            except Exception:
                return {}
            finally:
                db.close()

        db_trades = await asyncio.to_thread(_db_trade_meta)

        result = []
        for pos in raw:
            qty = pos.get("quantity") or pos.get("qty", 0)
            avg = pos.get("avg_price") or pos.get("avg_entry_price", 0.0)
            current = pos.get("current_price")
            pnl = pos.get("pnl_unrealized") or pos.get("unrealized_pl")
            pnl_pct = None
            if pnl is not None and avg and float(avg) > 0:
                pnl_pct = round(float(pnl) / (float(avg) * int(qty)) * 100, 2)
            t = db_trades.get(pos["symbol"])
            result.append(PositionResponse(
                symbol=pos["symbol"],
                quantity=int(qty),
                avg_price=float(avg),
                current_price=float(current) if current else None,
                pnl_unrealized=float(pnl) if pnl is not None else None,
                pnl_unrealized_pct=pnl_pct,
                stop_price=float(t.stop_price) if t and t.stop_price else None,
                target_price=float(t.target_price) if t and t.target_price else None,
                signal_type=t.signal_type if t else None,
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
        # Build trade_id → symbol map for quick lookup
        trade_ids = [d.trade_id for d in decisions if d.trade_id is not None]
        trade_symbols: dict = {}
        if trade_ids:
            trades = db.query(Trade).filter(Trade.id.in_(trade_ids)).all()
            trade_symbols = {t.id: t.symbol for t in trades}

        def _extract_symbol(d: AgentDecision) -> str | None:
            # 1. from trade join (TRADE_ENTERED / TRADE_EXITED)
            if d.trade_id and d.trade_id in trade_symbols:
                return trade_symbols[d.trade_id]
            r = d.reasoning or {}
            # 2. direct symbol/ticker key
            for key in ("symbol", "ticker"):
                if isinstance(r.get(key), str):
                    return r[key]
            # 3. nested proposal.symbol (TRADE_APPROVED from risk manager)
            proposal = r.get("proposal")
            if isinstance(proposal, dict) and isinstance(proposal.get("symbol"), str):
                return proposal["symbol"]
            # 4. symbols list or selected list (INSTRUMENTS_SELECTED)
            syms = r.get("symbols") or r.get("selected")
            if isinstance(syms, list) and syms:
                first = syms[0]
                return first if isinstance(first, str) else (first.get("symbol") if isinstance(first, dict) else None)
            return None

        return [
            AgentDecisionResponse(
                id=d.id,
                agent_name=d.agent_name,
                decision_type=d.decision_type,
                trade_id=d.trade_id,
                reasoning=d.reasoning,
                timestamp=d.timestamp,
                symbol=_extract_symbol(d),
            )
            for d in decisions
        ]
    except Exception as exc:
        logger.error("Decisions error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


# ─── Equity history ───────────────────────────────────────────────────────────

@router.get("/metrics/equity-history")
@ttl_cache(seconds=60)
async def get_equity_history(range: str = "1d"):
    """
    Equity curve from Alpaca portfolio history API — reflects all account
    activity regardless of whether trades were entered via this app.

    range: '1d' (5-min bars today), '1w' (daily, 7 days), '1m' (daily, 30 days)
    """
    def _fetch():
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        period_map = {"1d": "1D", "1w": "1W", "1m": "1M"}
        tf_map = {"1d": "5Min", "1w": "1D", "1m": "1D"}
        period = period_map.get(range, "1D")
        timeframe = tf_map.get(range, "5Min")
        req = GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
        hist = _alpaca().trading_client.get_portfolio_history(req)

        timestamps = hist.timestamp or []
        pnl_series = hist.profit_loss or []

        points = []
        for ts, pnl in zip(timestamps, pnl_series):
            if pnl is None:
                continue
            dt = datetime.utcfromtimestamp(ts)
            label = dt.strftime("%H:%M") if range == "1d" else dt.strftime("%b %d")
            points.append({"time": label, "pnl": round(float(pnl), 2)})
        return points

    try:
        points = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=10.0)
    except Exception as exc:
        logger.warning("Portfolio history unavailable: %s", exc)
        points = [{"time": "—", "pnl": 0.0}]

    return points


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
@ttl_cache(seconds=10)
async def get_health_alias():
    """Health check alias for dashboard polling."""
    db_ok = check_db_connection()
    try:
        redis_ok, alpaca_ok = await asyncio.wait_for(
            asyncio.gather(
                asyncio.to_thread(_redis().health_check),
                asyncio.to_thread(_alpaca().health_check),
            ),
            timeout=ALPACA_TIMEOUT,
        )
    except asyncio.TimeoutError:
        redis_ok, alpaca_ok = False, False
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
@ttl_cache(seconds=60)
async def get_system_health():
    """Full system health check."""
    db_ok = check_db_connection()
    try:
        redis_ok, alpaca_ok = await asyncio.wait_for(
            asyncio.gather(
                asyncio.to_thread(_redis().health_check),
                asyncio.to_thread(_alpaca().health_check),
            ),
            timeout=ALPACA_TIMEOUT,
        )
    except asyncio.TimeoutError:
        redis_ok, alpaca_ok = False, False

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
@ttl_cache(seconds=15)
async def get_live_trading_status():
    """Real-time live account health plus current capital stage."""
    from app.live_trading.monitoring import monitor
    from app.live_trading.capital_manager import capital_manager
    from app.live_trading.kill_switch import kill_switch
    from app.trading_modes import mode_manager

    health = await asyncio.to_thread(monitor.health_check)
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
@ttl_cache(seconds=60)
async def get_market_regime():
    """Return composite VIX+macro regime detail."""
    try:
        from app.strategy.regime_detector import regime_detector

        def _fetch():
            detail = regime_detector.get_regime_detail()
            detail["trend_following_active"] = regime_detector.trend_following_active()
            detail["mean_reversion_active"] = regime_detector.mean_reversion_active()
            detail["position_size_multiplier"] = regime_detector.position_size_multiplier()
            return detail

        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.error("Regime detection error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/analytics/macro")
@ttl_cache(seconds=300)
async def get_macro_indicators():
    """Return FRED macro indicators + macro risk score."""
    try:
        from app.macro.fred_client import fred_client

        def _fetch():
            return fred_client.get_all(), fred_client.macro_risk_score()

        indicators, macro_risk_score = await asyncio.to_thread(_fetch)
        return {
            "indicators": indicators,
            "macro_risk_score": macro_risk_score,
        }
    except Exception as exc:
        logger.error("Macro indicators error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/analytics/performance-review")
async def get_performance_review(days: int = 30):
    """Return paper trading performance review with backtest drift analysis."""
    try:
        from app.analytics.performance_review import get_performance_review as _review
        return _review(days=days)
    except Exception as exc:
        logger.error("Performance review error: %s", exc)
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


@router.get("/monitor/health")
async def get_monitor_health():
    """Run a live health check via LiveTradingMonitor."""
    try:
        from app.live_trading.monitoring import monitor
        return monitor.health_check()
    except Exception as exc:
        logger.error("Monitor health check error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/monitor/summary")
async def get_monitor_summary():
    """Return the last daily session summary (or None if not yet run today)."""
    try:
        from app.live_trading.monitoring import monitor
        return {"summary": monitor.last_summary}
    except Exception as exc:
        logger.error("Monitor summary error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/monitor/run-summary")
async def run_daily_summary():
    """Manually trigger a daily session summary (for testing / on-demand review)."""
    try:
        from app.live_trading.monitoring import monitor
        summary = monitor.daily_session_summary()
        return summary
    except Exception as exc:
        logger.error("Manual summary error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/monitor/history")
async def get_monitor_history(days: int = 7):
    """Return daily session summaries from AuditLog for the last N days."""
    try:
        from app.database.session import get_session
        from app.database.models import AuditLog
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        db = get_session()
        try:
            rows = (
                db.query(AuditLog)
                .filter(
                    AuditLog.action == "DAILY_SESSION_SUMMARY",
                    AuditLog.timestamp >= cutoff,
                )
                .order_by(AuditLog.timestamp.desc())
                .all()
            )
            return {"history": [r.details for r in rows if r.details]}
        finally:
            db.close()
    except Exception as exc:
        logger.error("Monitor history error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
