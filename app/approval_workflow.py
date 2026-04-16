"""
Approval workflow — validates paper-trading metrics against go-live criteria
and records the approval decision.
"""
import logging
import statistics
from datetime import datetime
from typing import Dict, Any, Tuple

from sqlalchemy import desc

from app.database.session import get_session
from app.database.models import TradingSession, Trade, RiskMetric, AuditLog

logger = logging.getLogger(__name__)

# ── Go-live thresholds ────────────────────────────────────────────────────────
GO_LIVE_CRITERIA = {
    "sharpe_ratio":    1.5,   # annualised Sharpe of daily P&L series
    "min_return_pct":  10.0,  # total return % over the paper period
    "max_drawdown_pct": 5.0,  # peak-to-trough drawdown must stay below
    "min_win_rate":    50.0,  # % of profitable trades
    "min_trades":      20,    # enough data points
    "min_days":        14,    # at least two weeks of live market data
}


class ApprovalWorkflow:
    """Check paper-trading performance and gate the paper→live transition."""

    # ── Public API ────────────────────────────────────────────────────────────

    def check_go_live_readiness(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate the most-recent active paper session against go-live criteria.

        Returns:
            (is_ready, metrics_dict)
        """
        db = get_session()
        try:
            session = (
                db.query(TradingSession)
                .filter_by(mode="paper", status="ACTIVE")
                .order_by(desc(TradingSession.started_at))
                .first()
            )

            if not session:
                # No live session — compute from all historical paper data
                metrics = self._compute_metrics_no_session(db)
            else:
                metrics = self._compute_metrics(session, db)

            passed, details = self._verify_criteria(metrics)
            metrics["criteria_details"] = details
            metrics["is_ready"] = passed
            return passed, metrics

        finally:
            db.close()

    def request_approval(self, approved_by: str = "automated") -> Dict[str, Any]:
        """
        Grant go-live approval if all criteria pass, and record it in the DB.
        """
        is_ready, metrics = self.check_go_live_readiness()

        if not is_ready:
            return {
                "status": "denied",
                "reason": "One or more go-live criteria not met",
                "metrics": metrics,
            }

        db = get_session()
        try:
            # Update the trading session record
            session = (
                db.query(TradingSession)
                .filter_by(mode="paper", status="ACTIVE")
                .order_by(desc(TradingSession.started_at))
                .first()
            )
            if session:
                session.approved_at = datetime.utcnow()
                session.approved_by = approved_by

            # Write audit log
            log = AuditLog(
                action="GO_LIVE_APPROVAL_GRANTED",
                details={
                    "approved_by": approved_by,
                    "metrics": {k: v for k, v in metrics.items()
                                if not isinstance(v, dict)},
                },
                timestamp=datetime.utcnow(),
            )
            db.add(log)
            db.commit()

            logger.warning("*** GO-LIVE APPROVAL GRANTED by %s ***", approved_by)
            return {
                "status": "approved",
                "approved_at": datetime.utcnow().isoformat(),
                "approved_by": approved_by,
                "metrics": metrics,
            }
        finally:
            db.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_metrics(self, session: TradingSession, db) -> Dict[str, Any]:
        duration_days = max((datetime.utcnow() - session.started_at).days, 1)

        trades = (
            db.query(Trade)
            .filter(Trade.created_at >= session.started_at, Trade.status == "CLOSED")
            .all()
        )

        return self._build_metrics(trades, duration_days, session.capital, db,
                                   since=session.started_at)

    def _compute_metrics_no_session(self, db) -> Dict[str, Any]:
        """Fallback when no TradingSession row exists — use all closed trades."""
        trades = db.query(Trade).filter_by(status="CLOSED").all()
        if not trades:
            return self._empty_metrics()

        first_trade = min(t.created_at for t in trades)
        duration_days = max((datetime.utcnow() - first_trade).days, 1)
        return self._build_metrics(trades, duration_days, 20_000.0, db, since=first_trade)

    def _build_metrics(self, trades, duration_days, initial_capital, db, since) -> Dict[str, Any]:
        total_trades = len(trades)
        winning = [t for t in trades if t.pnl and t.pnl > 0]
        total_pnl = sum(t.pnl for t in trades if t.pnl) or 0.0
        win_rate = (len(winning) / total_trades * 100) if total_trades > 0 else 0.0
        total_return_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0.0

        # Drawdown from daily risk_metric rows
        daily_rows = (
            db.query(RiskMetric)
            .filter(RiskMetric.date >= since.strftime("%Y-%m-%d"))
            .all()
        )
        max_drawdown_pct = max((r.max_drawdown or 0) * 100 for r in daily_rows) if daily_rows else 0.0

        # Sharpe from daily P&L
        daily_pnls = [r.daily_pnl for r in daily_rows if r.daily_pnl is not None]
        if len(daily_pnls) > 2:
            mean_r = statistics.mean(daily_pnls)
            std_r = statistics.stdev(daily_pnls)
            sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "duration_days": duration_days,
            "total_trades": total_trades,
            "winning_trades": len(winning),
            "win_rate_pct": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 3),
            "trades_per_day": round(total_trades / duration_days, 2),
            "initial_capital": initial_capital,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "duration_days": 0, "total_trades": 0, "winning_trades": 0,
            "win_rate_pct": 0.0, "total_pnl": 0.0, "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
            "trades_per_day": 0.0, "initial_capital": 20_000.0,
        }

    def _verify_criteria(self, metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, bool]]:
        c = GO_LIVE_CRITERIA
        details = {
            "sharpe_ratio":    metrics["sharpe_ratio"]    >= c["sharpe_ratio"],
            "return":          metrics["total_return_pct"] >= c["min_return_pct"],
            "drawdown":        metrics["max_drawdown_pct"] <= c["max_drawdown_pct"],
            "win_rate":        metrics["win_rate_pct"]     >= c["min_win_rate"],
            "trade_count":     metrics["total_trades"]     >= c["min_trades"],
            "duration":        metrics["duration_days"]    >= c["min_days"],
        }

        for name, passed in details.items():
            icon = "PASS" if passed else "FAIL"
            logger.info("Go-live check %-14s %s", name, icon)

        all_pass = all(details.values())
        logger.info("Go-live overall: %s", "READY" if all_pass else "NOT READY")
        return all_pass, details


# Module-level singleton
approval_workflow = ApprovalWorkflow()
