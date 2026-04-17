"""
Live trading health monitor.

Performs comprehensive health checks against the Alpaca account and the
local database, determines an overall status level, and fires alerts when
thresholds are breached.
"""
import logging
from datetime import datetime, date
from typing import Dict, Any, Optional

from app.database.session import get_session
from app.database.models import Trade, RiskMetric, AuditLog

logger = logging.getLogger(__name__)

# ── Alert throttle: minimum seconds between identical alert types ─────────────
_ALERT_COOLDOWN = 300  # 5 minutes


class LiveTradingMonitor:
    """Comprehensive health checks and threshold alerts for live trading."""

    def __init__(self):
        self._last_alert: Dict[str, datetime] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """
        Gather live account stats and today's DB metrics.
        Returns a health dict with a top-level "status" key.
        """
        alpaca = self._alpaca()
        try:
            account = alpaca.get_account()
            positions = alpaca.get_positions()
            alpaca_ok = True
        except Exception as exc:
            logger.error("Alpaca health_check failed: %s", exc)
            account = {}
            positions = []
            alpaca_ok = False

        db = get_session()
        try:
            today_str = date.today().isoformat()
            metric: Optional[RiskMetric] = db.query(RiskMetric).filter_by(date=today_str).first()

            today_start = datetime.combine(date.today(), datetime.min.time())
            trades_today = (
                db.query(Trade)
                .filter(Trade.created_at >= today_start, Trade.status == "CLOSED")
                .all()
            )

            portfolio_value = float(account.get("portfolio_value", 0)) or 1.0
            pnl_today = sum(t.pnl for t in trades_today if t.pnl) or 0.0
            pnl_today_pct = (pnl_today / portfolio_value) * 100

            max_drawdown = (metric.max_drawdown or 0.0) if metric else 0.0

            health = {
                "timestamp":       datetime.utcnow().isoformat(),
                "alpaca_connected": alpaca_ok,
                "account_value":   portfolio_value,
                "buying_power":    float(account.get("buying_power", 0)),
                "cash":            float(account.get("cash", 0)),
                "open_positions":  len(positions),
                "trades_today":    len(trades_today),
                "pnl_today":       round(pnl_today, 2),
                "pnl_today_pct":   round(pnl_today_pct, 2),
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "status":          self._determine_status(pnl_today_pct, max_drawdown),
            }
        finally:
            db.close()

        self._check_thresholds(health)
        return health

    # ── Threshold checks & alerts ─────────────────────────────────────────────

    def _check_thresholds(self, health: Dict[str, Any]):
        dd = health["max_drawdown_pct"]
        daily_loss = health["pnl_today_pct"]

        if dd > 5.0:
            self._alert("CRITICAL", "Max Drawdown Exceeded",
                        f"Drawdown: {dd:.2f}% (limit: 5%)")
        elif dd > 3.0:
            self._alert("WARNING", "Drawdown Warning",
                        f"Drawdown: {dd:.2f}% (limit: 5%)")

        if daily_loss < -2.0:
            self._alert("WARNING", "Daily Loss Limit Approached",
                        f"Daily P&L: {daily_loss:.2f}% (limit: -2%)")

        if daily_loss > 5.0:
            self._alert("INFO", "Strong Daily Performance",
                        f"Daily P&L: +{daily_loss:.2f}%")

    def _alert(self, severity: str, title: str, message: str):
        key = f"{severity}:{title}"
        now = datetime.utcnow()
        last = self._last_alert.get(key)

        if last and (now - last).total_seconds() < _ALERT_COOLDOWN:
            return  # throttled

        self._last_alert[key] = now
        log_fn = logger.critical if severity == "CRITICAL" else (
            logger.warning if severity == "WARNING" else logger.info
        )
        log_fn("[%s] %s — %s", severity, title, message)

        # Persist to audit log
        db = get_session()
        try:
            db.add(AuditLog(
                action="ALERT_SENT",
                details={"severity": severity, "title": title, "message": message},
                timestamp=now,
            ))
            db.commit()
        except Exception:
            pass
        finally:
            db.close()

        # Slack if configured
        self._slack(severity, title, message)

    @staticmethod
    def _slack(severity: str, title: str, message: str):
        from app.config import settings
        if not settings.slack_webhook_url:
            return
        try:
            import json
            import urllib.request
            emoji = {"CRITICAL": ":red_circle:", "WARNING": ":warning:", "INFO": ":white_check_mark:"}.get(severity, "")
            payload = json.dumps({"text": f"{emoji} *{title}*\n{message}"}).encode()
            req = urllib.request.Request(
                settings.slack_webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.debug("Slack alert failed: %s", exc)

    # ── Status determination ──────────────────────────────────────────────────

    @staticmethod
    def _determine_status(daily_pnl_pct: float, max_drawdown: float) -> str:
        if max_drawdown > 0.05 or daily_pnl_pct < -2.0:
            return "critical"
        if max_drawdown > 0.03 or daily_pnl_pct < -1.0:
            return "warning"
        return "healthy"

    @staticmethod
    def _alpaca():
        from app.integrations import get_alpaca_client
        return get_alpaca_client()


# Module-level singleton
monitor = LiveTradingMonitor()
