"""
Pre-flight checklist for switching to live trading.

Checks every critical safety criterion and returns a structured report.
All checks are read-only — nothing is changed.

Usage:
    from app.live_trading.readiness import ReadinessChecker
    report = ReadinessChecker().run()
    if report["ready"]:
        # safe to enable live trading
"""
import logging
from datetime import date, datetime
from typing import Any, Dict, List

from app.config import settings
from app.database.session import get_session
from app.database.models import Trade, RiskMetric
from app.integrations import get_alpaca_client, get_redis_queue
from app.database import check_db_connection

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_PAPER_TRADE_DAYS = 14
MIN_PAPER_TRADES = 20
MIN_WIN_RATE = 0.45
MAX_DRAWDOWN_PCT = 3.0
MIN_ACCOUNT_EQUITY = 1000.0


class CheckResult:
    def __init__(self, name: str, passed: bool, value: Any, detail: str):
        self.name = name
        self.passed = passed
        self.value = value
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.name,
            "passed": self.passed,
            "value": self.value,
            "detail": self.detail,
        }


class ReadinessChecker:
    """Run all pre-flight checks and return a summary report."""

    def run(self) -> Dict[str, Any]:
        checks: List[CheckResult] = [
            self._check_trading_mode(),
            self._check_kill_switch(),
            self._check_db_connection(),
            self._check_redis_connection(),
            self._check_alpaca_connection(),
            self._check_alpaca_equity(),
            self._check_paper_trade_days(),
            self._check_paper_trade_count(),
            self._check_win_rate(),
            self._check_drawdown(),
            self._check_smtp_configured(),
            self._check_slack_configured(),
            self._check_ml_model_exists(),
        ]

        passed = [c for c in checks if c.passed]
        failed = [c for c in checks if not c.passed]
        warnings = [c for c in failed if c.name in ("smtp_configured", "slack_configured")]
        blockers = [c for c in failed if c not in warnings]

        return {
            "ready": len(blockers) == 0,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": f"{len(passed)}/{len(checks)} checks passed",
            "blockers": [c.to_dict() for c in blockers],
            "warnings": [c.to_dict() for c in warnings],
            "passed": [c.to_dict() for c in passed],
            "all_checks": [c.to_dict() for c in checks],
        }

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_trading_mode(self) -> CheckResult:
        mode = settings.trading_mode
        passed = mode == "paper"
        return CheckResult(
            "trading_mode_is_paper", passed, mode,
            "Currently in paper mode — safe to switch to live" if passed
            else f"Mode is already '{mode}' — do not run this check against live",
        )

    def _check_kill_switch(self) -> CheckResult:
        try:
            from app.live_trading.kill_switch import kill_switch
            active = kill_switch.is_active()
            return CheckResult(
                "kill_switch_inactive", not active, active,
                "Kill switch is inactive — trading can proceed" if not active
                else "KILL SWITCH IS ACTIVE — deactivate before going live",
            )
        except Exception as e:
            return CheckResult("kill_switch_inactive", False, None, f"Could not check kill switch: {e}")

    def _check_db_connection(self) -> CheckResult:
        try:
            ok = check_db_connection()
            return CheckResult("db_connected", ok, ok, "Database reachable" if ok else "Cannot reach database")
        except Exception as e:
            return CheckResult("db_connected", False, None, f"DB check failed: {e}")

    def _check_redis_connection(self) -> CheckResult:
        try:
            rq = get_redis_queue()
            rq.ping()
            return CheckResult("redis_connected", True, True, "Redis reachable")
        except Exception as e:
            return CheckResult("redis_connected", False, None, f"Redis check failed: {e}")

    def _check_alpaca_connection(self) -> CheckResult:
        try:
            client = get_alpaca_client()
            client.get_account()
            return CheckResult("alpaca_connected", True, True, "Alpaca API reachable")
        except Exception as e:
            return CheckResult("alpaca_connected", False, None, f"Alpaca check failed: {e}")

    def _check_alpaca_equity(self) -> CheckResult:
        try:
            client = get_alpaca_client()
            account = client.get_account()
            equity = float(account.get("portfolio_value", 0) or account.get("equity", 0))
            passed = equity >= MIN_ACCOUNT_EQUITY
            return CheckResult(
                "alpaca_equity_sufficient", passed, round(equity, 2),
                f"Account equity ${equity:,.2f} ≥ ${MIN_ACCOUNT_EQUITY:,.0f}" if passed
                else f"Account equity ${equity:,.2f} is below minimum ${MIN_ACCOUNT_EQUITY:,.0f}",
            )
        except Exception as e:
            return CheckResult("alpaca_equity_sufficient", False, None, f"Could not fetch account: {e}")

    def _check_paper_trade_days(self) -> CheckResult:
        try:
            db = get_session()
            try:
                oldest = db.query(Trade).order_by(Trade.created_at.asc()).first()
                if not oldest or not oldest.created_at:
                    return CheckResult("paper_trade_days", False, 0,
                                       f"No trades found — need at least {MIN_PAPER_TRADE_DAYS} days")
                days = (datetime.utcnow() - oldest.created_at).days
                passed = days >= MIN_PAPER_TRADE_DAYS
                return CheckResult(
                    "paper_trade_days", passed, days,
                    f"{days} days of paper trading (need ≥ {MIN_PAPER_TRADE_DAYS})",
                )
            finally:
                db.close()
        except Exception as e:
            return CheckResult("paper_trade_days", False, None, f"DB query failed: {e}")

    def _check_paper_trade_count(self) -> CheckResult:
        try:
            db = get_session()
            try:
                count = db.query(Trade).filter(Trade.status == "CLOSED").count()
                passed = count >= MIN_PAPER_TRADES
                return CheckResult(
                    "paper_trade_count", passed, count,
                    f"{count} closed trades (need ≥ {MIN_PAPER_TRADES})",
                )
            finally:
                db.close()
        except Exception as e:
            return CheckResult("paper_trade_count", False, None, f"DB query failed: {e}")

    def _check_win_rate(self) -> CheckResult:
        try:
            db = get_session()
            try:
                closed = db.query(Trade).filter(Trade.status == "CLOSED").all()
                if not closed:
                    return CheckResult("win_rate", False, None, "No closed trades to evaluate")
                wins = sum(1 for t in closed if (t.pnl or 0) > 0)
                wr = wins / len(closed)
                passed = wr >= MIN_WIN_RATE
                return CheckResult(
                    "win_rate", passed, round(wr * 100, 1),
                    f"Win rate {wr*100:.1f}% ({'≥' if passed else '<'} {MIN_WIN_RATE*100:.0f}%)",
                )
            finally:
                db.close()
        except Exception as e:
            return CheckResult("win_rate", False, None, f"DB query failed: {e}")

    def _check_drawdown(self) -> CheckResult:
        try:
            db = get_session()
            try:
                today = date.today().isoformat()
                metric = db.query(RiskMetric).filter_by(date=today).first()
                dd_pct = round((metric.max_drawdown or 0) * 100, 2) if metric else 0.0
                passed = dd_pct <= MAX_DRAWDOWN_PCT
                return CheckResult(
                    "drawdown_within_limit", passed, dd_pct,
                    f"Max drawdown {dd_pct:.2f}% ({'≤' if passed else '>'} {MAX_DRAWDOWN_PCT}%)",
                )
            finally:
                db.close()
        except Exception as e:
            return CheckResult("drawdown_within_limit", False, None, f"DB query failed: {e}")

    def _check_smtp_configured(self) -> CheckResult:
        ok = bool(getattr(settings, "smtp_host", None) and getattr(settings, "alert_email", None))
        return CheckResult(
            "smtp_configured", ok, ok,
            "Email alerts configured" if ok else "SMTP not configured — no email alerts (warning only)",
        )

    def _check_slack_configured(self) -> CheckResult:
        ok = bool(settings.slack_webhook_url)
        return CheckResult(
            "slack_configured", ok, ok,
            "Slack alerts configured" if ok else "Slack not configured — no Slack alerts (warning only)",
        )

    def _check_ml_model_exists(self) -> CheckResult:
        try:
            from app.agents.portfolio_manager import PortfolioManagerAgent
            agent = PortfolioManagerAgent()
            has_model = hasattr(agent, "model") and agent.model is not None
            return CheckResult(
                "ml_model_trained", has_model, has_model,
                "ML model is loaded" if has_model else "ML model not yet trained — run retraining first",
            )
        except Exception as e:
            return CheckResult("ml_model_trained", False, None, f"Could not load portfolio manager: {e}")
