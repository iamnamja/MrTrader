"""
Performance Intelligence — Phase 22.

Tracks live system health and generates the Friday weekly report.

  22.1  Signal quality monitoring — rolling 30-trade win-rate per signal type.
         If any type drops below 45%, flag for DB-config downweight (no restart).
  22.2  Benchmark comparison — daily SPY cumulative return vs strategy return,
         alpha, Sharpe. Flag if underperforming SPY by > 15% over 60 days.
  22.3  Model health monitoring — average PM score of held positions, decision
         distribution (EXIT/HOLD/EXTEND_TARGET), concept drift flag when avg
         score < 0.45.
  22.4  Weekly auto-report — extends Phase 15 analytics, fires every Friday at
         market close via the orchestrator.
"""
from __future__ import annotations

import logging
import statistics
from collections import deque
from datetime import date, datetime, timedelta
from threading import Lock
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SIGNAL_QUALITY_WINDOW = 30          # rolling trades per signal type
SIGNAL_WIN_RATE_FLOOR = 0.45        # below this → flag for downweight
SPY_UNDERPERFORM_THRESHOLD = 0.15   # 15% behind SPY over 60 days → review flag
BENCHMARK_WINDOW_DAYS = 60
MODEL_SCORE_FLOOR = 0.45            # avg held position score below this → flag
DRIFT_SCORE_THRESHOLD = 0.10        # score stddev shift considered drift


class PerformanceMonitor:
    """
    Singleton monitor that collects performance telemetry in-memory and
    surfaces it via `status()` / `generate_weekly_report()`.

    Thread-safe — all mutable state guarded by a Lock.
    """

    def __init__(self):
        self._lock = Lock()

        # 22.1 Signal quality — {signal_type: deque of (pnl, won)}
        self._signal_trades: Dict[str, Deque[Tuple[float, bool]]] = {}

        # 22.2 Benchmark — cumulative daily returns [(date, strategy_ret, spy_ret)]
        self._daily_returns: List[Tuple[date, float, float]] = []

        # 22.3 Model health — deque of (timestamp, score) per PM cycle
        self._pm_scores: Deque[float] = deque(maxlen=200)
        self._decision_counts: Dict[str, int] = {"EXIT": 0, "HOLD": 0, "EXTEND_TARGET": 0}

    # ─── 22.1  Signal Quality ─────────────────────────────────────────────────

    def record_trade_result(self, signal_type: str, pnl: float) -> None:
        """Called after every closed trade to update rolling signal quality."""
        won = pnl > 0
        with self._lock:
            if signal_type not in self._signal_trades:
                self._signal_trades[signal_type] = deque(maxlen=SIGNAL_QUALITY_WINDOW)
            self._signal_trades[signal_type].append((pnl, won))

        win_rate = self._win_rate_for(signal_type)
        if win_rate is not None and win_rate < SIGNAL_WIN_RATE_FLOOR:
            logger.warning(
                "SIGNAL QUALITY: %s win rate %.1f%% < %.0f%% floor — consider downweighting",
                signal_type, win_rate * 100, SIGNAL_WIN_RATE_FLOOR * 100,
            )

    def signal_quality(self) -> Dict[str, Any]:
        """Return rolling win rate and avg P&L per signal type."""
        with self._lock:
            snapshot = {k: list(v) for k, v in self._signal_trades.items()}

        result: Dict[str, Any] = {}
        for signal, trades in snapshot.items():
            if not trades:
                continue
            pnls = [p for p, _ in trades]
            wins = sum(1 for _, w in trades if w)
            win_rate = wins / len(trades)
            result[signal] = {
                "trades": len(trades),
                "win_rate": round(win_rate * 100, 1),
                "avg_pnl": round(statistics.mean(pnls), 2),
                "total_pnl": round(sum(pnls), 2),
                "flagged": win_rate < SIGNAL_WIN_RATE_FLOOR,
            }
        return result

    def _win_rate_for(self, signal_type: str) -> Optional[float]:
        with self._lock:
            trades = list(self._signal_trades.get(signal_type, []))
        if not trades:
            return None
        return sum(1 for _, w in trades if w) / len(trades)

    def load_signal_quality_from_db(self, db, days: int = 60) -> None:
        """Backfill rolling window from DB on startup."""
        from app.database.models import Trade
        since = datetime.utcnow() - timedelta(days=days)
        try:
            trades = (
                db.query(Trade)
                .filter(Trade.status == "CLOSED", Trade.closed_at >= since)
                .order_by(Trade.closed_at)
                .all()
            )
            for t in trades:
                sig = t.signal_type or "UNKNOWN"
                pnl = float(t.pnl) if t.pnl is not None else 0.0
                self.record_trade_result(sig, pnl)
            logger.info("Signal quality loaded: %d trades", len(trades))
        except Exception as exc:
            logger.warning("Could not load signal quality from DB: %s", exc)

    # ─── 22.2  Benchmark Comparison ───────────────────────────────────────────

    def record_daily_return(self, strategy_ret: float, spy_ret: float,
                            for_date: Optional[date] = None) -> None:
        """
        Record today's strategy and SPY daily returns (as fractions, e.g. 0.01 = +1%).
        Called once per trading day at market close.
        """
        d = for_date or date.today()
        with self._lock:
            self._daily_returns.append((d, strategy_ret, spy_ret))
            # Keep only BENCHMARK_WINDOW_DAYS of history
            cutoff = d - timedelta(days=BENCHMARK_WINDOW_DAYS)
            self._daily_returns = [(dd, sr, br) for dd, sr, br in self._daily_returns
                                   if dd >= cutoff]

    def benchmark_comparison(self) -> Dict[str, Any]:
        """
        Return cumulative strategy return, cumulative SPY return, alpha, and flag
        if strategy underperforms SPY by > 15% over the window.
        """
        with self._lock:
            returns = list(self._daily_returns)

        if not returns:
            return {"strategy_cum_pct": 0.0, "spy_cum_pct": 0.0, "alpha_pct": 0.0,
                    "underperform_flag": False, "days": 0}

        strategy_cum = 1.0
        spy_cum = 1.0
        for _, sr, br in returns:
            strategy_cum *= (1 + sr)
            spy_cum *= (1 + br)

        strategy_cum_pct = round((strategy_cum - 1) * 100, 2)
        spy_cum_pct = round((spy_cum - 1) * 100, 2)
        alpha_pct = round(strategy_cum_pct - spy_cum_pct, 2)
        underperform = alpha_pct < -(SPY_UNDERPERFORM_THRESHOLD * 100)

        if underperform:
            logger.warning(
                "BENCHMARK: strategy %.1f%% vs SPY %.1f%% (alpha %.1f%%) — automated review flag",
                strategy_cum_pct, spy_cum_pct, alpha_pct,
            )

        return {
            "strategy_cum_pct": strategy_cum_pct,
            "spy_cum_pct": spy_cum_pct,
            "alpha_pct": alpha_pct,
            "underperform_flag": underperform,
            "days": len(returns),
        }

    def fetch_and_record_spy_return(self, strategy_daily_pnl: float,
                                    account_value: float) -> None:
        """
        Pull SPY daily return from Alpaca and record alongside strategy return.
        strategy_daily_pnl: $ P&L for the day.
        account_value: account value at start of day (for % calculation).
        """
        strategy_ret = strategy_daily_pnl / account_value if account_value > 0 else 0.0
        try:
            from app.integrations import get_alpaca_client
            client = get_alpaca_client()
            bars = client.get_bars("SPY", timeframe="1Day", limit=3)
            if bars is not None and len(bars) >= 2:
                prior = float(bars["close"].iloc[-2])
                latest = float(bars["close"].iloc[-1])
                spy_ret = (latest - prior) / prior if prior > 0 else 0.0
            else:
                spy_ret = 0.0
        except Exception as exc:
            logger.debug("SPY daily return fetch failed: %s", exc)
            spy_ret = 0.0
        self.record_daily_return(strategy_ret, spy_ret)

    # ─── 22.3  Model Health ───────────────────────────────────────────────────

    def record_pm_cycle(self, scores: List[float],
                        decisions: Dict[str, int]) -> None:
        """
        Called after each PM 30-min cycle.
        scores: list of ML scores for currently held positions.
        decisions: {"EXIT": n, "HOLD": n, "EXTEND_TARGET": n}
        """
        with self._lock:
            self._pm_scores.extend(scores)
            for k, v in decisions.items():
                self._decision_counts[k] = self._decision_counts.get(k, 0) + v

        if scores:
            avg = statistics.mean(scores)
            if avg < MODEL_SCORE_FLOOR:
                logger.warning(
                    "MODEL HEALTH: avg held score %.3f < %.2f floor — model may be stale",
                    avg, MODEL_SCORE_FLOOR,
                )

    def model_health(self) -> Dict[str, Any]:
        """Return model health snapshot."""
        with self._lock:
            scores = list(self._pm_scores)
            decisions = dict(self._decision_counts)

        if not scores:
            return {"avg_score": None, "score_std": None, "drift_flag": False,
                    "decision_counts": decisions, "samples": 0}

        avg = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        drift_flag = avg < MODEL_SCORE_FLOOR

        return {
            "avg_score": round(avg, 4),
            "score_std": round(std, 4),
            "drift_flag": drift_flag,
            "decision_counts": decisions,
            "samples": len(scores),
        }

    # ─── 22.4  Weekly Report ──────────────────────────────────────────────────

    def generate_weekly_report(self, db=None) -> Dict[str, Any]:
        """
        Assemble the Friday weekly report.
        Extends Phase 15 analytics with signal quality, benchmark, model health.
        """
        from app.analytics.performance_review import get_performance_review
        from app.analytics.signal_attribution import get_signal_attribution
        from app.analytics.drawdown_analyzer import get_drawdown_summary

        report: Dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat(),
            "period_days": 7,
        }

        # Core performance (Phase 15)
        try:
            report["performance"] = get_performance_review(days=7)
        except Exception as exc:
            logger.warning("Weekly report: performance_review failed: %s", exc)
            report["performance"] = {}

        # Signal attribution (Phase 15)
        try:
            report["signal_attribution"] = get_signal_attribution(days=7)
        except Exception as exc:
            logger.warning("Weekly report: signal_attribution failed: %s", exc)
            report["signal_attribution"] = {}

        # Drawdown (Phase 15)
        try:
            report["drawdown"] = get_drawdown_summary(days=7)
        except Exception as exc:
            logger.warning("Weekly report: drawdown failed: %s", exc)
            report["drawdown"] = {}

        # Phase 22 additions
        report["signal_quality"] = self.signal_quality()
        report["benchmark"] = self.benchmark_comparison()
        report["model_health"] = self.model_health()

        logger.info("Weekly report generated: %s", {k: "..." for k in report})
        return report

    # ─── Status ───────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return a summary dict for dashboard/API."""
        return {
            "signal_quality": self.signal_quality(),
            "benchmark": self.benchmark_comparison(),
            "model_health": self.model_health(),
        }


# Module-level singleton
performance_monitor = PerformanceMonitor()
