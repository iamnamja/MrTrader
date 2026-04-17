"""
Paper trading validation for Phase 42.

Computes performance statistics from closed paper trades and compares
them against the backtest to detect model drift before going live.

Key metrics:
  win_rate          — fraction of closed trades with PnL > 0
  avg_return_pct    — mean trade return (pnl / entry_price * 100)
  sharpe_ratio      — annualised Sharpe of daily trade returns
  profit_factor     — gross_wins / gross_losses
  max_drawdown_pct  — peak-to-trough equity curve drawdown
  drift_from_backtest — whether live metrics diverge materially from backtest
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum performance thresholds for live-trading approval
MIN_WIN_RATE = 0.45
MIN_PROFIT_FACTOR = 1.0
MIN_SHARPE = 0.3
MAX_DRAWDOWN_PCT = 10.0
MAX_DRIFT_PCT = 15.0    # live win rate can lag backtest by at most 15 pp
MIN_TRADES = 20


class PaperValidator:
    """
    Computes paper trading performance metrics and validates against thresholds.
    """

    def __init__(
        self,
        min_win_rate: float = MIN_WIN_RATE,
        min_profit_factor: float = MIN_PROFIT_FACTOR,
        min_sharpe: float = MIN_SHARPE,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
        max_drift_pct: float = MAX_DRIFT_PCT,
        min_trades: int = MIN_TRADES,
    ):
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.min_sharpe = min_sharpe
        self.max_drawdown_pct = max_drawdown_pct
        self.max_drift_pct = max_drift_pct
        self.min_trades = min_trades

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, trades: List[Dict[str, Any]], backtest_win_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate paper trading results.

        Args:
            trades:             List of closed trade dicts with 'pnl', 'entry_price', 'closed_at'.
            backtest_win_rate:  Expected win rate from backtesting (for drift check).

        Returns dict with keys:
            ready, metrics, checks, summary
        """
        metrics = self.compute_metrics(trades)
        checks = self._run_checks(metrics, backtest_win_rate)

        blockers = [c for c in checks if not c["passed"] and not c.get("warning")]
        warnings = [c for c in checks if not c["passed"] and c.get("warning")]

        return {
            "ready": len(blockers) == 0,
            "metrics": metrics,
            "checks": checks,
            "blockers": blockers,
            "warnings": warnings,
            "summary": f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed",
        }

    def compute_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute performance statistics from a list of closed trade dicts."""
        if not trades:
            return self._zero_metrics(0)

        pnls = [float(t.get("pnl") or 0) for t in trades]
        entry_prices = [float(t.get("entry_price") or 1) for t in trades]
        returns = [p / max(e, 1e-9) for p, e in zip(pnls, entry_prices)]

        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        win_rate = wins / len(pnls)

        gross_wins = sum(p for p in pnls if p > 0)
        gross_losses = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_wins / max(gross_losses, 1e-9)

        avg_return_pct = float(np.mean(returns)) * 100
        sharpe = self._compute_sharpe(returns)
        max_dd = self._compute_max_drawdown(pnls)

        return {
            "n_trades": len(pnls),
            "win_rate": round(win_rate, 4),
            "avg_return_pct": round(avg_return_pct, 4),
            "sharpe_ratio": round(sharpe, 4),
            "profit_factor": round(profit_factor, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "gross_wins": round(gross_wins, 2),
            "gross_losses": round(gross_losses, 2),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_checks(
        self, metrics: Dict[str, Any], backtest_win_rate: Optional[float]
    ) -> List[Dict[str, Any]]:
        checks = []

        def _check(name, passed, value, detail, warning=False):
            checks.append({"name": name, "passed": passed, "value": value, "detail": detail, "warning": warning})

        n = metrics.get("n_trades", 0)
        _check(
            "min_trades", n >= self.min_trades, n,
            f"{n} closed trades (need ≥ {self.min_trades})"
        )

        wr = metrics.get("win_rate", 0.0)
        _check(
            "win_rate", wr >= self.min_win_rate, round(wr * 100, 1),
            f"Win rate {wr*100:.1f}% ({'≥' if wr >= self.min_win_rate else '<'} {self.min_win_rate*100:.0f}%)"
        )

        pf = metrics.get("profit_factor", 0.0)
        _check(
            "profit_factor", pf >= self.min_profit_factor, round(pf, 3),
            f"Profit factor {pf:.3f} (threshold ≥ {self.min_profit_factor})"
        )

        sharpe = metrics.get("sharpe_ratio", 0.0)
        _check(
            "sharpe_ratio", sharpe >= self.min_sharpe, round(sharpe, 3),
            f"Sharpe {sharpe:.3f} (threshold ≥ {self.min_sharpe})",
            warning=True,  # advisory only — Sharpe needs many trades to be meaningful
        )

        dd = metrics.get("max_drawdown_pct", 0.0)
        _check(
            "max_drawdown", dd <= self.max_drawdown_pct, round(dd, 2),
            f"Max drawdown {dd:.2f}% (limit ≤ {self.max_drawdown_pct}%)"
        )

        if backtest_win_rate is not None:
            drift = (backtest_win_rate - wr) * 100
            passed = drift <= self.max_drift_pct
            _check(
                "drift_from_backtest", passed, round(drift, 1),
                f"Live win rate lags backtest by {drift:.1f}pp (limit ≤ {self.max_drift_pct}pp)"
            )

        return checks

    @staticmethod
    def _compute_sharpe(returns: List[float]) -> float:
        if len(returns) < 5:
            return 0.0
        arr = np.array(returns)
        mu, sigma = arr.mean(), arr.std(ddof=1)
        if sigma == 0:
            return 0.0
        # Annualise: assume ~252 trading days / year, trade ~1/day
        return float(mu / sigma * np.sqrt(252))

    @staticmethod
    def _compute_max_drawdown(pnls: List[float]) -> float:
        if not pnls:
            return 0.0
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.where(peak != 0, np.abs(peak), 1)
        return float(drawdown.max())

    @staticmethod
    def _zero_metrics(n: int) -> Dict[str, Any]:
        return {
            "n_trades": n, "win_rate": 0.0, "avg_return_pct": 0.0,
            "sharpe_ratio": 0.0, "profit_factor": 0.0,
            "max_drawdown_pct": 0.0, "gross_wins": 0.0, "gross_losses": 0.0,
        }
