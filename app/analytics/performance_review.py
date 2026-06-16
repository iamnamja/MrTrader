"""
Performance review — paper trading vs backtest expectations.

Computes:
  - Period P&L, win rate, avg trade, Sharpe estimate
  - Per-signal-type breakdown
  - Drift score: how much the live results deviate from what backtest predicted
  - Alpha vs SPY benchmark
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/inf) with None.

    FastAPI serialises route returns with the stdlib json encoder, which emits
    bare `NaN`/`Infinity` tokens that are NOT valid JSON — Starlette's strict
    encoder then raises `ValueError: Out of range float values are not JSON
    compliant`, surfacing as a 500. Any non-finite metric (e.g. an SPY return
    computed off a partial bar) must therefore be scrubbed before return.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


# ── Backtest reference targets (from phase-7 backtesting results) ─────────────
# These serve as the "expected" baseline for drift detection.
_BACKTEST_TARGETS = {
    "win_rate_pct": 55.0,  # % of winning trades
    "avg_pnl_per_trade": 12.0,  # $ average P&L per closed trade
    "max_drawdown_pct": 5.0,  # % max drawdown
    "sharpe_estimate": 1.2,  # annualised Sharpe
}


def _spy_return(start: date, end: date) -> float:
    try:
        import yfinance as yf
        df = yf.download("SPY", start=start.isoformat(), end=end.isoformat(),
                         progress=False, auto_adjust=True)
        close = df["Close"]
        if hasattr(close, "columns"):  # yfinance may return a single-column DataFrame
            close = close.iloc[:, 0]
        # Drop NaN bars — the latest row is often a partial/unpublished session
        # (e.g. queried premarket), whose NaN close would otherwise poison the
        # ratio with NaN and 500 the endpoint at JSON-encode time.
        close = close.dropna()
        if len(close) >= 2:
            first = float(close.iloc[0])
            last = float(close.iloc[-1])
            if first > 0:
                ret = round(last / first * 100 - 100, 2)
                if math.isfinite(ret):
                    return ret
    except Exception:
        pass
    return 0.0


def _sharpe(pnl_list: List[float], risk_free: float = 0.05) -> Optional[float]:
    """Rough annualised Sharpe from a list of per-trade P&Ls."""
    if len(pnl_list) < 2:
        return None
    import statistics
    mean = statistics.mean(pnl_list)
    std = statistics.stdev(pnl_list)
    if std == 0:
        return None
    return round((mean - risk_free / 252) / std * (252 ** 0.5), 2)


def get_performance_review(days: int = 30) -> Dict[str, Any]:
    """
    Fetch closed trades from DB, compute live metrics, compare to backtest targets.
    Returns a dict suitable for JSON serialisation.
    """
    from app.database.session import get_session
    from app.database.models import Trade

    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    start_dt = datetime.combine(start_date, datetime.min.time())

    db = get_session()
    try:
        trades = (
            db.query(Trade)
            .filter(Trade.status == "CLOSED", Trade.closed_at >= start_dt)
            .order_by(Trade.closed_at)
            .all()
        )
    finally:
        db.close()

    pnls = [float(t.pnl) for t in trades if t.pnl is not None]
    total_trades = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = round(wins / total_trades * 100, 1) if total_trades else 0.0
    total_pnl = round(sum(pnls), 2)
    avg_pnl = round(total_pnl / total_trades, 2) if total_trades else 0.0
    sharpe = _sharpe(pnls)

    # Per-strategy breakdown — key by STRATEGY SOURCE (selector: pead / quality_short /
    # factor_portfolio / trend / ...), falling back to the signal MECHANISM (signal_type) for
    # legacy untagged trades. Previously grouped only by signal_type, so PEAD (signal_type=
    # 'ML_RANK') was merged into ML_RANK and never showed as its own strategy (matches the
    # Analytics Signal-Attribution fix, 2026-06-16 UI audit).
    by_signal: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        sig = (getattr(t, "selector", None) or t.signal_type or "UNKNOWN")
        grp = by_signal.setdefault(sig, {"trades": 0, "wins": 0, "total_pnl": 0.0})
        grp["trades"] += 1
        pnl = float(t.pnl) if t.pnl is not None else 0.0
        grp["total_pnl"] = round(grp["total_pnl"] + pnl, 2)
        if pnl > 0:
            grp["wins"] += 1
    for grp in by_signal.values():
        grp["win_rate"] = round(grp["wins"] / grp["trades"] * 100, 1) if grp["trades"] else 0.0
        grp["avg_pnl"] = round(grp["total_pnl"] / grp["trades"], 2) if grp["trades"] else 0.0

    spy_ret = _spy_return(start_date, end_date)

    # Drift scoring — how far live metrics are from backtest targets
    drift_items: List[Dict[str, Any]] = []

    def _drift(metric: str, live: float, target: float, higher_is_better: bool = True) -> None:
        delta = live - target
        pct_diff = round(delta / abs(target) * 100, 1) if target != 0 else 0.0
        if higher_is_better:
            status = "ok" if delta >= 0 else ("warn" if delta > -target * 0.15 else "alert")
        else:
            status = "ok" if delta <= 0 else ("warn" if delta < target * 0.15 else "alert")
        drift_items.append({
            "metric": metric,
            "live": live,
            "target": target,
            "delta": round(delta, 2),
            "pct_diff": pct_diff,
            "status": status,
        })

    _drift("Win Rate (%)", win_rate, _BACKTEST_TARGETS["win_rate_pct"])
    _drift("Avg P&L / Trade ($)", avg_pnl, _BACKTEST_TARGETS["avg_pnl_per_trade"])
    if sharpe is not None:
        _drift("Sharpe (est.)", sharpe, _BACKTEST_TARGETS["sharpe_estimate"])

    alerts = [d for d in drift_items if d["status"] == "alert"]
    warnings = [d for d in drift_items if d["status"] == "warn"]

    # _json_safe scrubs any non-finite float (NaN/inf) so the dict is always
    # JSON-serialisable regardless of which metric went non-finite upstream.
    return _json_safe({
        "period_days": days,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_trades": total_trades,
        "wins": wins,
        "win_rate_pct": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "sharpe_estimate": sharpe,
        "spy_return_pct": spy_ret,
        "alpha_pct": round(total_pnl / 20000 * 100 - spy_ret, 2) if total_trades else None,
        "by_signal": by_signal,
        "backtest_targets": _BACKTEST_TARGETS,
        "drift": drift_items,
        "alerts": len(alerts),
        "warnings": len(warnings),
        "overall_status": "alert" if alerts else ("warn" if warnings else "ok"),
    })
