"""
Drawdown analytics.

Identifies the worst loss sequences, by symbol and by signal type, so we
can tell whether drawdowns cluster around specific stocks or conditions.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.database.models import RiskMetric, Trade
from app.database.session import get_session


def get_drawdown_summary(days: int = 90) -> Dict[str, Any]:
    """
    Return drawdown analytics:
      - max_drawdown_pct: worst single-day drawdown in the window
      - worst_sequences: top-3 consecutive-loss runs with total P&L
      - by_symbol: per-symbol { trades, losses, total_pnl }
    """
    from datetime import datetime, timedelta
    since = datetime.utcnow() - timedelta(days=days)
    since_str = since.strftime("%Y-%m-%d")

    db = get_session()
    try:
        trades: List[Trade] = (
            db.query(Trade)
            .filter(Trade.status == "CLOSED", Trade.closed_at >= since)
            .order_by(Trade.closed_at)
            .all()
        )
        metrics: List[RiskMetric] = (
            db.query(RiskMetric)
            .filter(RiskMetric.date >= since_str)
            .order_by(RiskMetric.date)
            .all()
        )
    finally:
        db.close()

    # Max drawdown from daily metrics
    drawdowns = [float(m.max_drawdown or 0) * 100 for m in metrics]
    max_dd = round(max(drawdowns), 2) if drawdowns else 0.0

    # Worst consecutive-loss sequences
    sequences = _find_loss_sequences(trades)
    top_sequences = sorted(sequences, key=lambda s: s["total_pnl"])[:3]

    # Per-symbol breakdown
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        s = by_symbol.setdefault(t.symbol, {"trades": 0, "losses": 0, "total_pnl": 0.0})
        s["trades"] += 1
        pnl = float(t.pnl) if t.pnl is not None else 0.0
        s["total_pnl"] = round(s["total_pnl"] + pnl, 2)
        if pnl < 0:
            s["losses"] += 1

    return {
        "period_days": days,
        "max_drawdown_pct": max_dd,
        "worst_sequences": top_sequences,
        "by_symbol": by_symbol,
        "total_trades": len(trades),
    }


def _find_loss_sequences(trades: List[Trade]) -> List[Dict[str, Any]]:
    """Walk the sorted trade list and extract every consecutive-loss run."""
    sequences = []
    run: List[Trade] = []

    for t in trades:
        pnl = float(t.pnl) if t.pnl is not None else 0.0
        if pnl < 0:
            run.append(t)
        else:
            if len(run) >= 2:
                sequences.append({
                    "length": len(run),
                    "total_pnl": round(sum(float(x.pnl or 0) for x in run), 2),
                    "symbols": list({x.symbol for x in run}),
                    "start": run[0].closed_at.isoformat() if run[0].closed_at else None,
                    "end": run[-1].closed_at.isoformat() if run[-1].closed_at else None,
                })
            run = []

    if len(run) >= 2:
        sequences.append({
            "length": len(run),
            "total_pnl": round(sum(float(x.pnl or 0) for x in run), 2),
            "symbols": list({x.symbol for x in run}),
            "start": run[0].closed_at.isoformat() if run[0].closed_at else None,
            "end": run[-1].closed_at.isoformat() if run[-1].closed_at else None,
        })

    return sequences
