"""
Signal attribution analytics.

Breaks down performance by signal type (EMA_CROSSOVER vs RSI_DIP) so we can
see which entry condition is actually generating returns.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List

from app.database.models import Trade
from app.database.session import get_session


def get_signal_attribution(days: int = 90) -> Dict[str, Any]:
    """
    Return per-signal-type performance stats over the last `days` calendar days.

    Returns a dict like:
        {
            "EMA_CROSSOVER": { "trades": 12, "win_rate": 58.3, "avg_pnl": 42.1, "total_pnl": 505.0 },
            "RSI_DIP":       { "trades": 8,  "win_rate": 50.0, "avg_pnl": 21.3, "total_pnl": 170.4 },
            "UNKNOWN":       { ... },
        }
    """
    from datetime import datetime, timedelta
    since = datetime.utcnow() - timedelta(days=days)

    db = get_session()
    try:
        trades: List[Trade] = (
            db.query(Trade)
            .filter(
                Trade.status == "CLOSED",
                Trade.closed_at >= since,
            )
            .all()
        )
    finally:
        db.close()

    buckets: Dict[str, List[float]] = {}
    for t in trades:
        key = t.signal_type or "UNKNOWN"
        buckets.setdefault(key, []).append(float(t.pnl) if t.pnl is not None else 0.0)

    result: Dict[str, Any] = {}
    for signal, pnls in buckets.items():
        wins = sum(1 for p in pnls if p > 0)
        result[signal] = {
            "trades": len(pnls),
            "win_rate": round(wins / len(pnls) * 100, 1) if pnls else 0.0,
            "avg_pnl": round(statistics.mean(pnls), 2) if pnls else 0.0,
            "total_pnl": round(sum(pnls), 2),
            "best_trade": round(max(pnls), 2) if pnls else 0.0,
            "worst_trade": round(min(pnls), 2) if pnls else 0.0,
        }
    return result
