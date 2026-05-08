"""
wf_reconciliation.py — WF-6 Live vs Walk-Forward reconciliation.

Computes realized live metrics from the trades table and compares them
against walk-forward predicted metrics stored in model_versions.performance.
One WfLiveReconciliation row per (strategy, date-range) run.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.database.models import Trade, WfLiveReconciliation
from app.database.session import get_db

logger = logging.getLogger(__name__)

# Annualisation factor for daily Sharpe (252 trading days)
_TRADING_DAYS = 252


# ── Live metrics computation ──────────────────────────────────────────────────

def _annualised_sharpe(pnl_list: List[float], hold_days_list: List[float]) -> float:
    """Compute annualised Sharpe from per-trade PnL and holding periods.

    Uses trade-level daily-equivalent returns: pnl / hold_days for each trade,
    then annualises by sqrt(252).
    """
    if len(pnl_list) < 3:
        return 0.0
    daily_rets = [
        p / max(h, 1.0) for p, h in zip(pnl_list, hold_days_list)
    ]
    n = len(daily_rets)
    mean = sum(daily_rets) / n
    variance = sum((r - mean) ** 2 for r in daily_rets) / max(n - 1, 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return round((mean / std) * math.sqrt(_TRADING_DAYS), 4)


def _max_drawdown(pnl_list: List[float]) -> float:
    """Peak-to-trough drawdown as a positive percentage of cumulative high."""
    if not pnl_list:
        return 0.0
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for pnl in pnl_list:
        cum += pnl
        peak = max(peak, cum)
        dd = (peak - cum) / max(abs(peak), 1e-9)
        max_dd = max(max_dd, dd)
    return round(max_dd * 100, 4)


def _profit_factor(pnl_list: List[float]) -> float:
    gross_win = sum(p for p in pnl_list if p > 0)
    gross_loss = abs(sum(p for p in pnl_list if p < 0))
    if gross_loss == 0:
        return 0.0
    return round(gross_win / gross_loss, 4)


def compute_live_metrics(
    strategy: str,
    range_start: date,
    range_end: date,
    db: Session,
) -> Dict[str, Any]:
    """Compute realized performance metrics from closed trades.

    Returns a dict with:
        live_sharpe, live_win_rate, live_total_return_pct,
        live_max_drawdown_pct, live_trade_count, live_profit_factor,
        live_avg_hold_days, per_symbol (list of dicts)
    """
    trades = (
        db.query(Trade)
        .filter(
            Trade.trade_type == strategy,
            Trade.status == "CLOSED",
            Trade.closed_at >= datetime.combine(range_start, datetime.min.time()),
            Trade.closed_at <= datetime.combine(range_end, datetime.max.time()),
            Trade.pnl.isnot(None),
        )
        .order_by(Trade.closed_at)
        .all()
    )

    if not trades:
        return {
            "live_sharpe": None,
            "live_win_rate": None,
            "live_total_return_pct": None,
            "live_max_drawdown_pct": None,
            "live_trade_count": 0,
            "live_profit_factor": None,
            "live_avg_hold_days": None,
            "per_symbol": [],
        }

    pnl_list = [t.pnl for t in trades]
    hold_days_list = [
        max(
            ((t.closed_at - t.created_at).total_seconds() / 86400) if t.closed_at and t.created_at else 1.0,
            0.5,
        )
        for t in trades
    ]

    wins = sum(1 for p in pnl_list if p > 0)
    total_return = sum(pnl_list)
    avg_entry = sum(
        t.entry_price * t.quantity for t in trades if t.entry_price and t.quantity
    ) / max(len(trades), 1)

    # Per-symbol breakdown
    sym_map: Dict[str, Dict] = {}
    for t in trades:
        s = t.symbol
        if s not in sym_map:
            sym_map[s] = {"symbol": s, "live_pnl": 0.0, "live_trades": 0, "wins": 0}
        sym_map[s]["live_pnl"] = round(sym_map[s]["live_pnl"] + (t.pnl or 0.0), 4)
        sym_map[s]["live_trades"] += 1
        if (t.pnl or 0) > 0:
            sym_map[s]["wins"] += 1

    per_symbol = []
    for s, v in sym_map.items():
        v["live_win_rate"] = round(v["wins"] / max(v["live_trades"], 1), 4)
        del v["wins"]
        per_symbol.append(v)
    per_symbol.sort(key=lambda x: x["live_pnl"], reverse=True)

    return {
        "live_sharpe": _annualised_sharpe(pnl_list, hold_days_list),
        "live_win_rate": round(wins / len(trades), 4),
        "live_total_return_pct": round(total_return / max(avg_entry, 1.0) * 100, 4),
        "live_max_drawdown_pct": _max_drawdown(pnl_list),
        "live_trade_count": len(trades),
        "live_profit_factor": _profit_factor(pnl_list),
        "live_avg_hold_days": round(sum(hold_days_list) / len(hold_days_list), 2),
        "per_symbol": per_symbol,
    }


# ── WF predicted metrics retrieval ───────────────────────────────────────────

def get_wf_predicted_metrics(
    strategy: str,
    db: Session,
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """Return (wf_sharpe, wf_win_rate, wf_trades, model_version) from latest active model.

    Reads from model_versions.performance JSON. Keys expected:
        sharpe, win_rate, total_trades  (written by train_model.py on WF pass)
    Returns (None, None, None, None) if no active model or no performance data.
    """
    from app.database.models import ModelVersion

    model_name = "intraday_selector" if strategy == "intraday" else "swing_selector"
    row = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.model_name == model_name,
            ModelVersion.status == "ACTIVE",
        )
        .order_by(ModelVersion.version.desc())
        .first()
    )
    if row is None or not row.performance:
        return None, None, None, None

    perf = row.performance
    return (
        perf.get("sharpe"),
        perf.get("win_rate"),
        perf.get("total_trades"),
        row.version,
    )


# ── Main reconciliation runner ────────────────────────────────────────────────

def run_reconciliation(
    strategy: str,
    range_start: Optional[date] = None,
    range_end: Optional[date] = None,
    trigger: str = "api",
    db: Optional[Session] = None,
) -> WfLiveReconciliation:
    """Compute and persist one WfLiveReconciliation row.

    If range_start/range_end are omitted, defaults to trailing 90 days.
    Opens its own DB session if db is not provided.
    """
    close_db = False
    if db is None:
        db = next(get_db())
        close_db = True

    if range_end is None:
        range_end = date.today()
    if range_start is None:
        range_start = range_end - timedelta(days=90)

    row = WfLiveReconciliation(
        strategy=strategy,
        range_start=range_start,
        range_end=range_end,
        trigger=trigger,
        status="pending",
    )
    db.add(row)
    db.flush()  # get id

    try:
        live = compute_live_metrics(strategy, range_start, range_end, db)
        wf_sharpe, wf_win_rate, wf_trades, model_version = get_wf_predicted_metrics(strategy, db)

        row.live_sharpe = live["live_sharpe"]
        row.live_win_rate = live["live_win_rate"]
        row.live_total_return_pct = live["live_total_return_pct"]
        row.live_max_drawdown_pct = live["live_max_drawdown_pct"]
        row.live_trade_count = live["live_trade_count"]
        row.live_profit_factor = live["live_profit_factor"]
        row.live_avg_hold_days = live["live_avg_hold_days"]
        row.per_symbol_json = live["per_symbol"]

        row.wf_predicted_sharpe = wf_sharpe
        row.wf_predicted_win_rate = wf_win_rate
        row.wf_predicted_trades = wf_trades
        row.wf_model_version = model_version

        if live["live_sharpe"] is not None and wf_sharpe is not None and wf_sharpe != 0:
            row.shortfall_sharpe = round(live["live_sharpe"] - wf_sharpe, 4)
            row.shortfall_pct = round(row.shortfall_sharpe / abs(wf_sharpe) * 100, 2)

        row.status = "complete"
        db.commit()
        db.refresh(row)
        logger.info(
            "WF-6 reconciliation complete: strategy=%s range=%s→%s "
            "live_sharpe=%.3f wf_sharpe=%s shortfall=%s",
            strategy, range_start, range_end,
            live["live_sharpe"] or 0,
            f"{wf_sharpe:.3f}" if wf_sharpe is not None else "N/A",
            f"{row.shortfall_sharpe:+.3f}" if row.shortfall_sharpe is not None else "N/A",
        )
    except Exception as exc:
        row.status = "error"
        row.error_detail = str(exc)
        db.commit()
        logger.exception("WF-6 reconciliation failed: %s", exc)
    finally:
        if close_db:
            db.close()

    return row
