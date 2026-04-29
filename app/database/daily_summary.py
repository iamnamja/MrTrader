"""
EOD daily summary writer (Gap 3).

Writes one RiskMetric row per trading day capturing:
  - swing / intraday / total P&L
  - trade count, win rate, avg hold bars
  - PM block rate by category (NIS, macro, correlation, other)
  - account value snapshot

Called by PM._run_eod_jobs() at 16:30 ET and can be run standalone.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def write_daily_summary(date_str: Optional[str] = None) -> None:
    """
    Compute and upsert a RiskMetric row for date_str (defaults to today ET).
    Idempotent — safe to call multiple times for the same date.
    """
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

    if date_str is None:
        date_str = datetime.now(ET).strftime("%Y-%m-%d")

    day_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=ET)
    day_end = day_start + timedelta(days=1)

    try:
        from app.database.session import get_session
        from app.database.models import Trade, DecisionAudit, RiskMetric

        with get_session() as db:
            # ── Closed trades for the day ─────────────────────────────────────
            trades = (
                db.query(Trade)
                .filter(
                    Trade.closed_at >= day_start,
                    Trade.closed_at < day_end,
                    Trade.status == "CLOSED",
                )
                .all()
            )

            swing_trades = [t for t in trades if (t.trade_type or "swing") == "swing"]
            intra_trades = [t for t in trades if (t.trade_type or "swing") == "intraday"]

            def _pnl(tlist):
                return sum(t.pnl for t in tlist if t.pnl is not None)

            def _win_rate(tlist):
                closed = [t for t in tlist if t.pnl is not None]
                if not closed:
                    return None
                return round(sum(1 for t in closed if t.pnl > 0) / len(closed), 4)

            def _avg_bars(tlist):
                bars = [t.bars_held for t in tlist if t.bars_held is not None]
                return round(sum(bars) / len(bars), 1) if bars else None

            swing_pnl = _pnl(swing_trades)
            intra_pnl = _pnl(intra_trades)
            total_pnl = swing_pnl + intra_pnl

            # ── Decision audit for the day ────────────────────────────────────
            decisions = (
                db.query(DecisionAudit)
                .filter(
                    DecisionAudit.decided_at >= day_start,
                    DecisionAudit.decided_at < day_end,
                )
                .all()
            )

            n_enter = sum(1 for d in decisions if d.final_decision == "enter")
            n_block = sum(1 for d in decisions if d.final_decision == "block")
            n_total = n_enter + n_block

            # Categorise blocks
            nis_blocks = sum(1 for d in decisions
                             if d.final_decision == "block"
                             and d.block_reason and "nis" in d.block_reason.lower())
            macro_blocks = sum(1 for d in decisions
                               if d.final_decision == "block"
                               and d.block_reason and "macro" in d.block_reason.lower())
            corr_blocks = sum(1 for d in decisions
                              if d.final_decision == "block"
                              and d.block_reason and "corr" in d.block_reason.lower())
            other_blocks = n_block - nis_blocks - macro_blocks - corr_blocks

            # ── Upsert RiskMetric row ─────────────────────────────────────────
            existing = db.query(RiskMetric).filter_by(date=date_str).first()
            if existing is None:
                existing = RiskMetric(date=date_str)
                db.add(existing)

            existing.daily_pnl = round(total_pnl, 4)
            existing.position_concentration = {
                "swing_pnl": round(swing_pnl, 4),
                "intraday_pnl": round(intra_pnl, 4),
                "swing_trades": len(swing_trades),
                "intraday_trades": len(intra_trades),
                "swing_win_rate": _win_rate(swing_trades),
                "intraday_win_rate": _win_rate(intra_trades),
                "swing_avg_bars_held": _avg_bars(swing_trades),
            }
            existing.sector_concentration = {
                "pm_decisions_total": n_total,
                "entries": n_enter,
                "blocks": n_block,
                "block_rate": round(n_block / max(n_total, 1), 4),
                "nis_blocks": nis_blocks,
                "macro_blocks": macro_blocks,
                "correlation_blocks": corr_blocks,
                "other_blocks": other_blocks,
            }
            existing.timestamp = datetime.now(timezone.utc)

            db.commit()
            logger.info(
                "Daily summary %s: P&L=%.2f  swing=%d intraday=%d  blocks=%d/%d",
                date_str, total_pnl, len(swing_trades), len(intra_trades), n_block, n_total,
            )

    except Exception as exc:
        logger.warning("write_daily_summary failed for %s: %s", date_str, exc)
