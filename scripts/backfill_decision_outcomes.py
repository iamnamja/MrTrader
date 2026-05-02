"""
Phase 77 — Backfill decision_audit outcomes.

Run at 16:30 ET (after market close) to populate:
  - outcome_pnl_pct: realized P&L % for trades that were entered
  - outcome_4h_pct: price change 4h after decision (uses Alpaca bar data)
  - outcome_1d_pct: price change 1 day after decision

Usage:
    python scripts/backfill_decision_outcomes.py              # last 14 days
    python scripts/backfill_decision_outcomes.py --days 30
    python scripts/backfill_decision_outcomes.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("backfill_outcomes")


def _fetch_price_change(symbol: str, from_dt: datetime, hours_forward: float) -> float | None:
    """Return price change % from from_dt over the next hours_forward hours."""
    try:
        from app.integrations import get_alpaca_client
        alpaca = get_alpaca_client()
        start = from_dt
        end = from_dt + timedelta(hours=hours_forward + 1)
        bars = alpaca.get_bars(symbol, timeframe="1H", start=start, end=end, limit=50)
        if bars is None or bars.empty or len(bars) < 2:
            return None
        open_price = float(bars.iloc[0]["open"])
        target_idx = min(int(hours_forward), len(bars) - 1)
        close_price = float(bars.iloc[target_idx]["close"])
        if open_price <= 0:
            return None
        return round((close_price - open_price) / open_price * 100, 4)
    except Exception as exc:
        logger.debug("Price fetch failed for %s: %s", symbol, exc)
        return None


def run(lookback_days: int = 14, dry_run: bool = False) -> int:
    from app.database.session import get_session
    from app.database.models import DecisionAudit, Trade

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    updated = 0

    db = get_session()
    try:
        rows = (
            db.query(DecisionAudit)
            .filter(
                DecisionAudit.decided_at >= cutoff,
            )
            .all()
        )

        logger.info("Backfilling %d decision_audit rows (lookback=%d days)", len(rows), lookback_days)

        for row in rows:
            changed = False

            # outcome_pnl_pct: realized P&L for entered trades
            if row.final_decision == "enter" and row.outcome_pnl_pct is None:
                trade = (
                    db.query(Trade)
                    .filter(
                        Trade.symbol == row.symbol,
                        Trade.created_at >= row.decided_at.replace(tzinfo=None) - timedelta(minutes=30),
                        Trade.status == "CLOSED",
                    )
                    .order_by(Trade.created_at)
                    .first()
                )
                if trade and trade.pnl is not None and trade.entry_price and trade.quantity:
                    cost_basis = trade.entry_price * trade.quantity
                    if cost_basis > 0:
                        row.outcome_pnl_pct = round(trade.pnl / cost_basis * 100, 4)
                        changed = True

            # outcome_4h_pct: 4-hour price change from decision time
            if row.outcome_4h_pct is None:
                pct = _fetch_price_change(row.symbol, row.decided_at, hours_forward=4)
                if pct is not None:
                    row.outcome_4h_pct = pct
                    changed = True

            # outcome_1d_pct: 1-day price change from decision time
            if row.outcome_1d_pct is None:
                pct = _fetch_price_change(row.symbol, row.decided_at, hours_forward=24)
                if pct is not None:
                    row.outcome_1d_pct = pct
                    changed = True

            if changed:
                updated += 1

        if not dry_run:
            db.commit()
            logger.info("Committed %d updated decision_audit rows", updated)
        else:
            db.rollback()
            logger.info("DRY RUN — would have updated %d rows (no commit)", updated)

    except Exception as exc:
        db.rollback()
        logger.error("Backfill failed: %s", exc)
    finally:
        db.close()

    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill decision_audit outcome columns")
    parser.add_argument("--days", type=int, default=14, help="Lookback days (default: 14)")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit changes")
    args = parser.parse_args()

    n = run(lookback_days=args.days, dry_run=args.dry_run)
    logger.info("Done. %d rows backfilled.", n)


if __name__ == "__main__":
    main()
