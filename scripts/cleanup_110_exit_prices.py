"""
One-off cleanup: nullify corrupted $110 exit_price records from ghost-trade reconciler.

Root cause: _lookup_close_fill matched stale Alpaca order history (NVDA, SPGI, etc.
at ~$110 in late 2022) because there was no date filter or price sanity check.

Fix applied: startup_reconciler.py now has 30-day date filter, 5% qty tolerance,
and 30% price-sanity guard. This script repairs the historical damage.

Targets:
  exit_reason = 'reconcile_ghost_expired'
  exit_price BETWEEN 109 AND 111

Action:
  exit_price = NULL
  pnl = NULL
  exit_reason = 'reconcile_ghost_data_corrupt'
  status = CLOSED (preserved — audit trail)
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database.session import get_session
from app.database.models import Trade

DRY_RUN = "--apply" not in sys.argv


def main():
    db = get_session()
    try:
        candidates = (
            db.query(Trade)
            .filter(
                Trade.exit_reason == "reconcile_ghost_expired",
                Trade.exit_price.isnot(None),
                Trade.exit_price.between(109.0, 111.0),
            )
            .all()
        )

        log.info("Found %d corrupted records (exit_price ≈ $110)", len(candidates))

        if not candidates:
            log.info("Nothing to clean up.")
            return

        for t in candidates:
            log.info(
                "  Trade id=%s symbol=%s entry=$%.2f exit=$%.2f pnl=%s",
                t.id, t.symbol, t.entry_price or 0, t.exit_price or 0, t.pnl,
            )

        if DRY_RUN:
            log.info("DRY RUN — pass --apply to commit changes")
            return

        for t in candidates:
            t.exit_price = None
            t.pnl = None
            t.exit_reason = "reconcile_ghost_data_corrupt"

        db.commit()
        log.info("Cleaned %d records — exit_price/pnl nulled, exit_reason='reconcile_ghost_data_corrupt'", len(candidates))

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
